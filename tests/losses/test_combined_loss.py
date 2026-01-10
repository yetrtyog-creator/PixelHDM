"""
PixelHDM-RPEA-DinoV3 Combined Loss Tests

Tests for CombinedLoss and CombinedLossSimple.

Triple Loss System:
    L = L_vloss + lambda_freq * L_freq + lambda_repa * L_REPA

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestCombinedLoss:
    """Combined Loss test suite (14 test cases)."""

    # =========================================================================
    # Fixtures
    # =========================================================================

    @pytest.fixture
    def config(self):
        """Create test configuration with all losses enabled."""
        from src.config import PixelHDMConfig
        config = PixelHDMConfig.for_testing()
        # Enable all losses for testing
        config.repa_enabled = True
        config.freq_loss_enabled = True
        config.repa_lambda = 0.5
        config.freq_loss_lambda = 1.0
        config.freq_loss_quality = 90
        config.repa_early_stop = 250000
        config.repa_hidden_size = 768
        return config

    @pytest.fixture
    def config_no_freq(self, config):
        """Create config with freq loss disabled."""
        config.freq_loss_enabled = False
        return config

    @pytest.fixture
    def mock_dino_encoder(self):
        """
        Mock DINOv3 encoder.

        Returns a callable mock that produces (B, L, 768) features.
        """
        encoder = Mock()
        def side_effect(x):
            B = x.shape[0]
            L = 256
            return torch.randn(B, L, 768, device=x.device, dtype=x.dtype)
        encoder.side_effect = side_effect
        return encoder

    @pytest.fixture
    def combined_loss(self, config):
        """Create Combined Loss."""
        from src.training.losses.combined_loss import CombinedLoss
        return CombinedLoss(config=config)

    @pytest.fixture
    def combined_loss_with_mock(self, config, mock_dino_encoder):
        """Create Combined Loss with mock DINOv3 encoder."""
        from src.training.losses.combined_loss import CombinedLoss
        loss = CombinedLoss(config=config)
        loss.set_dino_encoder(mock_dino_encoder)
        return loss

    @pytest.fixture
    def sample_inputs(self):
        """Create sample inputs for testing (V-Prediction)."""
        B, C, H, W = 2, 3, 64, 64
        return {
            "v_pred": torch.randn(B, C, H, W),  # V-Prediction: velocity
            "x_clean": torch.randn(B, C, H, W),
            "noise": torch.randn(B, C, H, W),
            "h_t": torch.randn(B, 16, 256),  # 16 patches for 64x64 with patch_size=16
        }

    @pytest.fixture
    def sample_inputs_bhwc(self):
        """Create sample inputs in BHWC format (V-Prediction)."""
        B, H, W, C = 2, 64, 64, 3
        return {
            "v_pred": torch.randn(B, H, W, C),  # V-Prediction: velocity
            "x_clean": torch.randn(B, H, W, C),
            "noise": torch.randn(B, H, W, C),
            "h_t": torch.randn(B, 16, 256),
        }

    # =========================================================================
    # Test 1-2: All Losses Computed
    # =========================================================================

    def test_all_losses_computed(self, combined_loss_with_mock, sample_inputs):
        """Test that all three losses are computed when enabled."""
        # Provide dino_features to avoid encoder requirement
        dino_features = torch.randn(2, 16, 768)

        result = combined_loss_with_mock(
            v_pred=sample_inputs["v_pred"],
            x_clean=sample_inputs["x_clean"],
            noise=sample_inputs["noise"],
            h_t=sample_inputs["h_t"],
            step=0,
            dino_features=dino_features,
        )

        assert "total" in result, "Result should contain 'total'"
        assert "vloss" in result, "Result should contain 'vloss'"
        assert "freq_loss" in result, "Result should contain 'freq_loss'"
        assert "repa_loss" in result, "Result should contain 'repa_loss'"

        # VLoss should always be non-zero
        assert result["vloss"].item() != 0.0, "VLoss should be computed"

    def test_vloss_always_computed(self, combined_loss, sample_inputs):
        """Test that VLoss is always computed regardless of other settings."""
        # Without h_t, REPA should be skipped
        result = combined_loss(
            v_pred=sample_inputs["v_pred"],
            x_clean=sample_inputs["x_clean"],
            noise=sample_inputs["noise"],
            h_t=None,  # No REPA features
            step=0,
        )

        assert torch.isfinite(result["vloss"]), "VLoss should always be finite"
        # VLoss contributes to total
        assert result["vloss"].item() != 0.0 or result["total"].item() != 0.0

    # =========================================================================
    # Test 3-4: Loss Enable/Disable
    # =========================================================================

    def test_freq_loss_respects_enabled(self, config):
        """Test that FreqLoss respects the enabled flag."""
        from src.training.losses.combined_loss import CombinedLoss

        # Create with freq loss disabled
        config.freq_loss_enabled = False
        loss_fn = CombinedLoss(config=config)

        B, C, H, W = 2, 3, 64, 64
        result = loss_fn(
            v_pred=torch.randn(B, C, H, W),
            x_clean=torch.randn(B, C, H, W),
            noise=torch.randn(B, C, H, W),
            step=0,
        )

        assert result["freq_loss"].item() == 0.0, "Freq loss should be 0 when disabled"

    def test_repa_loss_respects_early_stop(self, combined_loss, sample_inputs):
        """Test that REPA Loss respects the early stop threshold."""
        dino_features = torch.randn(2, 16, 768)

        # Test before early stop
        result_before = combined_loss(
            **sample_inputs,
            step=100,
            dino_features=dino_features,
        )

        # Test after early stop
        result_after = combined_loss(
            **sample_inputs,
            step=300000,  # > 250K
            dino_features=dino_features,
        )

        assert result_before["repa_loss"].item() != 0.0, \
            "REPA loss should be non-zero before early stop"
        assert result_after["repa_loss"].item() == 0.0, \
            "REPA loss should be zero after early stop"

    # =========================================================================
    # Test 5-6: Input Format Tests
    # =========================================================================

    def test_input_format_bhwc(self, combined_loss, sample_inputs_bhwc):
        """Test that (B, H, W, C) input format is automatically converted."""
        dino_features = torch.randn(2, 16, 768)

        result = combined_loss(
            v_pred=sample_inputs_bhwc["v_pred"],
            x_clean=sample_inputs_bhwc["x_clean"],
            noise=sample_inputs_bhwc["noise"],
            h_t=sample_inputs_bhwc["h_t"],
            step=0,
            dino_features=dino_features,
        )

        assert torch.isfinite(result["total"]), "Should handle BHWC format"
        assert torch.isfinite(result["vloss"]), "VLoss should be computed for BHWC"

    def test_input_format_bchw(self, combined_loss, sample_inputs):
        """Test that (B, C, H, W) input format works directly."""
        dino_features = torch.randn(2, 16, 768)

        result = combined_loss(
            v_pred=sample_inputs["v_pred"],
            x_clean=sample_inputs["x_clean"],
            noise=sample_inputs["noise"],
            h_t=sample_inputs["h_t"],
            step=0,
            dino_features=dino_features,
        )

        assert torch.isfinite(result["total"]), "Should handle BCHW format"
        assert torch.isfinite(result["vloss"]), "VLoss should be computed for BCHW"

    # =========================================================================
    # Test 7-8: Lambda Tests
    # =========================================================================

    def test_lambda_freq_zero_ignores(self):
        """Test that lambda_freq=0 effectively ignores freq loss."""
        from src.config import PixelHDMConfig
        from src.training.losses.combined_loss import CombinedLoss

        config = PixelHDMConfig.for_testing()
        config.freq_loss_enabled = True
        config.freq_loss_lambda = 0.0  # Zero weight

        loss_fn = CombinedLoss(config=config)

        B, C, H, W = 2, 3, 64, 64
        result = loss_fn(
            v_pred=torch.randn(B, C, H, W),
            x_clean=torch.randn(B, C, H, W),
            noise=torch.randn(B, C, H, W),
            step=0,
        )

        # Even if freq_loss is computed, it should not contribute to total
        # Note: The FrequencyLoss applies the weight internally
        # Total should be approximately equal to vloss when freq_loss weight is 0
        vloss_approx = result["vloss"].item()
        total = result["total"].item()

        # With lambda_freq=0, freq contribution should be minimal
        # Allow some numerical tolerance
        assert abs(total - vloss_approx) < 0.1 or result["freq_loss"].item() == 0.0

    def test_lambda_repa_zero_ignores(self):
        """Test that lambda_repa=0 effectively ignores REPA loss."""
        from src.config import PixelHDMConfig
        from src.training.losses.combined_loss import CombinedLoss

        config = PixelHDMConfig.for_testing()
        config.repa_enabled = True
        config.repa_lambda = 0.0  # Zero weight
        config.repa_hidden_size = 768

        loss_fn = CombinedLoss(config=config)

        B, C, H, W = 2, 3, 64, 64
        L = 16  # patches
        result = loss_fn(
            v_pred=torch.randn(B, C, H, W),
            x_clean=torch.randn(B, C, H, W),
            noise=torch.randn(B, C, H, W),
            h_t=torch.randn(B, L, 256),
            step=0,
            dino_features=torch.randn(B, L, 768),
        )

        # REPA loss should be zero due to lambda=0
        assert result["repa_loss"].item() == 0.0, "REPA loss should be 0 with lambda=0"

    # =========================================================================
    # Test 9-10: Enable/Disable REPA
    # =========================================================================

    def test_disable_repa(self, combined_loss, sample_inputs):
        """Test that disable_repa() sets REPA loss to 0."""
        dino_features = torch.randn(2, 16, 768)

        combined_loss.disable_repa()

        result = combined_loss(
            v_pred=sample_inputs["v_pred"],
            x_clean=sample_inputs["x_clean"],
            noise=sample_inputs["noise"],
            h_t=sample_inputs["h_t"],
            step=0,
            dino_features=dino_features,
        )

        assert result["repa_loss"].item() == 0.0, "REPA loss should be 0 after disable"

    def test_enable_repa(self, combined_loss, sample_inputs):
        """Test that enable_repa() restores REPA loss computation."""
        dino_features = torch.randn(2, 16, 768)

        # Disable then enable
        combined_loss.disable_repa()
        combined_loss.enable_repa()

        result = combined_loss(
            v_pred=sample_inputs["v_pred"],
            x_clean=sample_inputs["x_clean"],
            noise=sample_inputs["noise"],
            h_t=sample_inputs["h_t"],
            step=0,
            dino_features=dino_features,
        )

        # REPA should be computed (non-zero with high probability)
        assert result["repa_loss"].item() != 0.0, "REPA loss should be restored after enable"

    # =========================================================================
    # Test 11-12: Output Format Tests
    # =========================================================================

    def test_output_dict_keys(self, combined_loss, sample_inputs):
        """Test that output dict contains all required keys."""
        result = combined_loss(
            v_pred=sample_inputs["v_pred"],
            x_clean=sample_inputs["x_clean"],
            noise=sample_inputs["noise"],
            step=0,
        )

        required_keys = {"total", "vloss", "freq_loss", "repa_loss"}
        assert set(result.keys()) == required_keys, \
            f"Output keys {set(result.keys())} != expected {required_keys}"

    def test_output_values_scalar(self, combined_loss, sample_inputs):
        """Test that all output values are scalar tensors."""
        dino_features = torch.randn(2, 16, 768)

        result = combined_loss(
            v_pred=sample_inputs["v_pred"],
            x_clean=sample_inputs["x_clean"],
            noise=sample_inputs["noise"],
            h_t=sample_inputs["h_t"],
            step=0,
            dino_features=dino_features,
        )

        for key, value in result.items():
            assert isinstance(value, torch.Tensor), f"{key} should be a tensor"
            assert value.dim() == 0, f"{key} should be a scalar (0-dim), got {value.dim()}-dim"

    # =========================================================================
    # Test 13: CombinedLossSimple
    # =========================================================================

    def test_simple_variant_no_repa(self, config):
        """Test that CombinedLossSimple does not compute REPA loss."""
        from src.training.losses.combined_loss import CombinedLossSimple

        loss_fn = CombinedLossSimple(config=config)

        B, C, H, W = 2, 3, 64, 64
        result = loss_fn(
            v_pred=torch.randn(B, C, H, W),
            x_clean=torch.randn(B, C, H, W),
            noise=torch.randn(B, C, H, W),
        )

        # Simple variant should not have repa_loss key
        assert "total" in result
        assert "vloss" in result
        assert "freq_loss" in result
        assert "repa_loss" not in result, "Simple variant should not have REPA loss"

    # =========================================================================
    # Test 14: Gradient Flow
    # =========================================================================

    def test_gradient_flow_all_losses(self, config):
        """Test that gradients flow correctly through all losses."""
        from src.training.losses.combined_loss import CombinedLoss

        config.freq_loss_enabled = True
        config.repa_enabled = True
        loss_fn = CombinedLoss(config=config)

        B, C, H, W = 2, 3, 64, 64
        L = 16

        # Create inputs with requires_grad (V-Prediction)
        v_pred = torch.randn(B, C, H, W, requires_grad=True)
        x_clean = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)
        h_t = torch.randn(B, L, 256, requires_grad=True)
        dino_features = torch.randn(B, L, 768)

        result = loss_fn(
            v_pred=v_pred,
            x_clean=x_clean,
            noise=noise,
            h_t=h_t,
            step=0,
            dino_features=dino_features,
        )

        # Backward pass
        result["total"].backward()

        # Check gradients
        assert v_pred.grad is not None, "v_pred should receive gradients"
        assert not torch.all(v_pred.grad == 0), "v_pred gradients should be non-zero"

        assert h_t.grad is not None, "h_t should receive gradients (REPA)"
        assert not torch.all(h_t.grad == 0), "h_t gradients should be non-zero"


class TestCombinedLossFactories:
    """Test factory functions for Combined Loss."""

    def test_create_combined_loss(self):
        """Test create_combined_loss() factory."""
        from src.training.losses.combined_loss import create_combined_loss

        loss_fn = create_combined_loss(
            lambda_freq=0.5,
            lambda_repa=0.3,
            freq_quality=85,
            repa_early_stop=200000,
        )

        assert loss_fn.lambda_freq == 0.5
        assert loss_fn.lambda_repa == 0.3

    def test_create_combined_loss_from_config(self):
        """Test create_combined_loss_from_config() factory."""
        from src.config import PixelHDMConfig
        from src.training.losses.combined_loss import create_combined_loss_from_config

        config = PixelHDMConfig.for_testing()
        config.repa_lambda = 0.7
        config.freq_loss_lambda = 0.8

        loss_fn = create_combined_loss_from_config(config)

        assert loss_fn.lambda_freq == 0.8
        assert loss_fn.lambda_repa == 0.7

    def test_create_combined_loss_simple(self):
        """Test create_combined_loss_simple() factory."""
        from src.training.losses.combined_loss import create_combined_loss_simple

        loss_fn = create_combined_loss_simple(
            lambda_freq=0.5,
            freq_quality=85,
        )

        assert loss_fn.lambda_freq == 0.5

        # Verify it works
        B, C, H, W = 2, 3, 64, 64
        result = loss_fn(
            v_pred=torch.randn(B, C, H, W),
            x_clean=torch.randn(B, C, H, W),
            noise=torch.randn(B, C, H, W),
        )

        assert "total" in result
        assert "repa_loss" not in result


class TestCombinedLossNumericalStability:
    """Numerical stability tests for Combined Loss."""

    def test_numerical_stability(self):
        """Test numerical stability with various input conditions."""
        from src.config import PixelHDMConfig
        from src.training.losses.combined_loss import CombinedLoss

        config = PixelHDMConfig.for_testing()
        loss_fn = CombinedLoss(config=config)

        B, C, H, W = 2, 3, 64, 64

        # Test with normal inputs (V-Prediction doesn't need t)
        result = loss_fn(
            v_pred=torch.randn(B, C, H, W),
            x_clean=torch.randn(B, C, H, W),
            noise=torch.randn(B, C, H, W),
            step=0,
        )
        assert torch.isfinite(result["total"]), "Should handle normal inputs"

    def test_mixed_precision(self):
        """Test with mixed precision inputs."""
        from src.config import PixelHDMConfig
        from src.training.losses.combined_loss import CombinedLoss

        config = PixelHDMConfig.for_testing()
        loss_fn = CombinedLoss(config=config)

        B, C, H, W = 2, 3, 64, 64

        # Test with float16 inputs (V-Prediction)
        result = loss_fn(
            v_pred=torch.randn(B, C, H, W, dtype=torch.float16),
            x_clean=torch.randn(B, C, H, W, dtype=torch.float16),
            noise=torch.randn(B, C, H, W, dtype=torch.float16),
            step=0,
        )

        assert torch.isfinite(result["total"]), "Should handle float16 inputs"
