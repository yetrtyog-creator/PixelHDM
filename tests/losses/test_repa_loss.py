"""
PixelHDM-RPEA-DinoV3 iREPA Loss Tests

Tests for REPALoss (iREPA implementation) and REPALossWithProjector.

iREPA: Improved Representation Alignment Loss (arXiv:2512.10794)
- Conv2d projection (replaces MLP)
- Spatial normalization
- DINOv3 feature alignment
- Early stop mechanism (250K steps)
- Cosine Similarity

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestREPALoss:
    """REPA Loss test suite (16 test cases)."""

    # =========================================================================
    # Fixtures
    # =========================================================================

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        from src.config import PixelHDMConfig
        config = PixelHDMConfig.for_testing()
        # Override REPA settings for testing
        config.repa_enabled = True
        config.repa_lambda = 0.5
        config.repa_early_stop = 250000
        config.repa_hidden_size = 768
        return config

    @pytest.fixture
    def repa_loss(self, config):
        """Create REPA Loss without encoder."""
        from src.training.losses.repa_loss import REPALoss
        return REPALoss(config=config)

    @pytest.fixture
    def mock_dino_encoder(self):
        """
        Mock DINOv3 encoder.

        Returns a callable mock that produces (B, L, 768) features.
        """
        encoder = Mock()
        # The encoder should return (B, L, dino_dim) features
        def side_effect(x):
            # Infer batch size from input
            B = x.shape[0]
            # Assume L=256 patches for 256x256 images with patch_size=16
            L = 256
            return torch.randn(B, L, 768, device=x.device, dtype=x.dtype)
        encoder.side_effect = side_effect
        return encoder

    @pytest.fixture
    def repa_loss_with_mock(self, config, mock_dino_encoder):
        """Create REPA Loss with mock encoder."""
        from src.training.losses.repa_loss import REPALoss
        loss = REPALoss(config=config)
        loss.set_dino_encoder(mock_dino_encoder)
        return loss

    # =========================================================================
    # Test 1-3: Early Stop Tests
    # =========================================================================

    def test_early_stop_before_threshold(self, repa_loss):
        """Test that loss is computed normally before early stop threshold (step < 250K)."""
        model_features = torch.randn(2, 256, 256)  # Using config hidden_dim=256
        dino_features = torch.randn(2, 256, 768)

        loss = repa_loss(model_features, x_clean=None, step=10000, dino_features=dino_features)

        assert loss.item() != 0.0, "Loss should not be zero before early stop threshold"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_early_stop_at_threshold(self, repa_loss):
        """Test that loss is zero exactly at early stop threshold (step == 250K)."""
        model_features = torch.randn(2, 256, 256)
        dino_features = torch.randn(2, 256, 768)

        loss = repa_loss(model_features, x_clean=None, step=250000, dino_features=dino_features)

        assert loss.item() == 0.0, "Loss should be zero at early stop threshold"

    def test_early_stop_after_threshold(self, repa_loss):
        """Test that loss is zero after early stop threshold (step >> 250K)."""
        model_features = torch.randn(2, 256, 256)
        dino_features = torch.randn(2, 256, 768)

        loss = repa_loss(model_features, x_clean=None, step=500000, dino_features=dino_features)

        assert loss.item() == 0.0, "Loss should be zero after early stop threshold"

    # =========================================================================
    # Test 4-6: Encoder Tests
    # =========================================================================

    def test_no_encoder_uses_precomputed(self, repa_loss):
        """Test that precomputed dino_features are used when no encoder is set."""
        model_features = torch.randn(2, 256, 256)
        dino_features = torch.randn(2, 256, 768)

        # Should not raise error because we provide precomputed features
        loss = repa_loss(model_features, x_clean=None, step=0, dino_features=dino_features)

        assert torch.isfinite(loss), "Loss should be finite with precomputed features"

    def test_set_dino_encoder_works(self, repa_loss_with_mock):
        """Test that set_dino_encoder() allows normal operation."""
        model_features = torch.randn(2, 256, 256)
        x_clean = torch.randn(2, 3, 256, 256)  # BCHW format

        loss = repa_loss_with_mock(model_features, x_clean=x_clean, step=0)

        assert torch.isfinite(loss), "Loss should be finite after setting encoder"
        assert loss.item() != 0.0, "Loss should not be zero when computed"

    def test_precomputed_features_bypass(self, repa_loss_with_mock):
        """Test that providing dino_features bypasses encoder call."""
        model_features = torch.randn(2, 256, 256)
        dino_features = torch.randn(2, 256, 768)

        # Reset mock call count
        repa_loss_with_mock.dino_encoder.reset_mock()

        loss = repa_loss_with_mock(
            model_features, x_clean=None, step=0, dino_features=dino_features
        )

        # Encoder should not be called when precomputed features are provided
        repa_loss_with_mock.dino_encoder.assert_not_called()
        assert torch.isfinite(loss), "Loss should be finite"

    # =========================================================================
    # Test 7: Projector Shape Transformation (Conv2d)
    # =========================================================================

    def test_projector_shape_transformation(self, repa_loss, config):
        """Test that Conv2d projector transforms (B, D, H, W) to (B, D_dino, H, W)."""
        B, L = 2, 256
        H = W = int(L ** 0.5)  # 16x16

        # iREPA uses Conv2d which expects 4D input (B, D, H, W)
        model_features_2d = torch.randn(B, config.hidden_dim, H, W)

        # Apply projector directly
        projected = repa_loss.projector(model_features_2d)

        expected_shape = (B, 768, H, W)  # dino_dim
        assert projected.shape == expected_shape, \
            f"Projector output shape {projected.shape} != expected {expected_shape}"

    # =========================================================================
    # Test 8-10: Cosine Similarity Tests
    # =========================================================================

    def test_cosine_similarity_range(self, repa_loss):
        """Test that cosine similarity is in [-1, 1] range."""
        B, L = 2, 256
        model_features = torch.randn(B, L, 256)
        dino_features = torch.randn(B, L, 768)

        # Loss should be based on -cos_sim, so it should be in range [-lambda, lambda]
        loss = repa_loss(model_features, x_clean=None, step=0, dino_features=dino_features)

        # Since loss = lambda * (-cos_sim.mean()), and cos_sim in [-1, 1]
        # loss should be in [-lambda, lambda] = [-0.5, 0.5]
        assert -0.6 <= loss.item() <= 0.6, \
            f"Loss {loss.item()} should be in reasonable range based on lambda=0.5"

    def test_identical_features_loss_negative(self, repa_loss):
        """Test that identical features yield loss close to 0 (cos_sim approx 1)."""
        B, L = 2, 256
        H = W = int(L ** 0.5)

        # Create model features and project them using Conv2d
        model_features = torch.randn(B, L, 256)
        with torch.no_grad():
            # Reshape for Conv2d: (B, L, D) -> (B, D, H, W)
            h_2d = model_features.permute(0, 2, 1).reshape(B, 256, H, W)
            projected_2d = repa_loss.projector(h_2d)
            # Reshape back: (B, D_dino, H, W) -> (B, L, D_dino)
            projected = projected_2d.flatten(2).permute(0, 2, 1)

        # Use projected features as dino_features (but note: spatial_norm will be applied)
        # So we need to pre-apply spatial norm to match
        dino_features = repa_loss._spatial_normalize(projected.clone())

        loss = repa_loss(model_features, x_clean=None, step=0, dino_features=dino_features)

        # With identical features after spatial norm, cos_sim ≈ 1, loss ≈ 0
        assert abs(loss.item()) < 0.1, \
            f"Loss {loss.item()} should be close to 0 for identical features"

    def test_orthogonal_features_loss_half_lambda(self, repa_loss):
        """Test that orthogonal features yield loss close to lambda (cos_sim approx 0)."""
        B, L = 2, 256
        H = W = int(L ** 0.5)

        # Create orthogonal feature pairs
        model_features = torch.randn(B, L, 256)
        with torch.no_grad():
            # Reshape for Conv2d: (B, L, D) -> (B, D, H, W)
            h_2d = model_features.permute(0, 2, 1).reshape(B, 256, H, W)
            projected_2d = repa_loss.projector(h_2d)
            # Reshape back: (B, D_dino, H, W) -> (B, L, D_dino)
            projected = projected_2d.flatten(2).permute(0, 2, 1)

        # Create orthogonal dino features
        # For each token, create a vector orthogonal to the projected vector
        dino_features = torch.randn(B, L, 768)
        # Make approximately orthogonal by subtracting projection onto projected
        proj_norm = torch.nn.functional.normalize(projected, dim=-1)
        dino_norm = torch.nn.functional.normalize(dino_features, dim=-1)
        dot = (proj_norm * dino_norm).sum(dim=-1, keepdim=True)
        dino_features = dino_features - dot * projected
        dino_features = torch.nn.functional.normalize(dino_features, dim=-1) * projected.norm(dim=-1, keepdim=True)

        loss = repa_loss(model_features, x_clean=None, step=0, dino_features=dino_features)

        # With orthogonal features, cos_sim approx 0, loss = (1 - 0) * lambda = 0.5
        assert loss.item() >= 0, "Loss should be non-negative"
        assert abs(loss.item() - 0.5) < 0.3, \
            f"Loss {loss.item()} should be close to 0.5 for orthogonal features"

    # =========================================================================
    # Test 11-12: Feature Interpolation Tests
    # =========================================================================

    def test_feature_interpolation_same_length(self, repa_loss):
        """Test that no interpolation occurs when sequence lengths match."""
        B, L = 2, 256
        model_features = torch.randn(B, L, 256)
        dino_features = torch.randn(B, L, 768)  # Same L

        # Should work without interpolation
        loss = repa_loss(model_features, x_clean=None, step=0, dino_features=dino_features)

        assert torch.isfinite(loss), "Loss should be finite"

    def test_feature_interpolation_different_length(self, repa_loss):
        """Test that interpolation works when sequence lengths differ."""
        B = 2
        model_L = 256
        dino_L = 196  # Different length (e.g., 14x14 vs 16x16)

        model_features = torch.randn(B, model_L, 256)
        dino_features = torch.randn(B, dino_L, 768)

        # Should work with interpolation
        loss = repa_loss(model_features, x_clean=None, step=0, dino_features=dino_features)

        assert torch.isfinite(loss), "Loss should be finite after interpolation"

    # =========================================================================
    # Test 13: REPALossWithProjector
    # =========================================================================

    def test_with_projector_returns_dict(self, config):
        """Test that REPALossWithProjector.forward_with_features() returns dict."""
        from src.training.losses.repa_loss import REPALossWithProjector

        loss_fn = REPALossWithProjector(config=config)
        model_features = torch.randn(2, 256, 256)
        dino_features = torch.randn(2, 256, 768)

        result = loss_fn.forward_with_features(
            model_features, x_clean=None, step=0, dino_features=dino_features
        )

        assert isinstance(result, dict), "Result should be a dictionary"
        assert "loss" in result, "Result should contain 'loss' key"
        assert "h_proj" in result, "Result should contain 'h_proj' key"
        assert "dino_features" in result, "Result should contain 'dino_features' key"
        assert "cos_sim" in result, "Result should contain 'cos_sim' key"

        # Verify shapes
        assert result["h_proj"].shape == (2, 256, 768), "h_proj should be (B, L, dino_dim)"
        assert result["cos_sim"].shape == (2, 256), "cos_sim should be (B, L)"

    # =========================================================================
    # Test 14-15: Batch and Sequence Size Tests
    # =========================================================================

    def test_batch_sizes(self, config):
        """Test REPA Loss with different batch sizes (B=1, B=32)."""
        from src.training.losses.repa_loss import REPALoss

        loss_fn = REPALoss(config=config)
        L = 256

        for B in [1, 32]:
            model_features = torch.randn(B, L, 256)
            dino_features = torch.randn(B, L, 768)

            loss = loss_fn(model_features, x_clean=None, step=0, dino_features=dino_features)

            assert torch.isfinite(loss), f"Loss should be finite for batch_size={B}"
            assert loss.dim() == 0, f"Loss should be scalar for batch_size={B}"

    def test_sequence_lengths(self, config):
        """Test REPA Loss with different sequence lengths (L=256, L=1024)."""
        from src.training.losses.repa_loss import REPALoss

        loss_fn = REPALoss(config=config)
        B = 2

        for L in [256, 1024]:
            model_features = torch.randn(B, L, 256)
            dino_features = torch.randn(B, L, 768)

            loss = loss_fn(model_features, x_clean=None, step=0, dino_features=dino_features)

            assert torch.isfinite(loss), f"Loss should be finite for seq_len={L}"
            assert loss.dim() == 0, f"Loss should be scalar for seq_len={L}"

    # =========================================================================
    # Test 16: Factory Function
    # =========================================================================

    def test_factory_from_config(self, config):
        """Test create_repa_loss_from_config() factory function."""
        from src.training.losses.repa_loss import create_repa_loss_from_config

        loss_fn = create_repa_loss_from_config(config)

        # Verify configuration was applied
        assert loss_fn.hidden_dim == config.hidden_dim
        assert loss_fn.dino_dim == config.repa_hidden_size
        assert loss_fn.lambda_repa == config.repa_lambda
        assert loss_fn.early_stop_step == config.repa_early_stop

        # Verify it works
        model_features = torch.randn(2, 256, 256)
        dino_features = torch.randn(2, 256, 768)
        loss = loss_fn(model_features, x_clean=None, step=0, dino_features=dino_features)

        assert torch.isfinite(loss), "Factory-created loss should work"


class TestREPALossEdgeCases:
    """Edge case tests for REPA Loss."""

    def test_no_encoder_no_features_raises(self):
        """Test that calling without encoder or precomputed features raises RuntimeError."""
        from src.config import PixelHDMConfig
        from src.training.losses.repa_loss import REPALoss

        config = PixelHDMConfig.for_testing()
        config.repa_enabled = True
        loss_fn = REPALoss(config=config)

        model_features = torch.randn(2, 256, 256)
        x_clean = torch.randn(2, 3, 256, 256)

        with pytest.raises(RuntimeError, match="iREPA Loss"):
            loss_fn(model_features, x_clean=x_clean, step=0)

    def test_get_projector(self):
        """Test get_projector() method returns Conv2d."""
        from src.config import PixelHDMConfig
        from src.training.losses.repa_loss import REPALoss

        config = PixelHDMConfig.for_testing()
        loss_fn = REPALoss(config=config)

        projector = loss_fn.get_projector()

        # iREPA uses Conv2d instead of Sequential MLP
        assert isinstance(projector, nn.Conv2d), "Projector should be nn.Conv2d"
        assert projector.kernel_size == (3, 3), "Kernel size should be 3x3"
        assert projector.padding == (1, 1), "Padding should be 1"

    def test_extra_repr(self):
        """Test extra_repr() method for debugging."""
        from src.config import PixelHDMConfig
        from src.training.losses.repa_loss import REPALoss

        config = PixelHDMConfig.for_testing()
        config.repa_lambda = 0.5
        config.repa_early_stop = 250000
        loss_fn = REPALoss(config=config)

        repr_str = loss_fn.extra_repr()

        assert "hidden_dim" in repr_str
        assert "dino_dim" in repr_str
        assert "lambda" in repr_str
        assert "early_stop" in repr_str


class TestREPALossGradients:
    """Gradient flow tests for REPA Loss."""

    def test_gradient_flow_through_projector(self):
        """Test that gradients flow through projector to model features."""
        from src.config import PixelHDMConfig
        from src.training.losses.repa_loss import REPALoss

        config = PixelHDMConfig.for_testing()
        loss_fn = REPALoss(config=config)

        model_features = torch.randn(2, 256, 256, requires_grad=True)
        dino_features = torch.randn(2, 256, 768)

        loss = loss_fn(model_features, x_clean=None, step=0, dino_features=dino_features)
        loss.backward()

        assert model_features.grad is not None, "Gradients should flow to model_features"
        assert not torch.all(model_features.grad == 0), "Gradients should be non-zero"

    def test_projector_weights_receive_gradients(self):
        """Test that projector weights receive gradients."""
        from src.config import PixelHDMConfig
        from src.training.losses.repa_loss import REPALoss

        config = PixelHDMConfig.for_testing()
        loss_fn = REPALoss(config=config)

        model_features = torch.randn(2, 256, 256)
        dino_features = torch.randn(2, 256, 768)

        loss = loss_fn(model_features, x_clean=None, step=0, dino_features=dino_features)
        loss.backward()

        for name, param in loss_fn.projector.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Projector {name} should have gradients"


class TestiREPASpatialNormalize:
    """Tests for iREPA spatial normalization."""

    def test_spatial_normalize_output_shape(self):
        """Test that spatial normalization preserves shape."""
        from src.config import PixelHDMConfig
        from src.training.losses.repa_loss import REPALoss

        config = PixelHDMConfig.for_testing()
        loss_fn = REPALoss(config=config)

        x = torch.randn(2, 256, 768)
        result = loss_fn._spatial_normalize(x)

        assert result.shape == x.shape, "Shape should be preserved"

    def test_spatial_normalize_zero_mean(self):
        """Test that spatial normalization produces zero mean along L dimension."""
        from src.config import PixelHDMConfig
        from src.training.losses.repa_loss import REPALoss

        config = PixelHDMConfig.for_testing()
        loss_fn = REPALoss(config=config)

        x = torch.randn(2, 256, 768)
        result = loss_fn._spatial_normalize(x)

        # Mean along L dimension should be close to 0
        mean = result.mean(dim=1)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5), \
            "Mean along L dimension should be close to 0"

    def test_spatial_normalize_unit_std(self):
        """Test that spatial normalization produces approximately unit std along L dimension."""
        from src.config import PixelHDMConfig
        from src.training.losses.repa_loss import REPALoss

        config = PixelHDMConfig.for_testing()
        loss_fn = REPALoss(config=config)

        x = torch.randn(2, 256, 768)
        result = loss_fn._spatial_normalize(x)

        # Std along L dimension should be close to 1
        std = result.std(dim=1)
        assert torch.allclose(std, torch.ones_like(std), atol=0.1), \
            "Std along L dimension should be close to 1"

    def test_spatial_normalize_enhances_contrast(self):
        """Test that spatial normalization enhances contrast between patches."""
        from src.config import PixelHDMConfig
        from src.training.losses.repa_loss import REPALoss

        config = PixelHDMConfig.for_testing()
        loss_fn = REPALoss(config=config)

        # Create features with low variance
        x = torch.randn(2, 256, 768) * 0.01 + 1.0  # Small variance around 1.0
        result = loss_fn._spatial_normalize(x)

        # After normalization, variance should be much higher
        input_var = x.var(dim=1).mean()
        output_var = result.var(dim=1).mean()

        assert output_var > input_var * 10, \
            f"Spatial normalization should enhance contrast: {output_var} vs {input_var}"
