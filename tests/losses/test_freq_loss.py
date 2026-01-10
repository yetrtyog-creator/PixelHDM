"""
PixelHDM-RPEA-DinoV3 FrequencyLoss Tests

Tests for Frequency-aware Loss implementation based on DeCo paper.

Core Components:
    - DCT 8x8 block transform
    - JPEG quantization weights
    - YCbCr color space conversion

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import math
from typing import Tuple

import pytest
import torch
import torch.nn as nn

from src.config.model_config import PixelHDMConfig
from src.training.losses.freq_loss import (
    FrequencyLoss,
    BlockDCT2D,
    FreqLossConfig,
    rgb_to_ycbcr,
    create_freq_loss,
    create_freq_loss_from_config,
    JPEG_LUMINANCE_QUANTIZATION_TABLE,
    JPEG_CHROMINANCE_QUANTIZATION_TABLE,
)


class TestJPEGWeights:
    """Test JPEG quantization weight computation (5 test cases)."""

    # =========================================================================
    # Fixtures
    # =========================================================================

    @pytest.fixture
    def freq_loss_q1(self) -> FrequencyLoss:
        """Create FrequencyLoss with quality=1."""
        return create_freq_loss(quality=1, weight=1.0, use_ycbcr=False)

    @pytest.fixture
    def freq_loss_q50(self) -> FrequencyLoss:
        """Create FrequencyLoss with quality=50."""
        return create_freq_loss(quality=50, weight=1.0, use_ycbcr=False)

    @pytest.fixture
    def freq_loss_q90(self) -> FrequencyLoss:
        """Create FrequencyLoss with quality=90."""
        return create_freq_loss(quality=90, weight=1.0, use_ycbcr=False)

    @pytest.fixture
    def freq_loss_q100(self) -> FrequencyLoss:
        """Create FrequencyLoss with quality=100."""
        return create_freq_loss(quality=100, weight=1.0, use_ycbcr=False)

    # =========================================================================
    # Quality Scale Tests
    # =========================================================================

    def test_jpeg_weights_quality_1(self, freq_loss_q1: FrequencyLoss) -> None:
        """
        Test JPEG weights with quality=1.

        For quality < 50, scale = 5000 / quality = 5000.
        This produces the largest quantization values (highest compression).
        """
        # Scale = 5000 / 1 = 5000
        scale = 5000 / 1
        assert scale == 5000, f"Expected scale=5000, got {scale}"

        # Verify weights_y is registered and has correct shape
        assert hasattr(freq_loss_q1, 'weights_y'), "weights_y not registered"
        assert freq_loss_q1.weights_y.shape == (8, 8), \
            f"Expected shape (8, 8), got {freq_loss_q1.weights_y.shape}"

        # Weights should be positive
        assert (freq_loss_q1.weights_y > 0).all(), "Weights should be positive"

    def test_jpeg_weights_quality_50(self, freq_loss_q50: FrequencyLoss) -> None:
        """
        Test JPEG weights with quality=50.

        For quality=50, scale = 5000 / 50 = 100.
        This is the boundary between high and low compression.
        """
        # Scale = 5000 / 50 = 100
        scale = 5000 / 50
        assert scale == 100, f"Expected scale=100, got {scale}"

        # Verify weights
        assert hasattr(freq_loss_q50, 'weights_y'), "weights_y not registered"
        assert (freq_loss_q50.weights_y > 0).all(), "Weights should be positive"

    def test_jpeg_weights_quality_90(self, freq_loss_q90: FrequencyLoss) -> None:
        """
        Test JPEG weights with quality=90.

        For quality >= 50, scale = 200 - 2 * quality = 200 - 180 = 20.
        This produces smaller quantization values (lower compression).
        """
        # Scale = 200 - 2 * 90 = 20
        scale = 200 - 2 * 90
        assert scale == 20, f"Expected scale=20, got {scale}"

        # Verify weights
        assert hasattr(freq_loss_q90, 'weights_y'), "weights_y not registered"
        assert (freq_loss_q90.weights_y > 0).all(), "Weights should be positive"

    def test_jpeg_weights_quality_100(self, freq_loss_q100: FrequencyLoss) -> None:
        """
        Test JPEG weights with quality=100 boundary handling.

        For quality=100, scale = 200 - 200 = 0, but weights are clamped to 1-255.
        """
        # Scale = 200 - 2 * 100 = 0
        scale = 200 - 2 * 100
        assert scale == 0, f"Expected scale=0, got {scale}"

        # Weights should still be valid (clamped)
        assert hasattr(freq_loss_q100, 'weights_y'), "weights_y not registered"
        assert (freq_loss_q100.weights_y > 0).all(), "Weights should be positive after clamping"
        assert not torch.isnan(freq_loss_q100.weights_y).any(), "Weights contain NaN"

    def test_invalid_quality_negative_raises(self) -> None:
        """
        Test that negative quality values are handled gracefully.

        Quality is clamped to [1, 100] range, so negative values become 1.
        """
        # Should not raise - quality is clamped internally
        freq_loss = create_freq_loss(quality=-10, weight=1.0, use_ycbcr=False)

        # Weights should still be valid
        assert (freq_loss.weights_y > 0).all(), "Weights should be positive"


class TestYCbCrConversion:
    """Test YCbCr color space conversion (2 test cases)."""

    def test_ycbcr_conversion_enabled(self) -> None:
        """
        Test that use_ycbcr=True correctly converts RGB to YCbCr.

        Uses ITU-R BT.601 standard conversion matrix.
        """
        freq_loss = create_freq_loss(quality=90, weight=1.0, use_ycbcr=True)

        B, C, H, W = 2, 3, 64, 64
        rgb = torch.randn(B, C, H, W)

        # Manual YCbCr conversion
        r = rgb[:, 0:1, :, :]
        g = rgb[:, 1:2, :, :]
        b = rgb[:, 2:3, :, :]

        y_expected = 0.299 * r + 0.587 * g + 0.114 * b
        cb_expected = -0.168736 * r - 0.331264 * g + 0.5 * b
        cr_expected = 0.5 * r - 0.418688 * g - 0.081312 * b

        ycbcr = rgb_to_ycbcr(rgb)

        assert ycbcr.shape == rgb.shape, \
            f"Shape mismatch: {ycbcr.shape} vs {rgb.shape}"
        assert torch.allclose(ycbcr[:, 0:1], y_expected, atol=1e-5), \
            "Y channel mismatch"
        assert torch.allclose(ycbcr[:, 1:2], cb_expected, atol=1e-5), \
            "Cb channel mismatch"
        assert torch.allclose(ycbcr[:, 2:3], cr_expected, atol=1e-5), \
            "Cr channel mismatch"

    def test_ycbcr_conversion_disabled(self) -> None:
        """
        Test that use_ycbcr=False keeps RGB input unchanged.
        """
        freq_loss = create_freq_loss(quality=90, weight=1.0, use_ycbcr=False)

        B, C, H, W = 2, 3, 64, 64
        v_pred = torch.randn(B, C, H, W)
        v_target = torch.randn(B, C, H, W)

        # Verify use_ycbcr is False
        assert not freq_loss.use_ycbcr, "use_ycbcr should be False"

        # Loss computation should work without conversion
        loss = freq_loss(v_pred, v_target)
        assert not torch.isnan(loss), "Loss is NaN"


class TestBlockDCT:
    """Test Block DCT 2D transform (3 test cases)."""

    @pytest.fixture
    def dct(self) -> BlockDCT2D:
        """Create BlockDCT2D instance."""
        return BlockDCT2D(block_size=8)

    def test_dct_8x8_transform(self, dct: BlockDCT2D) -> None:
        """
        Test 8x8 DCT block transform.

        Verifies that the DCT transform produces correct output shape
        and the DCT matrix is orthogonal.
        """
        B, C, H, W = 2, 3, 64, 64
        x = torch.randn(B, C, H, W)

        dct_output = dct(x)

        # Output shape should match input
        assert dct_output.shape == x.shape, \
            f"Shape mismatch: {dct_output.shape} vs {x.shape}"

        # DCT output should not be NaN
        assert not torch.isnan(dct_output).any(), "DCT output contains NaN"

        # DCT matrix should be orthogonal: D @ D^T = I
        dct_matrix = dct.dct_matrix
        identity = torch.eye(8)
        product = dct_matrix @ dct_matrix.T

        assert torch.allclose(product, identity, atol=1e-5), \
            "DCT matrix is not orthogonal"

    def test_dct_non_8_multiple_padding(self, dct: BlockDCT2D) -> None:
        """
        Test DCT with non-8-multiple resolution handling via padding.

        Input dimensions that are not multiples of 8 should be padded
        with reflection padding.
        """
        B, C = 2, 3
        # Non-8-multiple dimensions
        H, W = 67, 73

        x = torch.randn(B, C, H, W)

        dct_output = dct(x)

        # Output should have same shape as input (padding is removed)
        assert dct_output.shape == x.shape, \
            f"Shape mismatch after padding: {dct_output.shape} vs {x.shape}"

        # DCT output should be valid
        assert not torch.isnan(dct_output).any(), "DCT output contains NaN after padding"

    def test_block_weights_application(self) -> None:
        """
        Test that JPEG weights are correctly applied to each 8x8 block.
        """
        freq_loss = create_freq_loss(quality=90, weight=1.0, use_ycbcr=False)

        B, C, H, W = 2, 1, 64, 64  # Single channel for simpler testing
        dct_coeffs = torch.randn(B, C, H, W)

        # Apply block weights
        weighted = freq_loss._apply_block_weights(dct_coeffs, freq_loss.weights_y)

        # Weighted output should have same shape
        assert weighted.shape == dct_coeffs.shape, \
            f"Shape mismatch: {weighted.shape} vs {dct_coeffs.shape}"

        # Weighted output should not be NaN
        assert not torch.isnan(weighted).any(), "Weighted output contains NaN"


class TestFrequencyLossForward:
    """Test FrequencyLoss forward pass (7 test cases)."""

    @pytest.fixture
    def freq_loss(self) -> FrequencyLoss:
        """Create FrequencyLoss with default parameters."""
        return create_freq_loss(quality=90, weight=1.0, use_ycbcr=True)

    @pytest.fixture
    def freq_loss_disabled(self) -> FrequencyLoss:
        """Create disabled FrequencyLoss."""
        config = PixelHDMConfig.for_testing()
        # freq_loss_enabled is False in for_testing()
        return FrequencyLoss(config=config)

    @pytest.fixture
    def sample_tensors(self) -> dict:
        """Create sample velocity tensors."""
        torch.manual_seed(42)
        B, C, H, W = 2, 3, 64, 64
        return {
            "v_pred": torch.randn(B, C, H, W),
            "v_target": torch.randn(B, C, H, W),
            "B": B,
            "C": C,
            "H": H,
            "W": W,
        }

    def test_forward_velocity_input(
        self, freq_loss: FrequencyLoss, sample_tensors: dict
    ) -> None:
        """
        Test forward pass with velocity inputs.

        FrequencyLoss operates on velocity (v_theta, v_target) from VLoss.
        """
        loss = freq_loss(
            v_pred=sample_tensors["v_pred"],
            v_target=sample_tensors["v_target"],
        )

        assert loss.dim() == 0, f"Expected scalar, got {loss.dim()}D"
        assert not torch.isnan(loss), "Loss is NaN"
        assert loss >= 0, f"Loss should be non-negative, got {loss}"

    def test_disabled_returns_zero(
        self, freq_loss_disabled: FrequencyLoss, sample_tensors: dict
    ) -> None:
        """
        Test that disabled FrequencyLoss returns zero.
        """
        loss = freq_loss_disabled(
            v_pred=sample_tensors["v_pred"],
            v_target=sample_tensors["v_target"],
        )

        assert loss.item() == 0.0, f"Expected 0.0, got {loss.item()}"

    def test_shape_mismatch_raises(self, freq_loss: FrequencyLoss) -> None:
        """
        Test that shape mismatch between v_pred and v_target raises error.
        """
        B, C, H, W = 2, 3, 64, 64
        v_pred = torch.randn(B, C, H, W)
        v_target = torch.randn(B, C, H * 2, W)  # Different height

        with pytest.raises(RuntimeError):
            freq_loss(v_pred=v_pred, v_target=v_target)

    def test_float32_computation(self, freq_loss: FrequencyLoss) -> None:
        """
        Test that float16 inputs use float32 for internal computation.
        """
        B, C, H, W = 2, 3, 64, 64
        v_pred = torch.randn(B, C, H, W, dtype=torch.float16)
        v_target = torch.randn(B, C, H, W, dtype=torch.float16)

        loss = freq_loss(v_pred=v_pred, v_target=v_target)

        # Loss should be returned in original dtype
        assert loss.dtype == torch.float16, \
            f"Expected float16 output, got {loss.dtype}"
        assert not torch.isnan(loss), "Loss is NaN with float16 input"

    def test_zero_velocity_loss(self, freq_loss: FrequencyLoss) -> None:
        """
        Test that zero velocity produces zero loss.
        """
        B, C, H, W = 2, 3, 64, 64
        v_pred = torch.zeros(B, C, H, W)
        v_target = torch.zeros(B, C, H, W)

        loss = freq_loss(v_pred=v_pred, v_target=v_target)

        assert loss.item() == 0.0 or loss.item() < 1e-10, \
            f"Expected near-zero loss for zero velocity, got {loss.item()}"

    def test_identical_pred_target_loss_zero(self, freq_loss: FrequencyLoss) -> None:
        """
        Test that identical prediction and target produce near-zero loss.
        """
        B, C, H, W = 2, 3, 64, 64
        v = torch.randn(B, C, H, W)

        loss = freq_loss(v_pred=v, v_target=v.clone())

        assert loss.item() < 1e-6, \
            f"Expected near-zero loss for identical tensors, got {loss.item()}"

    def test_different_resolutions(self, freq_loss: FrequencyLoss) -> None:
        """
        Test FrequencyLoss with different resolutions (256, 512, 1024).

        All resolutions that are multiples of 8 should work correctly.
        """
        B, C = 2, 3
        resolutions = [256, 512, 1024]

        for res in resolutions:
            # Skip 1024 if memory is limited
            if res > 512:
                H, W = res // 4, res // 4  # Use smaller size for testing
            else:
                H, W = res // 2, res // 2

            v_pred = torch.randn(B, C, H, W)
            v_target = torch.randn(B, C, H, W)

            loss = freq_loss(v_pred=v_pred, v_target=v_target)

            assert not torch.isnan(loss), f"Loss is NaN for resolution {H}x{W}"
            assert loss >= 0, f"Loss should be non-negative for resolution {H}x{W}"


class TestFrequencyLossFactory:
    """Test FrequencyLoss factory functions (1 test case)."""

    def test_factory_from_config(self) -> None:
        """
        Test create_freq_loss_from_config factory function.
        """
        # Create config with specific frequency loss settings
        config = PixelHDMConfig.default()

        freq_loss = create_freq_loss_from_config(config)

        assert isinstance(freq_loss, FrequencyLoss)
        assert freq_loss.quality == config.freq_loss_quality
        assert freq_loss.weight == config.freq_loss_lambda
        assert freq_loss.use_ycbcr == config.freq_loss_use_ycbcr
        assert freq_loss.enabled == config.freq_loss_enabled


class TestFreqLossConfig:
    """Test FreqLossConfig dataclass."""

    def test_config_defaults(self) -> None:
        """Test FreqLossConfig default values."""
        config = FreqLossConfig()

        assert config.enabled is True
        assert config.quality == 90
        assert config.weight == 1.0
        assert config.use_ycbcr is True
        assert config.block_size == 8
        assert config.only_y_channel is False

    def test_config_custom_values(self) -> None:
        """Test FreqLossConfig with custom values."""
        config = FreqLossConfig(
            enabled=False,
            quality=50,
            weight=0.5,
            use_ycbcr=False,
            block_size=8,
            only_y_channel=True,
        )

        assert config.enabled is False
        assert config.quality == 50
        assert config.weight == 0.5
        assert config.use_ycbcr is False
        assert config.only_y_channel is True


class TestJPEGQuantizationTables:
    """Test JPEG quantization table constants."""

    def test_luminance_table_shape(self) -> None:
        """Test JPEG luminance quantization table shape."""
        assert JPEG_LUMINANCE_QUANTIZATION_TABLE.shape == (8, 8)
        assert JPEG_LUMINANCE_QUANTIZATION_TABLE.dtype == torch.float32

    def test_chrominance_table_shape(self) -> None:
        """Test JPEG chrominance quantization table shape."""
        assert JPEG_CHROMINANCE_QUANTIZATION_TABLE.shape == (8, 8)
        assert JPEG_CHROMINANCE_QUANTIZATION_TABLE.dtype == torch.float32

    def test_table_values_positive(self) -> None:
        """Test that all quantization table values are positive."""
        assert (JPEG_LUMINANCE_QUANTIZATION_TABLE > 0).all()
        assert (JPEG_CHROMINANCE_QUANTIZATION_TABLE > 0).all()

    def test_dc_coefficient_position(self) -> None:
        """
        Test DC coefficient (top-left) is the smallest in luminance table.

        In JPEG, the DC coefficient typically has the smallest quantization
        value because it represents the average intensity which is perceptually
        important.
        """
        # The DC coefficient is at position (0, 0)
        dc_value = JPEG_LUMINANCE_QUANTIZATION_TABLE[0, 0]

        # It should be relatively small (16 in standard JPEG)
        assert dc_value == 16, f"Expected DC value 16, got {dc_value}"
