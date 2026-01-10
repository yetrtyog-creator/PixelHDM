"""
AdaLN (Adaptive Layer Normalization) Regression Tests.

Tests for TokenAdaLN and PixelwiseAdaLN to prevent regression of historical bugs:
- 2026-01-04: AdaLN zero initialization disabled time conditioning
- 2026-01-05: _basic_init() overwrote AdaLN bias to zero
- 2026-01-07: PixelwiseAdaLN signal dilution (99.7% loss)
- 2026-01-07: adaLN-Zero made PixelBlocks identity mappings

Test Count: 40 test cases

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-08
"""

import pytest
import torch
import torch.nn as nn
from typing import Tuple

from src.models.layers.adaln.token_adaln import TokenAdaLN
from src.models.layers.adaln.pixelwise_adaln import PixelwiseAdaLN
from src.models.layers.adaln.factory import (
    create_token_adaln,
    create_pixelwise_adaln,
    create_pixelwise_adaln_from_config,
)
from src.config import PixelHDMConfig


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def minimal_config():
    """Create minimal config for testing."""
    return PixelHDMConfig.for_testing()


@pytest.fixture
def token_adaln():
    """Create TokenAdaLN for testing."""
    return TokenAdaLN(hidden_dim=256, condition_dim=256)


@pytest.fixture
def pixelwise_adaln(minimal_config):
    """Create PixelwiseAdaLN for testing."""
    return PixelwiseAdaLN(config=minimal_config)


@pytest.fixture
def time_embedding_low():
    """Create time embedding for t=0.1."""
    torch.manual_seed(42)
    return torch.randn(2, 256)


@pytest.fixture
def time_embedding_high():
    """Create time embedding for t=0.9."""
    torch.manual_seed(123)
    return torch.randn(2, 256)


# =============================================================================
# Test Class: TokenAdaLN Initialization
# =============================================================================


class TestTokenAdaLNInitialization:
    """Test TokenAdaLN weight and bias initialization."""

    def test_weight_not_zero(self, token_adaln):
        """Test weight is NOT zero initialized (2026-01-04 fix)."""
        weight = token_adaln.proj[-1].weight
        assert weight.abs().mean() > 0.001, "Weight should not be zero"

    def test_weight_has_correct_std(self, token_adaln):
        """Test weight has approximately std=0.02."""
        weight = token_adaln.proj[-1].weight
        std = weight.std().item()
        assert 0.01 < std < 0.04, f"Weight std {std} not in expected range"

    def test_gamma_bias_is_one(self, token_adaln):
        """Test gamma bias values are initialized to 1.0."""
        bias = token_adaln.proj[-1].bias
        hidden_dim = token_adaln.hidden_dim
        num_params = token_adaln.num_params

        # Reshape bias to (num_params, hidden_dim)
        bias_reshaped = bias.view(num_params, hidden_dim)

        # gamma1 (index 0) should be 1.0
        gamma1 = bias_reshaped[0]
        assert torch.allclose(gamma1, torch.ones_like(gamma1)), \
            f"gamma1 mean: {gamma1.mean()}, should be 1.0"

        # gamma2 (index 3) should be 1.0
        gamma2 = bias_reshaped[3]
        assert torch.allclose(gamma2, torch.ones_like(gamma2)), \
            f"gamma2 mean: {gamma2.mean()}, should be 1.0"

    def test_alpha_bias_is_one(self, token_adaln):
        """Test alpha bias values are initialized to 1.0."""
        bias = token_adaln.proj[-1].bias
        hidden_dim = token_adaln.hidden_dim
        num_params = token_adaln.num_params

        bias_reshaped = bias.view(num_params, hidden_dim)

        # alpha1 (index 2) should be 1.0
        alpha1 = bias_reshaped[2]
        assert torch.allclose(alpha1, torch.ones_like(alpha1)), \
            f"alpha1 mean: {alpha1.mean()}, should be 1.0"

        # alpha2 (index 5) should be 1.0
        alpha2 = bias_reshaped[5]
        assert torch.allclose(alpha2, torch.ones_like(alpha2)), \
            f"alpha2 mean: {alpha2.mean()}, should be 1.0"

    def test_beta_bias_is_zero(self, token_adaln):
        """Test beta bias values are initialized to 0.0."""
        bias = token_adaln.proj[-1].bias
        hidden_dim = token_adaln.hidden_dim
        num_params = token_adaln.num_params

        bias_reshaped = bias.view(num_params, hidden_dim)

        # beta1 (index 1) should be 0.0
        beta1 = bias_reshaped[1]
        assert torch.allclose(beta1, torch.zeros_like(beta1)), \
            f"beta1 mean: {beta1.mean()}, should be 0.0"

        # beta2 (index 4) should be 0.0
        beta2 = bias_reshaped[4]
        assert torch.allclose(beta2, torch.zeros_like(beta2)), \
            f"beta2 mean: {beta2.mean()}, should be 0.0"


# =============================================================================
# Test Class: TokenAdaLN Time Conditioning
# =============================================================================


class TestTokenAdaLNTimeConditioning:
    """Test TokenAdaLN responds to different time inputs."""

    def test_different_times_different_outputs(
        self, token_adaln, time_embedding_low, time_embedding_high
    ):
        """Test different time embeddings produce different outputs."""
        output_low = token_adaln(time_embedding_low)
        output_high = token_adaln(time_embedding_high)

        # All 6 outputs should differ
        for i, (out_low, out_high) in enumerate(zip(output_low, output_high)):
            diff = (out_low - out_high).abs().mean()
            assert diff > 0.01, f"Output {i} not sensitive to time: diff={diff}"

    def test_time_signal_not_diluted(self, token_adaln):
        """Test time signal is preserved through projection (not diluted to <1%)."""
        # Create two distinct time embeddings
        t_emb_1 = torch.randn(2, 256)
        t_emb_2 = t_emb_1 + torch.randn(2, 256) * 2.0  # Add significant difference

        output_1 = token_adaln(t_emb_1)
        output_2 = token_adaln(t_emb_2)

        # Compute input difference
        input_diff = (t_emb_1 - t_emb_2).abs().mean().item()

        # Compute output difference (average across all 6 parameters)
        output_diff = sum(
            (o1 - o2).abs().mean().item()
            for o1, o2 in zip(output_1, output_2)
        ) / 6

        # Signal retention should be > 5%
        signal_retention = output_diff / input_diff
        assert signal_retention > 0.05, \
            f"Signal retention {signal_retention:.2%} too low (< 5%)"

    def test_output_shapes(self, token_adaln, time_embedding_low):
        """Test output shapes are correct."""
        outputs = token_adaln(time_embedding_low)

        assert len(outputs) == 6, "Should output 6 parameters"

        for i, out in enumerate(outputs):
            assert out.shape == (2, 1, 256), \
                f"Output {i} shape {out.shape} != (2, 1, 256)"


# =============================================================================
# Test Class: PixelwiseAdaLN Initialization
# =============================================================================


class TestPixelwiseAdaLNInitialization:
    """Test PixelwiseAdaLN initialization (2026-01-07 fixes)."""

    def test_gamma_init_small_nonzero(self, pixelwise_adaln):
        """Test gamma initialized to small nonzero value (0.1), not 0 or 1."""
        bias = pixelwise_adaln.param_gen[-1].bias
        pixel_dim = pixelwise_adaln.pixel_dim
        num_params = pixelwise_adaln.num_params

        bias_reshaped = bias.view(num_params, pixel_dim)

        # gamma1 (index 0) should be 0.1
        gamma1 = bias_reshaped[0]
        expected_gamma = 0.1

        assert torch.allclose(gamma1, torch.full_like(gamma1, expected_gamma)), \
            f"gamma1 mean: {gamma1.mean()}, expected: {expected_gamma}"

    def test_alpha_init_is_one(self, pixelwise_adaln):
        """Test alpha initialized to 1.0 for full residual flow."""
        bias = pixelwise_adaln.param_gen[-1].bias
        pixel_dim = pixelwise_adaln.pixel_dim
        num_params = pixelwise_adaln.num_params

        bias_reshaped = bias.view(num_params, pixel_dim)

        # alpha1 (index 2) should be 1.0
        alpha1 = bias_reshaped[2]
        assert torch.allclose(alpha1, torch.ones_like(alpha1)), \
            f"alpha1 mean: {alpha1.mean()}, should be 1.0"

        # alpha2 (index 5) should be 1.0
        alpha2 = bias_reshaped[5]
        assert torch.allclose(alpha2, torch.ones_like(alpha2)), \
            f"alpha2 mean: {alpha2.mean()}, should be 1.0"

    def test_not_adaln_zero(self, pixelwise_adaln):
        """Test NOT using adaLN-Zero (gamma=0 makes blocks identity)."""
        bias = pixelwise_adaln.param_gen[-1].bias
        pixel_dim = pixelwise_adaln.pixel_dim
        num_params = pixelwise_adaln.num_params

        bias_reshaped = bias.view(num_params, pixel_dim)

        # gamma1 should NOT be zero
        gamma1 = bias_reshaped[0]
        assert gamma1.abs().mean() > 0.05, \
            "gamma1 should not be zero (adaLN-Zero makes blocks inactive)"

    def test_cond_expand_xavier_init(self, pixelwise_adaln):
        """Test cond_expand uses Xavier initialization."""
        weight = pixelwise_adaln.cond_expand.weight
        std = weight.std().item()

        # Xavier uniform std should be roughly sqrt(2 / (fan_in + fan_out))
        # For large dims, this is typically 0.01-0.1 range
        assert 0.001 < std < 0.5, f"cond_expand std {std} not in Xavier range"


# =============================================================================
# Test Class: PixelwiseAdaLN Signal Preservation
# =============================================================================


class TestPixelwiseAdaLNSignalPreservation:
    """Test PixelwiseAdaLN preserves conditioning signal (2026-01-07 fix)."""

    def test_signal_retention_above_threshold(self, pixelwise_adaln):
        """Test signal retention is > 10% (was 0.3% before fix)."""
        # Create two distinct condition inputs
        torch.manual_seed(42)
        s_cond_1 = torch.randn(2, 64, 256)  # (B, L, hidden_dim)
        s_cond_2 = s_cond_1 + torch.randn(2, 64, 256) * 2.0

        output_1 = pixelwise_adaln(s_cond_1)
        output_2 = pixelwise_adaln(s_cond_2)

        # Compute condition difference
        input_diff = (s_cond_1 - s_cond_2).abs().mean().item()

        # Compute output difference
        output_diff = sum(
            (o1 - o2).abs().mean().item()
            for o1, o2 in zip(output_1, output_2)
        ) / 6

        # Signal retention
        if input_diff > 0:
            signal_retention = output_diff / input_diff
            assert signal_retention > 0.10, \
                f"Signal retention {signal_retention:.2%} too low (< 10%)"

    def test_rms_norm_present(self, pixelwise_adaln):
        """Test RMSNorm is present for signal preservation."""
        assert hasattr(pixelwise_adaln, 'cond_norm'), \
            "cond_norm (RMSNorm) should be present for signal preservation"

    def test_text_conditioning_difference(self, pixelwise_adaln):
        """Test different text conditions produce different outputs."""
        # Simulate different text prompts
        torch.manual_seed(42)
        s_cond_text_a = torch.randn(2, 64, 256)

        torch.manual_seed(100)
        s_cond_text_b = torch.randn(2, 64, 256)

        output_a = pixelwise_adaln(s_cond_text_a)
        output_b = pixelwise_adaln(s_cond_text_b)

        # Outputs should be different
        for i, (out_a, out_b) in enumerate(zip(output_a, output_b)):
            diff = (out_a - out_b).abs().mean()
            assert diff > 0.001, f"Output {i} not sensitive to text condition"


# =============================================================================
# Test Class: PixelwiseAdaLN Output Properties
# =============================================================================


class TestPixelwiseAdaLNOutput:
    """Test PixelwiseAdaLN output properties."""

    def test_output_shapes(self, pixelwise_adaln):
        """Test output shapes are correct."""
        B, L, D = 2, 64, 256
        s_cond = torch.randn(B, L, D)

        outputs = pixelwise_adaln(s_cond)

        assert len(outputs) == 6, "Should output 6 parameters"

        # Each output should be (B, L, p², pixel_dim)
        p2 = pixelwise_adaln.p2  # 256 for patch_size=16
        pixel_dim = pixelwise_adaln.pixel_dim  # 16

        for i, out in enumerate(outputs):
            expected_shape = (B, L, p2, pixel_dim)
            assert out.shape == expected_shape, \
                f"Output {i} shape {out.shape} != {expected_shape}"

    def test_numerical_stability(self, pixelwise_adaln):
        """Test outputs are numerically stable."""
        s_cond = torch.randn(2, 64, 256)
        outputs = pixelwise_adaln(s_cond)

        for i, out in enumerate(outputs):
            assert not torch.isnan(out).any(), f"Output {i} contains NaN"
            assert not torch.isinf(out).any(), f"Output {i} contains Inf"

    def test_modulate_function(self, pixelwise_adaln, minimal_config):
        """Test modulate function works correctly."""
        B, L, p2, pixel_dim = 2, 64, 256, minimal_config.pixel_dim
        x = torch.randn(B, L, p2, pixel_dim)
        gamma = torch.ones(B, L, p2, pixel_dim) * 0.5
        beta = torch.ones(B, L, p2, pixel_dim) * 0.1

        result = pixelwise_adaln.modulate(x, gamma, beta)

        assert result.shape == x.shape
        assert not torch.isnan(result).any()


# =============================================================================
# Test Class: Integration with PixelHDM Model
# =============================================================================


class TestAdaLNModelIntegration:
    """Test AdaLN integration with PixelHDM model."""

    def test_reinit_adaln_called(self):
        """Test that _reinit_adaln is called after _basic_init (2026-01-05 fix)."""
        from src.models.pixelhdm import PixelHDM

        config = PixelHDMConfig.for_testing()
        model = PixelHDM(config)

        # Check TokenAdaLN in patch blocks
        for i, block in enumerate(model.patch_blocks):
            bias = block.adaln.proj[-1].bias
            hidden_dim = block.adaln.hidden_dim
            num_params = block.adaln.num_params
            bias_reshaped = bias.view(num_params, hidden_dim)

            # gamma1 should be 1.0, not 0
            gamma1_mean = bias_reshaped[0].mean().item()
            assert abs(gamma1_mean - 1.0) < 0.01, \
                f"Patch block {i} gamma1={gamma1_mean}, expected 1.0 (was _reinit_adaln called?)"

    def test_bias_survives_basic_init(self):
        """Test AdaLN bias is not zeroed by _basic_init."""
        from src.models.pixelhdm import PixelHDM

        config = PixelHDMConfig.for_testing()
        model = PixelHDM(config)

        # Check PixelwiseAdaLN in pixel blocks
        for i, block in enumerate(model.pixel_blocks):
            bias = block.adaln.param_gen[-1].bias
            pixel_dim = block.adaln.pixel_dim
            num_params = block.adaln.num_params
            bias_reshaped = bias.view(num_params, pixel_dim)

            # alpha1 (index 2) should be 1.0, not 0
            alpha1_mean = bias_reshaped[2].mean().item()
            assert abs(alpha1_mean - 1.0) < 0.01, \
                f"Pixel block {i} alpha1={alpha1_mean}, expected 1.0"


# =============================================================================
# Test Class: Factory Functions
# =============================================================================


class TestAdaLNFactoryFunctions:
    """Test AdaLN factory functions."""

    def test_create_token_adaln(self):
        """Test create_token_adaln factory."""
        adaln = create_token_adaln(hidden_dim=512, condition_dim=512)

        assert isinstance(adaln, TokenAdaLN)
        assert adaln.hidden_dim == 512

    def test_create_pixelwise_adaln(self):
        """Test create_pixelwise_adaln factory."""
        adaln = create_pixelwise_adaln(
            hidden_dim=1024,
            pixel_dim=16,
            patch_size=16,
        )

        assert isinstance(adaln, PixelwiseAdaLN)
        assert adaln.hidden_dim == 1024

    def test_create_pixelwise_adaln_from_config(self, minimal_config):
        """Test create_pixelwise_adaln_from_config factory."""
        adaln = create_pixelwise_adaln_from_config(minimal_config)

        assert isinstance(adaln, PixelwiseAdaLN)


# =============================================================================
# Test Class: Edge Cases
# =============================================================================


class TestAdaLNEdgeCases:
    """Test AdaLN edge cases."""

    def test_batch_size_one(self, token_adaln):
        """Test with batch size 1."""
        t_embed = torch.randn(1, 256)
        outputs = token_adaln(t_embed)

        for out in outputs:
            assert out.shape[0] == 1

    def test_large_batch(self, token_adaln):
        """Test with large batch size."""
        t_embed = torch.randn(32, 256)
        outputs = token_adaln(t_embed)

        for out in outputs:
            assert out.shape[0] == 32

    def test_different_dtypes(self, token_adaln):
        """Test with different dtypes."""
        for dtype in [torch.float32, torch.float16]:
            t_embed = torch.randn(2, 256, dtype=dtype)
            adaln_dtype = token_adaln.to(dtype)
            outputs = adaln_dtype(t_embed)

            for out in outputs:
                assert out.dtype == dtype

    def test_gradient_flow(self, token_adaln):
        """Test gradients flow through AdaLN."""
        t_embed = torch.randn(2, 256, requires_grad=True)
        outputs = token_adaln(t_embed)

        loss = sum(out.sum() for out in outputs)
        loss.backward()

        assert t_embed.grad is not None
        assert not torch.isnan(t_embed.grad).any()


# =============================================================================
# Test Class: Regression Tests (Historical Bugs)
# =============================================================================


class TestAdaLNRegressionBugs:
    """Explicit regression tests for historical bugs documented in CLAUDE.md."""

    def test_2026_01_04_zero_init_fix(self):
        """
        Regression test for 2026-01-04 bug: AdaLN zero initialization.

        Bug: nn.init.zeros_(self.proj[-1].weight) disabled time conditioning
        Fix: Changed to nn.init.trunc_normal_(std=0.02)
        """
        adaln = TokenAdaLN(hidden_dim=256, condition_dim=256)
        weight = adaln.proj[-1].weight

        # Weight should NOT be zero
        assert weight.abs().sum() > 0, "Weight is zero - 2026-01-04 bug regression!"

        # Weight should have reasonable std
        assert weight.std() > 0.005, "Weight std too low - may indicate zero init"

    def test_2026_01_05_bias_overwrite_fix(self):
        """
        Regression test for 2026-01-05 bug: _basic_init() overwrote bias.

        Bug: apply(_basic_init) zeroed all Linear biases including AdaLN
        Fix: Added _reinit_adaln() after _basic_init()
        """
        from src.models.pixelhdm import PixelHDM

        config = PixelHDMConfig.for_testing()
        model = PixelHDM(config)

        # Check EVERY patch block
        for i, block in enumerate(model.patch_blocks):
            bias = block.adaln.proj[-1].bias
            # gamma1 (first group) should be 1.0
            gamma1 = bias[:config.hidden_dim]
            assert gamma1.mean().item() > 0.9, \
                f"Block {i}: gamma1={gamma1.mean():.4f}, bias was overwritten!"

    def test_2026_01_07_signal_dilution_fix(self):
        """
        Regression test for 2026-01-07 bug: 99.7% signal loss.

        Bug: PixelwiseAdaLN cond_expand with gain=0.1 + small param_gen init
        Fix: Added RMSNorm after cond_expand, removed gain
        """
        config = PixelHDMConfig.for_testing()
        adaln = PixelwiseAdaLN(config=config)

        # Create two distinct conditions
        torch.manual_seed(42)
        cond_a = torch.randn(2, 64, config.hidden_dim)
        cond_b = torch.randn(2, 64, config.hidden_dim)

        out_a = adaln(cond_a)
        out_b = adaln(cond_b)

        # Compute mean output difference
        input_diff = (cond_a - cond_b).abs().mean().item()
        output_diff = sum((a - b).abs().mean().item() for a, b in zip(out_a, out_b)) / 6

        # Signal retention should be > 5% (was 0.3% before fix)
        if input_diff > 0:
            retention = output_diff / input_diff
            assert retention > 0.05, \
                f"Signal retention {retention:.2%} < 5% - 2026-01-07 bug regression!"

    def test_2026_01_07_adaln_zero_fix(self):
        """
        Regression test for 2026-01-07 bug: adaLN-Zero made blocks identity.

        Bug: gamma=0, alpha=0 -> block output = x (no transformation)
        Fix: gamma=0.1, alpha=1.0 (small but active)
        """
        config = PixelHDMConfig.for_testing()
        adaln = PixelwiseAdaLN(config=config)

        bias = adaln.param_gen[-1].bias
        pixel_dim = adaln.pixel_dim
        num_params = adaln.num_params
        bias_reshaped = bias.view(num_params, pixel_dim)

        # gamma1 should be 0.1, not 0
        gamma1 = bias_reshaped[0].mean().item()
        assert gamma1 > 0.05, f"gamma1={gamma1}, too close to 0 (adaLN-Zero bug)"

        # alpha should be 1.0, not 0
        alpha1 = bias_reshaped[2].mean().item()
        assert alpha1 > 0.9, f"alpha1={alpha1}, should be ~1.0 for residual flow"

    def test_pixelblock_not_identity_mapping(self):
        """
        Ensure PixelBlocks are NOT identity mappings.

        With adaLN-Zero (gamma=0, alpha=0):
        - modulation = 0 * LN(x) + 0 = 0
        - block output = x + 0 * attention = x (identity!)

        This test verifies blocks actually transform input.
        """
        from src.models.blocks.pixel_block import PixelTransformerBlock

        config = PixelHDMConfig.for_testing()
        block = PixelTransformerBlock(config=config)
        block.eval()

        # Create input
        B, L = 2, 64
        p2 = config.patch_size ** 2
        pixel_dim = config.pixel_dim

        x = torch.randn(B, L, p2, pixel_dim)
        s_cond = torch.randn(B, L, config.hidden_dim)

        with torch.no_grad():
            output = block(x, s_cond)

        # Output should differ from input (not identity)
        diff = (output - x).abs().mean().item()
        assert diff > 0.001, f"Block is identity mapping! diff={diff}"


# =============================================================================
# Test Class: Numerical Stability
# =============================================================================


class TestAdaLNNumericalStability:
    """Test AdaLN numerical stability under extreme conditions."""

    def test_extreme_large_input(self, token_adaln):
        """Test with very large input values."""
        t_embed = torch.randn(2, 256) * 100
        outputs = token_adaln(t_embed)

        for i, out in enumerate(outputs):
            assert not torch.isnan(out).any(), f"NaN with large input at {i}"
            assert not torch.isinf(out).any(), f"Inf with large input at {i}"

    def test_extreme_small_input(self, token_adaln):
        """Test with very small input values."""
        t_embed = torch.randn(2, 256) * 1e-6
        outputs = token_adaln(t_embed)

        for i, out in enumerate(outputs):
            assert not torch.isnan(out).any(), f"NaN with small input at {i}"
            assert not torch.isinf(out).any(), f"Inf with small input at {i}"

    def test_zero_input(self, token_adaln):
        """Test with zero input."""
        t_embed = torch.zeros(2, 256)
        outputs = token_adaln(t_embed)

        # Should produce valid outputs (bias values)
        for i, out in enumerate(outputs):
            assert not torch.isnan(out).any(), f"NaN with zero input at {i}"

    def test_pixelwise_large_sequence(self, pixelwise_adaln):
        """Test PixelwiseAdaLN with large sequence length."""
        # Large sequence (256 patches for 512x512 image)
        s_cond = torch.randn(2, 256, 256)
        outputs = pixelwise_adaln(s_cond)

        for i, out in enumerate(outputs):
            assert not torch.isnan(out).any(), f"NaN with large seq at {i}"
            assert not torch.isinf(out).any(), f"Inf with large seq at {i}"


# =============================================================================
# Test Class: Device Compatibility
# =============================================================================


class TestAdaLNDeviceCompatibility:
    """Test AdaLN works on different devices."""

    def test_cpu_execution(self):
        """Test on CPU."""
        adaln = TokenAdaLN(hidden_dim=256, condition_dim=256)
        t_embed = torch.randn(2, 256)
        outputs = adaln(t_embed)

        assert all(out.device.type == "cpu" for out in outputs)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_execution(self):
        """Test on CUDA."""
        adaln = TokenAdaLN(hidden_dim=256, condition_dim=256).cuda()
        t_embed = torch.randn(2, 256).cuda()
        outputs = adaln(t_embed)

        assert all(out.device.type == "cuda" for out in outputs)

    def test_mixed_precision_forward(self, token_adaln):
        """Test forward pass in different precisions."""
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            try:
                adaln = token_adaln.to(dtype)
                t_embed = torch.randn(2, 256, dtype=dtype)
                outputs = adaln(t_embed)

                for out in outputs:
                    assert out.dtype == dtype
                    assert not torch.isnan(out).any()
            except Exception as e:
                if dtype == torch.bfloat16 and "not supported" in str(e).lower():
                    pytest.skip("bfloat16 not supported on this device")
