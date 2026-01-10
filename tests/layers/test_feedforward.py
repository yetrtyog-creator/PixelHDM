"""
Feedforward Network Tests

Tests for:
- SwiGLU: Standard Swish-Gated Linear Unit FFN
- AdaptiveSwiGLU: Conditional modulated SwiGLU for DiT
- create_ffn: Factory function for SwiGLU creation
- create_ffn_from_config: Config-based factory function

Architecture:
    - SwiGLU: gate * up projection with SiLU activation
    - Default mlp_ratio: 3.0 (vs traditional 4.0)
    - AdaptiveSwiGLU: (1 + scale) * SwiGLU(x) modulation

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-08
"""

import pytest
import torch
import torch.nn as nn

from src.models.layers.feedforward import (
    SwiGLU,
    AdaptiveSwiGLU,
    create_ffn,
    create_ffn_from_config,
)
from src.config import PixelHDMConfig


# ============================================================================
# SwiGLU Tests
# ============================================================================

class TestSwiGLU:
    """Tests for SwiGLU (Swish-Gated Linear Unit) module."""

    @pytest.fixture
    def swiglu(self):
        """Create SwiGLU with default settings."""
        return SwiGLU(hidden_dim=256, mlp_dim=768, dropout=0.0, bias=False)

    @pytest.fixture
    def swiglu_with_bias(self):
        """Create SwiGLU with bias enabled."""
        return SwiGLU(hidden_dim=256, mlp_dim=768, dropout=0.0, bias=True)

    @pytest.fixture
    def swiglu_with_dropout(self):
        """Create SwiGLU with dropout."""
        return SwiGLU(hidden_dim=256, mlp_dim=768, dropout=0.1, bias=False)

    # -------------------------------------------------------------------------
    # Shape Validation Tests
    # -------------------------------------------------------------------------

    def test_output_shape_basic(self, swiglu):
        """Test basic output shape preservation."""
        B, L, D = 2, 32, 256
        x = torch.randn(B, L, D)
        output = swiglu(x)

        assert output.shape == x.shape
        assert output.shape == (B, L, D)

    def test_output_shape_batch_size_1(self, swiglu):
        """Test with batch size 1."""
        B, L, D = 1, 64, 256
        x = torch.randn(B, L, D)
        output = swiglu(x)

        assert output.shape == (B, L, D)

    def test_output_shape_large_batch(self, swiglu):
        """Test with large batch size."""
        B, L, D = 16, 32, 256
        x = torch.randn(B, L, D)
        output = swiglu(x)

        assert output.shape == (B, L, D)

    def test_output_shape_different_seq_lengths(self, swiglu):
        """Test with various sequence lengths."""
        B, D = 2, 256
        for L in [1, 16, 64, 256, 1024]:
            x = torch.randn(B, L, D)
            output = swiglu(x)
            assert output.shape == (B, L, D), f"Failed for seq_len={L}"

    def test_output_shape_single_token(self, swiglu):
        """Test with single token input."""
        B, L, D = 4, 1, 256
        x = torch.randn(B, L, D)
        output = swiglu(x)

        assert output.shape == (B, L, D)

    # -------------------------------------------------------------------------
    # mlp_ratio Tests
    # -------------------------------------------------------------------------

    def test_mlp_ratio_default(self):
        """Test default mlp_ratio (3.0)."""
        hidden_dim = 1024
        swiglu = SwiGLU(hidden_dim=hidden_dim)

        # Default mlp_dim should be hidden_dim * 3
        assert swiglu.mlp_dim == hidden_dim * 3
        assert swiglu.mlp_dim == 3072

    def test_mlp_ratio_3_0(self):
        """Test mlp_ratio=3.0 explicitly."""
        hidden_dim = 512
        mlp_dim = int(hidden_dim * 3.0)
        swiglu = SwiGLU(hidden_dim=hidden_dim, mlp_dim=mlp_dim)

        assert swiglu.mlp_dim == 1536
        assert swiglu.gate_proj.out_features == 1536
        assert swiglu.up_proj.out_features == 1536
        assert swiglu.down_proj.in_features == 1536

    def test_mlp_ratio_4_0(self):
        """Test mlp_ratio=4.0 (traditional ratio)."""
        hidden_dim = 512
        mlp_dim = int(hidden_dim * 4.0)
        swiglu = SwiGLU(hidden_dim=hidden_dim, mlp_dim=mlp_dim)

        assert swiglu.mlp_dim == 2048
        assert swiglu.gate_proj.out_features == 2048

    def test_mlp_ratio_custom(self):
        """Test custom mlp_ratio."""
        hidden_dim = 256
        mlp_dim = int(hidden_dim * 2.5)
        swiglu = SwiGLU(hidden_dim=hidden_dim, mlp_dim=mlp_dim)

        assert swiglu.mlp_dim == 640

    # -------------------------------------------------------------------------
    # Bias Tests
    # -------------------------------------------------------------------------

    def test_bias_false(self, swiglu):
        """Test that bias=False creates layers without bias."""
        assert swiglu.gate_proj.bias is None
        assert swiglu.up_proj.bias is None
        assert swiglu.down_proj.bias is None

    def test_bias_true(self, swiglu_with_bias):
        """Test that bias=True creates layers with bias."""
        assert swiglu_with_bias.gate_proj.bias is not None
        assert swiglu_with_bias.up_proj.bias is not None
        assert swiglu_with_bias.down_proj.bias is not None

        # Verify bias shapes
        assert swiglu_with_bias.gate_proj.bias.shape == (768,)
        assert swiglu_with_bias.up_proj.bias.shape == (768,)
        assert swiglu_with_bias.down_proj.bias.shape == (256,)

    def test_bias_forward_equivalence(self):
        """Test forward pass works correctly with both bias settings."""
        hidden_dim, mlp_dim = 256, 768

        swiglu_no_bias = SwiGLU(hidden_dim, mlp_dim, bias=False)
        swiglu_with_bias = SwiGLU(hidden_dim, mlp_dim, bias=True)

        x = torch.randn(2, 32, hidden_dim)

        out_no_bias = swiglu_no_bias(x)
        out_with_bias = swiglu_with_bias(x)

        # Both should produce valid outputs
        assert out_no_bias.shape == x.shape
        assert out_with_bias.shape == x.shape
        assert not torch.isnan(out_no_bias).any()
        assert not torch.isnan(out_with_bias).any()

    # -------------------------------------------------------------------------
    # Dropout Tests
    # -------------------------------------------------------------------------

    def test_dropout_zero(self, swiglu):
        """Test dropout=0.0 uses Identity."""
        assert isinstance(swiglu.dropout, nn.Identity)

    def test_dropout_nonzero(self, swiglu_with_dropout):
        """Test dropout>0.0 uses Dropout."""
        assert isinstance(swiglu_with_dropout.dropout, nn.Dropout)
        assert swiglu_with_dropout.dropout.p == 0.1

    def test_dropout_training_vs_eval(self, swiglu_with_dropout):
        """Test dropout behavior in training vs eval mode."""
        x = torch.randn(2, 32, 256)

        # Training mode: dropout active (outputs may vary)
        swiglu_with_dropout.train()
        torch.manual_seed(42)
        out_train1 = swiglu_with_dropout(x)
        torch.manual_seed(43)
        out_train2 = swiglu_with_dropout(x)

        # Eval mode: dropout inactive (outputs should be deterministic)
        swiglu_with_dropout.eval()
        out_eval1 = swiglu_with_dropout(x)
        out_eval2 = swiglu_with_dropout(x)

        assert torch.allclose(out_eval1, out_eval2)

    # -------------------------------------------------------------------------
    # Gradient Flow Tests
    # -------------------------------------------------------------------------

    def test_gradient_flow(self, swiglu):
        """Test that gradients flow through SwiGLU."""
        x = torch.randn(2, 32, 256, requires_grad=True)
        output = swiglu(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert x.grad.shape == x.shape

    def test_gradient_flow_all_params(self, swiglu):
        """Test gradients flow to all parameters."""
        x = torch.randn(2, 32, 256, requires_grad=True)
        output = swiglu(x)
        loss = output.mean()
        loss.backward()

        # Check all parameters have gradients
        for name, param in swiglu.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_gradient_magnitude(self, swiglu):
        """Test gradient magnitude is reasonable."""
        x = torch.randn(2, 32, 256, requires_grad=True)
        output = swiglu(x)
        loss = output.mean()
        loss.backward()

        # Gradients should not be too small or too large
        grad_norm = x.grad.norm().item()
        assert grad_norm > 1e-8, "Gradients too small (vanishing)"
        assert grad_norm < 1e8, "Gradients too large (exploding)"

    # -------------------------------------------------------------------------
    # Numerical Stability Tests
    # -------------------------------------------------------------------------

    def test_numerical_stability_normal_input(self, swiglu):
        """Test numerical stability with normal inputs."""
        x = torch.randn(2, 32, 256)
        output = swiglu(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_numerical_stability_large_input(self, swiglu):
        """Test numerical stability with large inputs."""
        x = torch.randn(2, 32, 256) * 10.0
        output = swiglu(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_numerical_stability_small_input(self, swiglu):
        """Test numerical stability with small inputs."""
        x = torch.randn(2, 32, 256) * 0.01
        output = swiglu(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_numerical_stability_extreme_large(self, swiglu):
        """Test with extremely large values (may saturate SiLU)."""
        x = torch.randn(2, 32, 256) * 100.0
        output = swiglu(x)

        # SiLU saturates to x for large positive x, so output should be finite
        assert not torch.isnan(output).any()

    def test_numerical_stability_extreme_small(self, swiglu):
        """Test with extremely small values."""
        x = torch.randn(2, 32, 256) * 1e-6
        output = swiglu(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_numerical_stability_zeros(self, swiglu):
        """Test with zero input."""
        x = torch.zeros(2, 32, 256)
        output = swiglu(x)

        assert not torch.isnan(output).any()
        # With zero init bias, zero input should give zero output
        # But random weights means output won't be exactly zero

    # -------------------------------------------------------------------------
    # SwiGLU Formula Verification
    # -------------------------------------------------------------------------

    def test_swiglu_formula(self):
        """Verify SwiGLU computes: (x @ W_up) * SiLU(x @ W_gate) @ W_down."""
        hidden_dim, mlp_dim = 64, 192
        swiglu = SwiGLU(hidden_dim, mlp_dim, dropout=0.0, bias=False)

        x = torch.randn(1, 1, hidden_dim)

        # Manual computation
        gate = torch.nn.functional.silu(swiglu.gate_proj(x))
        up = swiglu.up_proj(x)
        hidden = gate * up
        expected = swiglu.down_proj(hidden)

        # Module output
        output = swiglu(x)

        assert torch.allclose(output, expected, atol=1e-6)

    # -------------------------------------------------------------------------
    # extra_repr Test
    # -------------------------------------------------------------------------

    def test_extra_repr(self, swiglu):
        """Test extra_repr output."""
        repr_str = swiglu.extra_repr()

        assert "hidden_dim=256" in repr_str
        assert "mlp_dim=768" in repr_str


# ============================================================================
# AdaptiveSwiGLU Tests
# ============================================================================

class TestAdaptiveSwiGLU:
    """Tests for AdaptiveSwiGLU (conditional modulated SwiGLU) module."""

    @pytest.fixture
    def adaptive_swiglu(self):
        """Create AdaptiveSwiGLU with default settings."""
        return AdaptiveSwiGLU(
            hidden_dim=256,
            cond_dim=512,
            mlp_dim=768,
            dropout=0.0,
        )

    @pytest.fixture
    def adaptive_swiglu_different_dims(self):
        """Create AdaptiveSwiGLU with different dimensions."""
        return AdaptiveSwiGLU(
            hidden_dim=1024,
            cond_dim=256,
            mlp_dim=3072,
            dropout=0.0,
        )

    # -------------------------------------------------------------------------
    # Shape Validation Tests
    # -------------------------------------------------------------------------

    def test_output_shape_2d_cond(self, adaptive_swiglu):
        """Test output shape with 2D condition tensor (B, cond_dim)."""
        B, L, D = 2, 32, 256
        x = torch.randn(B, L, D)
        cond = torch.randn(B, 512)  # 2D: (B, cond_dim)

        output = adaptive_swiglu(x, cond)

        assert output.shape == x.shape
        assert output.shape == (B, L, D)

    def test_output_shape_3d_cond(self, adaptive_swiglu):
        """Test output shape with 3D condition tensor (B, L, cond_dim)."""
        B, L, D = 2, 32, 256
        x = torch.randn(B, L, D)
        cond = torch.randn(B, L, 512)  # 3D: (B, L, cond_dim)

        output = adaptive_swiglu(x, cond)

        assert output.shape == x.shape

    def test_output_shape_cond_broadcast(self, adaptive_swiglu):
        """Test that 2D cond broadcasts correctly to sequence length."""
        B, L, D = 2, 64, 256
        x = torch.randn(B, L, D)
        cond = torch.randn(B, 512)  # Will be expanded to (B, 1, cond_dim)

        output = adaptive_swiglu(x, cond)

        assert output.shape == (B, L, D)

    def test_output_shape_different_dims(self, adaptive_swiglu_different_dims):
        """Test with different dimension configuration."""
        B, L, D = 4, 16, 1024
        x = torch.randn(B, L, D)
        cond = torch.randn(B, 256)

        output = adaptive_swiglu_different_dims(x, cond)

        assert output.shape == (B, L, D)

    # -------------------------------------------------------------------------
    # Conditional Modulation Tests
    # -------------------------------------------------------------------------

    def test_modulation_zero_init(self, adaptive_swiglu):
        """Test modulation is initialized to zero (identity at start)."""
        # Check weight and bias are zero
        mod_linear = adaptive_swiglu.modulation[-1]

        assert torch.allclose(mod_linear.weight, torch.zeros_like(mod_linear.weight))
        assert torch.allclose(mod_linear.bias, torch.zeros_like(mod_linear.bias))

    def test_modulation_initial_identity(self, adaptive_swiglu):
        """Test that initial modulation is identity: (1 + 0) * output = output."""
        B, L, D = 2, 32, 256
        x = torch.randn(B, L, D)
        cond = torch.randn(B, 512)

        # Get inner SwiGLU output
        inner_output = adaptive_swiglu.swiglu(x)

        # Get full output (should be same due to zero init)
        full_output = adaptive_swiglu(x, cond)

        assert torch.allclose(full_output, inner_output, atol=1e-6)

    def test_modulation_different_cond_different_output(self, adaptive_swiglu):
        """Test different conditions produce different outputs after training."""
        # Manually set non-zero modulation weights
        with torch.no_grad():
            adaptive_swiglu.modulation[-1].weight.normal_(std=0.1)
            adaptive_swiglu.modulation[-1].bias.normal_(std=0.1)

        B, L, D = 2, 32, 256
        x = torch.randn(B, L, D)
        cond1 = torch.randn(B, 512)
        cond2 = torch.randn(B, 512)  # Different condition

        output1 = adaptive_swiglu(x, cond1)
        output2 = adaptive_swiglu(x, cond2)

        # Outputs should be different
        assert not torch.allclose(output1, output2, atol=1e-4)

    def test_modulation_same_cond_same_output(self, adaptive_swiglu):
        """Test same condition produces same output."""
        B, L, D = 2, 32, 256
        x = torch.randn(B, L, D)
        cond = torch.randn(B, 512)

        output1 = adaptive_swiglu(x, cond)
        output2 = adaptive_swiglu(x, cond)

        assert torch.allclose(output1, output2)

    def test_modulation_formula(self, adaptive_swiglu):
        """Verify modulation formula: (1 + scale) * SwiGLU(x)."""
        # Set non-zero modulation
        with torch.no_grad():
            adaptive_swiglu.modulation[-1].weight.normal_(std=0.1)
            adaptive_swiglu.modulation[-1].bias.fill_(0.5)

        B, L, D = 2, 32, 256
        x = torch.randn(B, L, D)
        cond = torch.randn(B, 512)

        # Manual computation
        swiglu_out = adaptive_swiglu.swiglu(x)
        cond_expanded = cond.unsqueeze(1)  # (B, 1, cond_dim)
        scale = adaptive_swiglu.modulation(cond_expanded)  # (B, 1, hidden_dim)
        expected = swiglu_out * (1 + scale)

        # Module output
        output = adaptive_swiglu(x, cond)

        assert torch.allclose(output, expected, atol=1e-6)

    # -------------------------------------------------------------------------
    # Gradient Flow Tests
    # -------------------------------------------------------------------------

    def test_gradient_flow_to_input(self, adaptive_swiglu):
        """Test gradients flow to input."""
        x = torch.randn(2, 32, 256, requires_grad=True)
        cond = torch.randn(2, 512, requires_grad=True)

        output = adaptive_swiglu(x, cond)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_gradient_flow_to_condition(self, adaptive_swiglu):
        """Test gradients flow to condition."""
        x = torch.randn(2, 32, 256, requires_grad=True)
        cond = torch.randn(2, 512, requires_grad=True)

        output = adaptive_swiglu(x, cond)
        loss = output.sum()
        loss.backward()

        assert cond.grad is not None
        assert not torch.isnan(cond.grad).any()

    def test_gradient_flow_all_params(self, adaptive_swiglu):
        """Test gradients flow to all parameters."""
        x = torch.randn(2, 32, 256, requires_grad=True)
        cond = torch.randn(2, 512, requires_grad=True)

        output = adaptive_swiglu(x, cond)
        loss = output.mean()
        loss.backward()

        for name, param in adaptive_swiglu.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    # -------------------------------------------------------------------------
    # Numerical Stability Tests
    # -------------------------------------------------------------------------

    def test_numerical_stability_normal(self, adaptive_swiglu):
        """Test numerical stability with normal inputs."""
        x = torch.randn(2, 32, 256)
        cond = torch.randn(2, 512)

        output = adaptive_swiglu(x, cond)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_numerical_stability_large_cond(self, adaptive_swiglu):
        """Test numerical stability with large condition values."""
        x = torch.randn(2, 32, 256)
        cond = torch.randn(2, 512) * 10.0

        output = adaptive_swiglu(x, cond)

        assert not torch.isnan(output).any()

    def test_numerical_stability_zero_cond(self, adaptive_swiglu):
        """Test numerical stability with zero condition."""
        x = torch.randn(2, 32, 256)
        cond = torch.zeros(2, 512)

        output = adaptive_swiglu(x, cond)

        assert not torch.isnan(output).any()

    # -------------------------------------------------------------------------
    # extra_repr Test
    # -------------------------------------------------------------------------

    def test_extra_repr(self, adaptive_swiglu):
        """Test extra_repr output."""
        repr_str = adaptive_swiglu.extra_repr()

        assert "hidden_dim=256" in repr_str
        assert "mlp_dim=768" in repr_str
        assert "cond_dim=512" in repr_str


# ============================================================================
# Factory Function Tests
# ============================================================================

class TestCreateFFN:
    """Tests for create_ffn factory function."""

    def test_create_ffn_default(self):
        """Test create_ffn with default parameters."""
        ffn = create_ffn(hidden_dim=1024)

        assert isinstance(ffn, SwiGLU)
        assert ffn.hidden_dim == 1024
        assert ffn.mlp_dim == int(1024 * 3.0)  # Default mlp_ratio

    def test_create_ffn_mlp_ratio_3(self):
        """Test create_ffn with mlp_ratio=3.0."""
        ffn = create_ffn(hidden_dim=512, mlp_ratio=3.0)

        assert ffn.mlp_dim == 1536

    def test_create_ffn_mlp_ratio_4(self):
        """Test create_ffn with mlp_ratio=4.0."""
        ffn = create_ffn(hidden_dim=512, mlp_ratio=4.0)

        assert ffn.mlp_dim == 2048

    def test_create_ffn_with_dropout(self):
        """Test create_ffn with dropout."""
        ffn = create_ffn(hidden_dim=256, dropout=0.1)

        assert isinstance(ffn.dropout, nn.Dropout)
        assert ffn.dropout.p == 0.1

    def test_create_ffn_with_bias(self):
        """Test create_ffn with bias=True."""
        ffn = create_ffn(hidden_dim=256, bias=True)

        assert ffn.gate_proj.bias is not None

    def test_create_ffn_without_bias(self):
        """Test create_ffn with bias=False (default)."""
        ffn = create_ffn(hidden_dim=256, bias=False)

        assert ffn.gate_proj.bias is None

    def test_create_ffn_forward(self):
        """Test forward pass of created FFN."""
        ffn = create_ffn(hidden_dim=256, mlp_ratio=3.0)

        x = torch.randn(2, 32, 256)
        output = ffn(x)

        assert output.shape == x.shape


class TestCreateFFNFromConfig:
    """Tests for create_ffn_from_config factory function."""

    def test_create_from_testing_config(self):
        """Test creating FFN from testing config."""
        config = PixelHDMConfig.for_testing()
        ffn = create_ffn_from_config(config)

        assert isinstance(ffn, SwiGLU)
        assert ffn.hidden_dim == config.hidden_dim
        assert ffn.mlp_dim == int(config.hidden_dim * config.mlp_ratio)

    def test_create_from_default_config(self):
        """Test creating FFN from default config."""
        config = PixelHDMConfig.default()
        ffn = create_ffn_from_config(config)

        assert ffn.hidden_dim == 1024
        assert ffn.mlp_dim == int(1024 * 3.0)

    def test_config_dropout_passed(self):
        """Test that config dropout is used."""
        config = PixelHDMConfig.for_testing()
        # Testing config has dropout=0.0
        ffn = create_ffn_from_config(config)

        assert isinstance(ffn.dropout, nn.Identity)

    def test_bias_always_false(self):
        """Test that bias is always False from config."""
        config = PixelHDMConfig.for_testing()
        ffn = create_ffn_from_config(config)

        # create_ffn_from_config always uses bias=False
        assert ffn.gate_proj.bias is None
        assert ffn.up_proj.bias is None
        assert ffn.down_proj.bias is None

    def test_forward_with_config(self):
        """Test forward pass with config-created FFN."""
        config = PixelHDMConfig.for_testing()
        ffn = create_ffn_from_config(config)

        x = torch.randn(2, 32, config.hidden_dim)
        output = ffn(x)

        assert output.shape == x.shape


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================

class TestSwiGLUEdgeCases:
    """Edge case tests for SwiGLU."""

    def test_single_batch_single_token(self):
        """Test with B=1, L=1."""
        swiglu = SwiGLU(hidden_dim=256, mlp_dim=768)
        x = torch.randn(1, 1, 256)
        output = swiglu(x)

        assert output.shape == (1, 1, 256)

    def test_very_long_sequence(self):
        """Test with very long sequence."""
        swiglu = SwiGLU(hidden_dim=64, mlp_dim=192)
        x = torch.randn(1, 4096, 64)
        output = swiglu(x)

        assert output.shape == (1, 4096, 64)

    def test_different_hidden_dims(self):
        """Test with various hidden dimensions."""
        for hidden_dim in [64, 128, 256, 512, 1024]:
            mlp_dim = hidden_dim * 3
            swiglu = SwiGLU(hidden_dim, mlp_dim)
            x = torch.randn(2, 16, hidden_dim)
            output = swiglu(x)

            assert output.shape == (2, 16, hidden_dim)


class TestAdaptiveSwiGLUEdgeCases:
    """Edge case tests for AdaptiveSwiGLU."""

    def test_cond_dim_larger_than_hidden(self):
        """Test when cond_dim > hidden_dim."""
        ffn = AdaptiveSwiGLU(hidden_dim=256, cond_dim=1024, mlp_dim=768)
        x = torch.randn(2, 32, 256)
        cond = torch.randn(2, 1024)

        output = ffn(x, cond)
        assert output.shape == x.shape

    def test_cond_dim_smaller_than_hidden(self):
        """Test when cond_dim < hidden_dim."""
        ffn = AdaptiveSwiGLU(hidden_dim=1024, cond_dim=256, mlp_dim=3072)
        x = torch.randn(2, 32, 1024)
        cond = torch.randn(2, 256)

        output = ffn(x, cond)
        assert output.shape == x.shape

    def test_cond_3d_per_token(self):
        """Test with per-token conditioning (3D cond)."""
        ffn = AdaptiveSwiGLU(hidden_dim=256, cond_dim=512, mlp_dim=768)
        x = torch.randn(2, 32, 256)
        cond = torch.randn(2, 32, 512)  # Per-token condition

        output = ffn(x, cond)
        assert output.shape == x.shape


class TestDeviceCompatibility:
    """Tests for device compatibility (CPU)."""

    def test_swiglu_cpu(self):
        """Test SwiGLU on CPU."""
        swiglu = SwiGLU(hidden_dim=256, mlp_dim=768)
        x = torch.randn(2, 32, 256)

        output = swiglu(x)

        assert output.device.type == "cpu"

    def test_adaptive_swiglu_cpu(self):
        """Test AdaptiveSwiGLU on CPU."""
        ffn = AdaptiveSwiGLU(hidden_dim=256, cond_dim=512, mlp_dim=768)
        x = torch.randn(2, 32, 256)
        cond = torch.randn(2, 512)

        output = ffn(x, cond)

        assert output.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_swiglu_cuda(self):
        """Test SwiGLU on CUDA."""
        swiglu = SwiGLU(hidden_dim=256, mlp_dim=768).cuda()
        x = torch.randn(2, 32, 256).cuda()

        output = swiglu(x)

        assert output.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_adaptive_swiglu_cuda(self):
        """Test AdaptiveSwiGLU on CUDA."""
        ffn = AdaptiveSwiGLU(hidden_dim=256, cond_dim=512, mlp_dim=768).cuda()
        x = torch.randn(2, 32, 256).cuda()
        cond = torch.randn(2, 512).cuda()

        output = ffn(x, cond)

        assert output.device.type == "cuda"


class TestDtypeCompatibility:
    """Tests for dtype compatibility."""

    def test_swiglu_float32(self):
        """Test SwiGLU with float32."""
        swiglu = SwiGLU(hidden_dim=256, mlp_dim=768)
        x = torch.randn(2, 32, 256, dtype=torch.float32)

        output = swiglu(x)

        assert output.dtype == torch.float32

    def test_swiglu_float16(self):
        """Test SwiGLU with float16."""
        swiglu = SwiGLU(hidden_dim=256, mlp_dim=768).half()
        x = torch.randn(2, 32, 256, dtype=torch.float16)

        output = swiglu(x)

        assert output.dtype == torch.float16
        assert not torch.isnan(output).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for bfloat16")
    def test_swiglu_bfloat16(self):
        """Test SwiGLU with bfloat16."""
        swiglu = SwiGLU(hidden_dim=256, mlp_dim=768).to(torch.bfloat16).cuda()
        x = torch.randn(2, 32, 256, dtype=torch.bfloat16, device="cuda")

        output = swiglu(x)

        assert output.dtype == torch.bfloat16
        assert not torch.isnan(output).any()

    def test_adaptive_swiglu_float16(self):
        """Test AdaptiveSwiGLU with float16."""
        ffn = AdaptiveSwiGLU(hidden_dim=256, cond_dim=512, mlp_dim=768).half()
        x = torch.randn(2, 32, 256, dtype=torch.float16)
        cond = torch.randn(2, 512, dtype=torch.float16)

        output = ffn(x, cond)

        assert output.dtype == torch.float16


class TestModuleAttributes:
    """Tests for module attributes and properties."""

    def test_swiglu_hidden_dim_attribute(self):
        """Test hidden_dim attribute."""
        swiglu = SwiGLU(hidden_dim=512, mlp_dim=1536)
        assert swiglu.hidden_dim == 512

    def test_swiglu_mlp_dim_attribute(self):
        """Test mlp_dim attribute."""
        swiglu = SwiGLU(hidden_dim=512, mlp_dim=1536)
        assert swiglu.mlp_dim == 1536

    def test_adaptive_swiglu_cond_dim_attribute(self):
        """Test cond_dim attribute."""
        ffn = AdaptiveSwiGLU(hidden_dim=256, cond_dim=512, mlp_dim=768)
        assert ffn.cond_dim == 512

    def test_swiglu_parameter_count(self):
        """Test parameter count calculation."""
        hidden_dim, mlp_dim = 256, 768
        swiglu = SwiGLU(hidden_dim, mlp_dim, bias=False)

        # gate_proj: hidden_dim * mlp_dim
        # up_proj: hidden_dim * mlp_dim
        # down_proj: mlp_dim * hidden_dim
        expected_params = 3 * hidden_dim * mlp_dim
        actual_params = sum(p.numel() for p in swiglu.parameters())

        assert actual_params == expected_params

    def test_swiglu_parameter_count_with_bias(self):
        """Test parameter count with bias."""
        hidden_dim, mlp_dim = 256, 768
        swiglu = SwiGLU(hidden_dim, mlp_dim, bias=True)

        # Weights + biases
        expected_params = 3 * hidden_dim * mlp_dim + 2 * mlp_dim + hidden_dim
        actual_params = sum(p.numel() for p in swiglu.parameters())

        assert actual_params == expected_params
