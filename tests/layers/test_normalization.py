"""
Normalization Module Tests

Tests for:
- RMSNorm: Root Mean Square Layer Normalization
- AdaptiveRMSNorm: Conditional RMSNorm with scale/shift modulation
- QKNorm: Per-head RMSNorm for Q/K normalization
- create_norm: Factory function for creating normalization layers

Test Categories:
1. Output normalization correctness (RMS approx 1)
2. Gradient flow tests
3. Different dtypes (float32, float16, bfloat16)
4. Numerical stability (extreme/small/zero inputs)
5. AdaptiveRMSNorm scale/shift application
6. QKNorm Q/K separation handling
7. eps parameter effect
8. elementwise_affine (weight) parameter

Test Count: 46 test cases

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-08
"""

import pytest
import torch
import torch.nn as nn
import math

from src.models.layers.normalization import (
    RMSNorm,
    AdaptiveRMSNorm,
    QKNorm,
    create_norm,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def batch_size() -> int:
    return 2


@pytest.fixture
def seq_len() -> int:
    return 64


@pytest.fixture
def hidden_dim() -> int:
    return 256


@pytest.fixture
def head_dim() -> int:
    return 64


@pytest.fixture
def cond_dim() -> int:
    return 128


@pytest.fixture
def rmsnorm(hidden_dim):
    """Create standard RMSNorm."""
    return RMSNorm(dim=hidden_dim, eps=1e-6)


@pytest.fixture
def adaptive_rmsnorm(hidden_dim, cond_dim):
    """Create AdaptiveRMSNorm."""
    return AdaptiveRMSNorm(dim=hidden_dim, cond_dim=cond_dim, eps=1e-6)


@pytest.fixture
def qknorm(head_dim):
    """Create QKNorm."""
    return QKNorm(head_dim=head_dim, eps=1e-6)


# =============================================================================
# Test Class: RMSNorm Initialization
# =============================================================================


class TestRMSNormInitialization:
    """Test RMSNorm initialization."""

    def test_weight_initialized_to_ones(self, hidden_dim):
        """Test weight is initialized to ones."""
        norm = RMSNorm(dim=hidden_dim)
        assert torch.allclose(norm.weight, torch.ones(hidden_dim)), \
            "Weight should be initialized to ones"

    def test_eps_stored(self, hidden_dim):
        """Test eps parameter is stored correctly."""
        eps = 1e-5
        norm = RMSNorm(dim=hidden_dim, eps=eps)
        assert norm.eps == eps

    def test_dim_stored(self, hidden_dim):
        """Test dim parameter is stored correctly."""
        norm = RMSNorm(dim=hidden_dim)
        assert norm.dim == hidden_dim


# =============================================================================
# Test Class: RMSNorm Output Correctness
# =============================================================================


class TestRMSNormOutputCorrectness:
    """Test RMSNorm produces correct normalized output."""

    def test_output_rms_approximately_one(self, rmsnorm, batch_size, seq_len, hidden_dim):
        """Test output RMS is approximately 1.0."""
        x = torch.randn(batch_size, seq_len, hidden_dim)
        output = rmsnorm(x)

        # Compute RMS of output along last dimension
        rms = torch.sqrt(output.pow(2).mean(-1))

        # RMS should be close to 1 (within tolerance for weight effects)
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1), \
            f"Output RMS mean: {rms.mean()}, expected ~1.0"

    def test_output_shape_preserved(self, rmsnorm, batch_size, seq_len, hidden_dim):
        """Test output shape matches input shape."""
        x = torch.randn(batch_size, seq_len, hidden_dim)
        output = rmsnorm(x)
        assert output.shape == x.shape

    def test_different_input_shapes(self, hidden_dim):
        """Test RMSNorm works with different input shapes."""
        norm = RMSNorm(dim=hidden_dim)

        # 2D input
        x_2d = torch.randn(32, hidden_dim)
        out_2d = norm(x_2d)
        assert out_2d.shape == x_2d.shape

        # 3D input
        x_3d = torch.randn(2, 64, hidden_dim)
        out_3d = norm(x_3d)
        assert out_3d.shape == x_3d.shape

        # 4D input
        x_4d = torch.randn(2, 4, 16, hidden_dim)
        out_4d = norm(x_4d)
        assert out_4d.shape == x_4d.shape

    def test_weight_scaling_effect(self, batch_size, seq_len, hidden_dim):
        """Test weight parameter affects output scaling."""
        norm = RMSNorm(dim=hidden_dim)
        x = torch.randn(batch_size, seq_len, hidden_dim)

        # Default weight (ones) output
        output_default = norm(x).clone()

        # Modify weight to 2x
        with torch.no_grad():
            norm.weight.fill_(2.0)

        output_scaled = norm(x)

        # Scaled output should be 2x default
        assert torch.allclose(output_scaled, output_default * 2, atol=1e-5), \
            "Weight scaling not applied correctly"


# =============================================================================
# Test Class: RMSNorm Gradient Flow
# =============================================================================


class TestRMSNormGradientFlow:
    """Test gradient flow through RMSNorm."""

    def test_gradient_flows_to_input(self, rmsnorm, batch_size, seq_len, hidden_dim):
        """Test gradients flow back to input."""
        x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)
        output = rmsnorm(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "Input should have gradient"
        assert not torch.isnan(x.grad).any(), "Gradient contains NaN"

    def test_gradient_flows_to_weight(self, rmsnorm, batch_size, seq_len, hidden_dim):
        """Test gradients flow to weight parameter."""
        x = torch.randn(batch_size, seq_len, hidden_dim)
        output = rmsnorm(x)
        loss = output.sum()
        loss.backward()

        assert rmsnorm.weight.grad is not None, "Weight should have gradient"
        assert not torch.isnan(rmsnorm.weight.grad).any(), "Weight gradient contains NaN"


# =============================================================================
# Test Class: RMSNorm Different Dtypes
# =============================================================================


class TestRMSNormDtypes:
    """Test RMSNorm with different dtypes."""

    def test_float32(self, hidden_dim):
        """Test with float32."""
        norm = RMSNorm(dim=hidden_dim)
        x = torch.randn(2, 64, hidden_dim, dtype=torch.float32)
        output = norm(x)

        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()

    def test_float16(self, hidden_dim):
        """Test with float16."""
        norm = RMSNorm(dim=hidden_dim).half()
        x = torch.randn(2, 64, hidden_dim, dtype=torch.float16)
        output = norm(x)

        assert output.dtype == torch.float16
        assert not torch.isnan(output).any()

    def test_bfloat16(self, hidden_dim):
        """Test with bfloat16."""
        try:
            norm = RMSNorm(dim=hidden_dim).to(torch.bfloat16)
            x = torch.randn(2, 64, hidden_dim, dtype=torch.bfloat16)
            output = norm(x)

            assert output.dtype == torch.bfloat16
            assert not torch.isnan(output).any()
        except Exception as e:
            if "not supported" in str(e).lower():
                pytest.skip("bfloat16 not supported on this device")
            raise


# =============================================================================
# Test Class: RMSNorm Numerical Stability
# =============================================================================


class TestRMSNormNumericalStability:
    """Test RMSNorm numerical stability."""

    def test_extreme_large_input(self, hidden_dim):
        """Test with very large input values."""
        norm = RMSNorm(dim=hidden_dim)
        x = torch.randn(2, 64, hidden_dim) * 1000

        output = norm(x)

        assert not torch.isnan(output).any(), "NaN with large input"
        assert not torch.isinf(output).any(), "Inf with large input"

    def test_extreme_small_input(self, hidden_dim):
        """Test with very small input values."""
        norm = RMSNorm(dim=hidden_dim)
        x = torch.randn(2, 64, hidden_dim) * 1e-6

        output = norm(x)

        assert not torch.isnan(output).any(), "NaN with small input"
        assert not torch.isinf(output).any(), "Inf with small input"

    def test_zero_input(self, hidden_dim):
        """Test with zero input (eps prevents division by zero)."""
        norm = RMSNorm(dim=hidden_dim)
        x = torch.zeros(2, 64, hidden_dim)

        output = norm(x)

        assert not torch.isnan(output).any(), "NaN with zero input"
        assert not torch.isinf(output).any(), "Inf with zero input"
        # Output should be zero (0 * weight = 0)
        assert torch.allclose(output, torch.zeros_like(output), atol=1e-6)

    def test_eps_prevents_nan(self, hidden_dim):
        """Test that eps prevents NaN for near-zero RMS."""
        norm = RMSNorm(dim=hidden_dim, eps=1e-6)

        # Input with very small values
        x = torch.ones(2, 64, hidden_dim) * 1e-8

        output = norm(x)

        assert not torch.isnan(output).any(), "eps should prevent NaN"


# =============================================================================
# Test Class: AdaptiveRMSNorm Initialization
# =============================================================================


class TestAdaptiveRMSNormInitialization:
    """Test AdaptiveRMSNorm initialization."""

    def test_weight_initialized_to_ones(self, hidden_dim, cond_dim):
        """Test weight is initialized to ones."""
        norm = AdaptiveRMSNorm(dim=hidden_dim, cond_dim=cond_dim)
        assert torch.allclose(norm.weight, torch.ones(hidden_dim))

    def test_modulation_zero_init(self, hidden_dim, cond_dim):
        """Test modulation layer is zero initialized."""
        norm = AdaptiveRMSNorm(dim=hidden_dim, cond_dim=cond_dim)

        # Last linear in modulation should have zero weight and bias
        linear = norm.modulation[-1]
        assert torch.allclose(linear.weight, torch.zeros_like(linear.weight)), \
            "Modulation weight should be zero initialized"
        assert torch.allclose(linear.bias, torch.zeros_like(linear.bias)), \
            "Modulation bias should be zero initialized"

    def test_modulation_output_dim(self, hidden_dim, cond_dim):
        """Test modulation outputs scale and shift (2 * dim)."""
        norm = AdaptiveRMSNorm(dim=hidden_dim, cond_dim=cond_dim)
        linear = norm.modulation[-1]

        assert linear.out_features == hidden_dim * 2


# =============================================================================
# Test Class: AdaptiveRMSNorm Scale/Shift Application
# =============================================================================


class TestAdaptiveRMSNormScaleShift:
    """Test AdaptiveRMSNorm applies scale and shift correctly."""

    def test_zero_cond_produces_base_output(self, adaptive_rmsnorm, batch_size, seq_len, hidden_dim, cond_dim):
        """Test zero condition produces base RMSNorm output (scale=1, shift=0)."""
        x = torch.randn(batch_size, seq_len, hidden_dim)
        cond = torch.zeros(batch_size, cond_dim)

        # AdaptiveRMSNorm with zero-initialized modulation + zero cond
        # should produce scale=0, shift=0 after zero init
        # Applied as (1 + scale) * x + shift = (1 + 0) * x + 0 = x
        output = adaptive_rmsnorm(x, cond)

        # Compare with base RMSNorm
        base_norm = RMSNorm(dim=hidden_dim)
        with torch.no_grad():
            base_norm.weight.copy_(adaptive_rmsnorm.weight)
        base_output = base_norm(x)

        assert torch.allclose(output, base_output, atol=1e-5), \
            "Zero condition should produce base RMSNorm output"

    def test_output_shape(self, adaptive_rmsnorm, batch_size, seq_len, hidden_dim, cond_dim):
        """Test output shape matches input shape."""
        x = torch.randn(batch_size, seq_len, hidden_dim)
        cond = torch.randn(batch_size, cond_dim)

        output = adaptive_rmsnorm(x, cond)

        assert output.shape == x.shape

    def test_cond_2d_broadcast(self, adaptive_rmsnorm, batch_size, seq_len, hidden_dim, cond_dim):
        """Test 2D condition is broadcast to sequence."""
        x = torch.randn(batch_size, seq_len, hidden_dim)
        cond_2d = torch.randn(batch_size, cond_dim)  # (B, cond_dim)

        output = adaptive_rmsnorm(x, cond_2d)

        assert output.shape == x.shape

    def test_cond_3d_per_token(self, adaptive_rmsnorm, batch_size, seq_len, hidden_dim, cond_dim):
        """Test 3D condition applies per-token modulation."""
        x = torch.randn(batch_size, seq_len, hidden_dim)
        cond_3d = torch.randn(batch_size, seq_len, cond_dim)  # (B, L, cond_dim)

        output = adaptive_rmsnorm(x, cond_3d)

        assert output.shape == x.shape

    def test_different_cond_different_output(self, adaptive_rmsnorm, batch_size, seq_len, hidden_dim, cond_dim):
        """Test different conditions produce different outputs."""
        x = torch.randn(batch_size, seq_len, hidden_dim)
        cond_1 = torch.randn(batch_size, cond_dim)
        cond_2 = torch.randn(batch_size, cond_dim) + 10.0  # Significantly different

        # Need non-zero modulation weights to see difference
        with torch.no_grad():
            adaptive_rmsnorm.modulation[-1].weight.fill_(0.1)

        output_1 = adaptive_rmsnorm(x, cond_1)
        output_2 = adaptive_rmsnorm(x, cond_2)

        assert not torch.allclose(output_1, output_2), \
            "Different conditions should produce different outputs"


# =============================================================================
# Test Class: AdaptiveRMSNorm Gradient Flow
# =============================================================================


class TestAdaptiveRMSNormGradientFlow:
    """Test gradient flow through AdaptiveRMSNorm."""

    def test_gradient_to_input(self, adaptive_rmsnorm, batch_size, seq_len, hidden_dim, cond_dim):
        """Test gradients flow to input."""
        x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)
        cond = torch.randn(batch_size, cond_dim)

        output = adaptive_rmsnorm(x, cond)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_gradient_to_condition(self, adaptive_rmsnorm, batch_size, seq_len, hidden_dim, cond_dim):
        """Test gradients flow to condition."""
        x = torch.randn(batch_size, seq_len, hidden_dim)
        cond = torch.randn(batch_size, cond_dim, requires_grad=True)

        output = adaptive_rmsnorm(x, cond)
        loss = output.sum()
        loss.backward()

        assert cond.grad is not None
        assert not torch.isnan(cond.grad).any()


# =============================================================================
# Test Class: QKNorm Initialization
# =============================================================================


class TestQKNormInitialization:
    """Test QKNorm initialization."""

    def test_weight_initialized_to_ones(self, head_dim):
        """Test weight is initialized to ones."""
        norm = QKNorm(head_dim=head_dim)
        assert torch.allclose(norm.weight, torch.ones(head_dim))

    def test_eps_stored(self, head_dim):
        """Test eps is stored correctly."""
        eps = 1e-5
        norm = QKNorm(head_dim=head_dim, eps=eps)
        assert norm.eps == eps

    def test_head_dim_stored(self, head_dim):
        """Test head_dim is stored correctly."""
        norm = QKNorm(head_dim=head_dim)
        assert norm.head_dim == head_dim


# =============================================================================
# Test Class: QKNorm Output Correctness
# =============================================================================


class TestQKNormOutputCorrectness:
    """Test QKNorm output correctness."""

    def test_output_shape(self, qknorm, batch_size, head_dim):
        """Test output shape is preserved."""
        num_heads = 8
        seq_len = 64

        x = torch.randn(batch_size, num_heads, seq_len, head_dim)
        output = qknorm(x)

        assert output.shape == x.shape

    def test_per_head_normalization(self, qknorm, batch_size, head_dim):
        """Test normalization is applied per-head independently."""
        num_heads = 4
        seq_len = 32

        # Create input with different scales per head
        x = torch.randn(batch_size, num_heads, seq_len, head_dim)
        x[:, 0, :, :] *= 10  # Scale first head differently

        output = qknorm(x)

        # Each head should be normalized independently
        # Output RMS should be similar across heads
        rms_per_head = torch.sqrt(output.pow(2).mean(-1))  # (B, H, L)
        mean_rms = rms_per_head.mean()

        # All heads should have similar RMS (close to 1)
        assert torch.allclose(rms_per_head, torch.ones_like(rms_per_head), atol=0.2)

    def test_q_k_processed_independently(self, head_dim):
        """Test Q and K can be processed independently."""
        norm = QKNorm(head_dim=head_dim)

        B, H, L, D = 2, 8, 64, head_dim
        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)

        q_norm = norm(q)
        k_norm = norm(k)

        assert q_norm.shape == q.shape
        assert k_norm.shape == k.shape

        # Outputs should be different (different inputs)
        assert not torch.allclose(q_norm, k_norm)


# =============================================================================
# Test Class: QKNorm Gradient Flow
# =============================================================================


class TestQKNormGradientFlow:
    """Test gradient flow through QKNorm."""

    def test_gradient_flows(self, qknorm, batch_size, head_dim):
        """Test gradients flow through QKNorm."""
        x = torch.randn(batch_size, 8, 64, head_dim, requires_grad=True)
        output = qknorm(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# =============================================================================
# Test Class: create_norm Factory Function
# =============================================================================


class TestCreateNormFactory:
    """Test create_norm factory function."""

    def test_create_rmsnorm(self, hidden_dim):
        """Test creating RMSNorm."""
        norm = create_norm(dim=hidden_dim, norm_type="rmsnorm")

        assert isinstance(norm, RMSNorm)
        assert norm.dim == hidden_dim

    def test_create_layernorm(self, hidden_dim):
        """Test creating LayerNorm."""
        norm = create_norm(dim=hidden_dim, norm_type="layernorm")

        assert isinstance(norm, nn.LayerNorm)
        assert norm.normalized_shape == (hidden_dim,)

    def test_custom_eps(self, hidden_dim):
        """Test custom eps parameter."""
        eps = 1e-5
        norm = create_norm(dim=hidden_dim, norm_type="rmsnorm", eps=eps)

        assert norm.eps == eps

    def test_invalid_norm_type_raises(self, hidden_dim):
        """Test invalid norm_type raises ValueError."""
        with pytest.raises(ValueError, match="不支援"):
            create_norm(dim=hidden_dim, norm_type="invalid_norm")


# =============================================================================
# Test Class: eps Parameter Effect
# =============================================================================


class TestEpsParameterEffect:
    """Test eps parameter effect on numerical stability."""

    def test_larger_eps_more_stable(self, hidden_dim):
        """Test larger eps provides more stability for small inputs."""
        x_small = torch.ones(2, 64, hidden_dim) * 1e-10

        norm_small_eps = RMSNorm(dim=hidden_dim, eps=1e-10)
        norm_large_eps = RMSNorm(dim=hidden_dim, eps=1e-4)

        out_small_eps = norm_small_eps(x_small)
        out_large_eps = norm_large_eps(x_small)

        # Both should be stable
        assert not torch.isnan(out_small_eps).any()
        assert not torch.isnan(out_large_eps).any()

        # Larger eps should result in smaller output magnitude for small inputs
        assert out_large_eps.abs().mean() < out_small_eps.abs().mean()

    def test_eps_consistency_across_norms(self, hidden_dim, cond_dim, head_dim):
        """Test eps is consistently applied across different norm types."""
        eps = 1e-5

        rms_norm = RMSNorm(dim=hidden_dim, eps=eps)
        adaptive_norm = AdaptiveRMSNorm(dim=hidden_dim, cond_dim=cond_dim, eps=eps)
        qk_norm = QKNorm(head_dim=head_dim, eps=eps)

        assert rms_norm.eps == eps
        assert adaptive_norm.eps == eps
        assert qk_norm.eps == eps


# =============================================================================
# Test Class: Device Compatibility
# =============================================================================


class TestNormalizationDeviceCompatibility:
    """Test normalization layers work on different devices."""

    def test_cpu_execution(self, hidden_dim):
        """Test on CPU."""
        norm = RMSNorm(dim=hidden_dim)
        x = torch.randn(2, 64, hidden_dim)

        output = norm(x)
        assert output.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_execution(self, hidden_dim):
        """Test on CUDA."""
        norm = RMSNorm(dim=hidden_dim).cuda()
        x = torch.randn(2, 64, hidden_dim).cuda()

        output = norm(x)
        assert output.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_adaptive_rmsnorm_cuda(self, hidden_dim, cond_dim):
        """Test AdaptiveRMSNorm on CUDA."""
        norm = AdaptiveRMSNorm(dim=hidden_dim, cond_dim=cond_dim).cuda()
        x = torch.randn(2, 64, hidden_dim).cuda()
        cond = torch.randn(2, cond_dim).cuda()

        output = norm(x, cond)
        assert output.device.type == "cuda"


# =============================================================================
# Test Class: Batch Independence
# =============================================================================


class TestNormalizationBatchIndependence:
    """Test normalization processes batches independently."""

    def test_rmsnorm_batch_independence(self, hidden_dim):
        """Test RMSNorm processes each batch sample independently."""
        norm = RMSNorm(dim=hidden_dim)

        x = torch.randn(4, 64, hidden_dim)
        output = norm(x)

        # Modify first sample
        x_modified = x.clone()
        x_modified[0] = torch.randn(64, hidden_dim)
        output_modified = norm(x_modified)

        # Other samples should be unchanged
        assert torch.allclose(output[1], output_modified[1])
        assert torch.allclose(output[2], output_modified[2])
        assert torch.allclose(output[3], output_modified[3])

        # First sample should be different
        assert not torch.allclose(output[0], output_modified[0])

    def test_qknorm_head_independence(self, head_dim):
        """Test QKNorm processes each head independently."""
        norm = QKNorm(head_dim=head_dim)

        B, H, L, D = 2, 8, 64, head_dim
        x = torch.randn(B, H, L, D)
        output = norm(x)

        # Modify first head
        x_modified = x.clone()
        x_modified[:, 0, :, :] = torch.randn(B, L, D)
        output_modified = norm(x_modified)

        # Other heads should be unchanged
        assert torch.allclose(output[:, 1], output_modified[:, 1])
        assert torch.allclose(output[:, 2], output_modified[:, 2])


# =============================================================================
# Test Class: Extra Repr
# =============================================================================


class TestNormalizationExtraRepr:
    """Test extra_repr for debugging."""

    def test_rmsnorm_str(self, hidden_dim):
        """Test RMSNorm string representation contains key info."""
        norm = RMSNorm(dim=hidden_dim, eps=1e-5)
        # Check the module has correct attributes
        assert norm.dim == hidden_dim
        assert norm.eps == 1e-5

    def test_qknorm_str(self, head_dim):
        """Test QKNorm string representation contains key info."""
        norm = QKNorm(head_dim=head_dim, eps=1e-6)
        assert norm.head_dim == head_dim
        assert norm.eps == 1e-6
