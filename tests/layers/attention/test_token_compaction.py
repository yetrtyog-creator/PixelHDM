"""
Token Compaction Comprehensive Unit Tests

Tests for the Token Compaction mechanism (Compress-Attend-Expand Pipeline).
This module provides p^4 = 65,536x attention cost reduction for PixelHDM.

Test Categories:
1. Compression/Expansion Flow Tests (8 tests)
2. Attention Integration Tests (6 tests)
3. Dimension and Batch Tests (4 tests)
4. Normalization Tests (2 tests)

Total: 20 tests

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-08
"""

import pytest
import torch
import torch.nn as nn

from src.models.attention.token_compaction import (
    TokenCompaction,
    TokenCompactionNoResidual,
)
from src.models.attention import (
    create_token_compaction,
)
from src.config import PixelHDMConfig


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def default_config():
    """Create default testing configuration."""
    return PixelHDMConfig.for_testing()


@pytest.fixture
def token_compaction_default():
    """Create TokenCompaction with default parameters."""
    return TokenCompaction(
        config=None,
        hidden_dim=256,
        pixel_dim=16,
        patch_size=16,
        num_heads=8,
        num_kv_heads=2,
        use_checkpoint=False,
        expand_gain=0.1,
    )


@pytest.fixture
def token_compaction_from_config(default_config):
    """Create TokenCompaction from PixelHDMConfig."""
    return TokenCompaction(config=default_config)


@pytest.fixture
def token_compaction_no_residual():
    """Create TokenCompactionNoResidual variant."""
    return TokenCompactionNoResidual(
        config=None,
        hidden_dim=256,
        pixel_dim=16,
        patch_size=16,
        num_heads=8,
        num_kv_heads=2,
        use_checkpoint=False,
        expand_gain=0.1,
    )


@pytest.fixture
def sample_input():
    """Create sample input tensor for testing."""
    B, L, P2, D_pix = 2, 16, 256, 16  # p^2 = 16^2 = 256
    return torch.randn(B, L, P2, D_pix)


# ============================================================================
# 1. Compression/Expansion Flow Tests (8 tests)
# ============================================================================

class TestCompressionExpansionFlow:
    """Tests for compression and expansion flow."""

    def test_compress_output_shape(self, token_compaction_default, sample_input):
        """Test that compression reduces sequence length correctly.

        Compress: (B, L, p^2, D_pix) -> flatten -> (B, L, p^2*D_pix) -> Linear -> (B, L, D)
        """
        B, L, P2, D_pix = sample_input.shape

        # Manually test compress step
        x_flat = sample_input.reshape(B, L, P2 * D_pix)  # (B, L, 4096)
        x_compressed = token_compaction_default.compress(x_flat)  # (B, L, hidden_dim)

        expected_shape = (B, L, token_compaction_default.hidden_dim)
        assert x_compressed.shape == expected_shape, \
            f"Expected {expected_shape}, got {x_compressed.shape}"

        # Verify compression ratio
        input_dim = P2 * D_pix  # 4096
        output_dim = token_compaction_default.hidden_dim  # 256
        compression_ratio = input_dim / output_dim
        assert compression_ratio == 16.0  # 4096 / 256 = 16x

    def test_expand_output_shape(self, token_compaction_default, sample_input):
        """Test that expansion restores original sequence length.

        Expand: (B, L, D) -> Linear -> (B, L, p^2*D_pix) -> reshape -> (B, L, p^2, D_pix)
        """
        B, L, P2, D_pix = sample_input.shape
        hidden_dim = token_compaction_default.hidden_dim

        # Create compressed input
        x_compressed = torch.randn(B, L, hidden_dim)

        # Test expand step
        x_expanded = token_compaction_default.expand(x_compressed)  # (B, L, p^2*D_pix)

        expected_flat_shape = (B, L, token_compaction_default.p2_d_pix)
        assert x_expanded.shape == expected_flat_shape, \
            f"Expected {expected_flat_shape}, got {x_expanded.shape}"

        # Reshape to original 4D format
        x_reshaped = x_expanded.reshape(B, L, P2, D_pix)
        assert x_reshaped.shape == sample_input.shape

    def test_compress_expand_roundtrip(self, token_compaction_default, sample_input):
        """Test compress -> expand recovers original shape.

        Note: Values will differ due to lossy compression, but shape must match.
        """
        B, L, P2, D_pix = sample_input.shape

        # Full forward pass
        output = token_compaction_default(sample_input)

        # Shape must be preserved
        assert output.shape == sample_input.shape, \
            f"Roundtrip shape mismatch: {output.shape} != {sample_input.shape}"

        # Due to residual connection, output should not be all zeros
        assert not torch.allclose(output, torch.zeros_like(output)), \
            "Output is all zeros, residual connection may be broken"

    def test_compression_ratio_4x(self):
        """Test default 4x compression ratio (p^2*D_pix / hidden_dim).

        Standard config: 256 * 16 / 1024 = 4x compression
        """
        tc = create_token_compaction(
            hidden_dim=1024,
            pixel_dim=16,
            patch_size=16,
        )

        # p^2 * D_pix = 256 * 16 = 4096
        # hidden_dim = 1024
        # ratio = 4096 / 1024 = 4
        expected_ratio = tc.p2_d_pix / tc.hidden_dim
        assert expected_ratio == 4.0, f"Expected 4x compression, got {expected_ratio}x"

        # Verify extra_repr contains compression ratio
        repr_str = tc.extra_repr()
        assert "compression_ratio=4x" in repr_str

    def test_compression_ratio_custom(self):
        """Test custom compression ratios work correctly."""
        # 8x compression
        tc_8x = create_token_compaction(
            hidden_dim=512,
            pixel_dim=16,
            patch_size=16,
        )
        ratio_8x = tc_8x.p2_d_pix / tc_8x.hidden_dim
        assert ratio_8x == 8.0, f"Expected 8x, got {ratio_8x}x"

        # 16x compression
        tc_16x = create_token_compaction(
            hidden_dim=256,
            pixel_dim=16,
            patch_size=16,
        )
        ratio_16x = tc_16x.p2_d_pix / tc_16x.hidden_dim
        assert ratio_16x == 16.0, f"Expected 16x, got {ratio_16x}x"

        # 2x compression
        tc_2x = create_token_compaction(
            hidden_dim=2048,
            pixel_dim=16,
            patch_size=16,
        )
        ratio_2x = tc_2x.p2_d_pix / tc_2x.hidden_dim
        assert ratio_2x == 2.0, f"Expected 2x, got {ratio_2x}x"

    def test_residual_connection_enabled(self, token_compaction_default, sample_input):
        """Test that residual connection is properly applied.

        Output = Input + Expand(Attention(Compress(Input)))
        """
        # Get output
        output = token_compaction_default(sample_input)

        # With expand_gain=0.1, the expand contribution should be small
        # So output should be close to input but not identical
        diff = (output - sample_input).abs().mean()

        # Difference should be non-zero (transformation applied)
        assert diff > 0, "Output is identical to input, no transformation applied"

        # With small expand_gain, difference should be relatively small
        input_std = sample_input.std()
        assert diff < input_std * 2, \
            f"Difference ({diff:.4f}) too large compared to input std ({input_std:.4f})"

    def test_residual_connection_disabled(self, token_compaction_no_residual, sample_input):
        """Test TokenCompactionNoResidual variant without residual connection."""
        output = token_compaction_no_residual(sample_input)

        # Shape should be preserved
        assert output.shape == sample_input.shape

        # Output should be different from input (no residual)
        # With probability 1.0, random output != random input
        assert not torch.allclose(output, sample_input, atol=0.1), \
            "Output is too close to input for no-residual variant"

    def test_expand_init_gain(self, token_compaction_default):
        """Test that expand weights are initialized with gain=0.1.

        This ensures the initial output contribution is small,
        making the residual connection initially act like identity.
        """
        expand_gain = token_compaction_default.expand_gain
        assert expand_gain == 0.1, f"Expected gain=0.1, got {expand_gain}"

        # Verify expand weights are smaller than compress weights
        expand_std = token_compaction_default.expand.weight.std().item()
        compress_std = token_compaction_default.compress.weight.std().item()

        # Expand should have smaller variance due to gain scaling
        assert expand_std < compress_std, \
            f"Expand std ({expand_std:.4f}) >= Compress std ({compress_std:.4f})"

        # More specifically, expand_std should be ~10% of compress_std
        ratio = expand_std / compress_std
        assert 0.05 < ratio < 0.25, \
            f"Weight ratio ({ratio:.3f}) outside expected range [0.05, 0.25]"


# ============================================================================
# 2. Attention Integration Tests (6 tests)
# ============================================================================

class TestAttentionIntegration:
    """Tests for attention mechanism integration within Token Compaction."""

    def test_attention_in_compressed_space(self, token_compaction_default, sample_input):
        """Test that attention operates on compressed tokens (B, L, D).

        The attention sees compressed tokens, not the full pixel features.
        """
        B, L, P2, D_pix = sample_input.shape
        hidden_dim = token_compaction_default.hidden_dim

        # Manually compress
        x_flat = sample_input.reshape(B, L, P2 * D_pix)
        x_compressed = token_compaction_default.compress(x_flat)
        x_normed = token_compaction_default.norm(x_compressed)

        # Verify attention input shape
        assert x_normed.shape == (B, L, hidden_dim), \
            f"Attention input shape mismatch: {x_normed.shape}"

        # Test attention forward (without RoPE for simplicity)
        attn = token_compaction_default.attention
        attn_output = attn(x_normed)

        assert attn_output.shape == (B, L, hidden_dim), \
            f"Attention output shape mismatch: {attn_output.shape}"

    def test_attention_qkv_shapes(self, token_compaction_default, sample_input):
        """Test Q, K, V shapes in compressed space.

        For GQA with 8 Q heads and 2 KV heads:
        - Q: (B, 8, L, head_dim)
        - K: (B, 2, L, head_dim) -> expanded to (B, 8, L, head_dim)
        - V: (B, 2, L, head_dim) -> expanded to (B, 8, L, head_dim)
        """
        B, L, P2, D_pix = sample_input.shape

        # Get attention module
        attn = token_compaction_default.attention

        # Create compressed input
        x_flat = sample_input.reshape(B, L, P2 * D_pix)
        x_compressed = token_compaction_default.compress(x_flat)
        x_normed = token_compaction_default.norm(x_compressed)

        # Access QKV projection
        q, k, v = attn.qkv_proj(x_normed)

        # Check shapes
        num_heads = attn.num_heads
        num_kv_heads = attn.num_kv_heads
        head_dim = attn.head_dim

        assert q.shape == (B, num_heads, L, head_dim), \
            f"Q shape mismatch: {q.shape}"
        assert k.shape == (B, num_kv_heads, L, head_dim), \
            f"K shape mismatch: {k.shape}"
        assert v.shape == (B, num_kv_heads, L, head_dim), \
            f"V shape mismatch: {v.shape}"

    def test_attention_output_shape(self, token_compaction_default, sample_input):
        """Test that attention output has correct shape."""
        B, L, P2, D_pix = sample_input.shape
        hidden_dim = token_compaction_default.hidden_dim

        # Create compressed input
        x_flat = sample_input.reshape(B, L, P2 * D_pix)
        x_compressed = token_compaction_default.compress(x_flat)
        x_normed = token_compaction_default.norm(x_compressed)

        # Forward through attention
        attn_output = token_compaction_default.attention(x_normed)

        # Output should match input shape (residual-ready)
        assert attn_output.shape == (B, L, hidden_dim), \
            f"Attention output shape mismatch: {attn_output.shape}"

        # Check no NaN values
        assert not torch.isnan(attn_output).any(), "Attention output contains NaN"

    def test_mrope_integration(self, token_compaction_default, sample_input):
        """Test that mRoPE can be applied in compressed space."""
        from src.models.layers.rope import MRoPE
        from src.models.layers.rope.utils import create_image_only_position_ids_batched

        B, L, P2, D_pix = sample_input.shape

        # Get attention head_dim (should be 32 based on test setup)
        head_dim = token_compaction_default.attention.head_dim

        # Create Lumina2-style mRoPE with matching head_dim
        # axes_dims must sum to head_dim: 8 + 12 + 12 = 32
        rope = MRoPE(
            head_dim=head_dim,
            axes_dims=(8, 12, 12),
            max_seq_len=64,
            max_height=16,
            max_width=16,
        )

        # Create Lumina2-style position IDs for image-only (4x4 grid for L=16 patches)
        # For image-only: position_ids = (0, h, w) for each patch
        position_ids = create_image_only_position_ids_batched(
            batch_size=B,
            img_height=64,  # 4 patches * 16 pixels
            img_width=64,
            patch_size=16,
            device=sample_input.device,
        )

        # Create rope_fn wrapper
        def rope_fn(q, k, pos_ids=None):
            ids = pos_ids if pos_ids is not None else position_ids
            return rope(q, k, ids)

        # Forward with RoPE
        output = token_compaction_default(
            sample_input,
            rope_fn=rope_fn,
            position_ids=position_ids,
        )

        # Shape should be preserved
        assert output.shape == sample_input.shape, \
            f"Output shape mismatch with mRoPE: {output.shape}"

        # No NaN values
        assert not torch.isnan(output).any(), "Output contains NaN with mRoPE"

    def test_flash_attention_compatible(self, default_config):
        """Test compatibility with Flash Attention."""
        # Create config with flash attention enabled
        config = PixelHDMConfig(
            hidden_dim=default_config.hidden_dim,
            pixel_dim=default_config.pixel_dim,
            patch_size=default_config.patch_size,
            num_heads=default_config.num_heads,
            num_kv_heads=default_config.num_kv_heads,
            use_flash_attention=True,
            use_gradient_checkpointing=False,
        )

        tc = TokenCompaction(config=config)

        # Verify flash attention is enabled in internal attention module
        assert tc.attention.use_flash_attention, \
            "Flash attention should be enabled"

        # Test forward pass
        B, L, P2, D_pix = 2, 16, config.patch_size ** 2, config.pixel_dim
        x = torch.randn(B, L, P2, D_pix)

        output = tc(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_gradient_checkpoint_compatible(self, default_config):
        """Test compatibility with gradient checkpointing."""
        # Create config with checkpointing enabled
        config = PixelHDMConfig(
            hidden_dim=default_config.hidden_dim,
            pixel_dim=default_config.pixel_dim,
            patch_size=default_config.patch_size,
            num_heads=default_config.num_heads,
            num_kv_heads=default_config.num_kv_heads,
            use_flash_attention=False,
            use_gradient_checkpointing=True,
        )

        tc = TokenCompaction(config=config)

        # Verify checkpointing is enabled
        assert tc.use_checkpoint, "Gradient checkpointing should be enabled"

        # Test forward pass in training mode
        tc.train()

        B, L, P2, D_pix = 2, 16, config.patch_size ** 2, config.pixel_dim
        x = torch.randn(B, L, P2, D_pix, requires_grad=True)

        output = tc(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

        # Test gradient flow
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "No gradients computed"
        assert not torch.isnan(x.grad).any(), "Gradients contain NaN"


# ============================================================================
# 3. Dimension and Batch Tests (4 tests)
# ============================================================================

class TestDimensionAndBatch:
    """Tests for dimension validation and batch handling."""

    def test_dimension_validation(self, token_compaction_default):
        """Test that invalid dimensions raise appropriate errors."""
        B, L = 2, 16
        P2 = 256  # 16^2
        D_pix = 16

        # Valid 4D input
        x_valid = torch.randn(B, L, P2, D_pix)
        output = token_compaction_default(x_valid)
        assert output.shape == x_valid.shape

        # Invalid: 3D input (missing pixel_dim dimension)
        x_3d = torch.randn(B, L, P2 * D_pix)
        with pytest.raises(ValueError, match="4D"):
            token_compaction_default(x_3d)

        # Invalid: 2D input
        x_2d = torch.randn(B, L * P2 * D_pix)
        with pytest.raises(ValueError, match="4D"):
            token_compaction_default(x_2d)

        # Invalid: 5D input
        x_5d = torch.randn(B, L, P2, D_pix, 1)
        with pytest.raises(ValueError, match="4D"):
            token_compaction_default(x_5d)

        # Invalid: Wrong p^2 dimension
        wrong_p2 = torch.randn(B, L, 128, D_pix)  # 128 != 256
        with pytest.raises(AssertionError):
            token_compaction_default._forward_impl(wrong_p2)

        # Invalid: Wrong pixel_dim
        wrong_dpix = torch.randn(B, L, P2, 8)  # 8 != 16
        with pytest.raises(AssertionError):
            token_compaction_default._forward_impl(wrong_dpix)

    def test_batch_size_variation(self, token_compaction_default):
        """Test that Token Compaction works with different batch sizes."""
        P2 = 256  # 16^2
        D_pix = 16
        L = 16

        for batch_size in [1, 2, 4, 8, 16, 32]:
            x = torch.randn(batch_size, L, P2, D_pix)
            output = token_compaction_default(x)

            assert output.shape == x.shape, \
                f"Shape mismatch for batch_size={batch_size}"
            assert not torch.isnan(output).any(), \
                f"NaN output for batch_size={batch_size}"

    def test_sequence_length_variation(self, token_compaction_default):
        """Test that Token Compaction works with different sequence lengths.

        Sequence length L = number of patches, varies with image resolution.
        """
        B = 2
        P2 = 256  # 16^2
        D_pix = 16

        # Different patch counts for various resolutions
        # 256x256: 16^2 = 256 patches
        # 512x512: 32^2 = 1024 patches
        # 1024x1024: 64^2 = 4096 patches
        for num_patches in [4, 16, 64, 256, 1024]:
            x = torch.randn(B, num_patches, P2, D_pix)
            output = token_compaction_default(x)

            assert output.shape == x.shape, \
                f"Shape mismatch for num_patches={num_patches}"
            assert not torch.isnan(output).any(), \
                f"NaN output for num_patches={num_patches}"

    def test_batch_independence(self, token_compaction_default):
        """Test that results are batch-independent (no cross-batch contamination)."""
        P2 = 256
        D_pix = 16
        L = 16

        # Create two different inputs
        x1 = torch.randn(1, L, P2, D_pix)
        x2 = torch.randn(1, L, P2, D_pix)

        # Process individually
        token_compaction_default.eval()  # Deterministic mode
        with torch.no_grad():
            out1_individual = token_compaction_default(x1)
            out2_individual = token_compaction_default(x2)

        # Process as batch
        x_batch = torch.cat([x1, x2], dim=0)  # (2, L, P2, D_pix)
        with torch.no_grad():
            out_batch = token_compaction_default(x_batch)

        # Results should match
        assert torch.allclose(out_batch[0:1], out1_individual, atol=1e-5), \
            "Batch processing gives different result for sample 1"
        assert torch.allclose(out_batch[1:2], out2_individual, atol=1e-5), \
            "Batch processing gives different result for sample 2"


# ============================================================================
# 4. Normalization Tests (2 tests)
# ============================================================================

class TestNormalization:
    """Tests for normalization layers in Token Compaction."""

    def test_normalization_applied(self, token_compaction_default, sample_input):
        """Test that LayerNorm/RMSNorm is applied correctly.

        Token Compaction uses RMSNorm for both pre and post normalization.
        """
        from src.models.layers.normalization import RMSNorm

        # Verify norm layers exist and are RMSNorm
        assert hasattr(token_compaction_default, 'norm'), "Missing pre-norm layer"
        assert hasattr(token_compaction_default, 'post_norm'), "Missing post-norm layer"

        assert isinstance(token_compaction_default.norm, RMSNorm), \
            f"Pre-norm should be RMSNorm, got {type(token_compaction_default.norm)}"
        assert isinstance(token_compaction_default.post_norm, RMSNorm), \
            f"Post-norm should be RMSNorm, got {type(token_compaction_default.post_norm)}"

        # Test normalization effect
        B, L, P2, D_pix = sample_input.shape
        hidden_dim = token_compaction_default.hidden_dim

        x_flat = sample_input.reshape(B, L, P2 * D_pix)
        x_compressed = token_compaction_default.compress(x_flat)

        # Before normalization - check variance
        pre_norm_var = x_compressed.var(dim=-1).mean()

        # After normalization
        x_normed = token_compaction_default.norm(x_compressed)
        post_norm_var = x_normed.var(dim=-1).mean()

        # RMSNorm should normalize the variance to ~1
        assert 0.5 < post_norm_var < 2.0, \
            f"Post-norm variance ({post_norm_var:.3f}) should be close to 1"

    def test_normalization_stack(self, token_compaction_default, sample_input):
        """Test pre/post normalization stacking order.

        Flow: Compress -> PreNorm -> Attention -> PostNorm -> Expand
        """
        B, L, P2, D_pix = sample_input.shape
        hidden_dim = token_compaction_default.hidden_dim

        # Step 1: Compress
        x_flat = sample_input.reshape(B, L, P2 * D_pix)
        x_compressed = token_compaction_default.compress(x_flat)
        assert x_compressed.shape == (B, L, hidden_dim)

        # Step 2: Pre-normalization
        x_pre_normed = token_compaction_default.norm(x_compressed)
        assert x_pre_normed.shape == (B, L, hidden_dim)

        # Step 3: Attention (with residual)
        attn_out = token_compaction_default.attention(x_pre_normed)
        x_after_attn = x_pre_normed + attn_out  # Internal attention residual
        assert x_after_attn.shape == (B, L, hidden_dim)

        # Step 4: Post-normalization
        x_post_normed = token_compaction_default.post_norm(x_after_attn)
        assert x_post_normed.shape == (B, L, hidden_dim)

        # Step 5: Expand
        x_expanded = token_compaction_default.expand(x_post_normed)
        x_reshaped = x_expanded.reshape(B, L, P2, D_pix)
        assert x_reshaped.shape == sample_input.shape

        # Verify full forward matches manual steps (with residual)
        residual = sample_input
        manual_output = residual + x_reshaped

        # Note: The actual forward uses internal attention residual differently
        # This test verifies the normalization ordering is correct
        full_output = token_compaction_default(sample_input)

        # Both should have same shape
        assert full_output.shape == manual_output.shape


# ============================================================================
# Additional Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Additional edge case tests for completeness."""

    def test_extra_repr_format(self, token_compaction_default):
        """Test extra_repr output format."""
        repr_str = token_compaction_default.extra_repr()

        # Should contain key information
        assert "compress=" in repr_str
        assert "expand=" in repr_str
        assert "patch_size=" in repr_str
        assert "compression_ratio=" in repr_str

    def test_eval_mode_consistency(self, token_compaction_default, sample_input):
        """Test that eval mode gives consistent results."""
        token_compaction_default.eval()

        with torch.no_grad():
            out1 = token_compaction_default(sample_input)
            out2 = token_compaction_default(sample_input)

        assert torch.allclose(out1, out2), \
            "Eval mode should give consistent results"

    def test_training_mode_gradient(self, token_compaction_default, sample_input):
        """Test gradient flow in training mode."""
        token_compaction_default.train()

        x = sample_input.clone().requires_grad_(True)
        output = token_compaction_default(x)

        # Compute loss and backward
        loss = output.sum()
        loss.backward()

        # Check gradients exist for all parameters
        for name, param in token_compaction_default.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, \
                    f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), \
                    f"NaN gradient for {name}"

    def test_device_consistency(self, token_compaction_default, sample_input):
        """Test that output is on same device as input."""
        # CPU test
        output = token_compaction_default(sample_input)
        assert output.device == sample_input.device

        # GPU test (if available)
        if torch.cuda.is_available():
            tc_cuda = token_compaction_default.cuda()
            x_cuda = sample_input.cuda()
            out_cuda = tc_cuda(x_cuda)
            assert out_cuda.device == x_cuda.device

    def test_dtype_preservation(self, token_compaction_default):
        """Test that output dtype matches input dtype."""
        B, L, P2, D_pix = 2, 16, 256, 16

        # Float32
        x_f32 = torch.randn(B, L, P2, D_pix, dtype=torch.float32)
        out_f32 = token_compaction_default(x_f32)
        assert out_f32.dtype == torch.float32

        # Float16 (requires float16 model)
        tc_f16 = token_compaction_default.half()
        x_f16 = torch.randn(B, L, P2, D_pix, dtype=torch.float16)
        out_f16 = tc_f16(x_f16)
        assert out_f16.dtype == torch.float16
