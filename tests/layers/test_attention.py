"""
Attention Module Tests

Tests for:
- GatedMultiHeadAttention: GQA + QK Norm + Gating
- TokenCompaction: Compress-Attend-Expand Pipeline
- repeat_kv: KV head expansion for GQA
- configure_selective_dropout: Selective dropout configuration
- get_dropout_stats: Dropout statistics

Architecture:
    - GQA: 16 Q heads, 4 KV heads (4:1 ratio)
    - QK Norm: per-head RMSNorm
    - Gated Attention: headwise or elementwise gating
    - Token Compaction: p^4 = 65,536x attention cost reduction

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

import pytest
import torch
import torch.nn as nn

from src.models.attention import (
    GatedMultiHeadAttention,
    repeat_kv,
    create_attention,
    create_attention_from_config,
    configure_selective_dropout,
    get_dropout_stats,
    TokenCompaction,
    TokenCompactionNoResidual,
    create_token_compaction,
    create_token_compaction_from_config,
)
from src.config import PixelHDMConfig


# ============================================================================
# GatedMultiHeadAttention Tests
# ============================================================================

class TestRepeatKV:
    """Tests for repeat_kv function (GQA expansion)."""

    def test_repeat_kv_gqa_expansion(self):
        """Test GQA KV expansion from 4 heads to 16 heads (4:1 ratio)."""
        B, L, H_kv, D = 2, 32, 4, 64

        k = torch.randn(B, H_kv, L, D)
        v = torch.randn(B, H_kv, L, D)

        # Expand to 16 heads (4:1 ratio)
        k_expanded, v_expanded = repeat_kv(k, v, n_rep=4)

        assert k_expanded.shape == (B, 16, L, D)
        assert v_expanded.shape == (B, 16, L, D)

    def test_repeat_kv_no_expansion(self):
        """Test that n_rep=1 returns original tensors."""
        B, L, H, D = 2, 32, 8, 64

        k = torch.randn(B, H, L, D)
        v = torch.randn(B, H, L, D)

        k_out, v_out = repeat_kv(k, v, n_rep=1)

        # Should return same tensors
        assert torch.equal(k_out, k)
        assert torch.equal(v_out, v)

    def test_repeat_kv_correct_values(self):
        """Test that repeated values are correctly copied."""
        B, L, H_kv, D = 2, 32, 2, 32

        k = torch.randn(B, H_kv, L, D)
        v = torch.randn(B, H_kv, L, D)

        # Expand by 4 (2 -> 8 heads)
        k_expanded, v_expanded = repeat_kv(k, v, n_rep=4)

        # Check that heads 0-3 match original head 0
        assert torch.allclose(k_expanded[:, 0, :, :], k[:, 0, :, :])
        assert torch.allclose(k_expanded[:, 1, :, :], k[:, 0, :, :])
        assert torch.allclose(k_expanded[:, 2, :, :], k[:, 0, :, :])
        assert torch.allclose(k_expanded[:, 3, :, :], k[:, 0, :, :])

        # Check that heads 4-7 match original head 1
        assert torch.allclose(k_expanded[:, 4, :, :], k[:, 1, :, :])
        assert torch.allclose(k_expanded[:, 5, :, :], k[:, 1, :, :])
        assert torch.allclose(k_expanded[:, 6, :, :], k[:, 1, :, :])
        assert torch.allclose(k_expanded[:, 7, :, :], k[:, 1, :, :])


class TestGatedMultiHeadAttention:
    """Tests for GatedMultiHeadAttention module."""

    @pytest.fixture
    def attention(self):
        """Create GatedMultiHeadAttention with GQA 4:1."""
        return GatedMultiHeadAttention(
            hidden_dim=256,
            num_heads=8,
            num_kv_heads=2,  # GQA 4:1
            head_dim=32,
            dropout=0.0,
            use_qk_norm=True,
            use_gated_attention=True,
            gate_type="headwise",
            use_flash_attention=False,  # Use manual attention for testing
            use_checkpoint=False,
        )

    @pytest.fixture
    def attention_elementwise(self):
        """Create attention with elementwise gating."""
        return GatedMultiHeadAttention(
            hidden_dim=256,
            num_heads=8,
            num_kv_heads=2,
            head_dim=32,
            dropout=0.0,
            use_gated_attention=True,
            gate_type="elementwise",
            use_flash_attention=False,
            use_checkpoint=False,
        )

    def test_gqa_ratio_validation(self):
        """Test GQA ratio validation."""
        # Valid: 16 % 4 = 0
        attn = GatedMultiHeadAttention(
            hidden_dim=256,
            num_heads=16,
            num_kv_heads=4,
        )
        assert attn.n_rep == 4

        # Invalid: 16 % 3 != 0
        with pytest.raises(AssertionError):
            GatedMultiHeadAttention(
                hidden_dim=256,
                num_heads=16,
                num_kv_heads=3,
            )

    def test_qk_norm_application(self, attention):
        """Test that QK Norm is applied correctly."""
        assert attention.use_qk_norm
        assert hasattr(attention, 'q_norm')
        assert hasattr(attention, 'k_norm')

        # Test forward pass
        B, L, D = 2, 32, 256
        x = torch.randn(B, L, D)

        output = attention(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_gate_headwise(self, attention):
        """Test headwise gating (16 gates for 16 Q heads)."""
        assert attention.gate_type == "headwise"

        # Gate projection should output num_heads scalars
        assert attention.gate_proj.out_features == attention.num_heads

        # Test forward
        B, L, D = 2, 32, 256
        x = torch.randn(B, L, D)
        output = attention(x)

        assert output.shape == x.shape

    def test_gate_elementwise(self, attention_elementwise):
        """Test elementwise gating."""
        assert attention_elementwise.gate_type == "elementwise"

        # Gate projection should output num_heads * head_dim
        expected = attention_elementwise.num_heads * attention_elementwise.head_dim
        assert attention_elementwise.gate_proj.out_features == expected

        # Test forward
        B, L, D = 2, 32, 256
        x = torch.randn(B, L, D)
        output = attention_elementwise(x)

        assert output.shape == x.shape

    def test_flash_attention_enabled(self):
        """Test attention with Flash Attention enabled."""
        attn = GatedMultiHeadAttention(
            hidden_dim=256,
            num_heads=8,
            num_kv_heads=2,
            use_flash_attention=True,
            use_checkpoint=False,
        )

        assert attn.use_flash_attention

        B, L, D = 2, 32, 256
        x = torch.randn(B, L, D)
        output = attn(x)

        assert output.shape == x.shape

    def test_flash_attention_disabled(self, attention):
        """Test attention with Flash Attention disabled (manual implementation)."""
        assert not attention.use_flash_attention

        B, L, D = 2, 32, 256
        x = torch.randn(B, L, D)
        output = attention(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_gradient_checkpoint(self):
        """Test attention with gradient checkpointing."""
        attn = GatedMultiHeadAttention(
            hidden_dim=256,
            num_heads=8,
            num_kv_heads=2,
            use_checkpoint=True,
        )

        assert attn.use_checkpoint

        # In training mode, should use checkpointing
        attn.train()
        B, L, D = 2, 32, 256
        x = torch.randn(B, L, D, requires_grad=True)
        output = attn(x)

        assert output.shape == x.shape

    def test_dropout_training_mode(self):
        """Test dropout behavior in training mode."""
        attn = GatedMultiHeadAttention(
            hidden_dim=256,
            num_heads=8,
            num_kv_heads=2,
            dropout=0.1,
            use_flash_attention=False,
            use_checkpoint=False,
        )

        assert attn.dropout_p == 0.1

        attn.train()
        B, L, D = 2, 32, 256
        x = torch.randn(B, L, D)

        # Should not error
        output = attn(x)
        assert output.shape == x.shape

    def test_input_validation(self, attention):
        """Test input shape validation."""
        # Valid 3D input
        x_valid = torch.randn(2, 32, 256)
        output = attention(x_valid)
        assert output.shape == x_valid.shape

        # Invalid: wrong hidden_dim
        x_invalid = torch.randn(2, 32, 128)
        with pytest.raises(ValueError, match="輸入形狀"):
            attention(x_invalid)

        # Invalid: 2D input
        x_2d = torch.randn(32, 256)
        with pytest.raises(ValueError):
            attention(x_2d)

    def test_create_from_config(self):
        """Test creating attention from config."""
        config = PixelHDMConfig.for_testing()
        attn = create_attention_from_config(config)

        assert attn.hidden_dim == config.hidden_dim
        assert attn.num_heads == config.num_heads
        assert attn.num_kv_heads == config.num_kv_heads

    def test_attention_with_rope(self, attention):
        """Test attention with RoPE function."""
        from src.models.layers.rope import MRoPE
        from src.models.layers.rope.utils import create_position_ids_batched

        B, L, D = 2, 32, 256

        # Create Lumina2-style mRoPE (axes_dims must sum to head_dim=32)
        rope = MRoPE(
            head_dim=attention.head_dim,
            axes_dims=(8, 12, 12),
            max_seq_len=64,
            max_height=16,
            max_width=16,
        )

        x = torch.randn(B, L, D)

        # Create Lumina2-style position IDs
        # For simplicity, treat all as text tokens (L tokens)
        position_ids = create_position_ids_batched(
            batch_size=B,
            text_len=L,
            img_height=0,  # No image tokens
            img_width=0,
            patch_size=16,
            device=x.device,
        )

        # Create rope_fn wrapper
        def rope_fn(q, k, pos_ids=None):
            ids = pos_ids if pos_ids is not None else position_ids
            return rope(q, k, ids)

        output = attention(
            x,
            rope_fn=rope_fn,
            position_ids=position_ids,
        )

        assert output.shape == x.shape
        assert not torch.isnan(output).any()


class TestConfigureSelectiveDropout:
    """Tests for configure_selective_dropout utility."""

    def test_configure_selective_dropout_disable_attention(self):
        """Test disabling attention dropout."""
        attn = GatedMultiHeadAttention(
            hidden_dim=256,
            num_heads=8,
            num_kv_heads=2,
            dropout=0.2,
        )

        configure_selective_dropout(attn, disable_attention_dropout=True)

        assert attn.dropout_p == 0.0

    def test_configure_selective_dropout_set_rate(self):
        """Test setting dropout rate."""
        attn = GatedMultiHeadAttention(
            hidden_dim=256,
            num_heads=8,
            num_kv_heads=2,
            dropout=0.1,
        )

        configure_selective_dropout(attn, dropout_rate=0.3)

        assert attn.dropout_p == 0.3


class TestGetDropoutStats:
    """Tests for get_dropout_stats utility."""

    def test_get_dropout_stats_basic(self):
        """Test getting dropout statistics."""
        attn = GatedMultiHeadAttention(
            hidden_dim=256,
            num_heads=8,
            num_kv_heads=2,
            dropout=0.15,
        )

        stats = get_dropout_stats(attn)

        assert "total_dropout_modules" in stats
        assert "dropout_rates" in stats


# ============================================================================
# TokenCompaction Tests
# ============================================================================

class TestTokenCompaction:
    """Tests for TokenCompaction module."""

    @pytest.fixture
    def token_compaction(self):
        """Create TokenCompaction with default settings."""
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
    def token_compaction_from_config(self):
        """Create TokenCompaction from config."""
        config = PixelHDMConfig.for_testing()
        return TokenCompaction(config=config)

    def test_compress_flow(self, token_compaction):
        """Test compression flow (4096 -> 1024)."""
        B, L = 2, 16
        P2 = 256  # 16^2
        D_pix = 16

        x = torch.randn(B, L, P2, D_pix)

        output = token_compaction(x)

        # Output should have same shape as input (residual connection)
        assert output.shape == x.shape
        assert output.shape == (B, L, P2, D_pix)

    def test_expand_flow(self, token_compaction):
        """Test that expand restores original dimensions."""
        B, L = 2, 16
        P2 = 256
        D_pix = 16

        x = torch.randn(B, L, P2, D_pix)

        # Manually test compress then expand
        x_flat = x.reshape(B, L, P2 * D_pix)  # (B, L, 4096)
        x_comp = token_compaction.compress(x_flat)  # (B, L, hidden_dim)

        assert x_comp.shape == (B, L, token_compaction.hidden_dim)

        x_exp = token_compaction.expand(x_comp)  # (B, L, 4096)
        x_out = x_exp.reshape(B, L, P2, D_pix)

        assert x_out.shape == x.shape

    def test_compression_ratio(self, token_compaction):
        """Test compression ratio is 4x."""
        # p^2 * D_pix / hidden_dim = 256 * 16 / 1024 = 4
        expected_ratio = token_compaction.p2_d_pix / token_compaction.hidden_dim

        # Default: 4096 / 256 = 16 (in test config)
        # But in this fixture: 256 * 16 / 256 = 16
        assert expected_ratio == 16  # 256 * 16 / 256

    def test_attention_integration(self, token_compaction):
        """Test that attention is properly integrated."""
        assert hasattr(token_compaction, 'attention')
        assert isinstance(token_compaction.attention, GatedMultiHeadAttention)

        # Verify attention dimensions
        assert token_compaction.attention.hidden_dim == token_compaction.hidden_dim

    def test_normalization_stack(self, token_compaction):
        """Test normalization layers are present."""
        assert hasattr(token_compaction, 'norm')
        assert hasattr(token_compaction, 'post_norm')

    def test_residual_connection(self, token_compaction):
        """Test that residual connection is applied."""
        B, L = 2, 16
        P2 = 256
        D_pix = 16

        x = torch.randn(B, L, P2, D_pix)

        # With small expand_gain, output should be close to input initially
        output = token_compaction(x)

        # Residual should make output != all zeros
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_expand_init_gain(self, token_compaction):
        """Test expand layer initialization with gain=0.1."""
        assert token_compaction.expand_gain == 0.1

        # Check that expand weights are small
        expand_weight_std = token_compaction.expand.weight.std().item()
        compress_weight_std = token_compaction.compress.weight.std().item()

        # Expand should have smaller weights due to gain
        assert expand_weight_std < compress_weight_std

    def test_no_residual_variant(self):
        """Test TokenCompactionNoResidual variant."""
        tc_no_res = TokenCompactionNoResidual(
            config=None,
            hidden_dim=256,
            pixel_dim=16,
            patch_size=16,
            num_heads=8,
            num_kv_heads=2,
            use_checkpoint=False,
        )

        B, L = 2, 16
        P2 = 256
        D_pix = 16

        x = torch.randn(B, L, P2, D_pix)
        output = tc_no_res(x)

        assert output.shape == x.shape

        # No residual - output should be different from input
        # (unless by coincidence, which is unlikely)
        assert not torch.allclose(output, x, atol=0.1)

    def test_dimension_validation(self, token_compaction):
        """Test input dimension validation."""
        B, L = 2, 16
        P2 = 256
        D_pix = 16

        # Valid input
        x_valid = torch.randn(B, L, P2, D_pix)
        output = token_compaction(x_valid)
        assert output.shape == x_valid.shape

        # Invalid: 3D input
        x_3d = torch.randn(B, L, P2 * D_pix)
        with pytest.raises(ValueError, match="4D 張量"):
            token_compaction(x_3d)

    def test_batch_size_variation(self, token_compaction):
        """Test with different batch sizes."""
        P2 = 256
        D_pix = 16

        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 16, P2, D_pix)
            output = token_compaction(x)
            assert output.shape == x.shape

    def test_sequence_length_variation(self, token_compaction):
        """Test with different sequence lengths (number of patches)."""
        B = 2
        P2 = 256
        D_pix = 16

        for num_patches in [4, 16, 64, 256]:
            x = torch.randn(B, num_patches, P2, D_pix)
            output = token_compaction(x)
            assert output.shape == x.shape

    def test_mrope_integration(self, token_compaction):
        """Test TokenCompaction with mRoPE integration."""
        from src.models.layers.rope import MRoPE
        from src.models.layers.rope.utils import create_image_only_position_ids_batched

        B, L = 2, 16
        P2 = 256
        D_pix = 16

        # Create Lumina2-style mRoPE for attention head_dim
        head_dim = token_compaction.attention.head_dim
        rope = MRoPE(
            head_dim=head_dim,
            axes_dims=(8, 12, 12),
            max_seq_len=64,
            max_height=16,
            max_width=16,
        )

        x = torch.randn(B, L, P2, D_pix)

        # Create Lumina2-style position IDs for image-only (4x4 grid for L=16 patches)
        position_ids = create_image_only_position_ids_batched(
            batch_size=B,
            img_height=64,  # 4 patches * 16 pixels
            img_width=64,
            patch_size=16,
            device=x.device,
        )

        # Create rope_fn wrapper
        def rope_fn(q, k, pos_ids=None):
            ids = pos_ids if pos_ids is not None else position_ids
            return rope(q, k, ids)

        output = token_compaction(
            x,
            rope_fn=rope_fn,
            position_ids=position_ids,
        )

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_create_from_config(self, token_compaction_from_config):
        """Test creating TokenCompaction from config."""
        config = PixelHDMConfig.for_testing()

        assert token_compaction_from_config.hidden_dim == config.hidden_dim
        assert token_compaction_from_config.pixel_dim == config.pixel_dim
        assert token_compaction_from_config.patch_size == config.patch_size

    def test_gradient_flow(self, token_compaction):
        """Test that gradients flow through TokenCompaction."""
        B, L = 2, 16
        P2 = 256
        D_pix = 16

        x = torch.randn(B, L, P2, D_pix, requires_grad=True)
        output = token_compaction(x)
        loss = output.sum()
        loss.backward()

        # Input should have gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestCreateTokenCompaction:
    """Tests for TokenCompaction factory functions."""

    def test_create_token_compaction_default(self):
        """Test creating TokenCompaction with default parameters."""
        tc = create_token_compaction()

        assert tc.hidden_dim == 1024
        assert tc.pixel_dim == 16
        assert tc.patch_size == 16

    def test_create_token_compaction_custom(self):
        """Test creating TokenCompaction with custom parameters."""
        tc = create_token_compaction(
            hidden_dim=512,
            pixel_dim=8,
            patch_size=8,
            num_heads=8,
            num_kv_heads=2,
        )

        assert tc.hidden_dim == 512
        assert tc.pixel_dim == 8
        assert tc.patch_size == 8
        assert tc.p2 == 64  # 8^2

    def test_create_token_compaction_from_config(self):
        """Test creating TokenCompaction from config."""
        config = PixelHDMConfig.for_testing()
        tc = create_token_compaction_from_config(config)

        assert tc.hidden_dim == config.hidden_dim
        assert tc.pixel_dim == config.pixel_dim
        assert tc.patch_size == config.patch_size


class TestAttentionExtraRepr:
    """Tests for extra_repr methods."""

    def test_gated_attention_extra_repr(self):
        """Test GatedMultiHeadAttention extra_repr."""
        attn = GatedMultiHeadAttention(
            hidden_dim=256,
            num_heads=8,
            num_kv_heads=2,
            gate_type="headwise",
        )

        repr_str = attn.extra_repr()

        assert "hidden_dim=256" in repr_str
        assert "num_heads=8" in repr_str
        assert "num_kv_heads=2" in repr_str
        assert "headwise" in repr_str

    def test_token_compaction_extra_repr(self):
        """Test TokenCompaction extra_repr."""
        tc = create_token_compaction(
            hidden_dim=1024,
            pixel_dim=16,
            patch_size=16,
        )

        repr_str = tc.extra_repr()

        assert "compress" in repr_str
        assert "expand" in repr_str
        assert "patch_size=16" in repr_str
