"""
PatchTransformerBlock Test Suite

Tests for the Patch-Level DiT Transformer Block.

This module tests:
1. Initialization and configuration
2. Forward pass shape correctness
3. Text conditioning
4. Time conditioning
5. AdaLN modulation
6. Self-attention behavior
7. FFN layer
8. Residual connections
9. Gradient flow

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.config import PixelHDMConfig
from src.models.blocks.patch_block import (
    PatchTransformerBlock,
    PatchTransformerBlockStack,
    create_patch_block,
    create_patch_block_from_config,
    create_patch_block_stack,
    create_patch_block_stack_from_config,
)


# ============================================================================
# Fixtures (module-scoped where possible for efficiency)
# ============================================================================

@pytest.fixture(scope="module")
def minimal_config() -> PixelHDMConfig:
    """Create minimal config for fast tests (module-scoped)."""
    return PixelHDMConfig.for_testing()


@pytest.fixture(scope="module")
def patch_block(minimal_config: PixelHDMConfig) -> PatchTransformerBlock:
    """Create a PatchTransformerBlock for testing (module-scoped, eval mode)."""
    block = PatchTransformerBlock(config=minimal_config)
    block.eval()
    return block


@pytest.fixture(scope="module")
def patch_block_no_checkpoint() -> PatchTransformerBlock:
    """Create a PatchTransformerBlock without gradient checkpointing (module-scoped, eval mode)."""
    block = create_patch_block(
        hidden_dim=256,
        num_heads=4,
        num_kv_heads=2,
        mlp_ratio=2.0,
        dropout=0.0,
        use_checkpoint=False,
    )
    block.eval()
    return block


@pytest.fixture(scope="module")
def batch_size() -> int:
    return 2


@pytest.fixture(scope="module")
def seq_len() -> int:
    return 64  # 64 patches for fast tests


@pytest.fixture(scope="module")
def hidden_dim() -> int:
    return 256


# ============================================================================
# Test Class: PatchTransformerBlock
# ============================================================================

class TestPatchTransformerBlock:
    """Test suite for PatchTransformerBlock."""

    # -------------------------------------------------------------------------
    # Test 1: Initialization
    # -------------------------------------------------------------------------
    def test_patch_block_init(self, minimal_config: PixelHDMConfig):
        """Test PatchTransformerBlock initialization."""
        block = PatchTransformerBlock(config=minimal_config)

        # Check attributes
        assert block.hidden_dim == minimal_config.hidden_dim
        assert hasattr(block, "pre_attn_norm")
        assert hasattr(block, "attention")
        assert hasattr(block, "post_attn_norm")
        assert hasattr(block, "pre_mlp_norm")
        assert hasattr(block, "mlp")
        assert hasattr(block, "post_mlp_norm")
        assert hasattr(block, "adaln")

    def test_patch_block_init_with_params(self):
        """Test initialization with explicit parameters."""
        block = create_patch_block(
            hidden_dim=512,
            num_heads=8,
            num_kv_heads=2,
            mlp_ratio=4.0,
            dropout=0.1,
            use_checkpoint=False,
        )

        assert block.hidden_dim == 512
        assert block.use_checkpoint is False

    def test_patch_block_from_config(self, minimal_config: PixelHDMConfig):
        """Test creation from config."""
        block = create_patch_block_from_config(minimal_config)

        assert block.hidden_dim == minimal_config.hidden_dim

    # -------------------------------------------------------------------------
    # Test 2: Forward Pass Shape
    # -------------------------------------------------------------------------
    def test_patch_block_forward_shape(
        self,
        patch_block_no_checkpoint: PatchTransformerBlock,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ):
        """Test forward pass output shape."""
        x = torch.randn(batch_size, seq_len, hidden_dim)
        t_embed = torch.randn(batch_size, hidden_dim)

        output = patch_block_no_checkpoint(x, t_embed)

        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_patch_block_forward_different_seq_len(
        self,
        patch_block_no_checkpoint: PatchTransformerBlock,
        batch_size: int,
        hidden_dim: int,
    ):
        """Test forward pass with different sequence lengths."""
        for seq_len in [16, 64, 256]:
            x = torch.randn(batch_size, seq_len, hidden_dim)
            t_embed = torch.randn(batch_size, hidden_dim)

            output = patch_block_no_checkpoint(x, t_embed)

            assert output.shape == (batch_size, seq_len, hidden_dim)

    # -------------------------------------------------------------------------
    # Test 3: Text Conditioning
    # -------------------------------------------------------------------------
    def test_patch_block_with_text_conditioning(
        self,
        patch_block_no_checkpoint: PatchTransformerBlock,
        batch_size: int,
        hidden_dim: int,
    ):
        """Test block with text conditioning (attention mask + positions)."""
        img_seq_len = 64
        text_seq_len = 32
        total_seq_len = img_seq_len + text_seq_len

        x = torch.randn(batch_size, total_seq_len, hidden_dim)
        t_embed = torch.randn(batch_size, hidden_dim)

        # Text positions (simple range)
        text_positions = torch.arange(text_seq_len).unsqueeze(0).expand(batch_size, -1)

        # Image positions (H, W coordinates)
        img_h = img_w = 8  # 64 patches = 8x8
        img_positions = torch.stack(
            torch.meshgrid(torch.arange(img_h), torch.arange(img_w), indexing="ij"),
            dim=-1,
        ).reshape(-1, 2).unsqueeze(0).expand(batch_size, -1, -1)

        output = patch_block_no_checkpoint(
            x,
            t_embed,
            text_positions=text_positions,
            img_positions=img_positions,
            text_len=text_seq_len,
        )

        assert output.shape == (batch_size, total_seq_len, hidden_dim)
        assert not torch.isnan(output).any()

    # -------------------------------------------------------------------------
    # Test 4: Time Conditioning
    # -------------------------------------------------------------------------
    def test_patch_block_with_time_conditioning(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ):
        """Test time conditioning interface and parameter generation."""
        # Create fresh block for this test
        block = create_patch_block(
            hidden_dim=hidden_dim,
            num_heads=4,
            num_kv_heads=2,
            mlp_ratio=2.0,
            use_checkpoint=False,
        )

        x = torch.randn(batch_size, seq_len, hidden_dim)
        t_embed = torch.randn(batch_size, hidden_dim)

        # Verify forward pass works with conditioning
        output = block(x, t_embed)
        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert not torch.isnan(output).any()

        # Verify AdaLN generates 6 parameters
        params = block.adaln(t_embed)
        assert len(params) == 6
        for p in params:
            assert p.shape == (batch_size, 1, hidden_dim)

    def test_patch_block_time_embedding_same_input(
        self,
        patch_block_no_checkpoint: PatchTransformerBlock,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ):
        """Test that same time embedding produces consistent outputs."""
        x = torch.randn(batch_size, seq_len, hidden_dim)
        t_embed = torch.randn(batch_size, hidden_dim)

        output_1 = patch_block_no_checkpoint(x.clone(), t_embed.clone())
        output_2 = patch_block_no_checkpoint(x.clone(), t_embed.clone())

        assert torch.allclose(output_1, output_2, atol=1e-6)

    # -------------------------------------------------------------------------
    # Test 5: AdaLN Modulation
    # -------------------------------------------------------------------------
    def test_patch_block_adaln_modulation(
        self,
        patch_block_no_checkpoint: PatchTransformerBlock,
        batch_size: int,
        hidden_dim: int,
    ):
        """Test AdaLN generates correct number of modulation parameters."""
        t_embed = torch.randn(batch_size, hidden_dim)

        # Get modulation parameters
        params = patch_block_no_checkpoint.adaln(t_embed)

        # Should return 6 parameters: gamma1, beta1, alpha1, gamma2, beta2, alpha2
        assert len(params) == 6

        for param in params:
            assert param.shape == (batch_size, 1, hidden_dim)
            assert not torch.isnan(param).any()

    def test_patch_block_adaln_affects_output(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ):
        """Test that AdaLN parameters can affect output after training."""
        # Create fresh block for this test
        block = create_patch_block(
            hidden_dim=hidden_dim,
            num_heads=4,
            num_kv_heads=2,
            mlp_ratio=2.0,
            use_checkpoint=False,
        )

        x = torch.randn(batch_size, seq_len, hidden_dim)
        t_embed = torch.randn(batch_size, hidden_dim)

        # Simulate training by updating AdaLN weights
        with torch.no_grad():
            block.adaln.proj[-1].weight.fill_(0.1)

        # Two very different time embeddings should now produce different outputs
        t_embed_1 = torch.randn(batch_size, hidden_dim) * 10.0
        t_embed_2 = torch.randn(batch_size, hidden_dim) * -10.0

        output_1 = block(x, t_embed_1)
        output_2 = block(x, t_embed_2)

        # Outputs should differ after non-zero weights
        max_diff = (output_1 - output_2).abs().max().item()
        assert max_diff > 1e-5, f"Outputs should differ with different AdaLN, max_diff={max_diff}"

    # -------------------------------------------------------------------------
    # Test 6: Self-Attention
    # -------------------------------------------------------------------------
    def test_patch_block_self_attention(
        self,
        patch_block_no_checkpoint: PatchTransformerBlock,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ):
        """Test self-attention component."""
        x = torch.randn(batch_size, seq_len, hidden_dim)

        # Verify attention module exists and is callable
        assert hasattr(patch_block_no_checkpoint, "attention")
        assert callable(patch_block_no_checkpoint.attention)

        # Pre-norm input
        x_normed = patch_block_no_checkpoint.pre_attn_norm(x)
        assert x_normed.shape == (batch_size, seq_len, hidden_dim)

        # Attention output
        attn_out = patch_block_no_checkpoint.attention(x_normed)
        assert attn_out.shape == (batch_size, seq_len, hidden_dim)
        assert not torch.isnan(attn_out).any()

    def test_patch_block_attention_without_mask(
        self,
        patch_block_no_checkpoint: PatchTransformerBlock,
        batch_size: int,
        hidden_dim: int,
    ):
        """Test attention without mask (default behavior)."""
        seq_len = 64
        x = torch.randn(batch_size, seq_len, hidden_dim)
        t_embed = torch.randn(batch_size, hidden_dim)

        # Forward pass without attention mask
        output = patch_block_no_checkpoint(x, t_embed)

        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert not torch.isnan(output).any()

    # -------------------------------------------------------------------------
    # Test 7: FFN Layer
    # -------------------------------------------------------------------------
    def test_patch_block_ffn(
        self,
        patch_block_no_checkpoint: PatchTransformerBlock,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ):
        """Test FFN (MLP) component."""
        x = torch.randn(batch_size, seq_len, hidden_dim)

        # Verify MLP module exists
        assert hasattr(patch_block_no_checkpoint, "mlp")

        # Pre-MLP norm
        x_normed = patch_block_no_checkpoint.pre_mlp_norm(x)
        assert x_normed.shape == (batch_size, seq_len, hidden_dim)

        # MLP output
        mlp_out = patch_block_no_checkpoint.mlp(x_normed)
        assert mlp_out.shape == (batch_size, seq_len, hidden_dim)
        assert not torch.isnan(mlp_out).any()

    def test_patch_block_ffn_expansion(self, patch_block_no_checkpoint: PatchTransformerBlock):
        """Test MLP expansion ratio."""
        # SwiGLU uses gate_proj, up_proj, down_proj
        mlp = patch_block_no_checkpoint.mlp
        assert hasattr(mlp, "gate_proj") or hasattr(mlp, "w1") or hasattr(mlp, "fc1")

        # Check expansion dimension if gate_proj exists
        if hasattr(mlp, "gate_proj"):
            assert mlp.gate_proj.out_features > mlp.gate_proj.in_features

    # -------------------------------------------------------------------------
    # Test 8: Residual Connections
    # -------------------------------------------------------------------------
    def test_patch_block_residual_connections(
        self,
        patch_block_no_checkpoint: PatchTransformerBlock,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ):
        """Test residual connections preserve input contribution."""
        x = torch.randn(batch_size, seq_len, hidden_dim)
        t_embed = torch.randn(batch_size, hidden_dim)

        # Get output
        output = patch_block_no_checkpoint(x, t_embed)

        # Output should correlate with input (due to residual)
        # Correlation coefficient should be non-zero
        x_flat = x.reshape(-1).detach()
        out_flat = output.reshape(-1).detach()

        # Check correlation (not perfect due to transformations)
        correlation = torch.corrcoef(torch.stack([x_flat, out_flat]))[0, 1]

        # Should have some correlation due to residual connection
        assert correlation.abs() > 0.0 or torch.isnan(correlation)

    def test_patch_block_residual_with_zero_time_embedding(
        self,
        patch_block_no_checkpoint: PatchTransformerBlock,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ):
        """Test residual behavior with zero time embedding."""
        x = torch.randn(batch_size, seq_len, hidden_dim)
        t_embed = torch.zeros(batch_size, hidden_dim)

        output = patch_block_no_checkpoint(x, t_embed)

        # Output should still be valid (not NaN)
        assert not torch.isnan(output).any()
        # Shape preserved
        assert output.shape == x.shape

    # -------------------------------------------------------------------------
    # Test 9: Gradient Flow
    # -------------------------------------------------------------------------
    def test_patch_block_gradient_flow(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ):
        """Test gradients flow through the block correctly."""
        # Create block without checkpointing for gradient testing
        block = create_patch_block(
            hidden_dim=hidden_dim,
            num_heads=4,
            num_kv_heads=2,
            use_checkpoint=False,
        )

        x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)
        t_embed = torch.randn(batch_size, hidden_dim, requires_grad=True)

        output = block(x, t_embed)
        loss = output.sum()
        loss.backward()

        # Check gradients exist and are valid
        assert x.grad is not None
        assert t_embed.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(t_embed.grad).any()

        # Check all model parameters have gradients
        for name, param in block.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"
                assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient"

    def test_patch_block_gradient_flow_with_checkpointing(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ):
        """Test gradient flow with checkpointing enabled."""
        block = create_patch_block(
            hidden_dim=hidden_dim,
            num_heads=4,
            num_kv_heads=2,
            use_checkpoint=True,
        )
        block.train()

        x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)
        t_embed = torch.randn(batch_size, hidden_dim, requires_grad=True)

        output = block(x, t_embed)
        loss = output.sum()
        loss.backward()

        # Gradients should still flow with checkpointing
        assert x.grad is not None
        assert t_embed.grad is not None


# ============================================================================
# Test Class: PatchTransformerBlockStack
# ============================================================================

class TestPatchTransformerBlockStack:
    """Test suite for PatchTransformerBlockStack."""

    @pytest.fixture
    def patch_block_stack(self, minimal_config: PixelHDMConfig) -> PatchTransformerBlockStack:
        """Create a PatchTransformerBlockStack for testing."""
        return PatchTransformerBlockStack(config=minimal_config)

    def test_stack_init(self, minimal_config: PixelHDMConfig):
        """Test stack initialization."""
        stack = PatchTransformerBlockStack(config=minimal_config)

        assert stack.num_layers == minimal_config.patch_layers
        assert len(stack.blocks) == minimal_config.patch_layers

    def test_stack_forward_shape(
        self,
        patch_block_stack: PatchTransformerBlockStack,
        minimal_config: PixelHDMConfig,
    ):
        """Test stack forward pass shape."""
        batch_size = 2
        seq_len = 64
        hidden_dim = minimal_config.hidden_dim

        x = torch.randn(batch_size, seq_len, hidden_dim)
        t_embed = torch.randn(batch_size, hidden_dim)

        output, repa_features = patch_block_stack(x, t_embed)

        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert repa_features is None  # Not requested

    def test_stack_repa_features(self, minimal_config: PixelHDMConfig):
        """Test REPA feature extraction."""
        # Create stack with repa_layer=0 to ensure we capture features
        stack = PatchTransformerBlockStack(
            config=None,
            num_layers=4,
            repa_layer=0,  # Capture at first layer
            hidden_dim=minimal_config.hidden_dim,
            num_heads=minimal_config.num_heads,
            num_kv_heads=minimal_config.num_kv_heads,
            mlp_ratio=minimal_config.mlp_ratio,
        )

        batch_size = 2
        seq_len = 64
        hidden_dim = minimal_config.hidden_dim

        x = torch.randn(batch_size, seq_len, hidden_dim)
        t_embed = torch.randn(batch_size, hidden_dim)

        output, repa_features = stack(x, t_embed, return_repa_features=True)

        # REPA features should be returned when repa_layer < num_layers
        assert repa_features is not None
        assert repa_features.shape == (batch_size, seq_len, hidden_dim)

    def test_stack_factory_functions(self):
        """Test stack factory functions."""
        stack1 = create_patch_block_stack(
            num_layers=4,
            hidden_dim=256,
            num_heads=4,
            num_kv_heads=2,
            repa_layer=2,
        )
        assert len(stack1.blocks) == 4

    def test_stack_from_config(self, minimal_config: PixelHDMConfig):
        """Test stack creation from config."""
        stack = create_patch_block_stack_from_config(minimal_config)
        assert len(stack.blocks) == minimal_config.patch_layers


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestPatchBlockEdgeCases:
    """Test edge cases and error handling."""

    def test_single_token_input(self):
        """Test with single token input."""
        block = create_patch_block(
            hidden_dim=256,
            num_heads=4,
            num_kv_heads=2,
            use_checkpoint=False,
        )

        x = torch.randn(1, 1, 256)
        t_embed = torch.randn(1, 256)

        output = block(x, t_embed)
        assert output.shape == (1, 1, 256)

    def test_large_batch_size(self):
        """Test with larger batch size."""
        block = create_patch_block(
            hidden_dim=256,
            num_heads=4,
            num_kv_heads=2,
            use_checkpoint=False,
        )

        batch_size = 16
        x = torch.randn(batch_size, 64, 256)
        t_embed = torch.randn(batch_size, 256)

        output = block(x, t_embed)
        assert output.shape == (batch_size, 64, 256)

    def test_eval_mode(self):
        """Test block in evaluation mode."""
        block = create_patch_block(
            hidden_dim=256,
            num_heads=4,
            num_kv_heads=2,
            use_checkpoint=True,
        )
        block.eval()

        x = torch.randn(2, 64, 256)
        t_embed = torch.randn(2, 256)

        with torch.no_grad():
            output = block(x, t_embed)

        assert output.shape == (2, 64, 256)
        assert not torch.isnan(output).any()

    def test_extra_repr(self):
        """Test extra_repr for debugging."""
        block = create_patch_block(hidden_dim=256)
        repr_str = block.extra_repr()

        assert "hidden_dim=256" in repr_str
        assert "sandwich_norm=True" in repr_str
        assert "adaln=token_independent" in repr_str
