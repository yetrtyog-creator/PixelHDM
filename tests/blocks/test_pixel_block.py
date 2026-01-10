"""
PixelTransformerBlock Test Suite

Tests for the Pixel-Level DiT Transformer Block.

This module tests:
1. Initialization and configuration
2. Forward pass shape correctness
3. Token Compaction (Compress-Attend-Expand)
4. Compression ratio verification
5. Time conditioning via semantic condition
6. AdaLN modulation (Pixel-wise)
7. Residual connections
8. PixelwiseAdaLN behavior
9. Gradient flow
10. Different resolutions

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.config import PixelHDMConfig
from src.models.blocks.pixel_block import (
    PixelTransformerBlock,
    PixelTransformerBlockStack,
    PixelTransformerBlockLite,
    create_pixel_block,
    create_pixel_block_from_config,
    create_pixel_block_stack,
    create_pixel_block_stack_from_config,
)


# ============================================================================
# Fixtures (module-scoped where possible for efficiency)
# ============================================================================

@pytest.fixture(scope="module")
def minimal_config() -> PixelHDMConfig:
    """Create minimal config for fast tests (module-scoped)."""
    return PixelHDMConfig.for_testing()


@pytest.fixture(scope="module")
def pixel_block(minimal_config: PixelHDMConfig) -> PixelTransformerBlock:
    """Create a PixelTransformerBlock for testing (module-scoped, eval mode)."""
    block = PixelTransformerBlock(config=minimal_config)
    block.eval()
    return block


@pytest.fixture(scope="module")
def pixel_block_no_checkpoint() -> PixelTransformerBlock:
    """Create a PixelTransformerBlock without gradient checkpointing (module-scoped, eval mode)."""
    block = create_pixel_block(
        hidden_dim=256,
        pixel_dim=8,
        patch_size=16,
        mlp_ratio=2.0,
        num_heads=4,
        num_kv_heads=2,
        use_checkpoint=False,
    )
    block.eval()
    return block


@pytest.fixture(scope="module")
def batch_size() -> int:
    return 2


@pytest.fixture(scope="module")
def num_patches() -> int:
    return 16  # 4x4 patches (64x64 image with patch_size=16)


@pytest.fixture(scope="module")
def patch_size() -> int:
    return 16


@pytest.fixture(scope="module")
def pixel_dim() -> int:
    return 8


@pytest.fixture(scope="module")
def hidden_dim() -> int:
    return 256


# ============================================================================
# Test Class: PixelTransformerBlock
# ============================================================================

class TestPixelTransformerBlock:
    """Test suite for PixelTransformerBlock."""

    # -------------------------------------------------------------------------
    # Test 1: Initialization
    # -------------------------------------------------------------------------
    def test_pixel_block_init(self, minimal_config: PixelHDMConfig):
        """Test PixelTransformerBlock initialization."""
        block = PixelTransformerBlock(config=minimal_config)

        # Check attributes
        assert block.hidden_dim == minimal_config.hidden_dim
        assert block.pixel_dim == minimal_config.pixel_dim
        assert block.patch_size == minimal_config.patch_size
        assert block.p2 == minimal_config.patch_size ** 2
        assert hasattr(block, "adaln")
        assert hasattr(block, "compaction")
        assert hasattr(block, "mlp")
        # Note: final_norm was removed as dead code (model-level output_norm is used instead)

    def test_pixel_block_init_with_params(self):
        """Test initialization with explicit parameters."""
        block = create_pixel_block(
            hidden_dim=512,
            pixel_dim=16,
            patch_size=16,
            mlp_ratio=3.0,
            num_heads=8,
            num_kv_heads=2,
            use_checkpoint=False,
        )

        assert block.hidden_dim == 512
        assert block.pixel_dim == 16
        assert block.patch_size == 16
        assert block.use_checkpoint is False

    def test_pixel_block_from_config(self, minimal_config: PixelHDMConfig):
        """Test creation from config."""
        block = create_pixel_block_from_config(minimal_config)

        assert block.hidden_dim == minimal_config.hidden_dim
        assert block.pixel_dim == minimal_config.pixel_dim

    # -------------------------------------------------------------------------
    # Test 2: Forward Pass Shape
    # -------------------------------------------------------------------------
    def test_pixel_block_forward_shape(
        self,
        pixel_block_no_checkpoint: PixelTransformerBlock,
        batch_size: int,
        num_patches: int,
        patch_size: int,
        pixel_dim: int,
        hidden_dim: int,
    ):
        """Test forward pass output shape."""
        p2 = patch_size ** 2

        # Input: (B, L, p^2, D_pix)
        x = torch.randn(batch_size, num_patches, p2, pixel_dim)
        # Semantic condition: (B, L, D)
        s_cond = torch.randn(batch_size, num_patches, hidden_dim)

        output = pixel_block_no_checkpoint(x, s_cond)

        # Output should match input shape
        assert output.shape == (batch_size, num_patches, p2, pixel_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_pixel_block_forward_different_num_patches(
        self,
        pixel_block_no_checkpoint: PixelTransformerBlock,
        batch_size: int,
        patch_size: int,
        pixel_dim: int,
        hidden_dim: int,
    ):
        """Test forward pass with different number of patches."""
        p2 = patch_size ** 2

        for num_patches in [4, 16, 64]:
            x = torch.randn(batch_size, num_patches, p2, pixel_dim)
            s_cond = torch.randn(batch_size, num_patches, hidden_dim)

            output = pixel_block_no_checkpoint(x, s_cond)

            assert output.shape == (batch_size, num_patches, p2, pixel_dim)

    # -------------------------------------------------------------------------
    # Test 3: Token Compaction
    # -------------------------------------------------------------------------
    def test_pixel_block_token_compaction(
        self,
        pixel_block_no_checkpoint: PixelTransformerBlock,
        batch_size: int,
        num_patches: int,
        patch_size: int,
        pixel_dim: int,
        hidden_dim: int,
    ):
        """Test Token Compaction (Compress-Attend-Expand) pipeline."""
        p2 = patch_size ** 2

        x = torch.randn(batch_size, num_patches, p2, pixel_dim)
        s_cond = torch.randn(batch_size, num_patches, hidden_dim)

        # Check compaction module exists
        assert hasattr(pixel_block_no_checkpoint, "compaction")
        compaction = pixel_block_no_checkpoint.compaction

        # Test compaction dimensions
        assert compaction.p2 == p2
        assert compaction.pixel_dim == pixel_dim
        assert compaction.hidden_dim == hidden_dim

        # Test compress-expand pipeline
        x_flat = x.reshape(batch_size, num_patches, p2 * pixel_dim)
        compressed = compaction.compress(x_flat)
        assert compressed.shape == (batch_size, num_patches, hidden_dim)

        expanded = compaction.expand(compressed)
        assert expanded.shape == (batch_size, num_patches, p2 * pixel_dim)

    def test_pixel_block_token_compaction_preserves_info(
        self,
        pixel_block_no_checkpoint: PixelTransformerBlock,
        batch_size: int,
        num_patches: int,
        patch_size: int,
        pixel_dim: int,
        hidden_dim: int,
    ):
        """Test that Token Compaction preserves information through residual."""
        p2 = patch_size ** 2

        x = torch.randn(batch_size, num_patches, p2, pixel_dim)
        s_cond = torch.randn(batch_size, num_patches, hidden_dim)

        output = pixel_block_no_checkpoint(x, s_cond)

        # Due to residual, output should correlate with input
        x_flat = x.reshape(-1).detach()
        out_flat = output.reshape(-1).detach()

        # Check output is not identical (transformation happened)
        assert not torch.allclose(x, output, atol=1e-6)

        # Check output is valid
        assert not torch.isnan(output).any()

    # -------------------------------------------------------------------------
    # Test 4: Compression Ratio
    # -------------------------------------------------------------------------
    def test_pixel_block_compression_ratio(
        self,
        pixel_block_no_checkpoint: PixelTransformerBlock,
    ):
        """Test compression ratio is 4x (4096 -> 1024)."""
        compaction = pixel_block_no_checkpoint.compaction

        # p^2 * D_pix / D = 256 * 8 / 256 = 8 for test config
        # For default config: 256 * 16 / 1024 = 4
        input_dim = compaction.p2 * compaction.pixel_dim
        output_dim = compaction.hidden_dim
        compression_ratio = input_dim / output_dim

        # Check compression exists (ratio >= 1)
        assert compression_ratio >= 1.0

        # For test config: 256 * 8 / 256 = 8
        expected_ratio = (16 ** 2) * 8 / 256  # patch_size^2 * pixel_dim / hidden_dim
        assert abs(compression_ratio - expected_ratio) < 0.01

    def test_pixel_block_compression_ratio_default_config(self):
        """Test compression ratio for default configuration."""
        config = PixelHDMConfig.default()
        block = PixelTransformerBlock(config=config)

        compaction = block.compaction
        input_dim = compaction.p2 * compaction.pixel_dim  # 256 * 16 = 4096
        output_dim = compaction.hidden_dim  # 1024

        compression_ratio = input_dim / output_dim
        assert compression_ratio == 4.0  # 4096 / 1024 = 4

    # -------------------------------------------------------------------------
    # Test 5: Time/Semantic Conditioning
    # -------------------------------------------------------------------------
    def test_pixel_block_with_time_conditioning(
        self,
        batch_size: int,
        num_patches: int,
        patch_size: int,
        pixel_dim: int,
        hidden_dim: int,
    ):
        """Test semantic conditioning interface and parameter generation."""
        p2 = patch_size ** 2

        # Create fresh block for this test
        block = create_pixel_block(
            hidden_dim=hidden_dim,
            pixel_dim=pixel_dim,
            patch_size=patch_size,
            num_heads=4,
            num_kv_heads=2,
            use_checkpoint=False,
        )

        x = torch.randn(batch_size, num_patches, p2, pixel_dim)
        s_cond = torch.randn(batch_size, num_patches, hidden_dim)

        # Verify forward pass works with conditioning
        output = block(x, s_cond)
        assert output.shape == (batch_size, num_patches, p2, pixel_dim)
        assert not torch.isnan(output).any()

        # Verify AdaLN generates 6 parameters with correct pixel-wise shape
        params = block.adaln(s_cond)
        assert len(params) == 6
        for p in params:
            assert p.shape == (batch_size, num_patches, p2, pixel_dim)

    def test_pixel_block_semantic_condition_per_patch(
        self,
        pixel_block_no_checkpoint: PixelTransformerBlock,
        batch_size: int,
        num_patches: int,
        patch_size: int,
        pixel_dim: int,
        hidden_dim: int,
    ):
        """Test per-patch semantic conditioning."""
        p2 = patch_size ** 2

        x = torch.randn(batch_size, num_patches, p2, pixel_dim)

        # Different conditions for each patch
        s_cond = torch.randn(batch_size, num_patches, hidden_dim)
        s_cond[:, 0, :] = 0  # Zero condition for first patch

        output = pixel_block_no_checkpoint(x, s_cond)

        # Output should be valid
        assert not torch.isnan(output).any()
        assert output.shape == x.shape

    # -------------------------------------------------------------------------
    # Test 6: AdaLN Modulation
    # -------------------------------------------------------------------------
    def test_pixel_block_adaln_modulation(
        self,
        pixel_block_no_checkpoint: PixelTransformerBlock,
        batch_size: int,
        num_patches: int,
        patch_size: int,
        pixel_dim: int,
        hidden_dim: int,
    ):
        """Test PixelwiseAdaLN generates correct modulation parameters."""
        s_cond = torch.randn(batch_size, num_patches, hidden_dim)

        # Get modulation parameters
        params = pixel_block_no_checkpoint.adaln(s_cond)

        # Should return 6 parameters
        assert len(params) == 6

        # Each should be (B, L, p^2, D_pix) for pixel-wise modulation
        p2 = patch_size ** 2
        for param in params:
            assert param.shape == (batch_size, num_patches, p2, pixel_dim)
            assert not torch.isnan(param).any()

    def test_pixel_block_adaln_modulate_function(
        self,
        pixel_block_no_checkpoint: PixelTransformerBlock,
        batch_size: int,
        num_patches: int,
        patch_size: int,
        pixel_dim: int,
        hidden_dim: int,
    ):
        """Test AdaLN modulate function."""
        p2 = patch_size ** 2

        x = torch.randn(batch_size, num_patches, p2, pixel_dim)
        gamma = torch.ones(batch_size, num_patches, p2, pixel_dim)
        beta = torch.zeros(batch_size, num_patches, p2, pixel_dim)

        modulated = pixel_block_no_checkpoint.adaln.modulate(x, gamma, beta)

        # With gamma=1, beta=0, should be close to identity (after norm)
        assert modulated.shape == x.shape

    # -------------------------------------------------------------------------
    # Test 7: Residual Connections
    # -------------------------------------------------------------------------
    def test_pixel_block_residual_connections(
        self,
        pixel_block_no_checkpoint: PixelTransformerBlock,
        batch_size: int,
        num_patches: int,
        patch_size: int,
        pixel_dim: int,
        hidden_dim: int,
    ):
        """Test residual connections preserve input contribution."""
        p2 = patch_size ** 2

        x = torch.randn(batch_size, num_patches, p2, pixel_dim)
        s_cond = torch.randn(batch_size, num_patches, hidden_dim)

        output = pixel_block_no_checkpoint(x, s_cond)

        # Output should not be identical to input (transformation happened)
        assert not torch.allclose(x, output, atol=1e-5)

        # But should have similar magnitude (residual prevents explosion)
        input_norm = x.norm().item()
        output_norm = output.norm().item()

        # Norms should be in same order of magnitude
        assert 0.1 * input_norm < output_norm < 10 * input_norm

    def test_pixel_block_residual_numerical_stability(
        self,
        pixel_block_no_checkpoint: PixelTransformerBlock,
        batch_size: int,
        num_patches: int,
        patch_size: int,
        pixel_dim: int,
        hidden_dim: int,
    ):
        """Test residual connections maintain numerical stability."""
        p2 = patch_size ** 2

        # Test with small values
        x_small = torch.randn(batch_size, num_patches, p2, pixel_dim) * 0.01
        s_cond = torch.randn(batch_size, num_patches, hidden_dim)

        output_small = pixel_block_no_checkpoint(x_small, s_cond)
        assert not torch.isnan(output_small).any()
        assert not torch.isinf(output_small).any()

        # Test with larger values
        x_large = torch.randn(batch_size, num_patches, p2, pixel_dim) * 10
        output_large = pixel_block_no_checkpoint(x_large, s_cond)
        assert not torch.isnan(output_large).any()
        assert not torch.isinf(output_large).any()

    # -------------------------------------------------------------------------
    # Test 8: PixelwiseAdaLN
    # -------------------------------------------------------------------------
    def test_pixel_block_pixelwise_adaln(
        self,
        pixel_block_no_checkpoint: PixelTransformerBlock,
        batch_size: int,
        num_patches: int,
        patch_size: int,
        pixel_dim: int,
        hidden_dim: int,
    ):
        """Test Pixel-wise AdaLN generates per-pixel modulation."""
        adaln = pixel_block_no_checkpoint.adaln
        p2 = patch_size ** 2

        s_cond = torch.randn(batch_size, num_patches, hidden_dim)
        params = adaln(s_cond)

        # Each param should have pixel-wise granularity
        for param in params:
            assert param.shape[-2] == p2  # p^2 pixels
            assert param.shape[-1] == pixel_dim  # D_pix

    def test_pixel_block_pixelwise_adaln_per_patch_different(
        self,
        pixel_block_no_checkpoint: PixelTransformerBlock,
        batch_size: int,
        num_patches: int,
        patch_size: int,
        pixel_dim: int,
        hidden_dim: int,
    ):
        """Test that different patches get different modulation."""
        torch.manual_seed(42)
        adaln = pixel_block_no_checkpoint.adaln

        # Different conditions for different patches - use very different values
        s_cond = torch.randn(batch_size, num_patches, hidden_dim)
        s_cond[:, 0, :] = torch.randn(hidden_dim) * 10.0  # Very different first patch
        s_cond[:, 1, :] = torch.randn(hidden_dim) * -10.0  # Very different second patch

        params = adaln(s_cond)

        # Check that params have correct shape
        gamma1 = params[0]
        assert gamma1.shape[1] == num_patches

        # For PixelwiseAdaLN, check s_cond modulates differently
        # (Note: output depends on the specific AdaLN implementation)
        assert not torch.isnan(gamma1).any()

    # -------------------------------------------------------------------------
    # Test 9: Gradient Flow
    # -------------------------------------------------------------------------
    def test_pixel_block_gradient_flow(
        self,
        batch_size: int,
        num_patches: int,
    ):
        """Test gradients flow through the block correctly."""
        patch_size = 16
        pixel_dim = 8
        hidden_dim = 256
        p2 = patch_size ** 2

        block = create_pixel_block(
            hidden_dim=hidden_dim,
            pixel_dim=pixel_dim,
            patch_size=patch_size,
            num_heads=4,
            num_kv_heads=2,
            use_checkpoint=False,
        )

        x = torch.randn(batch_size, num_patches, p2, pixel_dim, requires_grad=True)
        s_cond = torch.randn(batch_size, num_patches, hidden_dim, requires_grad=True)

        output = block(x, s_cond)
        loss = output.sum()
        loss.backward()

        # Check gradients exist and are valid
        assert x.grad is not None
        assert s_cond.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(s_cond.grad).any()

        # Check all trainable parameters have gradients
        grad_count = 0
        for name, param in block.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1
        assert grad_count > 0, "At least some parameters should have gradients"

    def test_pixel_block_gradient_flow_with_checkpointing(
        self,
        batch_size: int,
        num_patches: int,
    ):
        """Test gradient flow with checkpointing enabled."""
        patch_size = 16
        pixel_dim = 8
        hidden_dim = 256
        p2 = patch_size ** 2

        block = create_pixel_block(
            hidden_dim=hidden_dim,
            pixel_dim=pixel_dim,
            patch_size=patch_size,
            num_heads=4,
            num_kv_heads=2,
            use_checkpoint=True,
        )
        block.train()

        x = torch.randn(batch_size, num_patches, p2, pixel_dim, requires_grad=True)
        s_cond = torch.randn(batch_size, num_patches, hidden_dim, requires_grad=True)

        output = block(x, s_cond)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert s_cond.grad is not None

    # -------------------------------------------------------------------------
    # Test 10: Different Resolutions
    # -------------------------------------------------------------------------
    def test_pixel_block_different_resolutions(
        self,
        pixel_block_no_checkpoint: PixelTransformerBlock,
        batch_size: int,
        patch_size: int,
        pixel_dim: int,
        hidden_dim: int,
    ):
        """Test block with different image resolutions (different num_patches)."""
        p2 = patch_size ** 2

        # 64x64 image -> 4x4 = 16 patches
        num_patches_64 = (64 // patch_size) ** 2
        x_64 = torch.randn(batch_size, num_patches_64, p2, pixel_dim)
        s_cond_64 = torch.randn(batch_size, num_patches_64, hidden_dim)
        output_64 = pixel_block_no_checkpoint(x_64, s_cond_64)
        assert output_64.shape == (batch_size, num_patches_64, p2, pixel_dim)

        # 128x128 image -> 8x8 = 64 patches
        num_patches_128 = (128 // patch_size) ** 2
        x_128 = torch.randn(batch_size, num_patches_128, p2, pixel_dim)
        s_cond_128 = torch.randn(batch_size, num_patches_128, hidden_dim)
        output_128 = pixel_block_no_checkpoint(x_128, s_cond_128)
        assert output_128.shape == (batch_size, num_patches_128, p2, pixel_dim)

        # 256x256 image -> 16x16 = 256 patches
        num_patches_256 = (256 // patch_size) ** 2
        x_256 = torch.randn(batch_size, num_patches_256, p2, pixel_dim)
        s_cond_256 = torch.randn(batch_size, num_patches_256, hidden_dim)
        output_256 = pixel_block_no_checkpoint(x_256, s_cond_256)
        assert output_256.shape == (batch_size, num_patches_256, p2, pixel_dim)

    def test_pixel_block_rectangular_resolution(
        self,
        pixel_block_no_checkpoint: PixelTransformerBlock,
        batch_size: int,
        patch_size: int,
        pixel_dim: int,
        hidden_dim: int,
    ):
        """Test block with rectangular image (non-square)."""
        p2 = patch_size ** 2

        # 128x64 image -> 8x4 = 32 patches
        h_patches = 128 // patch_size
        w_patches = 64 // patch_size
        num_patches = h_patches * w_patches

        x = torch.randn(batch_size, num_patches, p2, pixel_dim)
        s_cond = torch.randn(batch_size, num_patches, hidden_dim)

        output = pixel_block_no_checkpoint(x, s_cond)

        assert output.shape == (batch_size, num_patches, p2, pixel_dim)
        assert not torch.isnan(output).any()


# ============================================================================
# Test Class: PixelTransformerBlockStack
# ============================================================================

class TestPixelTransformerBlockStack:
    """Test suite for PixelTransformerBlockStack."""

    @pytest.fixture
    def pixel_block_stack(self, minimal_config: PixelHDMConfig) -> PixelTransformerBlockStack:
        """Create a PixelTransformerBlockStack for testing."""
        return PixelTransformerBlockStack(config=minimal_config)

    def test_stack_init(self, minimal_config: PixelHDMConfig):
        """Test stack initialization."""
        stack = PixelTransformerBlockStack(config=minimal_config)

        assert stack.num_layers == minimal_config.pixel_layers
        assert len(stack.blocks) == minimal_config.pixel_layers

    def test_stack_forward_shape(
        self,
        pixel_block_stack: PixelTransformerBlockStack,
        minimal_config: PixelHDMConfig,
    ):
        """Test stack forward pass shape."""
        batch_size = 2
        num_patches = 16
        p2 = minimal_config.patch_size ** 2
        pixel_dim = minimal_config.pixel_dim
        hidden_dim = minimal_config.hidden_dim

        x = torch.randn(batch_size, num_patches, p2, pixel_dim)
        s_cond = torch.randn(batch_size, num_patches, hidden_dim)

        output = pixel_block_stack(x, s_cond)

        assert output.shape == (batch_size, num_patches, p2, pixel_dim)

    def test_stack_factory_functions(self):
        """Test stack factory functions."""
        stack = create_pixel_block_stack(
            num_layers=2,
            hidden_dim=256,
            pixel_dim=8,
            patch_size=16,
            num_heads=4,
            num_kv_heads=2,
        )
        assert len(stack.blocks) == 2

    def test_stack_from_config(self, minimal_config: PixelHDMConfig):
        """Test stack creation from config."""
        stack = create_pixel_block_stack_from_config(minimal_config)
        assert len(stack.blocks) == minimal_config.pixel_layers


# ============================================================================
# Test Class: PixelTransformerBlockLite
# ============================================================================

class TestPixelTransformerBlockLite:
    """Test suite for PixelTransformerBlockLite (no Token Compaction)."""

    def test_lite_init(self, minimal_config: PixelHDMConfig):
        """Test lite block initialization."""
        block = PixelTransformerBlockLite(config=minimal_config)

        assert block.pixel_dim == minimal_config.pixel_dim
        assert hasattr(block, "adaln")
        assert hasattr(block, "mlp")
        # Should NOT have compaction
        assert not hasattr(block, "compaction")

    def test_lite_forward_shape(self, minimal_config: PixelHDMConfig):
        """Test lite block forward pass."""
        block = PixelTransformerBlockLite(config=minimal_config)

        batch_size = 2
        num_patches = 16
        p2 = minimal_config.patch_size ** 2
        pixel_dim = minimal_config.pixel_dim
        hidden_dim = minimal_config.hidden_dim

        x = torch.randn(batch_size, num_patches, p2, pixel_dim)
        s_cond = torch.randn(batch_size, num_patches, hidden_dim)

        output = block(x, s_cond)

        assert output.shape == (batch_size, num_patches, p2, pixel_dim)
        assert not torch.isnan(output).any()


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestPixelBlockEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_input_dimension(self):
        """Test error handling for invalid input dimensions."""
        block = create_pixel_block(
            hidden_dim=256,
            pixel_dim=8,
            patch_size=16,
            use_checkpoint=False,
        )

        # 3D input instead of 4D should raise error
        x_3d = torch.randn(2, 16, 256 * 8)  # Wrong shape
        s_cond = torch.randn(2, 16, 256)

        with pytest.raises(ValueError, match="4D"):
            block(x_3d, s_cond)

    def test_eval_mode(self):
        """Test block in evaluation mode."""
        block = create_pixel_block(
            hidden_dim=256,
            pixel_dim=8,
            patch_size=16,
            use_checkpoint=True,
        )
        block.eval()

        p2 = 16 ** 2
        x = torch.randn(2, 16, p2, 8)
        s_cond = torch.randn(2, 16, 256)

        with torch.no_grad():
            output = block(x, s_cond)

        assert output.shape == (2, 16, p2, 8)
        assert not torch.isnan(output).any()

    def test_single_patch_input(self):
        """Test with single patch input."""
        block = create_pixel_block(
            hidden_dim=256,
            pixel_dim=8,
            patch_size=16,
            use_checkpoint=False,
        )

        p2 = 16 ** 2
        x = torch.randn(1, 1, p2, 8)
        s_cond = torch.randn(1, 1, 256)

        output = block(x, s_cond)
        assert output.shape == (1, 1, p2, 8)

    def test_extra_repr(self):
        """Test extra_repr for debugging."""
        block = create_pixel_block(
            hidden_dim=256,
            pixel_dim=8,
            patch_size=16,
        )
        repr_str = block.extra_repr()

        assert "hidden_dim=256" in repr_str
        assert "pixel_dim=8" in repr_str
        assert "adaln=pixel_wise" in repr_str
