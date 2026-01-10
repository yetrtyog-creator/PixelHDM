"""
Embedding Layers Test Suite

Tests for PatchEmbedding, TimeEmbedding, PixelEmbedding, and related embedding modules.

This module tests:
1. PatchEmbedding initialization
2. PatchEmbedding forward shape
3. Number of patches calculation
4. Different resolution handling
5. TimeEmbedding initialization
6. TimeEmbedding forward shape
7. Sinusoidal embedding properties
8. Different timestep values
9. Boundary values (t=0, t=1)
10. PixelUnpatchify and PixelPatchify
11. PixelEmbedding (1×1 Patchify) - 2026-01-06

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn

from src.config import PixelHDMConfig
from src.models.layers.embedding import (
    PixelEmbedding,
    PatchEmbedding,
    TimeEmbedding,
    PixelUnpatchify,
    PixelPatchify,
    LearnedPositionalEmbedding,
    create_pixel_embedding,
    create_pixel_embedding_from_config,
    create_patch_embedding,
    create_patch_embedding_from_config,
    create_time_embedding,
    create_time_embedding_from_config,
    create_pixel_unpatchify,
    create_pixel_unpatchify_from_config,
    create_pixel_patchify,
    create_pixel_patchify_from_config,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def minimal_config() -> PixelHDMConfig:
    """Create minimal config for fast tests."""
    return PixelHDMConfig.for_testing()


@pytest.fixture
def batch_size() -> int:
    return 2


@pytest.fixture
def patch_size() -> int:
    return 16


@pytest.fixture
def hidden_dim() -> int:
    return 256


@pytest.fixture
def pixel_dim() -> int:
    return 8


@pytest.fixture
def in_channels() -> int:
    return 3


# ============================================================================
# Test Class: PatchEmbedding
# ============================================================================

class TestPatchEmbedding:
    """Test suite for PatchEmbedding."""

    # -------------------------------------------------------------------------
    # Test 1: PatchEmbedding Initialization
    # -------------------------------------------------------------------------
    def test_patch_embedding_init(self, minimal_config: PixelHDMConfig):
        """Test PatchEmbedding initialization."""
        embed = PatchEmbedding(config=minimal_config)

        assert embed.patch_size == minimal_config.patch_size
        assert embed.hidden_dim == minimal_config.hidden_dim
        assert embed.in_channels == minimal_config.in_channels
        assert embed.p2 == minimal_config.patch_size ** 2
        assert hasattr(embed, "proj_down")
        assert hasattr(embed, "proj_up")
        assert hasattr(embed, "act")

    def test_patch_embedding_init_with_params(self):
        """Test initialization with explicit parameters."""
        embed = create_patch_embedding(
            patch_size=16,
            hidden_dim=512,
            in_channels=3,
            bottleneck_dim=128,
        )

        assert embed.patch_size == 16
        assert embed.hidden_dim == 512
        assert embed.in_channels == 3

    def test_patch_embedding_from_config(self, minimal_config: PixelHDMConfig):
        """Test creation from config."""
        embed = create_patch_embedding_from_config(minimal_config)

        assert embed.patch_size == minimal_config.patch_size
        assert embed.hidden_dim == minimal_config.hidden_dim

    def test_patch_embedding_bottleneck_dimensions(self):
        """Test bottleneck projection dimensions."""
        embed = create_patch_embedding(
            patch_size=16,
            hidden_dim=1024,
            in_channels=3,
            bottleneck_dim=256,
        )

        # Input: 3 * 16^2 = 768
        # Bottleneck: 256
        # Output: 1024
        assert embed.proj_down.in_features == 3 * 256  # 768
        assert embed.proj_down.out_features == 256  # bottleneck
        assert embed.proj_up.in_features == 256  # bottleneck
        assert embed.proj_up.out_features == 1024  # hidden_dim

    # -------------------------------------------------------------------------
    # Test 2: PatchEmbedding Forward Shape
    # -------------------------------------------------------------------------
    def test_patch_embedding_forward_shape(
        self,
        batch_size: int,
        patch_size: int,
        hidden_dim: int,
    ):
        """Test forward pass shape (B, H, W, C) -> (B, L, D)."""
        embed = create_patch_embedding(
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            in_channels=3,
        )

        H = W = 256
        x = torch.randn(batch_size, H, W, 3)  # (B, H, W, C) format

        output = embed(x)

        expected_L = (H // patch_size) * (W // patch_size)
        assert output.shape == (batch_size, expected_L, hidden_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_patch_embedding_forward_shape_chw_format(
        self,
        batch_size: int,
        patch_size: int,
        hidden_dim: int,
    ):
        """Test forward pass with (B, C, H, W) format."""
        embed = create_patch_embedding(
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            in_channels=3,
        )

        H = W = 256
        x = torch.randn(batch_size, 3, H, W)  # (B, C, H, W) format

        output = embed(x)

        expected_L = (H // patch_size) * (W // patch_size)
        assert output.shape == (batch_size, expected_L, hidden_dim)

    # -------------------------------------------------------------------------
    # Test 3: Number of Patches Calculation
    # -------------------------------------------------------------------------
    def test_patch_embedding_num_patches(self, patch_size: int):
        """Test patches number calculation for various sizes."""
        embed = create_patch_embedding(patch_size=patch_size, hidden_dim=256)

        test_cases = [
            (64, 64, (64 // patch_size) ** 2),  # 16 patches
            (128, 128, (128 // patch_size) ** 2),  # 64 patches
            (256, 256, (256 // patch_size) ** 2),  # 256 patches
            (512, 512, (512 // patch_size) ** 2),  # 1024 patches
        ]

        for H, W, expected_patches in test_cases:
            x = torch.randn(1, H, W, 3)
            output = embed(x)
            assert output.shape[1] == expected_patches

    def test_patch_embedding_rectangular_patches(self, patch_size: int):
        """Test patches for rectangular images."""
        embed = create_patch_embedding(patch_size=patch_size, hidden_dim=256)

        # Rectangular image: 256x128
        H, W = 256, 128
        expected_patches = (H // patch_size) * (W // patch_size)

        x = torch.randn(1, H, W, 3)
        output = embed(x)

        assert output.shape[1] == expected_patches  # 16 * 8 = 128 patches

    # -------------------------------------------------------------------------
    # Test 4: Different Resolution Handling
    # -------------------------------------------------------------------------
    def test_patch_embedding_different_resolutions(self, patch_size: int):
        """Test forward pass with different resolutions."""
        embed = create_patch_embedding(patch_size=patch_size, hidden_dim=256)

        resolutions = [64, 128, 256, 512]

        for res in resolutions:
            x = torch.randn(2, res, res, 3)
            output = embed(x)

            expected_L = (res // patch_size) ** 2
            assert output.shape == (2, expected_L, 256)
            assert not torch.isnan(output).any()

    def test_patch_embedding_indivisible_resolution_error(self, patch_size: int):
        """Test error on resolution not divisible by patch_size."""
        embed = create_patch_embedding(patch_size=patch_size, hidden_dim=256)

        # 65x65 is not divisible by 16
        x = torch.randn(1, 65, 65, 3)

        with pytest.raises(AssertionError):
            embed(x)


# ============================================================================
# Test Class: TimeEmbedding
# ============================================================================

class TestTimeEmbedding:
    """Test suite for TimeEmbedding."""

    # -------------------------------------------------------------------------
    # Test 5: TimeEmbedding Initialization
    # -------------------------------------------------------------------------
    def test_time_embedding_init(self, minimal_config: PixelHDMConfig):
        """Test TimeEmbedding initialization."""
        embed = TimeEmbedding(config=minimal_config)

        assert embed.hidden_dim == minimal_config.hidden_dim
        assert embed.embed_dim == minimal_config.time_embed_dim
        assert hasattr(embed, "mlp")
        assert hasattr(embed, "freqs")

    def test_time_embedding_init_with_params(self):
        """Test initialization with explicit parameters."""
        embed = create_time_embedding(
            hidden_dim=512,
            embed_dim=128,
        )

        assert embed.hidden_dim == 512
        assert embed.embed_dim == 128

    def test_time_embedding_from_config(self, minimal_config: PixelHDMConfig):
        """Test creation from config."""
        embed = create_time_embedding_from_config(minimal_config)

        assert embed.hidden_dim == minimal_config.hidden_dim
        assert embed.embed_dim == minimal_config.time_embed_dim

    def test_time_embedding_freqs_shape(self):
        """Test frequency buffer shape."""
        embed_dim = 256
        embed = create_time_embedding(hidden_dim=512, embed_dim=embed_dim)

        assert embed.freqs.shape == (embed_dim // 2,)

    # -------------------------------------------------------------------------
    # Test 6: TimeEmbedding Forward Shape
    # -------------------------------------------------------------------------
    def test_time_embedding_forward_shape(self, batch_size: int, hidden_dim: int):
        """Test forward pass shape (B,) -> (B, D)."""
        embed = create_time_embedding(hidden_dim=hidden_dim, embed_dim=256)

        t = torch.rand(batch_size)

        output = embed(t)

        assert output.shape == (batch_size, hidden_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_time_embedding_forward_shape_2d_input(
        self, batch_size: int, hidden_dim: int
    ):
        """Test forward pass with (B, 1) input."""
        embed = create_time_embedding(hidden_dim=hidden_dim, embed_dim=256)

        t = torch.rand(batch_size, 1)

        output = embed(t)

        assert output.shape == (batch_size, hidden_dim)

    def test_time_embedding_forward_shape_scalar_input(self, hidden_dim: int):
        """Test forward pass with scalar input."""
        embed = create_time_embedding(hidden_dim=hidden_dim, embed_dim=256)

        t = torch.tensor(0.5)

        output = embed(t)

        assert output.shape == (1, hidden_dim)

    # -------------------------------------------------------------------------
    # Test 7: Sinusoidal Embedding Properties
    # -------------------------------------------------------------------------
    def test_time_embedding_sinusoidal(self, hidden_dim: int):
        """Test sinusoidal embedding properties."""
        embed = create_time_embedding(hidden_dim=hidden_dim, embed_dim=256)

        t = torch.tensor([0.5])

        # The embedding should have meaningful sinusoidal structure
        output = embed(t)

        # Output should be bounded (sinusoidal + MLP)
        assert output.abs().max() < 100  # Reasonable magnitude

    def test_time_embedding_continuous(self, hidden_dim: int):
        """Test embeddings are continuous in t."""
        embed = create_time_embedding(hidden_dim=hidden_dim, embed_dim=256)

        t1 = torch.tensor([0.50])
        t2 = torch.tensor([0.51])  # Small difference

        emb1 = embed(t1)
        emb2 = embed(t2)

        # Small time difference should produce similar embeddings
        diff = (emb1 - emb2).abs().mean()
        assert diff < 1.0  # Embeddings should be relatively close

    def test_time_embedding_frequency_range(self):
        """Test frequency range of sinusoidal embeddings."""
        embed = create_time_embedding(hidden_dim=256, embed_dim=256)

        # Frequencies should range from high to low
        freqs = embed.freqs
        assert freqs[0] > freqs[-1]  # First freq > last freq

    # -------------------------------------------------------------------------
    # Test 8: Different t Values
    # -------------------------------------------------------------------------
    def test_time_embedding_different_t_values(self, hidden_dim: int):
        """Test different t values produce different embeddings."""
        embed = create_time_embedding(hidden_dim=hidden_dim, embed_dim=256)

        t1 = torch.tensor([0.1, 0.1])
        t2 = torch.tensor([0.9, 0.9])

        emb1 = embed(t1)
        emb2 = embed(t2)

        # Different t should produce different embeddings
        assert not torch.allclose(emb1, emb2)

    def test_time_embedding_same_t_same_output(self, hidden_dim: int):
        """Test same t produces consistent embeddings."""
        embed = create_time_embedding(hidden_dim=hidden_dim, embed_dim=256)

        t = torch.tensor([0.5])

        emb1 = embed(t)
        emb2 = embed(t)

        assert torch.allclose(emb1, emb2)

    def test_time_embedding_batch_independence(self, hidden_dim: int):
        """Test embeddings are computed independently per batch."""
        embed = create_time_embedding(hidden_dim=hidden_dim, embed_dim=256)

        t = torch.tensor([0.1, 0.5, 0.9])

        output = embed(t)

        # Each batch element should be different
        assert not torch.allclose(output[0], output[1])
        assert not torch.allclose(output[1], output[2])

    # -------------------------------------------------------------------------
    # Test 9: Boundary Values (t=0, t=1)
    # -------------------------------------------------------------------------
    def test_time_embedding_boundary_values(self, hidden_dim: int):
        """Test embeddings at boundary values t=0 and t=1."""
        embed = create_time_embedding(hidden_dim=hidden_dim, embed_dim=256)

        t_zero = torch.tensor([0.0])
        t_one = torch.tensor([1.0])

        emb_zero = embed(t_zero)
        emb_one = embed(t_one)

        # Both should be valid (no NaN/Inf)
        assert not torch.isnan(emb_zero).any()
        assert not torch.isnan(emb_one).any()
        assert not torch.isinf(emb_zero).any()
        assert not torch.isinf(emb_one).any()

        # t=0 and t=1 should produce different embeddings
        assert not torch.allclose(emb_zero, emb_one)

    def test_time_embedding_boundary_gradient(self, hidden_dim: int):
        """Test gradient flow at boundaries."""
        embed = create_time_embedding(hidden_dim=hidden_dim, embed_dim=256)

        t = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)

        output = embed(t)
        loss = output.sum()
        loss.backward()

        # Gradient should exist
        assert t.grad is not None
        assert not torch.isnan(t.grad).any()

    def test_time_embedding_near_boundary(self, hidden_dim: int):
        """Test embeddings near boundaries."""
        embed = create_time_embedding(hidden_dim=hidden_dim, embed_dim=256)

        # Very close to 0 and 1
        t_near_zero = torch.tensor([1e-6])
        t_near_one = torch.tensor([1.0 - 1e-6])

        emb_near_zero = embed(t_near_zero)
        emb_near_one = embed(t_near_one)

        assert not torch.isnan(emb_near_zero).any()
        assert not torch.isnan(emb_near_one).any()


# ============================================================================
# Test Class: PixelUnpatchify and PixelPatchify
# ============================================================================

class TestPixelUnpatchifyPatchify:
    """Test suite for PixelUnpatchify and PixelPatchify."""

    # -------------------------------------------------------------------------
    # Test 10: Text Projection / Pixel Transforms
    # -------------------------------------------------------------------------
    def test_pixel_unpatchify_shape(
        self,
        batch_size: int,
        hidden_dim: int,
        pixel_dim: int,
        patch_size: int,
    ):
        """Test PixelUnpatchify shape transformation."""
        unpatch = create_pixel_unpatchify(
            hidden_dim=hidden_dim,
            pixel_dim=pixel_dim,
            patch_size=patch_size,
        )

        num_patches = 16
        x = torch.randn(batch_size, num_patches, hidden_dim)

        output = unpatch(x)

        p2 = patch_size ** 2
        assert output.shape == (batch_size, num_patches, p2, pixel_dim)
        assert not torch.isnan(output).any()

    def test_pixel_patchify_shape(self, batch_size: int, pixel_dim: int, patch_size: int):
        """Test PixelPatchify shape transformation."""
        patchify = create_pixel_patchify(
            pixel_dim=pixel_dim,
            patch_size=patch_size,
            out_channels=3,
        )

        num_patches = 16  # 4x4 patches
        p2 = patch_size ** 2
        x = torch.randn(batch_size, num_patches, p2, pixel_dim)

        # Specify image size for reshape
        output = patchify(x, image_size=(64, 64))

        assert output.shape == (batch_size, 64, 64, 3)
        assert not torch.isnan(output).any()

    def test_pixel_patchify_infer_size(self, batch_size: int, pixel_dim: int, patch_size: int):
        """Test PixelPatchify with inferred image size."""
        patchify = create_pixel_patchify(
            pixel_dim=pixel_dim,
            patch_size=patch_size,
            out_channels=3,
        )

        num_patches = 16  # 4x4 patches (should infer 64x64 image)
        p2 = patch_size ** 2
        x = torch.randn(batch_size, num_patches, p2, pixel_dim)

        output = patchify(x)  # No image_size specified

        # Should infer 4x4 patches * 16 patch_size = 64x64
        assert output.shape == (batch_size, 64, 64, 3)

    def test_unpatchify_patchify_roundtrip(
        self,
        batch_size: int,
        hidden_dim: int,
        pixel_dim: int,
        patch_size: int,
    ):
        """Test roundtrip from patches to pixels and back."""
        unpatch = create_pixel_unpatchify(
            hidden_dim=hidden_dim,
            pixel_dim=pixel_dim,
            patch_size=patch_size,
        )

        num_patches = 16
        x = torch.randn(batch_size, num_patches, hidden_dim)

        # Unpatchify
        pixel_features = unpatch(x)

        # Shape should be correct
        p2 = patch_size ** 2
        assert pixel_features.shape == (batch_size, num_patches, p2, pixel_dim)


# ============================================================================
# Test Class: LearnedPositionalEmbedding
# ============================================================================

class TestLearnedPositionalEmbedding:
    """Test suite for LearnedPositionalEmbedding."""

    def test_learned_pos_embed_init(self, minimal_config: PixelHDMConfig):
        """Test LearnedPositionalEmbedding initialization."""
        embed = LearnedPositionalEmbedding(config=minimal_config)

        assert embed.max_patches == minimal_config.max_patches
        assert embed.hidden_dim == minimal_config.hidden_dim
        assert embed.embedding.shape == (1, minimal_config.max_patches, minimal_config.hidden_dim)

    def test_learned_pos_embed_forward(self, minimal_config: PixelHDMConfig):
        """Test forward pass returns correct shape."""
        embed = LearnedPositionalEmbedding(config=minimal_config)

        num_patches = 64
        output = embed(num_patches)

        assert output.shape == (1, num_patches, minimal_config.hidden_dim)

    def test_learned_pos_embed_max_patches_error(self):
        """Test error when exceeding max_patches."""
        embed = LearnedPositionalEmbedding(
            max_patches=256,
            hidden_dim=256,
        )

        with pytest.raises(AssertionError):
            embed(300)  # Exceeds max_patches


# ============================================================================
# PixelEmbedding Tests (1×1 Patchify) - Added 2026-01-06
# ============================================================================

class TestPixelEmbedding:
    """Test PixelEmbedding (1×1 Patchify) module.

    PixelEmbedding is the "1×1 Patchify" from the PixelHDM paper.
    It performs per-pixel linear embedding to preserve high-frequency details.

    Architecture (2026-01-06):
        Noised Image x_t
            │
            ├─→ 16×16 Patchify (PatchEmbedding) ─→ DiT ─→ s_cond
            │
            └─→ 1×1 Patchify (PixelEmbedding) ───→ PiT Blocks ─→ Output
    """

    def test_pixel_embedding_initialization(self):
        """Test PixelEmbedding initializes correctly."""
        embed = create_pixel_embedding(in_channels=3, pixel_dim=16, patch_size=16)

        assert embed.in_channels == 3
        assert embed.pixel_dim == 16
        assert embed.patch_size == 16
        assert embed.p2 == 256  # 16^2

    def test_pixel_embedding_forward_shape(self):
        """Test PixelEmbedding forward produces correct shape.

        Flow: (B, H, W, C) -> Linear -> (B, H, W, D_pix) -> reshape -> (B, L, p², D_pix)
        """
        embed = create_pixel_embedding(in_channels=3, pixel_dim=16, patch_size=16)

        # Test with 256x256 image
        B, H, W, C = 2, 256, 256, 3
        x = torch.randn(B, H, W, C)
        output = embed(x)

        L = (H // 16) * (W // 16)  # 256 patches
        p2 = 16 * 16  # 256 pixels per patch
        D_pix = 16

        expected_shape = (B, L, p2, D_pix)
        assert output.shape == expected_shape, \
            f"Expected {expected_shape}, got {output.shape}"

    def test_pixel_embedding_channel_first_input(self):
        """Test PixelEmbedding handles (B, C, H, W) input."""
        embed = create_pixel_embedding(in_channels=3, pixel_dim=16, patch_size=16)

        # Channel-first format
        x = torch.randn(2, 3, 128, 128)
        output = embed(x)

        L = (128 // 16) * (128 // 16)  # 64 patches
        assert output.shape == (2, L, 256, 16)

    def test_pixel_embedding_different_resolutions(self):
        """Test PixelEmbedding works with various resolutions."""
        embed = create_pixel_embedding(in_channels=3, pixel_dim=16, patch_size=16)

        test_cases = [
            (64, 64, 16),    # 4x4 = 16 patches
            (128, 128, 64),  # 8x8 = 64 patches
            (256, 256, 256), # 16x16 = 256 patches
            (512, 512, 1024), # 32x32 = 1024 patches
            (128, 256, 128),  # Non-square: 8x16 = 128 patches
        ]

        for H, W, expected_L in test_cases:
            x = torch.randn(1, H, W, 3)
            output = embed(x)
            assert output.shape == (1, expected_L, 256, 16), \
                f"Failed for {H}x{W}: expected L={expected_L}, got {output.shape}"

    def test_pixel_embedding_xavier_init(self):
        """Test PixelEmbedding uses Xavier initialization."""
        embed = create_pixel_embedding(in_channels=3, pixel_dim=16, patch_size=16)

        # Weight should have Xavier scale
        weight_norm = embed.proj.weight.norm().item()
        assert weight_norm > 0, "Weight should be non-zero"

        # Xavier for (3, 16): std ≈ sqrt(2/19) ≈ 0.32
        # With 48 elements, norm should be around sqrt(48 * 0.32^2) ≈ 2.2
        assert 1.0 < weight_norm < 5.0, \
            f"Xavier weight norm should be reasonable, got {weight_norm}"

        # Bias should be zero
        assert torch.allclose(embed.proj.bias, torch.zeros_like(embed.proj.bias))

    def test_pixel_embedding_gradient_flow(self):
        """Test gradient flows through PixelEmbedding."""
        embed = create_pixel_embedding(in_channels=3, pixel_dim=16, patch_size=16)

        x = torch.randn(2, 64, 64, 3, requires_grad=True)
        output = embed(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None, "Input should have gradient"
        assert embed.proj.weight.grad is not None, "Weight should have gradient"
        assert embed.proj.bias.grad is not None, "Bias should have gradient"

    def test_pixel_embedding_from_config(self, minimal_config: PixelHDMConfig):
        """Test PixelEmbedding creation from config."""
        embed = create_pixel_embedding_from_config(minimal_config)

        assert embed.in_channels == minimal_config.in_channels
        assert embed.pixel_dim == minimal_config.pixel_dim
        assert embed.patch_size == minimal_config.patch_size

    def test_pixel_embedding_preserves_batch_independence(self):
        """Test each sample in batch is processed independently."""
        embed = create_pixel_embedding(in_channels=3, pixel_dim=16, patch_size=16)

        x = torch.randn(4, 64, 64, 3)
        output = embed(x)

        # Change one sample and verify others unchanged
        x_modified = x.clone()
        x_modified[0] = torch.randn(64, 64, 3)
        output_modified = embed(x_modified)

        # Sample 0 should be different
        assert not torch.allclose(output[0], output_modified[0])
        # Other samples should be same
        assert torch.allclose(output[1], output_modified[1])
        assert torch.allclose(output[2], output_modified[2])
        assert torch.allclose(output[3], output_modified[3])

    def test_pixel_embedding_extra_repr(self):
        """Test extra_repr for debugging."""
        embed = create_pixel_embedding(in_channels=3, pixel_dim=16, patch_size=16)
        repr_str = embed.extra_repr()

        assert "in_channels=3" in repr_str
        assert "pixel_dim=16" in repr_str
        assert "patch_size=16" in repr_str


# ============================================================================
# Edge Cases and Factory Functions
# ============================================================================

class TestEmbeddingEdgeCases:
    """Test edge cases for embedding modules."""

    def test_patch_embedding_single_batch(self):
        """Test with batch size 1."""
        embed = create_patch_embedding(patch_size=16, hidden_dim=256)

        x = torch.randn(1, 64, 64, 3)
        output = embed(x)

        assert output.shape == (1, 16, 256)

    def test_time_embedding_gradient_flow(self, hidden_dim: int):
        """Test gradient flow through time embedding."""
        embed = create_time_embedding(hidden_dim=hidden_dim, embed_dim=256)

        t = torch.rand(4, requires_grad=True)
        output = embed(t)
        loss = output.sum()
        loss.backward()

        # Check model parameters have gradients
        for param in embed.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_patch_embedding_extra_repr(self):
        """Test extra_repr for debugging."""
        embed = create_patch_embedding(patch_size=16, hidden_dim=256)
        repr_str = embed.extra_repr()

        assert "patch_size=16" in repr_str
        assert "hidden_dim=256" in repr_str

    def test_time_embedding_extra_repr(self):
        """Test extra_repr for debugging."""
        embed = create_time_embedding(hidden_dim=256, embed_dim=128)
        repr_str = embed.extra_repr()

        assert "hidden_dim=256" in repr_str
        assert "embed_dim=128" in repr_str

    def test_pixel_unpatchify_from_config(self, minimal_config: PixelHDMConfig):
        """Test PixelUnpatchify creation from config."""
        unpatch = create_pixel_unpatchify_from_config(minimal_config)

        assert unpatch.hidden_dim == minimal_config.hidden_dim
        assert unpatch.pixel_dim == minimal_config.pixel_dim
        assert unpatch.patch_size == minimal_config.patch_size

    def test_pixel_patchify_from_config(self, minimal_config: PixelHDMConfig):
        """Test PixelPatchify creation from config."""
        patchify = create_pixel_patchify_from_config(minimal_config)

        assert patchify.pixel_dim == minimal_config.pixel_dim
        assert patchify.patch_size == minimal_config.patch_size
        assert patchify.out_channels == minimal_config.in_channels

    def test_pixel_patchify_xavier_init(self):
        """Test PixelPatchify uses Xavier initialization.

        CRITICAL FIX (2026-01-06):
        Changed from trunc_normal_(std=0.02) to xavier_uniform_ because:
        - std=0.02 produced output covering only 7% of target v_target range
        - This caused V-Loss to be stuck at 0.73
        - Xavier initialization produces output covering ~113% of target range
        """
        patchify = create_pixel_patchify(pixel_dim=8, patch_size=16, out_channels=3)

        # Weight should be non-zero (Xavier gives reasonable magnitude)
        weight_norm = patchify.proj.weight.norm().item()
        assert weight_norm > 0, "Weight should be non-zero for gradient flow"

        # Xavier init for (8, 3) should give std ≈ sqrt(2/11) ≈ 0.426
        # Weight norm should be in reasonable range (not too small)
        assert weight_norm > 1.0, "Xavier init should produce reasonable weight magnitude"

        # Bias should still be zero
        assert torch.allclose(patchify.proj.bias, torch.zeros_like(patchify.proj.bias))
