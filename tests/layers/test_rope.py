"""
RoPE (Rotary Position Embedding) Tests

Tests for:
- RoPE1D: 1D rotary position embedding for text sequences
- RoPE2D: 2D rotary position embedding for image patches
- mRoPE: Lumina2-style multi-axis rotary position embedding (16-24-24 dimension split)
- create_image_positions: Image position index generation
- create_position_ids: Lumina2-style 3-axis position ID generation
- create_rope_from_config: Config-based factory

mRoPE Dimension Configuration (Lumina2 style):
    - axis0_dim: 16 (sequence position - text=0..L-1, image=L)
    - axis1_dim: 24 (height position - text=0, image=0..H-1)
    - axis2_dim: 24 (width position - text=0, image=0..W-1)
    - Total: 64 = head_dim

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
Updated: 2026-01-09 (Lumina2-style mRoPE interface)
"""

import pytest
import torch
import torch.nn as nn
import math

from src.models.layers.rope import (
    RoPE1D,
    RoPE2D,
    MRoPE,
    precompute_freqs_cis,
    apply_rotary_emb,
    create_rope_from_config,
    create_mrope,
    create_image_positions,
    create_image_positions_batched,
    create_position_ids,
    create_position_ids_batched,
    create_image_only_position_ids,
    create_image_only_position_ids_batched,
)
from src.config import PixelHDMConfig


class TestPrecomputeFreqs:
    """Tests for frequency precomputation."""

    def test_precompute_freqs_shape(self):
        """Test that precomputed frequencies have correct shape."""
        dim = 64
        max_seq_len = 512

        freqs = precompute_freqs_cis(dim, max_seq_len)

        # Shape should be (max_seq_len, dim // 2)
        assert freqs.shape == (max_seq_len, dim // 2)
        assert freqs.shape == (512, 32)

    def test_precompute_freqs_odd_dim_raises(self):
        """Test that odd dimension raises ValueError."""
        dim = 63  # Odd dimension
        max_seq_len = 512

        with pytest.raises(ValueError, match="RoPE 維度必須是偶數"):
            precompute_freqs_cis(dim, max_seq_len)

    def test_precompute_freqs_values_valid(self):
        """Test that precomputed frequencies contain valid values."""
        dim = 64
        max_seq_len = 128

        freqs = precompute_freqs_cis(dim, max_seq_len)

        # Should not contain NaN or Inf
        assert not torch.isnan(freqs).any()
        assert not torch.isinf(freqs).any()

        # First position should be 0 (since pos * freq = 0 * freq = 0)
        assert torch.allclose(freqs[0], torch.zeros(dim // 2), atol=1e-6)

    def test_precompute_freqs_different_theta(self):
        """Test that different theta values produce different frequencies."""
        dim = 64
        max_seq_len = 128

        freqs_10k = precompute_freqs_cis(dim, max_seq_len, theta=10000.0)
        freqs_50k = precompute_freqs_cis(dim, max_seq_len, theta=50000.0)

        # Different theta should produce different frequencies (except position 0)
        assert not torch.allclose(freqs_10k[1:], freqs_50k[1:])


class TestApplyRotaryEmb:
    """Tests for apply_rotary_emb function."""

    def test_apply_rotary_emb_norm_preserving(self):
        """Test that rotary embedding preserves vector norm."""
        batch_size = 2
        seq_len = 32
        num_heads = 8
        dim = 64

        x = torch.randn(batch_size, num_heads, seq_len, dim)

        # Compute frequencies
        freqs = torch.randn(batch_size, 1, seq_len, dim // 2)
        freqs_cos = torch.cos(freqs)
        freqs_sin = torch.sin(freqs)

        original_norm = x.norm(dim=-1)
        rotated = apply_rotary_emb(x, freqs_cos, freqs_sin)
        rotated_norm = rotated.norm(dim=-1)

        # Rotation should preserve norm
        assert torch.allclose(original_norm, rotated_norm, atol=1e-5)

    def test_apply_rotary_emb_output_shape(self):
        """Test that rotary embedding preserves shape."""
        batch_size = 2
        seq_len = 32
        num_heads = 8
        dim = 64

        x = torch.randn(batch_size, num_heads, seq_len, dim)
        freqs_cos = torch.randn(batch_size, 1, seq_len, dim // 2)
        freqs_sin = torch.randn(batch_size, 1, seq_len, dim // 2)

        rotated = apply_rotary_emb(x, freqs_cos, freqs_sin)

        assert rotated.shape == x.shape


class TestRoPE1D:
    """Tests for 1D Rotary Position Embedding."""

    @pytest.fixture
    def rope1d(self):
        """Create RoPE1D instance."""
        return RoPE1D(dim=64, max_seq_len=512)

    def test_rope1d_initialization(self, rope1d):
        """Test RoPE1D initialization."""
        assert rope1d.dim == 64
        assert rope1d.max_seq_len == 512
        assert rope1d.theta == 10000.0
        assert hasattr(rope1d, 'freqs')

    def test_rope1d_forward_shape(self, rope1d):
        """Test RoPE1D forward pass shape."""
        B, H, L, D = 2, 16, 32, 64

        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)
        positions = torch.arange(L).unsqueeze(0).expand(B, -1)

        q_rot, k_rot = rope1d(q, k, positions)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rope1d_different_positions(self, rope1d):
        """Test that different positions produce different rotations."""
        B, H, L, D = 2, 16, 32, 64

        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)

        positions1 = torch.arange(L).unsqueeze(0).expand(B, -1)
        positions2 = torch.arange(1, L + 1).unsqueeze(0).expand(B, -1)

        q_rot1, k_rot1 = rope1d(q, k, positions1)
        q_rot2, k_rot2 = rope1d(q, k, positions2)

        # Different positions should produce different rotations
        assert not torch.allclose(q_rot1, q_rot2)
        assert not torch.allclose(k_rot1, k_rot2)

    def test_rope1d_batch_independence(self, rope1d):
        """Test that batches are processed independently."""
        B, H, L, D = 2, 16, 32, 64

        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)
        positions = torch.arange(L).unsqueeze(0).expand(B, -1)

        q_rot, k_rot = rope1d(q, k, positions)

        # Process each batch separately
        q_rot_0, k_rot_0 = rope1d(q[0:1], k[0:1], positions[0:1])
        q_rot_1, k_rot_1 = rope1d(q[1:2], k[1:2], positions[1:2])

        # Should match batched result
        assert torch.allclose(q_rot[0:1], q_rot_0, atol=1e-5)
        assert torch.allclose(q_rot[1:2], q_rot_1, atol=1e-5)


class TestRoPE2D:
    """Tests for 2D Rotary Position Embedding."""

    @pytest.fixture
    def rope2d(self):
        """Create RoPE2D instance."""
        return RoPE2D(dim=64, max_size=128)

    def test_rope2d_initialization(self, rope2d):
        """Test RoPE2D initialization."""
        assert rope2d.dim == 64
        assert rope2d.max_size == 128
        assert rope2d.h_dim == 32  # Half for height
        assert rope2d.w_dim == 32  # Half for width

    def test_rope2d_dimension_split(self, rope2d):
        """Test that RoPE2D splits dimensions correctly for H/W."""
        # Dimension should be split evenly
        assert rope2d.h_dim + rope2d.w_dim == rope2d.dim

        # For dim=64, each should get 32
        assert rope2d.h_dim == 32
        assert rope2d.w_dim == 32

    def test_rope2d_forward_shape(self, rope2d):
        """Test RoPE2D forward pass shape."""
        B, H, L, D = 2, 16, 256, 64  # L = 16x16 patches

        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)

        # Create 2D positions for 16x16 grid
        h_pos = torch.arange(16).repeat(16)  # [0,1,2,...,15, 0,1,2,...,15, ...]
        w_pos = torch.arange(16).repeat_interleave(16)  # [0,0,0,..., 1,1,1,..., ...]

        q_rot, k_rot = rope2d(q, k, h_pos, w_pos)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rope2d_different_positions(self, rope2d):
        """Test that different positions produce different rotations."""
        B, H, L, D = 2, 16, 64, 64

        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)

        # Different position grids
        h_pos1 = torch.arange(8).repeat(8)
        w_pos1 = torch.arange(8).repeat_interleave(8)

        h_pos2 = torch.arange(1, 9).repeat(8)  # Shifted
        w_pos2 = torch.arange(1, 9).repeat_interleave(8)

        q_rot1, k_rot1 = rope2d(q, k, h_pos1, w_pos1)
        q_rot2, k_rot2 = rope2d(q, k, h_pos2, w_pos2)

        assert not torch.allclose(q_rot1, q_rot2)


class TestMRoPE:
    """Tests for Lumina2-style Multi-axis Rotary Position Embedding."""

    @pytest.fixture
    def mrope(self):
        """Create mRoPE instance with default 16-24-24 dimensions."""
        return MRoPE(
            head_dim=64,
            axes_dims=(16, 24, 24),
            max_seq_len=512,
            max_height=256,
            max_width=256,
        )

    def test_mrope_dimension_validation(self, mrope):
        """Test mRoPE dimension validation (16+24+24=64)."""
        # Dimensions should sum to head_dim
        total = mrope.axis0_dim + mrope.axis1_dim + mrope.axis2_dim
        assert total == mrope.head_dim
        assert total == 64

    def test_mrope_dimension_validation_error(self):
        """Test that invalid dimension sum raises error."""
        with pytest.raises(ValueError, match="axes_dims sum must equal head_dim"):
            MRoPE(
                head_dim=64,
                axes_dims=(16, 24, 20),  # 16+24+20=60 != 64
            )

    def test_mrope_default_dimensions(self, mrope):
        """Test mRoPE default dimension allocation (16-24-24)."""
        assert mrope.axis0_dim == 16
        assert mrope.axis1_dim == 24
        assert mrope.axis2_dim == 24
        assert mrope.axes_dims == (16, 24, 24)

    def test_mrope_text_only_encoding(self, mrope):
        """Test mRoPE with text-only sequence (image-only positions)."""
        B, H, L = 2, 16, 64
        D = 64

        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)

        # Text-only position IDs: (n, 0, 0) for each token
        position_ids = torch.zeros(B, L, 3, dtype=torch.long)
        position_ids[:, :, 0] = torch.arange(L).unsqueeze(0).expand(B, -1)

        q_rot, k_rot = mrope(q, k, position_ids)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        assert not torch.isnan(q_rot).any()

    def test_mrope_image_only_encoding(self, mrope):
        """Test mRoPE with image-only sequence."""
        B, H = 2, 16
        img_len = 256  # 16x16 patches
        D = 64

        q = torch.randn(B, H, img_len, D)
        k = torch.randn(B, H, img_len, D)

        # Image-only position IDs: (0, h, w) for each patch
        position_ids = create_image_only_position_ids_batched(
            batch_size=B,
            img_height=256,  # 16 patches
            img_width=256,   # 16 patches
            patch_size=16,
        )

        q_rot, k_rot = mrope(q, k, position_ids)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        assert not torch.isnan(q_rot).any()

    def test_mrope_mixed_sequence(self, mrope):
        """Test mRoPE with mixed text and image sequence."""
        B, H = 2, 16
        text_len = 64
        img_len = 256
        total_len = text_len + img_len
        D = 64

        q = torch.randn(B, H, total_len, D)
        k = torch.randn(B, H, total_len, D)

        # Create Lumina2-style position IDs
        position_ids = create_position_ids_batched(
            batch_size=B,
            text_len=text_len,
            img_height=256,  # 16 patches
            img_width=256,   # 16 patches
            patch_size=16,
        )

        q_rot, k_rot = mrope(q, k, position_ids)

        # Verify output shape
        assert q_rot.shape == (B, H, total_len, D)
        assert k_rot.shape == (B, H, total_len, D)

        # Verify no NaN values
        assert not torch.isnan(q_rot).any()
        assert not torch.isnan(k_rot).any()

    def test_mrope_batch_independence(self, mrope):
        """Test that batches are processed independently in mRoPE."""
        B, H, L, D = 4, 16, 64, 64

        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)

        # Text-only position IDs
        position_ids = torch.zeros(B, L, 3, dtype=torch.long)
        position_ids[:, :, 0] = torch.arange(L).unsqueeze(0).expand(B, -1)

        q_rot, k_rot = mrope(q, k, position_ids)

        # Process batch 0 separately
        q_rot_0, k_rot_0 = mrope(
            q[0:1], k[0:1],
            position_ids[0:1],
        )

        # Should match
        assert torch.allclose(q_rot[0:1], q_rot_0, atol=1e-5)
        assert torch.allclose(k_rot[0:1], k_rot_0, atol=1e-5)

    def test_mrope_different_positions_produce_different_results(self, mrope):
        """Test that different positions produce different rotations."""
        B, H, L, D = 2, 16, 64, 64

        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)

        # Position set 1: sequential
        pos_ids_1 = torch.zeros(B, L, 3, dtype=torch.long)
        pos_ids_1[:, :, 0] = torch.arange(L).unsqueeze(0).expand(B, -1)

        # Position set 2: shifted
        pos_ids_2 = torch.zeros(B, L, 3, dtype=torch.long)
        pos_ids_2[:, :, 0] = torch.arange(1, L + 1).unsqueeze(0).expand(B, -1)

        q_rot_1, k_rot_1 = mrope(q, k, pos_ids_1)
        q_rot_2, k_rot_2 = mrope(q, k, pos_ids_2)

        # Different positions should produce different rotations
        assert not torch.allclose(q_rot_1, q_rot_2)
        assert not torch.allclose(k_rot_1, k_rot_2)


class TestCreateRopeFromConfig:
    """Tests for config-based RoPE factory."""

    def test_create_rope_from_config(self):
        """Test creating mRoPE from config."""
        config = PixelHDMConfig.default()

        rope = create_rope_from_config(config)

        assert isinstance(rope, MRoPE)
        assert rope.head_dim == config.head_dim
        # New Lumina2-style attributes
        assert rope.axis0_dim == config.mrope_text_dim
        assert rope.axis1_dim == config.mrope_img_h_dim
        assert rope.axis2_dim == config.mrope_img_w_dim

    def test_create_rope_from_config_default_values(self):
        """Test that config factory uses correct default values."""
        config = PixelHDMConfig.default()
        rope = create_rope_from_config(config)

        # Verify default mRoPE dimensions (16-24-24) - Lumina2 style
        assert rope.axis0_dim == 16
        assert rope.axis1_dim == 24
        assert rope.axis2_dim == 24
        assert rope.axes_dims == (16, 24, 24)

    def test_create_rope_from_config_custom(self):
        """Test creating mRoPE from custom config."""
        config = PixelHDMConfig.large()

        # Large config has head_dim=72, so mRoPE dimensions are 16-28-28
        rope = create_rope_from_config(config)

        assert rope.head_dim == 72
        assert rope.axis0_dim == 16
        assert rope.axis1_dim == 28
        assert rope.axis2_dim == 28
        assert rope.axes_dims == (16, 28, 28)


class TestCreateImagePositions:
    """Tests for image position creation utilities."""

    def test_create_image_positions_basic(self):
        """Test basic image position creation."""
        positions = create_image_positions(
            height=256,
            width=256,
            patch_size=16,
        )

        # 16x16 patches = 256 positions
        expected_patches = (256 // 16) ** 2
        assert positions.shape == (expected_patches, 2)
        assert positions.shape == (256, 2)

    def test_create_image_positions_512x512(self):
        """Test image positions for 512x512 image."""
        positions = create_image_positions(
            height=512,
            width=512,
            patch_size=16,
        )

        # 32x32 patches = 1024 positions
        assert positions.shape == (1024, 2)

    def test_create_image_positions_rectangular(self):
        """Test image positions for rectangular image."""
        positions = create_image_positions(
            height=256,
            width=512,
            patch_size=16,
        )

        # 16x32 patches = 512 positions
        num_h = 256 // 16
        num_w = 512 // 16
        assert positions.shape == (num_h * num_w, 2)
        assert positions.shape == (512, 2)

    def test_create_image_positions_values(self):
        """Test that image positions have correct values."""
        positions = create_image_positions(
            height=64,
            width=64,
            patch_size=16,
        )

        # 4x4 = 16 patches
        assert positions.shape == (16, 2)

        # First position should be (0, 0)
        assert positions[0, 0] == 0  # h
        assert positions[0, 1] == 0  # w

        # Second position should be (0, 1) - row-major order
        assert positions[1, 0] == 0  # h
        assert positions[1, 1] == 1  # w

        # Position 4 should be (1, 0) - next row
        assert positions[4, 0] == 1  # h
        assert positions[4, 1] == 0  # w

    def test_create_image_positions_invalid_height(self):
        """Test that invalid height raises error."""
        with pytest.raises(ValueError, match="圖像高度"):
            create_image_positions(
                height=100,  # Not divisible by 16
                width=256,
                patch_size=16,
            )

    def test_create_image_positions_invalid_width(self):
        """Test that invalid width raises error."""
        with pytest.raises(ValueError, match="圖像寬度"):
            create_image_positions(
                height=256,
                width=100,  # Not divisible by 16
                patch_size=16,
            )

    def test_create_image_positions_batched(self):
        """Test batched image position creation."""
        batch_size = 4
        positions = create_image_positions_batched(
            batch_size=batch_size,
            height=256,
            width=256,
            patch_size=16,
        )

        # Shape should be (B, num_patches, 2)
        assert positions.shape == (batch_size, 256, 2)

    def test_create_image_positions_batched_consistency(self):
        """Test that batched positions are consistent across batch."""
        batch_size = 3
        positions = create_image_positions_batched(
            batch_size=batch_size,
            height=256,
            width=256,
            patch_size=16,
        )

        # All batches should have same positions
        assert torch.allclose(positions[0], positions[1])
        assert torch.allclose(positions[1], positions[2])

    def test_create_image_positions_device(self):
        """Test image positions on specific device."""
        device = torch.device("cpu")
        positions = create_image_positions(
            height=256,
            width=256,
            patch_size=16,
            device=device,
        )

        assert positions.device == device


class TestCreateMrope:
    """Tests for create_mrope factory function."""

    def test_create_mrope_default(self):
        """Test create_mrope with default parameters."""
        rope = create_mrope()

        assert rope.head_dim == 64
        assert rope.axes_dims == (16, 24, 24)
        assert rope.axis0_dim == 16
        assert rope.axis1_dim == 24
        assert rope.axis2_dim == 24

    def test_create_mrope_custom(self):
        """Test create_mrope with custom parameters."""
        rope = create_mrope(
            head_dim=128,
            axes_dims=(32, 48, 48),
            max_seq_len=1024,
            max_height=512,
            max_width=512,
        )

        assert rope.head_dim == 128
        assert rope.axes_dims == (32, 48, 48)
        assert rope.axis0_dim == 32
        assert rope.axis1_dim == 48
        assert rope.axis2_dim == 48


class TestPositionIds:
    """Tests for Lumina2-style position ID generation."""

    def test_create_position_ids_basic(self):
        """Test basic position ID generation."""
        text_len = 64
        img_height = 256
        img_width = 256
        patch_size = 16

        position_ids = create_position_ids(
            text_len=text_len,
            img_height=img_height,
            img_width=img_width,
            patch_size=patch_size,
        )

        num_patches = (img_height // patch_size) * (img_width // patch_size)
        total_len = text_len + num_patches

        assert position_ids.shape == (total_len, 3)

    def test_create_position_ids_text_part(self):
        """Test that text positions are correctly generated."""
        text_len = 32
        position_ids = create_position_ids(
            text_len=text_len,
            img_height=64,
            img_width=64,
            patch_size=16,
        )

        # Text tokens: (n, 0, 0) for n = 0, 1, ..., text_len-1
        for i in range(text_len):
            assert position_ids[i, 0].item() == i  # axis0 = n
            assert position_ids[i, 1].item() == 0  # axis1 = 0
            assert position_ids[i, 2].item() == 0  # axis2 = 0

    def test_create_position_ids_image_part(self):
        """Test that image positions are correctly generated."""
        text_len = 32
        img_height = 64  # 4 patches
        img_width = 64   # 4 patches
        patch_size = 16

        position_ids = create_position_ids(
            text_len=text_len,
            img_height=img_height,
            img_width=img_width,
            patch_size=patch_size,
        )

        # Image tokens: (text_len, h, w)
        # First image token at index text_len
        assert position_ids[text_len, 0].item() == text_len  # axis0 = text_len
        assert position_ids[text_len, 1].item() == 0  # axis1 = 0 (first row)
        assert position_ids[text_len, 2].item() == 0  # axis2 = 0 (first col)

        # Second image token (same row, next col)
        assert position_ids[text_len + 1, 0].item() == text_len
        assert position_ids[text_len + 1, 1].item() == 0
        assert position_ids[text_len + 1, 2].item() == 1

        # Token in second row
        num_w = img_width // patch_size
        assert position_ids[text_len + num_w, 1].item() == 1  # axis1 = 1

    def test_create_position_ids_batched(self):
        """Test batched position ID generation."""
        batch_size = 4
        text_len = 32
        img_height = 256
        img_width = 256
        patch_size = 16

        position_ids = create_position_ids_batched(
            batch_size=batch_size,
            text_len=text_len,
            img_height=img_height,
            img_width=img_width,
            patch_size=patch_size,
        )

        num_patches = (img_height // patch_size) * (img_width // patch_size)
        total_len = text_len + num_patches

        assert position_ids.shape == (batch_size, total_len, 3)

        # All batches should have identical positions
        for i in range(1, batch_size):
            assert torch.equal(position_ids[0], position_ids[i])

    def test_create_image_only_position_ids(self):
        """Test image-only position ID generation."""
        img_height = 256
        img_width = 256
        patch_size = 16

        position_ids = create_image_only_position_ids(
            img_height=img_height,
            img_width=img_width,
            patch_size=patch_size,
        )

        num_patches = (img_height // patch_size) * (img_width // patch_size)
        assert position_ids.shape == (num_patches, 3)

        # All axis0 should be 0 (no text prefix)
        assert (position_ids[:, 0] == 0).all()

    def test_create_image_only_position_ids_batched(self):
        """Test batched image-only position ID generation."""
        batch_size = 3
        img_height = 128
        img_width = 128
        patch_size = 16

        position_ids = create_image_only_position_ids_batched(
            batch_size=batch_size,
            img_height=img_height,
            img_width=img_width,
            patch_size=patch_size,
        )

        num_patches = (img_height // patch_size) * (img_width // patch_size)
        assert position_ids.shape == (batch_size, num_patches, 3)
