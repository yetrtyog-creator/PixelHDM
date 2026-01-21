"""
RoPE Utility Functions

Contains:
    - precompute_freqs_cis: Precompute RoPE frequencies
    - apply_rotary_emb: Apply rotary embeddings
    - create_image_positions: Create image position indices
    - create_image_positions_batched: Create batched image position indices

Author: PixelHDM-RPEA-DinoV3 Project
"""

from __future__ import annotations

from typing import Optional

import torch


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Precompute RoPE frequencies.

    Args:
        dim: Dimension (must be even)
        max_seq_len: Maximum sequence length
        theta: Frequency base

    Returns:
        Frequency tensor (max_seq_len, dim // 2)
    """
    if dim % 2 != 0:
        raise ValueError(
            f"RoPE 維度必須是偶數，當前: {dim}。"
            f"建議修復: 使用 dim={dim + 1} 或 dim={dim - 1}"
        )

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    positions = torch.arange(max_seq_len, device=device).float()
    freqs_cis = torch.outer(positions, freqs)

    return freqs_cis


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary position embedding.

    Args:
        x: Input tensor (..., dim)
        freqs_cos: Cosine frequencies (..., dim // 2)
        freqs_sin: Sine frequencies (..., dim // 2)

    Returns:
        Rotated tensor (..., dim)
    """
    x_r = x[..., ::2]
    x_i = x[..., 1::2]

    out_r = x_r * freqs_cos - x_i * freqs_sin
    out_i = x_r * freqs_sin + x_i * freqs_cos

    out = torch.stack([out_r, out_i], dim=-1).flatten(-2)

    return out


def create_image_positions(
    height: int,
    width: int,
    patch_size: int = 16,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create image position indices.

    Args:
        height: Image height (pixels)
        width: Image width (pixels)
        patch_size: Patch size (default: 16, matches DINOv3)
        device: Compute device

    Returns:
        img_positions: (num_patches, 2) - [h_pos, w_pos] position indices
    """
    if height % patch_size != 0:
        raise ValueError(
            f"圖像高度 {height} 必須是 patch_size ({patch_size}) 的倍數"
        )
    if width % patch_size != 0:
        raise ValueError(
            f"圖像寬度 {width} 必須是 patch_size ({patch_size}) 的倍數"
        )

    num_patches_h = height // patch_size
    num_patches_w = width // patch_size

    h_pos = torch.arange(num_patches_h, device=device)
    w_pos = torch.arange(num_patches_w, device=device)

    h_grid, w_grid = torch.meshgrid(h_pos, w_pos, indexing="ij")

    h_flat = h_grid.flatten()
    w_flat = w_grid.flatten()

    positions = torch.stack([h_flat, w_flat], dim=-1)

    return positions


def create_image_positions_batched(
    batch_size: int,
    height: int,
    width: int,
    patch_size: int = 16,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create batched image position indices.

    Args:
        batch_size: Batch size
        height: Image height (pixels)
        width: Image width (pixels)
        patch_size: Patch size
        device: Compute device

    Returns:
        img_positions: (batch_size, num_patches, 2)
    """
    positions = create_image_positions(height, width, patch_size, device)
    return positions.unsqueeze(0).expand(batch_size, -1, -1)


def create_position_ids(
    text_len: int,
    img_height: int,
    img_width: int,
    patch_size: int = 16,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create Lumina2-style 3-axis position IDs.

    Position ID format:
    - Text tokens: (n, 0, 0) where n = 0, 1, ..., text_len-1
    - Image tokens: (text_len, h, w) where h, w are patch coordinates

    This provides:
    - Clear text/image boundary (axis0: text uses 0..L-1, image uses L)
    - Spatial encoding for images (axis1: height, axis2: width)

    Args:
        text_len: Number of text tokens
        img_height: Image height in pixels
        img_width: Image width in pixels
        patch_size: Patch size (default: 16)
        device: Device

    Returns:
        position_ids: (seq_len, 3) where seq_len = text_len + num_patches
    """
    num_patches_h = img_height // patch_size
    num_patches_w = img_width // patch_size
    num_patches = num_patches_h * num_patches_w
    total_len = text_len + num_patches

    position_ids = torch.zeros(total_len, 3, dtype=torch.long, device=device)

    # Text tokens: (n, 0, 0)
    if text_len > 0:
        position_ids[:text_len, 0] = torch.arange(text_len, device=device)
        # axis1, axis2 already 0

    # Image tokens: (text_len, h, w)
    if num_patches > 0:
        # axis0: all image tokens have position = text_len
        position_ids[text_len:, 0] = text_len

        # axis1, axis2: spatial grid
        h_indices = torch.arange(num_patches_h, device=device)
        w_indices = torch.arange(num_patches_w, device=device)
        h_grid, w_grid = torch.meshgrid(h_indices, w_indices, indexing="ij")

        position_ids[text_len:, 1] = h_grid.flatten()
        position_ids[text_len:, 2] = w_grid.flatten()

    return position_ids


def create_position_ids_batched(
    batch_size: int,
    text_len: int,
    img_height: int,
    img_width: int,
    patch_size: int = 16,
    device: Optional[torch.device] = None,
    text_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Create batched 3-axis position IDs with per-sample text length support.

    Args:
        batch_size: Batch size
        text_len: Number of text tokens (padded length)
        img_height: Image height in pixels
        img_width: Image width in pixels
        patch_size: Patch size
        device: Device
        text_mask: Optional (B, T) bool mask where True=valid token.
                   If provided, image tokens use per-sample actual_text_len as axis0.

    Returns:
        position_ids: (batch_size, seq_len, 3)
    """
    position_ids = create_position_ids(
        text_len, img_height, img_width, patch_size, device
    )
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1, -1).clone()

    # Per-sample text length: image tokens should use actual text length as axis0
    if text_mask is not None:
        actual_text_lens = text_mask.sum(dim=1)  # (B,)
        num_patches = (img_height // patch_size) * (img_width // patch_size)
        # Image tokens start at text_len, update their axis0 to per-sample actual_text_len
        for b in range(batch_size):
            position_ids[b, text_len:, 0] = actual_text_lens[b]

    return position_ids


def create_image_only_position_ids(
    img_height: int,
    img_width: int,
    patch_size: int = 16,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create position IDs for image-only sequences (Pixel Blocks).

    For pixel blocks, there's no text prefix, so axis0 = 0 for all tokens.

    Args:
        img_height: Image height in pixels
        img_width: Image width in pixels
        patch_size: Patch size
        device: Device

    Returns:
        position_ids: (num_patches, 3)
    """
    num_patches_h = img_height // patch_size
    num_patches_w = img_width // patch_size
    num_patches = num_patches_h * num_patches_w

    position_ids = torch.zeros(num_patches, 3, dtype=torch.long, device=device)

    # axis0: all 0 (no text prefix)
    # axis1, axis2: spatial grid
    h_indices = torch.arange(num_patches_h, device=device)
    w_indices = torch.arange(num_patches_w, device=device)
    h_grid, w_grid = torch.meshgrid(h_indices, w_indices, indexing="ij")

    position_ids[:, 1] = h_grid.flatten()
    position_ids[:, 2] = w_grid.flatten()

    return position_ids


def create_image_only_position_ids_batched(
    batch_size: int,
    img_height: int,
    img_width: int,
    patch_size: int = 16,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create batched image-only position IDs.

    Args:
        batch_size: Batch size
        img_height: Image height in pixels
        img_width: Image width in pixels
        patch_size: Patch size
        device: Device

    Returns:
        position_ids: (batch_size, num_patches, 3)
    """
    position_ids = create_image_only_position_ids(
        img_height, img_width, patch_size, device
    )
    return position_ids.unsqueeze(0).expand(batch_size, -1, -1)


__all__ = [
    "precompute_freqs_cis",
    "apply_rotary_emb",
    "create_image_positions",
    "create_image_positions_batched",
    "create_position_ids",
    "create_position_ids_batched",
    "create_image_only_position_ids",
    "create_image_only_position_ids_batched",
]
