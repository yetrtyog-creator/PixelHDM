"""
RoPE Factory Functions

Author: PixelHDM-RPEA-DinoV3 Project
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .rope_2d import RoPE2D
from .mrope import MRoPE

if TYPE_CHECKING:
    from ....config.model_config import PixelHDMConfig


def create_rope_2d(
    head_dim: int = 64,
    max_size: int = 128,
    theta: float = 10000.0,
) -> RoPE2D:
    """Create 2D RoPE module."""
    return RoPE2D(head_dim, max_size, theta)


def create_mrope(
    head_dim: int = 64,
    axes_dims: tuple = (16, 24, 24),
    max_seq_len: int = 512,
    max_height: int = 256,
    max_width: int = 256,
    theta: float = 10000.0,
) -> MRoPE:
    """Create Lumina2-style mRoPE module with unified 3-axis position encoding."""
    return MRoPE(
        head_dim=head_dim,
        axes_dims=axes_dims,
        max_seq_len=max_seq_len,
        max_height=max_height,
        max_width=max_width,
        theta=theta,
    )


def create_rope_from_config(config: "PixelHDMConfig") -> MRoPE:
    """Create mRoPE module from config."""
    # Map config parameters to Lumina2-style interface
    axes_dims = (
        config.mrope_text_dim,
        config.mrope_img_h_dim,
        config.mrope_img_w_dim,
    )

    # Use explicit config values for max_height/max_width (no sqrt calculation)
    max_height = config.mrope_img_max_height
    max_width = config.mrope_img_max_width

    # CRITICAL: max_seq_len must be text_max_len + 1 because image tokens
    # use axis0 = text_len as position (see create_position_ids in utils.py).
    # If text_len = 512 and max_seq_len = 512, index 512 would be out of bounds.
    max_seq_len = config.mrope_text_max_len + 1

    return create_mrope(
        head_dim=config.head_dim,
        axes_dims=axes_dims,
        max_seq_len=max_seq_len,
        max_height=max_height,
        max_width=max_width,
        theta=config.mrope_theta,
    )


__all__ = [
    "create_rope_2d",
    "create_mrope",
    "create_rope_from_config",
]
