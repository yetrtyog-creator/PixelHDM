"""
Factory Functions for Pixel-Level Blocks
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .core import PixelTransformerBlock
from .stack import PixelTransformerBlockStack

if TYPE_CHECKING:
    from ....config.model_config import PixelHDMConfig


def create_pixel_block(
    hidden_dim: int = 1024,
    pixel_dim: int = 16,
    patch_size: int = 16,
    mlp_ratio: float = 3.0,
    num_heads: int = 16,
    num_kv_heads: int = 4,
    use_checkpoint: bool = True,
) -> PixelTransformerBlock:
    """Create a single Pixel-Level Block."""
    return PixelTransformerBlock(
        config=None,
        hidden_dim=hidden_dim,
        pixel_dim=pixel_dim,
        patch_size=patch_size,
        mlp_ratio=mlp_ratio,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        use_checkpoint=use_checkpoint,
    )


def create_pixel_block_from_config(
    config: "PixelHDMConfig",
) -> PixelTransformerBlock:
    """Create Pixel-Level Block from config."""
    return PixelTransformerBlock(config=config)


def create_pixel_block_stack(
    num_layers: int = 4,
    hidden_dim: int = 1024,
    pixel_dim: int = 16,
    patch_size: int = 16,
    mlp_ratio: float = 3.0,
    num_heads: int = 16,
    num_kv_heads: int = 4,
) -> PixelTransformerBlockStack:
    """Create Pixel-Level Block stack."""
    return PixelTransformerBlockStack(
        config=None,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        pixel_dim=pixel_dim,
        patch_size=patch_size,
        mlp_ratio=mlp_ratio,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )


def create_pixel_block_stack_from_config(
    config: "PixelHDMConfig",
) -> PixelTransformerBlockStack:
    """Create Pixel-Level Block stack from config."""
    return PixelTransformerBlockStack(config=config)


__all__ = [
    "create_pixel_block",
    "create_pixel_block_from_config",
    "create_pixel_block_stack",
    "create_pixel_block_stack_from_config",
]
