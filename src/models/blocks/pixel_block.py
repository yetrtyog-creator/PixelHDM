"""
Pixel-Level Transformer Block (Legacy Compatibility)

This module re-exports from the new modular structure.
For new code, import directly from models.blocks.pixel.
"""

from .pixel import (
    PixelTransformerBlock,
    PixelTransformerBlockStack,
    PixelTransformerBlockLite,
    create_pixel_block,
    create_pixel_block_from_config,
    create_pixel_block_stack,
    create_pixel_block_stack_from_config,
)

__all__ = [
    "PixelTransformerBlock",
    "PixelTransformerBlockStack",
    "PixelTransformerBlockLite",
    "create_pixel_block",
    "create_pixel_block_from_config",
    "create_pixel_block_stack",
    "create_pixel_block_stack_from_config",
]
