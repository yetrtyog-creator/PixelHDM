"""
PixelHDM-RPEA-DinoV3 Transformer Blocks

包含:
    - PatchTransformerBlock: Patch-Level DiT Block (Token-Independent AdaLN)
    - PixelTransformerBlock: Pixel-Level DiT Block (Pixel-wise AdaLN + Token Compaction)
    - ImageProcessorBlock: Image Processor Block (Pre-joint, no RoPE, TokenAdaLN)
    - TextProcessorBlock: Text Processor Block (Pre-joint, no RoPE, no AdaLN)
"""

from .patch_block import (
    PatchTransformerBlock,
    PatchTransformerBlockStack,
    create_patch_block,
    create_patch_block_from_config,
    create_patch_block_stack,
    create_patch_block_stack_from_config,
)

from .pixel_block import (
    PixelTransformerBlock,
    PixelTransformerBlockStack,
    PixelTransformerBlockLite,
    create_pixel_block,
    create_pixel_block_from_config,
    create_pixel_block_stack,
    create_pixel_block_stack_from_config,
)

from .image import (
    ImageProcessorBlock,
    ImageProcessorStack,
)

from .text import (
    TextProcessorBlock,
    TextProcessorStack,
)


__all__ = [
    # Patch-Level
    "PatchTransformerBlock",
    "PatchTransformerBlockStack",
    "create_patch_block",
    "create_patch_block_from_config",
    "create_patch_block_stack",
    "create_patch_block_stack_from_config",
    # Pixel-Level
    "PixelTransformerBlock",
    "PixelTransformerBlockStack",
    "PixelTransformerBlockLite",
    "create_pixel_block",
    "create_pixel_block_from_config",
    "create_pixel_block_stack",
    "create_pixel_block_stack_from_config",
    # Image Processor (Pre-joint)
    "ImageProcessorBlock",
    "ImageProcessorStack",
    # Text Processor (Pre-joint)
    "TextProcessorBlock",
    "TextProcessorStack",
]
