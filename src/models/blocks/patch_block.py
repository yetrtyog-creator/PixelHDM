"""
Patch-Level Transformer Block (Legacy Compatibility)

This module re-exports from the new modular structure.
For new code, import directly from models.blocks.patch.
"""

from .patch import (
    PatchTransformerBlock,
    PatchTransformerBlockStack,
    create_patch_block,
    create_patch_block_from_config,
    create_patch_block_stack,
    create_patch_block_stack_from_config,
)

__all__ = [
    "PatchTransformerBlock",
    "PatchTransformerBlockStack",
    "create_patch_block",
    "create_patch_block_from_config",
    "create_patch_block_stack",
    "create_patch_block_stack_from_config",
]
