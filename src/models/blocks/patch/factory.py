"""
Factory Functions for Patch-Level Blocks
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .core import PatchTransformerBlock
from .stack import PatchTransformerBlockStack

if TYPE_CHECKING:
    from ....config.model_config import PixelHDMConfig


def create_patch_block(
    hidden_dim: int = 1024,
    num_heads: int = 16,
    num_kv_heads: int = 4,
    mlp_ratio: float = 3.0,
    dropout: float = 0.0,
    use_checkpoint: bool = True,
) -> PatchTransformerBlock:
    """Create a single Patch-Level Block."""
    return PatchTransformerBlock(
        config=None,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        use_checkpoint=use_checkpoint,
    )


def create_patch_block_from_config(
    config: "PixelHDMConfig",
) -> PatchTransformerBlock:
    """Create Patch-Level Block from config."""
    return PatchTransformerBlock(config=config)


def create_patch_block_stack(
    num_layers: int = 16,
    hidden_dim: int = 1024,
    num_heads: int = 16,
    num_kv_heads: int = 4,
    mlp_ratio: float = 3.0,
    repa_layer: int = 8,
) -> PatchTransformerBlockStack:
    """Create Patch-Level Block stack."""
    return PatchTransformerBlockStack(
        config=None,
        num_layers=num_layers,
        repa_layer=repa_layer,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        mlp_ratio=mlp_ratio,
    )


def create_patch_block_stack_from_config(
    config: "PixelHDMConfig",
) -> PatchTransformerBlockStack:
    """Create Patch-Level Block stack from config."""
    return PatchTransformerBlockStack(config=config)


__all__ = [
    "create_patch_block",
    "create_patch_block_from_config",
    "create_patch_block_stack",
    "create_patch_block_stack_from_config",
]
