"""
Pixel-Level Transformer Block Stack

M-layer stack of PixelTransformerBlocks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Callable

import torch
import torch.nn as nn

from ...layers.normalization import RMSNorm
from .core import PixelTransformerBlock

if TYPE_CHECKING:
    from ....config.model_config import PixelHDMConfig


class PixelTransformerBlockStack(nn.Module):
    """
    Pixel-Level Transformer Block Stack

    M layers of PixelTransformerBlock.

    Args:
        config: PixelHDMConfig configuration
        num_layers: Number of layers (M)
    """

    def __init__(
        self,
        config: Optional["PixelHDMConfig"] = None,
        num_layers: int = 4,
        hidden_dim: int = 1024,
        pixel_dim: int = 16,
        patch_size: int = 16,
        mlp_ratio: float = 3.0,
        num_heads: int = 16,
        num_kv_heads: int = 4,
    ) -> None:
        super().__init__()

        if config is not None:
            num_layers = config.pixel_layers
            hidden_dim = config.hidden_dim
            pixel_dim = config.pixel_dim
            patch_size = config.patch_size
            mlp_ratio = config.mlp_ratio
            num_heads = config.num_heads
            num_kv_heads = config.num_kv_heads

        self.num_layers = num_layers
        self.pixel_dim = pixel_dim

        self.blocks = nn.ModuleList([
            PixelTransformerBlock(
                config=config,
                hidden_dim=hidden_dim,
                pixel_dim=pixel_dim,
                patch_size=patch_size,
                mlp_ratio=mlp_ratio,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(pixel_dim)

    def forward(
        self,
        x: torch.Tensor,
        s_cond: torch.Tensor,
        rope_fn: Optional[Callable] = None,
        img_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, L, p^2, D_pix) pixel features
            s_cond: (B, L, D) semantic+time condition
            rope_fn: Optional RoPE function
            img_positions: Optional image positions

        Returns:
            Output features (B, L, p^2, D_pix)
        """
        for block in self.blocks:
            x = block(x, s_cond, rope_fn, img_positions)

        x = self.final_norm(x)

        return x


__all__ = ["PixelTransformerBlockStack"]
