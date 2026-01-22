"""
Lightweight Pixel-Level Block

Pixel-Level Block without Token Compaction, only Pixel-wise AdaLN + MLP.
Used for resource-constrained scenarios or ablation experiments.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

from ...layers.feedforward import SwiGLU
from ...layers.adaln import PixelwiseAdaLN

if TYPE_CHECKING:
    from ....config.model_config import PixelHDMConfig




class PixelTransformerBlockLite(nn.Module):
    """
    Lightweight Pixel-Level Block

    No Token Compaction, only Pixel-wise AdaLN + MLP.
    """

    def __init__(
        self,
        config: Optional["PixelHDMConfig"] = None,
        hidden_dim: int = 1024,
        pixel_dim: int = 16,
        patch_size: int = 16,
        mlp_ratio: float = 3.0,
    ) -> None:
        super().__init__()

        if config is not None:
            hidden_dim = config.hidden_dim
            pixel_dim = config.pixel_dim
            patch_size = config.patch_size
            mlp_ratio = config.mlp_ratio

        self.pixel_dim = pixel_dim
        self.p2 = patch_size ** 2

        # Depth scaling (k=2 residual branches per block)
        pixel_layers = config.pixel_layers if config is not None else 1
        self.residual_scale = 1.0 / math.sqrt(2 * pixel_layers)


        self.adaln = PixelwiseAdaLN(
            hidden_dim=hidden_dim,
            pixel_dim=pixel_dim,
            patch_size=patch_size,
            num_params=6,
        )

        mlp_dim = int(pixel_dim * mlp_ratio)
        self.mlp = SwiGLU(hidden_dim=pixel_dim, mlp_dim=mlp_dim)

    def forward(
        self,
        x: torch.Tensor,
        s_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass (no Token Compaction)."""
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaln(s_cond)

        h = self.adaln.modulate(x, gamma2, beta2)
        h = self.mlp(h)
        x = x + (alpha2 * self.residual_scale) * h

        return x


__all__ = ["PixelTransformerBlockLite"]
