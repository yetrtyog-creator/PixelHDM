"""
Pixel-wise Adaptive Layer Normalization

Generates independent modulation parameters for each pixel.
This is one of the core innovations of PixelHDM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn

from ..normalization import RMSNorm

if TYPE_CHECKING:
    from ....config.model_config import PixelHDMConfig


class PixelwiseAdaLN(nn.Module):
    """
    Pixel-wise Adaptive Layer Normalization

    Generates independent modulation parameters for each pixel.

    Args:
        config: PixelHDMConfig configuration
        hidden_dim: Condition dimension (Patch-Level output)
        pixel_dim: Pixel feature dimension
        patch_size: Patch size
        num_params: Number of modulation parameters

    Flow:
        1. Condition expansion: (B, L, D) -> (B, L, p^2, D_pix)
        2. Parameter generation: (B, L, p^2, D_pix) -> (B, L, p^2, num_params * D_pix)
        3. Split into modulation parameters

    Shape:
        - s_cond: (B, L, D) - semantic+time condition from Patch-Level
        - Output: num_params tensors of shape (B, L, p^2, D_pix)
    """

    def __init__(
        self,
        config: Optional["PixelHDMConfig"] = None,
        hidden_dim: int = 1024,
        pixel_dim: int = 16,
        patch_size: int = 16,
        num_params: int = 6,
        init_gain: float = 0.0001,
    ) -> None:
        super().__init__()

        if config is not None:
            hidden_dim = config.hidden_dim
            pixel_dim = config.pixel_dim
            patch_size = config.patch_size
            num_params = config.adaln_num_params
            init_gain = config.adaln_init_gain

        self.hidden_dim = hidden_dim
        self.pixel_dim = pixel_dim
        self.patch_size = patch_size
        self.p2 = patch_size ** 2
        self.num_params = num_params
        self.init_gain = init_gain

        self.cond_expand = nn.Linear(
            hidden_dim,
            self.p2 * pixel_dim,
            bias=False,
        )

        # NOTE: cond_norm (RMSNorm) was removed on 2026-01-19
        # It was destroying 99.7% of text conditioning signal because:
        # - cond_expand projects different s_cond to nearly parallel vectors (cosine_sim=0.998)
        # - RMSNorm normalizes to unit length, preserving only direction
        # - Since directions are nearly identical, outputs became nearly identical
        # Signal retention: WITH cond_norm=0.3%, WITHOUT=53.9%

        self.param_gen = nn.Sequential(
            nn.SiLU(),
            nn.Linear(pixel_dim, num_params * pixel_dim, bias=True),
        )

        self.norm = RMSNorm(pixel_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize weights for time conditioning.

        CRITICAL FIX (2026-01-04): param_gen zero init disabled time conditioning.
        CRITICAL FIX (2026-01-07): Small non-zero init to keep blocks active.
        CRITICAL FIX (2026-01-19): Removed cond_norm - it destroyed 99.7% of text signal.

        Initialization strategy:
            - gamma=init_gain (small but active modulation)
            - alpha=1.0 (full residual connection for gradient flow)
            - beta=0.0 (no shift initially)
        """
        # Xavier init for cond_expand
        nn.init.xavier_uniform_(self.cond_expand.weight)

        # Xavier for param_gen weight
        nn.init.xavier_uniform_(self.param_gen[-1].weight)

        # Small non-zero initialization (NOT full adaLN-Zero)
        # gamma=1e-4: small init for gradual modulation learning
        # alpha=1.0: full residual for gradient flow
        # beta=0.0: no shift
        with torch.no_grad():
            bias = self.param_gen[-1].bias.view(self.num_params, self.pixel_dim)
            bias.zero_()  # Start from zero
            bias[0].fill_(self.init_gain)   # gamma1 = 1e-4
            bias[2].fill_(1.0)   # alpha1 = 1.0 (full residual)
            bias[3].fill_(self.init_gain)   # gamma2 = 1e-4
            bias[5].fill_(1.0)   # alpha2 = 1.0

    def forward(
        self,
        s_cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Generate Pixel-wise AdaLN parameters.

        Args:
            s_cond: (B, L, D) - semantic+time condition

        Returns:
            num_params tensors of shape (B, L, p^2, D_pix)
        """
        B, L, D = s_cond.shape

        cond_exp = self.cond_expand(s_cond)
        cond_exp = cond_exp.view(B, L, self.p2, self.pixel_dim)

        # No normalization - cond_norm was removed (see __init__ comment)

        params = self.param_gen(cond_exp)
        params = params.chunk(self.num_params, dim=-1)

        return params

    def modulate(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply modulation: gamma * LayerNorm(x) + beta

        Args:
            x: (B, L, p^2, D_pix) input
            gamma: (B, L, p^2, D_pix) scale parameter
            beta: (B, L, p^2, D_pix) shift parameter

        Returns:
            Modulated tensor (B, L, p^2, D_pix)
        """
        return gamma * self.norm(x) + beta

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"pixel_dim={self.pixel_dim}, "
            f"p^2={self.p2}, "
            f"num_params={self.num_params}"
        )


__all__ = ["PixelwiseAdaLN"]
