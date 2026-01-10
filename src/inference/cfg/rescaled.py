"""
Rescaled CFG implementation.

Paper: Common Diffusion Noise Schedules and Sample Steps are Flawed

Contains:
    - RescaledCFG: CFG with variance correction to prevent oversaturation

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import torch

from .base import BaseCFG, compute_rescaled_output


class RescaledCFG(BaseCFG):
    """
    Rescaled CFG (prevents oversaturation).

    Paper: Common Diffusion Noise Schedules and Sample Steps are Flawed

    Features:
        - Uses guidance_rescale to mitigate CFG-induced oversaturation
        - Implemented via standard deviation adjustment
    """

    def __init__(self, rescale_factor: float = 0.7) -> None:
        self.rescale_factor = rescale_factor

    def apply(
        self,
        x_cond: torch.Tensor,
        x_uncond: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """
        Apply Rescaled CFG.

        Args:
            x_cond: Conditional prediction
            x_uncond: Unconditional prediction
            guidance_scale: CFG weight

        Returns:
            Rescaled CFG prediction
        """
        x_cfg = x_uncond + guidance_scale * (x_cond - x_uncond)
        return compute_rescaled_output(x_cond, x_cfg, self.rescale_factor)


__all__ = [
    "RescaledCFG",
]
