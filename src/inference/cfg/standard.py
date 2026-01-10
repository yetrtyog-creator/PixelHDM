"""
Standard CFG and Perplexity CFG implementations.

Contains:
    - StandardCFG: Basic classifier-free guidance
    - PerplexityCFG: Uncertainty-based adaptive CFG

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import torch

from .base import BaseCFG


class StandardCFG(BaseCFG):
    """
    Standard Classifier-Free Guidance.

    Formula: x = x_uncond + scale * (x_cond - x_uncond)
    """

    def apply(
        self,
        x_cond: torch.Tensor,
        x_uncond: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """
        Apply standard CFG.

        Args:
            x_cond: Conditional prediction
            x_uncond: Unconditional prediction
            guidance_scale: CFG weight

        Returns:
            CFG weighted prediction
        """
        return x_uncond + guidance_scale * (x_cond - x_uncond)


class PerplexityCFG(BaseCFG):
    """
    Perplexity-guided CFG.

    Dynamically adjusts CFG weight based on prediction uncertainty.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def apply(
        self,
        x_cond: torch.Tensor,
        x_uncond: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """
        Apply Perplexity CFG.

        Args:
            x_cond: Conditional prediction
            x_uncond: Unconditional prediction
            guidance_scale: Base CFG weight

        Returns:
            Perplexity-adjusted prediction
        """
        diff = (x_cond - x_uncond).abs()
        uncertainty = diff.mean(dim=list(range(1, diff.dim())), keepdim=True)

        adaptive_scale = guidance_scale * torch.sigmoid(
            -uncertainty / self.temperature
        )

        return x_uncond + adaptive_scale * (x_cond - x_uncond)


__all__ = [
    "StandardCFG",
    "PerplexityCFG",
]
