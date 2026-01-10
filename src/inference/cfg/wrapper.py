"""
CFG Model Wrapper.

Contains:
    - CFGWrapper: Wraps a model with CFG logic

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .base import BaseCFG, CFGScheduler
from .standard import StandardCFG


class CFGWrapper(nn.Module):
    """
    CFG model wrapper.

    Wraps CFG logic into model calls.

    Args:
        model: Base model
        cfg_method: CFG method
        default_scale: Default CFG weight
        scheduler: CFG scheduler (optional)
    """

    def __init__(
        self,
        model: nn.Module,
        cfg_method: BaseCFG = None,
        default_scale: float = 7.5,
        scheduler: Optional[CFGScheduler] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.cfg_method = cfg_method or StandardCFG()
        self.default_scale = default_scale
        self.scheduler = scheduler
        self._batched_cfg = False

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
        null_text_embed: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = None,
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with CFG.

        Args:
            x_t: Noisy image
            t: Timestep
            text_embed: Conditional text embedding
            null_text_embed: Unconditional text embedding
            guidance_scale: CFG weight (overrides default)
            step: Current step (for scheduling)
            total_steps: Total steps (for scheduling)
            **kwargs: Extra arguments

        Returns:
            CFG weighted prediction
        """
        guidance_scale = self._get_guidance_scale(guidance_scale, t, step, total_steps)

        if guidance_scale == 1.0 or null_text_embed is None:
            return self.model(x_t, t, text_embed=text_embed, **kwargs)

        x_cond = self.model(x_t, t, text_embed=text_embed, **kwargs)
        x_uncond = self.model(x_t, t, text_embed=null_text_embed, **kwargs)

        return self.cfg_method.apply(x_cond, x_uncond, guidance_scale)

    def _get_guidance_scale(
        self,
        guidance_scale: Optional[float],
        t: torch.Tensor,
        step: Optional[int],
        total_steps: Optional[int],
    ) -> float:
        """Determine guidance scale to use."""
        if guidance_scale is not None:
            return guidance_scale

        if self.scheduler is not None:
            t_float = t[0].item() if t.dim() > 0 else t.item()
            return self.scheduler.get_scale(t_float, step, total_steps)

        return self.default_scale

    def enable_batched_cfg(self) -> None:
        """Enable batched CFG (compute cond and uncond together)."""
        self._batched_cfg = True

    def disable_batched_cfg(self) -> None:
        """Disable batched CFG."""
        self._batched_cfg = False


__all__ = [
    "CFGWrapper",
]
