"""
CFG with time interval control.

Contains:
    - CFGWithInterval: CFG that only applies in specific time ranges

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import torch

from .base import BaseCFG, compute_rescaled_output


class CFGWithInterval(BaseCFG):
    """
    CFG with time interval control.

    Allows CFG to be applied only in specific time intervals, useful for:
    1. Early CFG stopping (save computation)
    2. Delayed CFG start (let early denoising be more free)
    3. Custom CFG application intervals

    Args:
        guidance_scale: CFG weight
        rescale_factor: Rescale factor (0=disabled)
        interval_start: CFG start time (PixelHDM: 0=noise, 1=clean)
        interval_end: CFG end time

    Time convention (PixelHDM style):
        - t=0: Pure noise
        - t=1: Clean image
        - Default interval_end=0.75 means stop CFG at 75% progress
    """

    def __init__(
        self,
        guidance_scale: float = 7.5,
        rescale_factor: float = 0.0,
        interval_start: float = 0.0,
        interval_end: float = 0.75,
    ) -> None:
        self._validate_interval(interval_start, interval_end)
        self.guidance_scale = guidance_scale
        self.rescale_factor = rescale_factor
        self.interval_start = interval_start
        self.interval_end = interval_end

    def _validate_interval(self, start: float, end: float) -> None:
        """Validate interval parameters."""
        if not (0.0 <= start < end <= 1.0):
            raise ValueError(
                f"Invalid interval: [{start}, {end}]. "
                f"Must satisfy 0 <= start < end <= 1"
            )

    def should_apply_cfg(self, t: float) -> bool:
        """
        Check if CFG should be applied at current timestep.

        Args:
            t: Current timestep (PixelHDM: 0=noise, 1=clean)

        Returns:
            Whether to apply CFG
        """
        return self.interval_start <= t < self.interval_end

    def get_effective_scale(self, t: float) -> float:
        """
        Get effective CFG weight for current timestep.

        Args:
            t: Current timestep

        Returns:
            CFG weight (1.0 if outside interval)
        """
        if self.should_apply_cfg(t):
            return self.guidance_scale
        return 1.0

    def apply(
        self,
        x_cond: torch.Tensor,
        x_uncond: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """
        Apply CFG (standard mode, ignores interval).

        Note: Caller should use should_apply_cfg() to check if needed.

        Args:
            x_cond: Conditional prediction
            x_uncond: Unconditional prediction
            guidance_scale: CFG weight

        Returns:
            CFG weighted prediction
        """
        x_cfg = x_uncond + guidance_scale * (x_cond - x_uncond)
        return compute_rescaled_output(x_cond, x_cfg, self.rescale_factor)

    def apply_with_interval_check(
        self,
        x_cond: torch.Tensor,
        x_uncond: torch.Tensor,
        t: float,
    ) -> torch.Tensor:
        """
        Apply CFG with interval check.

        Args:
            x_cond: Conditional prediction
            x_uncond: Unconditional prediction
            t: Current timestep

        Returns:
            Processed prediction (x_cond if outside interval)
        """
        if not self.should_apply_cfg(t):
            return x_cond
        return self.apply(x_cond, x_uncond, self.guidance_scale)


__all__ = [
    "CFGWithInterval",
]
