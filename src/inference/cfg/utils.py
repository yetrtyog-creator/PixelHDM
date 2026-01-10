"""
CFG Utility Functions.

Contains:
    - apply_cfg: Convenience function for CFG application
    - compute_guidance_scale_schedule: Pre-compute CFG schedule
    - create_cfg: Factory function for CFG methods
    - create_cfg_scheduler: Factory function for CFG schedulers

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import math

import torch

from .base import BaseCFG, CFGScheduler, compute_rescaled_output
from .standard import StandardCFG, PerplexityCFG
from .rescaled import RescaledCFG
from .interval import CFGWithInterval


def apply_cfg(
    x_cond: torch.Tensor,
    x_uncond: torch.Tensor,
    guidance_scale: float,
    rescale_factor: float = 0.0,
) -> torch.Tensor:
    """
    Convenience CFG application function.

    Args:
        x_cond: Conditional prediction
        x_uncond: Unconditional prediction
        guidance_scale: CFG weight
        rescale_factor: Rescale factor (0=disabled)

    Returns:
        CFG weighted prediction
    """
    x_cfg = x_uncond + guidance_scale * (x_cond - x_uncond)
    return compute_rescaled_output(x_cond, x_cfg, rescale_factor)


def compute_guidance_scale_schedule(
    num_steps: int,
    start_scale: float = 7.5,
    end_scale: float = 1.0,
    schedule_type: str = "linear",
) -> torch.Tensor:
    """
    Compute CFG schedule sequence.

    Args:
        num_steps: Sampling steps
        start_scale: Starting CFG weight
        end_scale: Ending CFG weight
        schedule_type: Schedule type

    Returns:
        scales: (num_steps,) CFG weight sequence
    """
    t = torch.linspace(0, 1, num_steps)

    if schedule_type == "constant":
        scales = torch.full((num_steps,), start_scale)
    elif schedule_type == "linear":
        scales = start_scale + (end_scale - start_scale) * t
    elif schedule_type == "cosine":
        scales = end_scale + (start_scale - end_scale) * (
            1 + torch.cos(math.pi * t)
        ) / 2
    elif schedule_type == "quadratic":
        scales = start_scale + (end_scale - start_scale) * (t ** 2)
    else:
        raise ValueError(f"未知的調度類型: {schedule_type}")

    return scales


def create_cfg(
    method: str = "standard",
    rescale_factor: float = 0.0,
    temperature: float = 1.0,
    guidance_scale: float = 7.5,
    interval_start: float = 0.0,
    interval_end: float = 0.75,
) -> BaseCFG:
    """
    Create CFG method.

    Args:
        method: CFG method (standard, rescaled, perplexity, interval)
        rescale_factor: Rescale factor (for rescaled and interval)
        temperature: Temperature (for perplexity)
        guidance_scale: CFG weight (for interval)
        interval_start: CFG start time (for interval)
        interval_end: CFG end time (for interval)

    Returns:
        CFG method instance
    """
    if method == "standard":
        return StandardCFG()
    elif method == "rescaled":
        return RescaledCFG(rescale_factor)
    elif method == "perplexity":
        return PerplexityCFG(temperature)
    elif method == "interval":
        return CFGWithInterval(
            guidance_scale=guidance_scale,
            rescale_factor=rescale_factor,
            interval_start=interval_start,
            interval_end=interval_end,
        )
    else:
        valid_methods = ["standard", "rescaled", "perplexity", "interval"]
        raise ValueError(
            f"未知的 CFG 方法: {method}。"
            f"有效選項: {valid_methods}"
        )


def create_cfg_scheduler(
    schedule_type: str = "constant",
    min_scale: float = 1.0,
    max_scale: float = 7.5,
) -> CFGScheduler:
    """Create CFG scheduler."""
    return CFGScheduler(schedule_type, min_scale, max_scale)


__all__ = [
    "apply_cfg",
    "compute_guidance_scale_schedule",
    "create_cfg",
    "create_cfg_scheduler",
]
