"""
CFG Base Classes and Shared Components

Contains:
    - CFGScheduleType enum
    - CFGConfig dataclass
    - CFGScheduler
    - BaseCFG abstract class

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import torch


class CFGScheduleType(Enum):
    """CFG schedule types."""
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    QUADRATIC = "quadratic"


@dataclass
class CFGConfig:
    """CFG configuration."""
    guidance_scale: float = 7.5
    guidance_rescale: float = 0.0
    schedule_type: CFGScheduleType = CFGScheduleType.CONSTANT
    min_guidance_scale: float = 1.0
    max_guidance_scale: float = 7.5
    use_negative_prompt: bool = True
    null_prompt: str = ""


class CFGScheduler:
    """
    CFG Scheduler for dynamic CFG weight adjustment.

    Args:
        schedule_type: Schedule type (constant, linear, cosine, quadratic)
        min_scale: Minimum CFG weight
        max_scale: Maximum CFG weight
    """

    def __init__(
        self,
        schedule_type: Union[str, CFGScheduleType] = "constant",
        min_scale: float = 1.0,
        max_scale: float = 7.5,
    ) -> None:
        if isinstance(schedule_type, str):
            schedule_type = CFGScheduleType(schedule_type)
        self.schedule_type = schedule_type
        self.min_scale = min_scale
        self.max_scale = max_scale

    def get_scale(
        self,
        t: float,
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> float:
        """
        Get CFG weight for current timestep.

        Args:
            t: Timestep (PixelHDM: 0=noise, 1=clean)
            step: Current step number
            total_steps: Total steps

        Returns:
            CFG weight
        """
        if self.schedule_type == CFGScheduleType.CONSTANT:
            return self.max_scale

        progress = t

        if self.schedule_type == CFGScheduleType.LINEAR:
            scale = self.max_scale - (self.max_scale - self.min_scale) * progress
        elif self.schedule_type == CFGScheduleType.COSINE:
            scale = self.min_scale + (self.max_scale - self.min_scale) * (
                1 + math.cos(math.pi * progress)
            ) / 2
        elif self.schedule_type == CFGScheduleType.QUADRATIC:
            scale = self.max_scale - (self.max_scale - self.min_scale) * (progress ** 2)
        else:
            scale = self.max_scale

        return scale


class BaseCFG(ABC):
    """Abstract base class for CFG methods."""

    @abstractmethod
    def apply(
        self,
        x_cond: torch.Tensor,
        x_uncond: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Apply CFG."""
        pass


def compute_rescaled_output(
    x_cond: torch.Tensor,
    x_cfg: torch.Tensor,
    rescale_factor: float,
) -> torch.Tensor:
    """
    Apply rescaling to CFG output.

    Args:
        x_cond: Conditional prediction
        x_cfg: Standard CFG output
        rescale_factor: Rescale factor (0=no rescaling)

    Returns:
        Rescaled output
    """
    if rescale_factor == 0.0:
        return x_cfg

    std_cond = x_cond.std(dim=list(range(1, x_cond.dim())), keepdim=True)
    std_cfg = x_cfg.std(dim=list(range(1, x_cfg.dim())), keepdim=True)
    factor = std_cond / (std_cfg + 1e-8)
    rescaled = x_cfg * factor

    return rescale_factor * rescaled + (1 - rescale_factor) * x_cfg


__all__ = [
    "CFGScheduleType",
    "CFGConfig",
    "CFGScheduler",
    "BaseCFG",
    "compute_rescaled_output",
]
