"""
PixelHDM Flow Matching Core Module - Factory Functions

Author: PixelHDM-RPEA-DinoV3 Project
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .training import PixelHDMFlowMatching
from .pixelhdm_sampler import PixelHDMSampler

if TYPE_CHECKING:
    from ...config.model_config import PixelHDMConfig


def create_flow_matching(
    config: Optional["PixelHDMConfig"] = None,
    p_mean: float = 0.0,   # SD3/PixelHDM convention
    p_std: float = 1.0,    # SD3/PixelHDM convention
    t_eps: float = 0.05,
) -> PixelHDMFlowMatching:
    """Create Flow Matching module."""
    return PixelHDMFlowMatching(
        config=config,
        p_mean=p_mean,
        p_std=p_std,
        t_eps=t_eps,
    )


def create_sampler(
    num_steps: int = 50,
    method: str = "heun",
    t_eps: float = 0.05,
) -> PixelHDMSampler:
    """Create sampler."""
    return PixelHDMSampler(
        num_steps=num_steps,
        method=method,
        t_eps=t_eps,
    )


def create_flow_matching_from_config(config: "PixelHDMConfig") -> PixelHDMFlowMatching:
    """Create Flow Matching module from config."""
    return PixelHDMFlowMatching(config=config)


__all__ = [
    "PixelHDMFlowMatching",
    "PixelHDMSampler",
    "create_flow_matching",
    "create_flow_matching_from_config",
    "create_sampler",
]
