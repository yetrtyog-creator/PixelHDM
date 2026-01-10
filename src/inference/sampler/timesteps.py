"""
PixelHDM-RPEA-DinoV3 Sampler - Time Step Utilities (V-Prediction)

PixelHDM Flow Matching time step generation utilities.
V-Prediction: 不需要 x_to_v 轉換。

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def get_timesteps(
    num_steps: int,
    t_eps: float = 0.05,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    t_start: Optional[float] = None,
) -> torch.Tensor:
    """
    Generate sampling timesteps.

    PixelHDM style: from t_start (or t_eps) to 1-t_eps (near clean).

    Args:
        num_steps: Number of sampling steps
        t_eps: Time epsilon for numerical stability
        device: Target device
        dtype: Target dtype
        t_start: Start time (for I2I), defaults to t_eps

    Returns:
        Timesteps tensor of shape (num_steps + 1,)
    """
    start = t_start if t_start is not None else t_eps

    # Validate and clamp start time
    if start < t_eps:
        logger.warning(f"t_start={start:.4f} < t_eps={t_eps:.4f}, clamping to t_eps")
        start = t_eps
    if start > 1 - t_eps:
        logger.warning(f"t_start={start:.4f} > 1-t_eps={1-t_eps:.4f}, clamping to 1-t_eps")
        start = 1 - t_eps

    return torch.linspace(
        start, 1 - t_eps, num_steps + 1,
        device=device, dtype=dtype
    )


def get_lambda(t: torch.Tensor, t_eps: float = 0.05) -> torch.Tensor:
    """
    Compute lambda(t) = log(alpha(t)/sigma(t)).

    For PixelHDM flow matching: alpha(t) = t, sigma(t) = 1-t

    Args:
        t: Timestep tensor
        t_eps: Epsilon for numerical stability

    Returns:
        Lambda value
    """
    t = t.clamp(t_eps, 1 - t_eps)
    return torch.log(t / (1 - t))


__all__ = [
    "get_timesteps",
    "get_lambda",
]
