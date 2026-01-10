"""
Time Sampling for PixelHDM Flow Matching

Provides Logit-Normal time distribution sampling for training.

Time Convention (PixelHDM):
    - t=0: noise
    - t=1: clean image

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Callable

import torch

if TYPE_CHECKING:
    from ...config.model_config import PixelHDMConfig


def sample_logit_normal(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    p_mean: float = 0.0,
    p_std: float = 1.0,
    t_eps: float = 0.05,
) -> torch.Tensor:
    """
    Sample timesteps from Logit-Normal distribution.

    Formula: t = sigmoid(p_mean + p_std * N(0,1))
    Then rescale to [t_eps, 1-t_eps].

    Args:
        batch_size: Number of samples
        device: Target device
        dtype: Data type
        p_mean: Distribution mean (SD3=0.0, JiT=-0.8)
        p_std: Distribution std (SD3=1.0, JiT=0.8)
        t_eps: Epsilon for time clamping

    Returns:
        t: Timesteps, shape [B], range [t_eps, 1-t_eps]
    """
    u = torch.randn(batch_size, device=device, dtype=dtype)
    u = p_mean + p_std * u
    t = torch.sigmoid(u)
    t = t_eps + (1 - 2 * t_eps) * t
    return t


def get_sampling_timesteps(
    num_steps: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    t_eps: float = 0.05,
) -> torch.Tensor:
    """
    Get evenly spaced timesteps for inference sampling.

    PixelHDM sampling direction: t from 0 to 1 (noise to clean).

    Args:
        num_steps: Number of sampling steps
        device: Target device
        dtype: Data type
        t_eps: Epsilon for time bounds

    Returns:
        Timesteps from t_eps to 1-t_eps, shape [num_steps+1]
    """
    return torch.linspace(
        t_eps, 1 - t_eps, num_steps + 1,
        device=device, dtype=dtype
    )


class TimeSampler:
    """
    Configurable time sampler for Flow Matching.

    Supports both SD3/PixelHDM (p_mean=0.0) and JiT (p_mean=-0.8) distributions.

    Args:
        p_mean: Distribution mean
        p_std: Distribution std
        t_eps: Time epsilon
    """

    def __init__(
        self,
        p_mean: float = 0.0,
        p_std: float = 1.0,
        t_eps: float = 0.05,
    ) -> None:
        self.p_mean = p_mean
        self.p_std = p_std
        self.t_eps = t_eps

    @classmethod
    def from_config(cls, config: "PixelHDMConfig") -> "TimeSampler":
        """Create from PixelHDMConfig."""
        return cls(
            p_mean=config.time_p_mean,
            p_std=config.time_p_std,
            t_eps=config.time_eps,
        )

    def sample(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Sample timesteps for training."""
        return sample_logit_normal(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            p_mean=self.p_mean,
            p_std=self.p_std,
            t_eps=self.t_eps,
        )

    def get_inference_timesteps(
        self,
        num_steps: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Get timesteps for inference."""
        return get_sampling_timesteps(
            num_steps=num_steps,
            device=device,
            dtype=dtype,
            t_eps=self.t_eps,
        )


__all__ = [
    "sample_logit_normal",
    "get_sampling_timesteps",
    "TimeSampler",
]
