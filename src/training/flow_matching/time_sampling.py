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

import logging
import math
from typing import TYPE_CHECKING, Optional, Tuple, Callable, Dict

import torch

if TYPE_CHECKING:
    from ...config.model_config import PixelHDMConfig

logger = logging.getLogger(__name__)


def sample_logit_normal(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    p_mean: float = 0.0,
    p_std: float = 1.0,
    t_eps: float = 0.0001,
    num_tokens: Optional[int] = None,
    shift_config: Optional[Dict[str, float]] = None,
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

    if num_tokens is not None and shift_config is not None:
        t = _apply_dynamic_timestep_shift(
            t=t,
            t_eps=t_eps,
            num_tokens=num_tokens,
            shift_config=shift_config,
        )

    return t


def get_sampling_timesteps(
    num_steps: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    t_eps: float = 0.0001,
    use_logit_normal: bool = True,
    p_mean: float = 0.0,
    p_std: float = 1.0,
    num_tokens: Optional[int] = None,
    shift_config: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """
    Get timesteps for inference sampling.

    PixelHDM sampling direction: t from 0 to 1 (noise to clean).

    When use_logit_normal=True (default), uses Logit-Normal ICDF to match
    training distribution. This ensures model is evaluated at timesteps
    where it was most trained (concentrated around t=0.5).

    Args:
        num_steps: Number of sampling steps
        device: Target device
        dtype: Data type
        t_eps: Epsilon for time bounds
        use_logit_normal: Whether to use Logit-Normal distribution (True)
                         or uniform spacing (False). Default True for
                         consistency with training distribution.
        p_mean: Logit-Normal mean (same as training)
        p_std: Logit-Normal std (same as training)

    Returns:
        Timesteps from t_eps to 1-t_eps, shape [num_steps+1]
    """
    if not use_logit_normal:
        # Legacy: uniform spacing (not recommended, inconsistent with training)
        return torch.linspace(
            t_eps, 1 - t_eps, num_steps + 1,
            device=device, dtype=dtype
        )

    # Logit-Normal ICDF: t = sigmoid(p_mean + p_std * Φ^(-1)(u))
    # Generate uniform points in CDF space, then map to time via ICDF
    u = torch.linspace(0, 1, num_steps + 1, device=device, dtype=dtype)

    # Standard normal ICDF (probit function) via erfinv
    # Φ^(-1)(u) = sqrt(2) * erfinv(2u - 1)
    # Clamp u to avoid infinity at boundaries
    u_clamped = u.clamp(1e-6, 1 - 1e-6)
    z = torch.erfinv(2 * u_clamped - 1) * (2 ** 0.5)

    # Apply Logit-Normal ICDF
    t = torch.sigmoid(p_mean + p_std * z)

    # Rescale to [t_eps, 1-t_eps] range
    t = t_eps + (1 - 2 * t_eps) * t

    if num_tokens is not None and shift_config is not None:
        t = _apply_dynamic_timestep_shift(
            t=t,
            t_eps=t_eps,
            num_tokens=num_tokens,
            shift_config=shift_config,
        )

    return t


def compute_timestep_shift_mu(
    num_tokens: int,
    shift_type: str = "exponential",
    fixed_shift: float = 3.0,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    clamp_linear: bool = True,
) -> float:
    """Return mu (time_shift parameter).

    - exponential: linearly interpolates factor from 1.0 (at base_seq_len)
      to fixed_shift (at max_seq_len), then returns log(factor).
      This makes DTS resolution-dependent.
    - linear: mu is the shift factor directly
    """
    if shift_type == "exponential":
        if fixed_shift <= 0:
            raise ValueError("fixed_shift must be > 0")
        if max_seq_len == base_seq_len:
            raise ValueError("max_seq_len must differ from base_seq_len for exponential shift")
        m = (fixed_shift - 1.0) / (max_seq_len - base_seq_len)
        b = 1.0 - m * base_seq_len
        factor = max(1.0, num_tokens * m + b)
        return math.log(factor)
    if shift_type == "linear":
        if max_seq_len == base_seq_len:
            raise ValueError("max_seq_len must differ from base_seq_len for linear shift")
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = num_tokens * m + b
        if clamp_linear:
            mu = max(min(mu, max_shift), base_shift)
        return mu
    raise ValueError(f"Unknown shift_type: {shift_type}")


def apply_timestep_shift(
    sigmas: torch.Tensor,
    mu: float,
    shift_type: str = "exponential",
) -> torch.Tensor:
    """Apply time-shift to sigma (noise proportion)."""
    sigmas_safe = sigmas.clamp(min=1e-8, max=1.0 - 1e-8)

    if shift_type == "exponential":
        exp_mu = math.exp(mu)
        return exp_mu / (exp_mu + (1.0 / sigmas_safe - 1.0))
    if shift_type == "linear":
        return mu / (mu + (1.0 / sigmas_safe - 1.0))
    raise ValueError(f"Unknown shift_type: {shift_type}")


def _apply_dynamic_timestep_shift(
    t: torch.Tensor,
    t_eps: float,
    num_tokens: int,
    shift_config: Dict[str, float],
) -> torch.Tensor:
    allowed_keys = {
        "shift_type",
        "fixed_shift",
        "base_shift",
        "max_shift",
        "base_seq_len",
        "max_seq_len",
        "clamp_linear",
    }
    extra_keys = set(shift_config.keys()) - allowed_keys
    if extra_keys:
        logger.warning(
            "Ignoring unknown shift_config keys: %s",
            sorted(extra_keys),
        )

    shift_type = shift_config.get("shift_type", "exponential")
    mu = compute_timestep_shift_mu(
        num_tokens=num_tokens,
        shift_type=shift_type,
        fixed_shift=float(shift_config.get("fixed_shift", 3.0)),
        base_shift=float(shift_config.get("base_shift", 0.5)),
        max_shift=float(shift_config.get("max_shift", 1.15)),
        base_seq_len=int(shift_config.get("base_seq_len", 256)),
        max_seq_len=int(shift_config.get("max_seq_len", 4096)),
        clamp_linear=bool(shift_config.get("clamp_linear", True)),
    )

    sigma = 1.0 - t
    sigma = apply_timestep_shift(sigma, mu, shift_type)
    t = 1.0 - sigma
    return t.clamp(t_eps, 1.0 - t_eps)


def build_timestep_shift_config(config: "PixelHDMConfig") -> Optional[Dict[str, float]]:
    """Build shift_config dict from PixelHDMConfig. Returns None if disabled."""
    if not getattr(config, "use_dynamic_timestep_shift", False):
        return None
    return {
        "shift_type": getattr(config, "timestep_shift_type", "exponential"),
        "fixed_shift": getattr(config, "timestep_shift_fixed", 3.0),
        "base_shift": getattr(config, "timestep_shift_base_shift", 0.5),
        "max_shift": getattr(config, "timestep_shift_max_shift", 1.15),
        "base_seq_len": getattr(config, "timestep_shift_base_seq_len", 256),
        "max_seq_len": getattr(config, "timestep_shift_max_seq_len", 4096),
        "clamp_linear": getattr(config, "timestep_shift_clamp_linear", True),
    }


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
        t_eps: float = 0.0001,
        shift_config: Optional[Dict[str, float]] = None,
    ) -> None:
        self.p_mean = p_mean
        self.p_std = p_std
        self.t_eps = t_eps
        self.shift_config = shift_config

    @classmethod
    def from_config(cls, config: "PixelHDMConfig") -> "TimeSampler":
        """Create from PixelHDMConfig."""
        return cls(
            p_mean=config.time_p_mean,
            p_std=config.time_p_std,
            t_eps=config.time_eps,
            shift_config=build_timestep_shift_config(config),
        )

    def sample(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        num_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample timesteps for training."""
        return sample_logit_normal(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            p_mean=self.p_mean,
            p_std=self.p_std,
            t_eps=self.t_eps,
            num_tokens=num_tokens,
            shift_config=self.shift_config,
        )

    def get_inference_timesteps(
        self,
        num_steps: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        use_logit_normal: bool = True,
        num_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """Get timesteps for inference.

        Args:
            num_steps: Number of sampling steps
            device: Target device
            dtype: Data type
            use_logit_normal: Whether to use Logit-Normal distribution
                             matching training (default True)

        Returns:
            Timesteps tensor of shape (num_steps + 1,)
        """
        return get_sampling_timesteps(
            num_steps=num_steps,
            device=device,
            dtype=dtype,
            t_eps=self.t_eps,
            use_logit_normal=use_logit_normal,
            p_mean=self.p_mean,
            p_std=self.p_std,
            num_tokens=num_tokens,
            shift_config=self.shift_config,
        )


__all__ = [
    "sample_logit_normal",
    "get_sampling_timesteps",
    "TimeSampler",
    "compute_timestep_shift_mu",
    "apply_timestep_shift",
    "build_timestep_shift_config",
]
