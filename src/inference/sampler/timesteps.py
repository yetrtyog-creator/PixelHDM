"""
PixelHDM-RPEA-DinoV3 Sampler - Time Step Utilities (V-Prediction)

PixelHDM Flow Matching time step generation utilities.
V-Prediction: 不需要 x_to_v 轉換。

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

import logging
from typing import Optional, Dict

import torch

from ...training.flow_matching.time_sampling import (
    apply_timestep_shift,
    compute_timestep_shift_mu,
)

logger = logging.getLogger(__name__)


def get_timesteps(
    num_steps: int,
    t_eps: float = 0.0001,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    t_start: Optional[float] = None,
    use_logit_normal: bool = True,
    p_mean: float = 0.0,
    p_std: float = 1.0,
    num_tokens: Optional[int] = None,
    shift_config: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """
    Generate sampling timesteps.

    PixelHDM style: from t_start (or t_eps) to 1-t_eps (near clean).

    When use_logit_normal=True (default), uses Logit-Normal ICDF to match
    training distribution. This ensures consistency between training and
    inference timestep distributions.

    Args:
        num_steps: Number of sampling steps
        t_eps: Time epsilon for numerical stability
        device: Target device
        dtype: Target dtype
        t_start: Start time (for I2I), defaults to t_eps
        use_logit_normal: Whether to use Logit-Normal distribution (True)
                         or uniform spacing (False). Default True for
                         consistency with training distribution.
        p_mean: Logit-Normal mean (default 0.0, same as SD3/PixelHDM training)
        p_std: Logit-Normal std (default 1.0, same as SD3/PixelHDM training)

    Returns:
        Timesteps tensor of shape (num_steps + 1,)
    """
    # Validate t_eps: must be in (0, 0.5) to have valid time range
    max_t_eps = 0.4  # Conservative upper bound to prevent scale explosion
    if t_eps >= 0.5:
        logger.warning(f"t_eps={t_eps:.4f} >= 0.5 is invalid, clamping to {max_t_eps}")
        t_eps = max_t_eps
    elif t_eps > max_t_eps:
        logger.warning(f"t_eps={t_eps:.4f} > {max_t_eps} may cause numerical instability")

    start = t_start if t_start is not None else t_eps

    # Validate and clamp start time
    if start < t_eps:
        logger.warning(f"t_start={start:.4f} < t_eps={t_eps:.4f}, clamping to t_eps")
        start = t_eps
    if start > 1 - t_eps:
        logger.warning(f"t_start={start:.4f} > 1-t_eps={1-t_eps:.4f}, clamping to 1-t_eps")
        start = 1 - t_eps

    if not use_logit_normal:
        # Legacy: uniform spacing (not recommended, inconsistent with training)
        return torch.linspace(
            start, 1 - t_eps, num_steps + 1,
            device=device, dtype=dtype
        )

    # Logit-Normal ICDF: t = sigmoid(p_mean + p_std * Phi^{-1}(u))
    # Generate uniform points in CDF space, then map to time via ICDF.
    u = torch.linspace(0, 1, num_steps + 1, device=device, dtype=dtype)

    # Standard normal ICDF (probit function) via erfinv:
    # Phi^{-1}(u) = sqrt(2) * erfinv(2u - 1)
    # Clamp u to avoid infinities at the boundaries.
    u_clamped = u.clamp(1e-6, 1 - 1e-6)
    z = torch.erfinv(2 * u_clamped - 1) * (2 ** 0.5)

    # Apply Logit-Normal ICDF.
    t = torch.sigmoid(p_mean + p_std * z)

    # Rescale to [t_eps, 1-t_eps] range.
    t = t_eps + (1 - 2 * t_eps) * t

    if num_tokens is not None and shift_config is not None:
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
        t = t.clamp(t_eps, 1.0 - t_eps)

    # For I2I: linearly shift range to start at t_start while preserving count.
    if t_start is not None and t_start > t_eps:
        denom = 1 - 2 * t_eps
        if denom <= 0:
            logger.warning("Invalid t_eps; falling back to uniform timesteps")
            return torch.linspace(
                start, 1 - t_eps, num_steps + 1,
                device=device, dtype=dtype
            )
        scale = (1 - t_eps - start) / denom
        t = start + (t - t_eps) * scale

    # Ensure exact endpoints for stability/tests.
    if t.numel() >= 2:
        t[0] = start
        t[-1] = 1 - t_eps

    return t


def get_lambda(t: torch.Tensor, t_eps: float = 0.0001) -> torch.Tensor:
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
