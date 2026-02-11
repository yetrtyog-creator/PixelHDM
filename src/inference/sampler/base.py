"""
PixelHDM-RPEA-DinoV3 Sampler - Base Classes

Base sampler class and configuration dataclasses.

V-Prediction: 模型直接輸出 velocity v = x - ε

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import torch

from .timesteps import get_timesteps
from ..cfg.base import compute_rescaled_output


def _predict_v_with_cfg(
    model: Callable,
    z: torch.Tensor,
    t: torch.Tensor,
    text_embeddings: Optional[torch.Tensor],
    guidance_scale: float,
    null_text_embeddings: Optional[torch.Tensor],
    guidance_rescale: float = 0.0,
    **model_kwargs,
) -> torch.Tensor:
    """Predict velocity with optional CFG.

    Shared implementation for all samplers.

    Args:
        model: Model callable
        z: Current latent state
        t: Current timestep
        text_embeddings: Conditional text embeddings
        guidance_scale: CFG scale (1.0 = no CFG)
        null_text_embeddings: Unconditional text embeddings
        guidance_rescale: Rescale factor to prevent oversaturation (0.0 = no rescale)
        **model_kwargs: Additional model arguments

    Note:
        CFG branches need separate text_mask:
        - text_mask: for conditional branch
        - null_text_mask: for unconditional branch

        guidance_rescale (arXiv:2305.08891):
        When > 0, rescales CFG output to match conditional output's std.
        Helps prevent color shift and oversaturation at high guidance scales.
    """
    # Use get() instead of pop() to preserve model_kwargs for multi-call samplers
    # (e.g., Heun calls _predict_v twice with the same model_kwargs dict)
    text_mask = model_kwargs.get("text_mask", None)
    null_text_mask = model_kwargs.get("null_text_mask", None)

    # Create filtered kwargs without mask keys to avoid duplicate keyword arguments
    filtered_kwargs = {k: v for k, v in model_kwargs.items()
                       if k not in ("text_mask", "null_text_mask")}

    if guidance_scale > 1.0 and null_text_embeddings is not None:
        uncond_mask = null_text_mask if null_text_mask is not None else text_mask
        v_uncond = model(
            z, t,
            text_embed=null_text_embeddings,
            text_mask=uncond_mask,
            **filtered_kwargs
        )
        v_cond = model(
            z, t,
            text_embed=text_embeddings,
            text_mask=text_mask,
            **filtered_kwargs
        )
        v_cfg = v_uncond + guidance_scale * (v_cond - v_uncond)

        # Apply rescaling if enabled (prevents oversaturation at high CFG)
        if guidance_rescale > 0.0:
            v_cfg = compute_rescaled_output(v_cond, v_cfg, guidance_rescale)

        return v_cfg
    else:
        return model(
            z, t,
            text_embed=text_embeddings,
            text_mask=text_mask,
            **filtered_kwargs
        )


class SamplerMethod(Enum):
    """Sampler method enumeration."""
    EULER = "euler"
    HEUN = "heun"
    DPM_PP = "dpm_pp"
    DPM_PP_2S = "dpm_pp_2s"


@dataclass
class SamplerConfig:
    """Sampler configuration."""
    num_steps: int = 50
    method: SamplerMethod = SamplerMethod.HEUN
    t_eps: float = 0.0001
    guidance_scale: float = 7.5
    guidance_rescale: float = 0.0
    # Time distribution (must match training for consistency)
    use_logit_normal: bool = True  # Use Logit-Normal timesteps (matches training)
    time_p_mean: float = 0.0  # Logit-Normal mean (SD3/PixelHDM default)
    time_p_std: float = 1.0   # Logit-Normal std (SD3/PixelHDM default)
    use_dynamic_timestep_shift: bool = True
    timestep_shift_config: Optional[dict] = None


class BaseSampler(ABC):
    """Base sampler class."""

    def __init__(
        self,
        num_steps: int = 50,
        t_eps: float = 0.0001,
        use_logit_normal: bool = True,
        time_p_mean: float = 0.0,
        time_p_std: float = 1.0,
        use_dynamic_timestep_shift: bool = False,
        timestep_shift_config: Optional[dict] = None,
    ) -> None:
        self.num_steps = num_steps
        self.t_eps = t_eps
        # Logit-Normal distribution parameters (must match training)
        self.use_logit_normal = use_logit_normal
        self.time_p_mean = time_p_mean
        self.time_p_std = time_p_std
        self.use_dynamic_timestep_shift = use_dynamic_timestep_shift
        self.timestep_shift_config = timestep_shift_config

    @abstractmethod
    def step(
        self,
        model: Callable,
        z: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        text_embeddings: Optional[torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """Single sampling step."""
        pass

    def get_timesteps(
        self,
        num_steps: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        t_start: Optional[float] = None,
        num_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Get sampling timesteps.

        PixelHDM style: from t_start (or t_eps) to 1-t_eps (near clean).
        Uses Logit-Normal distribution by default for consistency with training.

        Args:
            num_steps: Number of sampling steps
            device: Target device
            dtype: Target dtype
            t_start: Start time (for I2I), defaults to t_eps
        """
        num_steps = num_steps if num_steps is not None else self.num_steps
        return get_timesteps(
            num_steps=num_steps,
            t_eps=self.t_eps,
            device=device,
            dtype=dtype,
            t_start=t_start,
            use_logit_normal=self.use_logit_normal,
            p_mean=self.time_p_mean,
            p_std=self.time_p_std,
            num_tokens=num_tokens,
            shift_config=self.timestep_shift_config if self.use_dynamic_timestep_shift else None,
        )

    def _predict_v(
        self,
        model: Callable,
        z: torch.Tensor,
        t: torch.Tensor,
        text_embeddings: Optional[torch.Tensor],
        guidance_scale: float,
        null_text_embeddings: Optional[torch.Tensor],
        guidance_rescale: float = 0.0,
        **model_kwargs,
    ) -> torch.Tensor:
        """Predict velocity with optional CFG.

        Args:
            model: Model callable
            z: Current latent state
            t: Current timestep
            text_embeddings: Conditional text embeddings
            guidance_scale: CFG scale
            null_text_embeddings: Unconditional text embeddings
            guidance_rescale: Rescale factor (0.0 = disabled)
            **model_kwargs: Additional model arguments
        """
        return _predict_v_with_cfg(
            model, z, t, text_embeddings,
            guidance_scale, null_text_embeddings,
            guidance_rescale=guidance_rescale,
            **model_kwargs
        )


__all__ = [
    "SamplerMethod",
    "SamplerConfig",
    "BaseSampler",
    "_predict_v_with_cfg",
]
