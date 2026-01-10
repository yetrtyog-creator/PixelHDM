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


def _predict_v_with_cfg(
    model: Callable,
    z: torch.Tensor,
    t: torch.Tensor,
    text_embeddings: Optional[torch.Tensor],
    guidance_scale: float,
    null_text_embeddings: Optional[torch.Tensor],
    **model_kwargs,
) -> torch.Tensor:
    """Predict velocity with optional CFG.

    Shared implementation for all samplers.

    Note:
        CFG branches need separate pooled_text_embed and text_mask:
        - pooled_text_embed / text_mask: for conditional branch
        - null_pooled_text_embed / null_text_mask: for unconditional branch
    """
    pooled_text_embed = model_kwargs.pop("pooled_text_embed", None)
    null_pooled_text_embed = model_kwargs.pop("null_pooled_text_embed", None)
    text_mask = model_kwargs.pop("text_mask", None)
    null_text_mask = model_kwargs.pop("null_text_mask", None)

    if guidance_scale > 1.0 and null_text_embeddings is not None:
        uncond_mask = null_text_mask if null_text_mask is not None else text_mask
        v_uncond = model(
            z, t,
            text_embed=null_text_embeddings,
            text_mask=uncond_mask,
            pooled_text_embed=null_pooled_text_embed,
            **model_kwargs
        )
        v_cond = model(
            z, t,
            text_embed=text_embeddings,
            text_mask=text_mask,
            pooled_text_embed=pooled_text_embed,
            **model_kwargs
        )
        return v_uncond + guidance_scale * (v_cond - v_uncond)
    else:
        return model(
            z, t,
            text_embed=text_embeddings,
            text_mask=text_mask,
            pooled_text_embed=pooled_text_embed,
            **model_kwargs
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
    t_eps: float = 0.05
    guidance_scale: float = 7.5
    guidance_rescale: float = 0.0


class BaseSampler(ABC):
    """Base sampler class."""

    def __init__(
        self,
        num_steps: int = 50,
        t_eps: float = 0.05,
    ) -> None:
        self.num_steps = num_steps
        self.t_eps = t_eps

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
    ) -> torch.Tensor:
        """
        Get sampling timesteps.

        PixelHDM style: from t_start (or t_eps) to 1-t_eps (near clean).

        Args:
            num_steps: Number of sampling steps
            device: Target device
            dtype: Target dtype
            t_start: Start time (for I2I), defaults to t_eps
        """
        num_steps = num_steps or self.num_steps
        return get_timesteps(num_steps, self.t_eps, device, dtype, t_start)

    def _predict_v(
        self,
        model: Callable,
        z: torch.Tensor,
        t: torch.Tensor,
        text_embeddings: Optional[torch.Tensor],
        guidance_scale: float,
        null_text_embeddings: Optional[torch.Tensor],
        **model_kwargs,
    ) -> torch.Tensor:
        """Predict velocity with optional CFG."""
        return _predict_v_with_cfg(
            model, z, t, text_embeddings,
            guidance_scale, null_text_embeddings, **model_kwargs
        )


__all__ = [
    "SamplerMethod",
    "SamplerConfig",
    "BaseSampler",
    "_predict_v_with_cfg",
]
