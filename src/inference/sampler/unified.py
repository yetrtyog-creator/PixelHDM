"""
PixelHDM-RPEA-DinoV3 Sampler - Unified Sampler

Unified sampler interface that integrates all sampling methods.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Union

import torch

from .base import BaseSampler, SamplerMethod
from .euler import EulerSampler
from .heun import HeunSampler
from .dpm import DPMPPSampler
from .full_heun import FullHeunCFGMixin

if TYPE_CHECKING:
    from ...config.model_config import PixelHDMConfig


class UnifiedSampler(FullHeunCFGMixin):
    """
    Unified sampler interface.

    Integrates all sampling methods and provides a consistent API.

    Args:
        method: Sampling method
        num_steps: Number of sampling steps
        t_eps: Time epsilon
    """

    def __init__(
        self,
        method: Union[str, SamplerMethod] = "heun",
        num_steps: int = 50,
        t_eps: float = 0.05,
    ) -> None:
        if isinstance(method, str):
            method = SamplerMethod(method.lower())

        self.method = method
        self.num_steps = num_steps
        self.t_eps = t_eps

        # Create sampler
        self._sampler = self._create_sampler()

    def _create_sampler(self) -> BaseSampler:
        """Create the corresponding sampler."""
        if self.method == SamplerMethod.EULER:
            return EulerSampler(self.num_steps, self.t_eps)
        elif self.method == SamplerMethod.HEUN:
            return HeunSampler(self.num_steps, self.t_eps)
        elif self.method in (SamplerMethod.DPM_PP, SamplerMethod.DPM_PP_2S):
            return DPMPPSampler(self.num_steps, self.t_eps)
        else:
            valid_methods = [m.value for m in SamplerMethod]
            raise ValueError(
                f"Unknown sampling method: {self.method}. "
                f"Valid options: {valid_methods}"
            )

    @torch.no_grad()
    def sample(
        self,
        model: Callable,
        z_0: torch.Tensor,
        text_embeddings: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        guidance_scale: float = 7.5,
        null_text_embeddings: Optional[torch.Tensor] = None,
        null_text_mask: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        t_start: Optional[float] = None,
        **model_kwargs,
    ) -> torch.Tensor:
        """
        Generate samples from noise.

        Args:
            model: PixelHDM model
            z_0: (B, H, W, 3) Initial noise or noised image (I2I)
            text_embeddings: (B, T, D) Conditional text embeddings
            text_mask: (B, T) Text mask
            num_steps: Number of sampling steps (overrides default)
            guidance_scale: CFG guidance scale
            null_text_embeddings: (B, T, D) Unconditional text embeddings
            null_text_mask: (B, T) Null text mask for unconditional branch
            callback: Progress callback (step, total_steps, z)
            t_start: Start time (for I2I), defaults to t_eps
            **model_kwargs: Additional model arguments

        Returns:
            x: (B, H, W, 3) Generated samples
        """
        num_steps = num_steps or self.num_steps
        device = z_0.device
        dtype = z_0.dtype

        # Get timesteps (supports I2I custom start time)
        timesteps = self._sampler.get_timesteps(num_steps, device, dtype, t_start)

        # Reset sampler state (for DPM++)
        if hasattr(self._sampler, "reset"):
            self._sampler.reset()

        z = z_0

        for i in range(num_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]

            # Sampling step
            z = self._sampler.step(
                model=model,
                z=z,
                t=t,
                t_next=t_next,
                text_embeddings=text_embeddings,
                guidance_scale=guidance_scale,
                null_text_embeddings=null_text_embeddings,
                text_mask=text_mask,
                null_text_mask=null_text_mask,
                **model_kwargs,
            )

            # Callback
            if callback is not None:
                callback(i, num_steps, z)

        return z

    @torch.no_grad()
    def sample_progressive(
        self,
        model: Callable,
        z_0: torch.Tensor,
        text_embeddings: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        guidance_scale: float = 7.5,
        null_text_embeddings: Optional[torch.Tensor] = None,
        null_text_mask: Optional[torch.Tensor] = None,
        return_every: int = 1,
        **model_kwargs,
    ) -> torch.Tensor:
        """
        Progressive sampling, returns all intermediate steps.

        Args:
            model: PixelHDM model
            z_0: Initial noise
            text_embeddings: Conditional text embeddings
            text_mask: Text mask
            num_steps: Number of sampling steps
            guidance_scale: CFG guidance scale
            null_text_embeddings: Unconditional text embeddings
            null_text_mask: Null text mask for unconditional branch
            return_every: Return every N steps
            **model_kwargs: Additional model arguments

        Returns:
            progressions: (N_saved, B, H, W, 3) All saved intermediate states
        """
        progressions = []

        def save_callback(step, total, z):
            if step % return_every == 0 or step == total - 1:
                progressions.append(z.clone())

        self.sample(
            model=model,
            z_0=z_0,
            text_embeddings=text_embeddings,
            text_mask=text_mask,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            null_text_embeddings=null_text_embeddings,
            null_text_mask=null_text_mask,
            callback=save_callback,
            **model_kwargs,
        )

        return torch.stack(progressions)

    def count_nfe(
        self,
        num_steps: Optional[int] = None,
        use_cfg: bool = False,
        full_heun_cfg: bool = False,
    ) -> int:
        """
        Count total Network Function Evaluations (NFE).

        Args:
            num_steps: Number of sampling steps (None uses default)
            use_cfg: Whether to use CFG (doubles NFE)
            full_heun_cfg: Whether to use Heun for both CFG branches

        Returns:
            Total NFE count
        """
        num_steps = num_steps or self.num_steps

        # Base NFE (without CFG)
        if self.method == SamplerMethod.EULER:
            base_nfe = num_steps
        elif self.method == SamplerMethod.HEUN:
            # Heun needs 2 evaluations per step, but last step is Euler
            base_nfe = num_steps * 2 - 1
        elif self.method in (SamplerMethod.DPM_PP, SamplerMethod.DPM_PP_2S):
            # DPM++ uses 1 evaluation per step (uses history)
            base_nfe = num_steps
        else:
            base_nfe = num_steps

        # CFG doubles evaluations
        if use_cfg:
            if full_heun_cfg:
                return base_nfe * 2
            else:
                return num_steps * 2

        return base_nfe


def create_sampler(
    method: str = "heun",
    num_steps: int = 50,
    t_eps: float = 0.05,
) -> UnifiedSampler:
    """Create a sampler."""
    return UnifiedSampler(
        method=method,
        num_steps=num_steps,
        t_eps=t_eps,
    )


def create_sampler_from_config(
    config: "PixelHDMConfig",
    method: Optional[str] = None,
) -> UnifiedSampler:
    """Create a sampler from config."""
    return UnifiedSampler(
        method=method or config.default_sampler_method,
        num_steps=config.default_num_steps,
        t_eps=config.time_eps,
    )


__all__ = [
    "UnifiedSampler",
    "create_sampler",
    "create_sampler_from_config",
]
