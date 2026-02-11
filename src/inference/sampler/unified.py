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
from ..cfg.base import CFGScheduler, CFGScheduleType
from ...training.flow_matching.time_sampling import build_timestep_shift_config

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
        use_logit_normal: Use Logit-Normal timesteps (matches training)
        time_p_mean: Logit-Normal mean (must match training)
        time_p_std: Logit-Normal std (must match training)
        cfg_schedule: CFG schedule type ("constant", "linear", "cosine", "quadratic")
        cfg_min_scale: Minimum CFG scale (for dynamic scheduling)
        cfg_max_scale: Maximum CFG scale (for dynamic scheduling)
    """

    def __init__(
        self,
        method: Union[str, SamplerMethod] = "heun",
        num_steps: int = 50,
        t_eps: float = 0.0001,
        use_logit_normal: bool = True,
        time_p_mean: float = 0.0,
        time_p_std: float = 1.0,
        use_dynamic_timestep_shift: bool = False,
        timestep_shift_config: Optional[dict] = None,
        cfg_schedule: str = "constant",
        cfg_min_scale: float = 1.0,
        cfg_max_scale: float = 7.5,
    ) -> None:
        if isinstance(method, str):
            method = SamplerMethod(method.lower())

        self.method = method
        self.num_steps = num_steps
        self.t_eps = t_eps
        self.use_logit_normal = use_logit_normal
        self.time_p_mean = time_p_mean
        self.time_p_std = time_p_std
        self.use_dynamic_timestep_shift = use_dynamic_timestep_shift
        self.timestep_shift_config = timestep_shift_config

        # CFG scheduler for dynamic guidance scale
        self.cfg_scheduler = CFGScheduler(
            schedule_type=cfg_schedule,
            min_scale=cfg_min_scale,
            max_scale=cfg_max_scale,
        )

        # Create sampler
        self._sampler = self._create_sampler()

    def _create_sampler(self) -> BaseSampler:
        """Create the corresponding sampler."""
        kwargs = {
            "num_steps": self.num_steps,
            "t_eps": self.t_eps,
            "use_logit_normal": self.use_logit_normal,
            "time_p_mean": self.time_p_mean,
            "time_p_std": self.time_p_std,
            "use_dynamic_timestep_shift": self.use_dynamic_timestep_shift,
            "timestep_shift_config": self.timestep_shift_config,
        }
        if self.method == SamplerMethod.EULER:
            return EulerSampler(**kwargs)
        elif self.method == SamplerMethod.HEUN:
            return HeunSampler(**kwargs)
        elif self.method in (SamplerMethod.DPM_PP, SamplerMethod.DPM_PP_2S):
            return DPMPPSampler(**kwargs)
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
        guidance_scale: Optional[float] = None,
        guidance_rescale: float = 0.0,
        null_text_embeddings: Optional[torch.Tensor] = None,
        null_text_mask: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        t_start: Optional[float] = None,
        use_dynamic_cfg: bool = False,
        num_tokens: Optional[int] = None,
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
            guidance_scale: CFG guidance scale. If None, uses cfg_max_scale.
                           Ignored when use_dynamic_cfg=True.
            guidance_rescale: Rescale factor to prevent oversaturation at high CFG
                             (0.0 = disabled, typical value 0.7)
            null_text_embeddings: (B, T, D) Unconditional text embeddings
            null_text_mask: (B, T) Null text mask for unconditional branch
            callback: Progress callback (step, total_steps, z)
            t_start: Start time (for I2I), defaults to t_eps
            use_dynamic_cfg: If True, use CFGScheduler for timestep-dependent
                            CFG scale. Helps prevent overshoot at high guidance.
            **model_kwargs: Additional model arguments

        Returns:
            x: (B, H, W, 3) Generated samples

        Note:
            When use_dynamic_cfg=True, the CFG scale varies based on timestep:
            - linear: high CFG early (noise removal), low CFG late (detail)
            - cosine: smooth transition between min and max
            - quadratic: faster decay from max to min
            This helps stabilize ODE integration at high guidance scales.
        """
        num_steps = num_steps if num_steps is not None else self.num_steps
        device = z_0.device
        dtype = z_0.dtype

        # Default guidance scale
        if guidance_scale is None:
            guidance_scale = self.cfg_scheduler.max_scale

        # Get timesteps (supports I2I custom start time)
        timesteps = self._sampler.get_timesteps(
            num_steps, device, dtype, t_start, num_tokens=num_tokens
        )

        # Reset sampler state (for DPM++)
        if hasattr(self._sampler, "reset"):
            self._sampler.reset()

        z = z_0

        for i in range(num_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]

            # Get CFG scale for this timestep
            if use_dynamic_cfg:
                current_cfg = self.cfg_scheduler.get_scale(
                    t=t.item() if isinstance(t, torch.Tensor) else t,
                    step=i,
                    total_steps=num_steps,
                )
            else:
                current_cfg = guidance_scale

            # Sampling step
            z = self._sampler.step(
                model=model,
                z=z,
                t=t,
                t_next=t_next,
                text_embeddings=text_embeddings,
                guidance_scale=current_cfg,
                null_text_embeddings=null_text_embeddings,
                guidance_rescale=guidance_rescale,
                step_idx=i,
                num_steps=num_steps,
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
        guidance_rescale: float = 0.0,
        null_text_embeddings: Optional[torch.Tensor] = None,
        null_text_mask: Optional[torch.Tensor] = None,
        return_every: int = 1,
        use_dynamic_cfg: bool = False,
        num_tokens: Optional[int] = None,
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
            guidance_rescale: Rescale factor to prevent oversaturation
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
            guidance_rescale=guidance_rescale,
            null_text_embeddings=null_text_embeddings,
            null_text_mask=null_text_mask,
            callback=save_callback,
            use_dynamic_cfg=use_dynamic_cfg,
            num_tokens=num_tokens,
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

        NFE = number of model forward passes during sampling.

        Args:
            num_steps: Number of sampling steps (None uses default)
            use_cfg: Whether to use CFG (doubles each model call)
            full_heun_cfg: Whether to use Heun for both CFG branches
                          (typically False, meaning corrector uses same v)

        Returns:
            Total NFE count

        Note (#11 fix):
            - Euler: 1 model call per step
            - Heun: 2 model calls per step (predictor + corrector)
            - Full-Heun CFG: each branch is Heun, last step skips corrector
            - DPM++: 1 model call per step (uses history for multi-step)
            - CFG: doubles each call (cond + uncond)
        """
        num_steps = num_steps if num_steps is not None else self.num_steps

        if self.method == SamplerMethod.EULER:
            base_nfe = num_steps
        elif self.method == SamplerMethod.HEUN:
            if full_heun_cfg:
                # Heun per-branch, last step skips corrector
                base_nfe = 2 * num_steps - 1
            else:
                # Standard Heun also skips corrector on last step (H-1)
                base_nfe = 2 * num_steps - 1
        elif self.method in (SamplerMethod.DPM_PP, SamplerMethod.DPM_PP_2S):
            base_nfe = num_steps
        else:
            base_nfe = num_steps

        return base_nfe * 2 if use_cfg else base_nfe


def create_sampler(
    method: str = "heun",
    num_steps: int = 50,
    t_eps: float = 0.0001,
    use_logit_normal: bool = True,
    time_p_mean: float = 0.0,
    time_p_std: float = 1.0,
    use_dynamic_timestep_shift: bool = False,
    timestep_shift_config: Optional[dict] = None,
    cfg_schedule: str = "constant",
    cfg_min_scale: float = 1.0,
    cfg_max_scale: float = 7.5,
) -> UnifiedSampler:
    """Create a sampler.

    Args:
        method: Sampling method (euler, heun, dpm_pp, dpm_pp_2s)
        num_steps: Number of sampling steps
        t_eps: Time epsilon for numerical stability
        use_logit_normal: Use Logit-Normal timesteps (matches training)
        time_p_mean: Logit-Normal mean (must match training config)
        time_p_std: Logit-Normal std (must match training config)
        cfg_schedule: CFG schedule type ("constant", "linear", "cosine", "quadratic")
        cfg_min_scale: Minimum CFG scale for dynamic scheduling
        cfg_max_scale: Maximum CFG scale for dynamic scheduling
    """
    return UnifiedSampler(
        method=method,
        num_steps=num_steps,
        t_eps=t_eps,
        use_logit_normal=use_logit_normal,
        time_p_mean=time_p_mean,
        time_p_std=time_p_std,
        use_dynamic_timestep_shift=use_dynamic_timestep_shift,
        timestep_shift_config=timestep_shift_config,
        cfg_schedule=cfg_schedule,
        cfg_min_scale=cfg_min_scale,
        cfg_max_scale=cfg_max_scale,
    )


def create_sampler_from_config(
    config: "PixelHDMConfig",
    method: Optional[str] = None,
    cfg_schedule: str = "constant",
    cfg_min_scale: float = 1.0,
) -> UnifiedSampler:
    """Create a sampler from config.

    Automatically uses training time distribution parameters for consistency.

    Args:
        config: PixelHDMConfig
        method: Override sampling method (default from config)
        cfg_schedule: CFG schedule type for dynamic guidance
        cfg_min_scale: Minimum CFG scale for dynamic scheduling
    """
    return UnifiedSampler(
        method=method or config.default_sampler_method,
        num_steps=config.default_num_steps,
        t_eps=config.time_eps,
        use_logit_normal=True,  # Always use Logit-Normal for consistency
        time_p_mean=config.time_p_mean,
        time_p_std=config.time_p_std,
        use_dynamic_timestep_shift=getattr(config, "use_dynamic_timestep_shift", False),
        timestep_shift_config=build_timestep_shift_config(config),
        cfg_schedule=cfg_schedule,
        cfg_min_scale=cfg_min_scale,
        cfg_max_scale=config.default_guidance_scale,
    )


__all__ = [
    "UnifiedSampler",
    "create_sampler",
    "create_sampler_from_config",
]
