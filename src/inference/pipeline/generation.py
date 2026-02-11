"""
Pipeline Generation Core

Handles the core sampling/generation logic.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Callable, List, Union

import torch

from ..cfg import CFGScheduler
from ..sampler import UnifiedSampler, create_sampler
from ...training.flow_matching.time_sampling import build_timestep_shift_config
from .preprocessing import GenerationInputs

if TYPE_CHECKING:
    from ...config.model_config import PixelHDMConfig

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """
    Generation configuration parameters.

    Attributes:
        height: Image height in pixels (default 512)
        width: Image width in pixels (default 512)
        num_steps: Number of sampling steps (default 50)
        sampler_method: Sampling method ("euler", "heun", "dpm_pp")
        guidance_scale: CFG guidance scale (default 7.5)
        guidance_rescale: Guidance rescale factor (default 0.0)
        seed: Random seed for reproducibility
        generator: PyTorch random generator
        batch_size: Batch size (default 1)
        num_images_per_prompt: Images per prompt (default 1)
        output_type: Output format ("pil", "tensor", "numpy")
    """
    height: int = 512
    width: int = 512
    num_steps: int = 50
    sampler_method: str = "heun"
    guidance_scale: float = 7.5
    guidance_rescale: float = 0.0
    use_dynamic_cfg: bool = False
    cfg_schedule: str = "constant"
    cfg_min_scale: float = 1.0
    cfg_max_scale: Optional[float] = None
    seed: Optional[int] = None
    generator: Optional[torch.Generator] = None
    batch_size: int = 1
    num_images_per_prompt: int = 1
    output_type: str = "pil"


class Generator:
    """
    Core generation logic using samplers.

    Manages sampler creation and execution.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: Optional["PixelHDMConfig"] = None,
    ) -> None:
        """
        Initialize generator.

        Args:
            model: PixelHDM model for denoising
            config: Model configuration
        """
        self.model = model
        self.config = config
        self._sampler: Optional[UnifiedSampler] = None
        self._current_method: Optional[str] = None

    def get_sampler(
        self,
        method: str = "heun",
        num_steps: int = 50,
    ) -> UnifiedSampler:
        """
        Get or create sampler.

        Args:
            method: Sampling method
            num_steps: Number of steps

        Returns:
            UnifiedSampler instance
        """
        if self._sampler is None or self._current_method != method:
            t_eps = self._get_t_eps()
            if self.config is not None:
                shift_config = build_timestep_shift_config(self.config)
                self._sampler = create_sampler(
                    method,
                    num_steps,
                    t_eps,
                    use_logit_normal=True,
                    time_p_mean=self.config.time_p_mean,
                    time_p_std=self.config.time_p_std,
                    use_dynamic_timestep_shift=getattr(self.config, "use_dynamic_timestep_shift", True),
                    timestep_shift_config=shift_config,
                )
            else:
                self._sampler = create_sampler(method, num_steps, t_eps)
            self._current_method = method
        return self._sampler

    def _configure_cfg_scheduler(
        self,
        sampler: UnifiedSampler,
        cfg_schedule: Optional[str],
        cfg_min_scale: Optional[float],
        cfg_max_scale: Optional[float],
    ) -> None:
        """Update CFG scheduler settings for dynamic CFG."""
        schedule = cfg_schedule if cfg_schedule is not None else sampler.cfg_scheduler.schedule_type
        min_scale = cfg_min_scale if cfg_min_scale is not None else sampler.cfg_scheduler.min_scale
        max_scale = cfg_max_scale if cfg_max_scale is not None else sampler.cfg_scheduler.max_scale
        sampler.cfg_scheduler = CFGScheduler(schedule, min_scale, max_scale)

    def _get_t_eps(self) -> float:
        """Get time epsilon from config or default."""
        if self.config is not None:
            return self.config.time_eps
        logger.debug("No config available, using default t_eps=0.0001")
        return 0.0001

    def _get_num_tokens(self, height: int, width: int) -> Optional[int]:
        if self.config is None:
            return None
        patch_size = getattr(self.config, "patch_size", None)
        if not patch_size:
            return None
        if height % patch_size != 0 or width % patch_size != 0:
            logger.warning(
                "Resolution (%dx%d) is not divisible by patch_size (%d); "
                "dynamic timestep shift will be skipped.",
                height, width, patch_size,
            )
            return None
        return (height // patch_size) * (width // patch_size)

    def _infer_num_tokens_from_latents(self, z: torch.Tensor) -> Optional[int]:
        if z.dim() != 4:
            return None
        if self.config is None:
            return None
        patch_size = getattr(self.config, "patch_size", None)
        if not patch_size:
            return None
        if z.shape[-1] in (1, 3, 4) and z.shape[1] not in (1, 3, 4):
            height, width = z.shape[1], z.shape[2]  # NHWC
        elif z.shape[1] in (1, 3, 4) and z.shape[-1] not in (1, 3, 4):
            height, width = z.shape[2], z.shape[3]  # NCHW
        else:
            return None
        if height % patch_size != 0 or width % patch_size != 0:
            return None
        return (height // patch_size) * (width // patch_size)

    def generate(
        self,
        inputs: GenerationInputs,
        num_steps: int,
        guidance_scale: float,
        sampler_method: str = "heun",
        return_intermediates: bool = False,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        guidance_rescale: float = 0.0,
        use_dynamic_cfg: bool = False,
        cfg_schedule: str = "constant",
        cfg_min_scale: float = 1.0,
        cfg_max_scale: Optional[float] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Run the generation process.

        Args:
            inputs: Preprocessed generation inputs
            num_steps: Sampling steps
            guidance_scale: CFG scale
            sampler_method: Sampling method
            return_intermediates: Whether to return intermediate steps
            callback: Progress callback function

        Returns:
            Generated images tensor, or list if return_intermediates
        """
        sampler = self.get_sampler(sampler_method, num_steps)
        if use_dynamic_cfg:
            if cfg_max_scale is None:
                cfg_max_scale = guidance_scale
            self._configure_cfg_scheduler(
                sampler, cfg_schedule, cfg_min_scale, cfg_max_scale
            )

        logger.debug(
            f"Sampling with {sampler_method}, {num_steps} steps, "
            f"CFG={guidance_scale}"
        )

        num_tokens = self._get_num_tokens(inputs.height, inputs.width)

        if return_intermediates:
            return self._generate_progressive(
                sampler, inputs, num_steps, guidance_scale,
                guidance_rescale, use_dynamic_cfg, num_tokens
            )
        else:
            return self._generate_single(
                sampler, inputs, num_steps, guidance_scale,
                guidance_rescale, use_dynamic_cfg, callback, num_tokens
            )

    def _generate_single(
        self,
        sampler: UnifiedSampler,
        inputs: GenerationInputs,
        num_steps: int,
        guidance_scale: float,
        guidance_rescale: float,
        use_dynamic_cfg: bool,
        callback: Optional[Callable[[int, int, torch.Tensor], None]],
        num_tokens: Optional[int],
    ) -> torch.Tensor:
        """Generate without intermediates."""
        return sampler.sample(
            model=self.model,
            z_0=inputs.z_0,
            text_embeddings=inputs.text_embed,
            text_mask=inputs.text_mask,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            null_text_embeddings=inputs.null_text_embed,
            null_text_mask=inputs.null_text_mask,
            use_dynamic_cfg=use_dynamic_cfg,
            callback=callback,
            num_tokens=num_tokens,
        )

    def _generate_progressive(
        self,
        sampler: UnifiedSampler,
        inputs: GenerationInputs,
        num_steps: int,
        guidance_scale: float,
        guidance_rescale: float,
        use_dynamic_cfg: bool,
        num_tokens: Optional[int],
    ) -> List[torch.Tensor]:
        """Generate with intermediate steps."""
        return sampler.sample_progressive(
            model=self.model,
            z_0=inputs.z_0,
            text_embeddings=inputs.text_embed,
            text_mask=inputs.text_mask,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            null_text_embeddings=inputs.null_text_embed,
            null_text_mask=inputs.null_text_mask,
            return_every=max(1, num_steps // 10),
            use_dynamic_cfg=use_dynamic_cfg,
            num_tokens=num_tokens,
        )

    def generate_i2i(
        self,
        z_t: torch.Tensor,
        text_embed: torch.Tensor,
        text_mask: torch.Tensor,
        null_text_embed: Optional[torch.Tensor],
        num_steps: int,
        guidance_scale: float,
        t_start: float,
        sampler_method: str = "heun",
        null_text_mask: Optional[torch.Tensor] = None,
        guidance_rescale: float = 0.0,
        use_dynamic_cfg: bool = False,
        cfg_schedule: str = "constant",
        cfg_min_scale: float = 1.0,
        cfg_max_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Run image-to-image generation.

        Args:
            z_t: Noised image latent
            text_embed: Text embeddings
            text_mask: Text mask
            null_text_embed: Null embeddings for CFG
            num_steps: Sampling steps
            guidance_scale: CFG scale
            t_start: Starting time (noise level)
            sampler_method: Sampling method
            null_text_mask: Null text mask for unconditional CFG branch

        Returns:
            Generated images tensor
        """
        sampler = self.get_sampler(sampler_method, num_steps)
        if use_dynamic_cfg:
            if cfg_max_scale is None:
                cfg_max_scale = guidance_scale
            self._configure_cfg_scheduler(
                sampler, cfg_schedule, cfg_min_scale, cfg_max_scale
            )

        num_tokens = self._infer_num_tokens_from_latents(z_t)
        return sampler.sample(
            model=self.model,
            z_0=z_t,
            text_embeddings=text_embed,
            text_mask=text_mask,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            null_text_embeddings=null_text_embed,
            null_text_mask=null_text_mask,
            t_start=t_start,
            use_dynamic_cfg=use_dynamic_cfg,
            num_tokens=num_tokens,
        )

    def count_nfe(
        self,
        num_steps: int = 50,
        use_cfg: bool = True,
        sampler_method: str = "heun",
        full_heun_cfg: bool = False,
    ) -> int:
        """
        Count network function evaluations.

        Args:
            num_steps: Sampling steps
            use_cfg: Whether CFG is used
            sampler_method: Sampling method

        Returns:
            Total NFE count
        """
        sampler = self.get_sampler(sampler_method, num_steps)
        return sampler.count_nfe(
            num_steps=num_steps,
            use_cfg=use_cfg,
            full_heun_cfg=full_heun_cfg,
        )


__all__ = [
    "GenerationConfig",
    "Generator",
]
