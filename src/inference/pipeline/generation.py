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

from ..sampler import UnifiedSampler, create_sampler
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
            self._sampler = create_sampler(method, num_steps, t_eps)
            self._current_method = method
        return self._sampler

    def _get_t_eps(self) -> float:
        """Get time epsilon from config or default."""
        if self.config is not None:
            return self.config.time_eps
        logger.debug("No config available, using default t_eps=0.05")
        return 0.05

    def generate(
        self,
        inputs: GenerationInputs,
        num_steps: int,
        guidance_scale: float,
        sampler_method: str = "heun",
        return_intermediates: bool = False,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
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

        logger.debug(
            f"Sampling with {sampler_method}, {num_steps} steps, "
            f"CFG={guidance_scale}"
        )

        if return_intermediates:
            return self._generate_progressive(sampler, inputs, num_steps, guidance_scale)
        else:
            return self._generate_single(
                sampler, inputs, num_steps, guidance_scale, callback
            )

    def _generate_single(
        self,
        sampler: UnifiedSampler,
        inputs: GenerationInputs,
        num_steps: int,
        guidance_scale: float,
        callback: Optional[Callable[[int, int, torch.Tensor], None]],
    ) -> torch.Tensor:
        """Generate without intermediates."""
        return sampler.sample(
            model=self.model,
            z_0=inputs.z_0,
            text_embeddings=inputs.text_embed,
            text_mask=inputs.text_mask,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            null_text_embeddings=inputs.null_text_embed,
            null_text_mask=inputs.null_text_mask,
            callback=callback,
            pooled_text_embed=inputs.pooled_text_embed,
            null_pooled_text_embed=inputs.null_pooled_text_embed,
        )

    def _generate_progressive(
        self,
        sampler: UnifiedSampler,
        inputs: GenerationInputs,
        num_steps: int,
        guidance_scale: float,
    ) -> List[torch.Tensor]:
        """Generate with intermediate steps."""
        return sampler.sample_progressive(
            model=self.model,
            z_0=inputs.z_0,
            text_embeddings=inputs.text_embed,
            text_mask=inputs.text_mask,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            null_text_embeddings=inputs.null_text_embed,
            null_text_mask=inputs.null_text_mask,
            return_every=max(1, num_steps // 10),
            pooled_text_embed=inputs.pooled_text_embed,
            null_pooled_text_embed=inputs.null_pooled_text_embed,
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
        pooled_text_embed: Optional[torch.Tensor] = None,
        null_pooled_text_embed: Optional[torch.Tensor] = None,
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
            pooled_text_embed: Pooled text embeddings for AdaLN
            null_pooled_text_embed: Null pooled embeddings for CFG

        Returns:
            Generated images tensor
        """
        sampler = self.get_sampler(sampler_method, num_steps)

        return sampler.sample(
            model=self.model,
            z_0=z_t,
            text_embeddings=text_embed,
            text_mask=text_mask,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            null_text_embeddings=null_text_embed,
            t_start=t_start,
            pooled_text_embed=pooled_text_embed,
            null_pooled_text_embed=null_pooled_text_embed,
        )

    def count_nfe(
        self,
        num_steps: int = 50,
        use_cfg: bool = True,
        sampler_method: str = "heun",
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
        return sampler.count_nfe(num_steps=num_steps, use_cfg=use_cfg)


__all__ = [
    "GenerationConfig",
    "Generator",
]
