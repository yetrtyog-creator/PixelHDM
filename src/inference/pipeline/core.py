"""
PixelHDM Pipeline Core

Main pipeline class for text-to-image generation.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Callable, List, Union

import torch

from .validation import InputValidator
from .preprocessing import Preprocessor
from .postprocessing import Postprocessor, PipelineOutput
from .generation import Generator, GenerationConfig
from .optimization import PipelineOptimizationMixin
from .api_mixin import PipelineAPIMixin

if TYPE_CHECKING:
    from ...config.model_config import PixelHDMConfig
    from ...models.pixelhdm import PixelHDM, PixelHDMForT2I
    from ...models.encoders.text_encoder import Qwen3TextEncoder

logger = logging.getLogger(__name__)


class PixelHDMPipeline(PipelineOptimizationMixin, PipelineAPIMixin):
    """
    PixelHDM Text-to-Image Pipeline.

    Provides complete text-to-image generation workflow.

    Args:
        model: PixelHDM or PixelHDMForT2I model
        text_encoder: Text encoder (optional if using PixelHDMForT2I)
        device: Computation device
        dtype: Data type

    Example:
        >>> from src.models.pixelhdm import create_pixelhdm_for_t2i
        >>> model = create_pixelhdm_for_t2i()
        >>> pipeline = PixelHDMPipeline(model)
        >>> output = pipeline("a cat sitting on a chair")
        >>> output.images[0].save("cat.png")
    """

    def __init__(
        self,
        model: Union["PixelHDM", "PixelHDMForT2I"],
        text_encoder: Optional["Qwen3TextEncoder"] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model = model
        self._text_encoder = text_encoder
        self.dtype = dtype

        # Device setup
        if device is None:
            device = next(model.parameters()).device
        self.device = device

        # Get config from model
        self.config = getattr(model, "config", None)

        # Initialize components
        self._validator = InputValidator(config=self.config)
        self._preprocessor = Preprocessor(
            text_encoder=self.text_encoder,
            validator=self._validator,
            device=self.device,
            dtype=self.dtype,
        )
        self._postprocessor = Postprocessor()
        self._generator = Generator(model=self.model, config=self.config)

        # Compilation state
        self._compiled = False

        # Backward compatibility cache
        self._null_text_embed = None
        self._null_text_mask = None

    @property
    def text_encoder(self) -> Optional["Qwen3TextEncoder"]:
        """Get text encoder from pipeline or model."""
        if self._text_encoder is not None:
            return self._text_encoder
        if hasattr(self.model, "text_encoder"):
            return self.model.text_encoder
        return None

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 512,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
        sampler_method: str = "heun",
        output_type: str = "pil",
        return_intermediates: bool = False,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        **kwargs,
    ) -> PipelineOutput:
        """Generate images from text prompts."""
        # Ensure model is in eval mode for inference
        self.model.eval()

        generator = self._create_generator(seed)

        logger.debug("Encoding prompts...")
        inputs = self._preprocessor.prepare_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
        )

        logger.debug(f"Preparing latents: {inputs.batch_size}x{height}x{width}")
        result = self._generator.generate(
            inputs=inputs,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            sampler_method=sampler_method,
            return_intermediates=return_intermediates,
            callback=callback,
        )

        logger.debug("Postprocessing images...")
        return self._create_output(
            result, inputs, prompt, negative_prompt, height, width,
            num_steps, guidance_scale, seed, sampler_method, output_type,
            return_intermediates
        )

    def _create_generator(self, seed: Optional[int]) -> Optional[torch.Generator]:
        """Create random generator from seed."""
        if seed is None:
            return None
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        return generator

    def _create_output(
        self,
        result: Union[torch.Tensor, List[torch.Tensor]],
        inputs,
        prompt, negative_prompt, height, width,
        num_steps, guidance_scale, seed, sampler_method, output_type,
        return_intermediates,
    ) -> PipelineOutput:
        """Create pipeline output from generation result."""
        if return_intermediates:
            images = result[-1]
            intermediates = result
        else:
            images = result
            intermediates = None

        processed = self._postprocessor.process(images, output_type)
        prompts = [prompt] if isinstance(prompt, str) else prompt

        return self._postprocessor.create_output(
            images=processed,
            latents=inputs.z_0,
            intermediates=intermediates,
            metadata={
                "prompt": prompts,
                "negative_prompt": negative_prompt,
                "height": height,
                "width": width,
                "num_steps": num_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "sampler_method": sampler_method,
            },
        )


__all__ = ["PixelHDMPipeline"]
