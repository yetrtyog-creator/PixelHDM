"""
PixelHDM Image-to-Image Pipeline

Pipeline for image-to-image generation.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, List, Union

import torch
from PIL import Image
import numpy as np

from .core import PixelHDMPipeline
from .postprocessing import PipelineOutput

if TYPE_CHECKING:
    from ...models.pixelhdm import PixelHDM, PixelHDMForT2I
    from ...models.encoders.text_encoder import Qwen3TextEncoder

logger = logging.getLogger(__name__)


class PixelHDMPipelineForImg2Img(PixelHDMPipeline):
    """
    PixelHDM Image-to-Image Pipeline.

    Generates images based on a reference image and text prompt.
    """

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: Union[Image.Image, torch.Tensor, List[Image.Image]],
        strength: float = 0.8,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        sampler_method: str = "heun",
        output_type: str = "pil",
        **kwargs,
    ) -> PipelineOutput:
        """
        Generate from reference image.

        Args:
            prompt: Text prompt(s)
            image: Reference image(s)
            strength: Variation strength (0-1), 1 = ignore reference
            negative_prompt: Negative prompt(s)
            num_steps: Sampling steps
            guidance_scale: CFG scale
            seed: Random seed
            sampler_method: Sampling method
            output_type: Output format
            **kwargs: Additional arguments

        Returns:
            PipelineOutput with generated images
        """
        # Validate strength range
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"strength must be in [0.0, 1.0], got {strength}")

        # Boundary warnings
        if strength < 0.05:
            logger.warning(f"strength={strength} < 0.05, image will barely change")
        if strength > 0.95:
            logger.warning(f"strength={strength} > 0.95, close to full regeneration")

        # Process input image
        image_tensor = self._prepare_image(image)
        prompts = self._validator.validate_prompt(prompt)
        batch_size = len(prompts)

        # Expand image batch if needed
        if image_tensor.shape[0] == 1 and batch_size > 1:
            image_tensor = image_tensor.expand(batch_size, -1, -1, -1)

        # Setup generator
        generator = self._create_generator(seed)

        # Encode prompts
        text_embed, text_mask, pooled_embed, null_embed, null_mask, null_pooled = self._preprocessor.encode_prompt(
            prompts, negative_prompt, num_images_per_prompt=1
        )

        # Add noise to image
        z_t, t_start, actual_steps = self._add_noise_to_image(
            image_tensor, strength, num_steps, batch_size, generator
        )

        # Generate
        images = self._generator.generate_i2i(
            z_t=z_t,
            text_embed=text_embed,
            text_mask=text_mask,
            null_text_embed=null_embed,
            num_steps=actual_steps,
            guidance_scale=guidance_scale,
            t_start=t_start,
            sampler_method=sampler_method,
            pooled_text_embed=pooled_embed,
            null_pooled_text_embed=null_pooled,
        )

        # Postprocess and return
        return self._build_output(
            images, z_t, prompts, strength, actual_steps, guidance_scale, seed, output_type
        )

    def _build_output(
        self, images, latents, prompts, strength, num_steps, guidance_scale, seed, output_type
    ) -> PipelineOutput:
        """Build pipeline output with metadata."""
        processed = self._postprocessor.process(images, output_type)
        return self._postprocessor.create_output(
            images=processed,
            latents=latents,
            metadata={
                "prompt": prompts,
                "strength": strength,
                "num_steps": num_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
            },
        )

    def _prepare_image(
        self,
        image: Union[Image.Image, torch.Tensor, List[Image.Image]],
    ) -> torch.Tensor:
        """Prepare image tensor from various inputs."""
        if isinstance(image, torch.Tensor):
            return image
        if isinstance(image, list):
            tensors = [self._preprocess_image(img) for img in image]
            return torch.cat(tensors, dim=0)
        return self._preprocess_image(image)

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess PIL image."""
        # Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Get patch size
        patch_size = 16
        if self.config is not None:
            patch_size = self.config.patch_size

        # Align to patch size
        w, h = image.size
        new_w = (w // patch_size) * patch_size
        new_h = (h // patch_size) * patch_size

        if new_w != w or new_h != h:
            image = image.resize((new_w, new_h), Image.LANCZOS)

        # Convert to tensor
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = img_array * 2.0 - 1.0  # Scale to [-1, 1]
        tensor = torch.from_numpy(img_array)

        return tensor.unsqueeze(0)  # (1, H, W, 3)

    def _add_noise_to_image(
        self,
        image: torch.Tensor,
        strength: float,
        num_steps: int,
        batch_size: int,
        generator: Optional[torch.Generator],
    ) -> tuple:
        """Add noise to image based on strength."""
        # Get t_eps from config or use default
        t_eps = 0.05
        if self.config is not None:
            t_eps = self.config.time_eps

        # PixelHDM convention: t=1 clean, t=0 noise
        t_start_raw = 1.0 - strength

        # Clamp to valid training range [t_eps, 1-t_eps]
        t_start = max(t_eps, min(1.0 - t_eps, t_start_raw))

        if t_start != t_start_raw:
            logger.warning(f"t_start clamped from {t_start_raw:.4f} to {t_start:.4f}")

        image = image.to(device=self.device, dtype=self.dtype)
        noise = torch.randn(
            image.shape,
            generator=generator,
            device=image.device,
            dtype=image.dtype,
        )

        t_tensor = torch.full(
            (batch_size,), t_start, device=self.device, dtype=self.dtype
        )
        t_expanded = t_tensor.view(-1, 1, 1, 1)
        z_t = t_expanded * image + (1 - t_expanded) * noise

        # Ensure at least 1 step
        actual_steps = max(1, int(num_steps * strength)) if strength > 0 else 0

        return z_t, t_start, actual_steps


__all__ = [
    "PixelHDMPipelineForImg2Img",
]
