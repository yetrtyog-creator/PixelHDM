"""
Pipeline Public API Methods

Mixin providing public API methods for the pipeline.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, List, Union, Dict, Any

import torch

if TYPE_CHECKING:
    pass


class PipelineAPIMixin:
    """
    Mixin providing public API methods for pipelines.

    Requires the host class to have:
        - self._generator: Generator
        - self._validator: InputValidator
        - self._preprocessor: Preprocessor
        - self._postprocessor: Postprocessor
    """

    def get_sampler(self, method: str = "heun", num_steps: int = 50):
        """Get sampler instance."""
        return self._generator.get_sampler(method, num_steps)

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
    ):
        """Encode text prompts."""
        prompts = self._validator.validate_prompt(prompt)
        return self._preprocessor.encode_prompt(
            prompts, negative_prompt, num_images_per_prompt
        )

    def prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Prepare initial noise latents."""
        self._validator.validate_resolution(height, width)
        return self._preprocessor.prepare_latents(batch_size, height, width, generator)

    def postprocess(
        self,
        images: torch.Tensor,
        output_type: str = "pil",
    ):
        """Postprocess generated images."""
        return self._postprocessor.process(images, output_type)

    def _get_null_text_embed(self, batch_size: int):
        """Get null text embeddings."""
        return self._preprocessor._get_null_text_embed(batch_size)

    def get_nfe_stats(
        self,
        num_steps: int = 50,
        use_cfg: bool = True,
        sampler_method: str = "heun",
    ) -> Dict[str, Any]:
        """Get NFE statistics."""
        nfe = self._generator.count_nfe(num_steps, use_cfg, sampler_method)
        return {
            "nfe": nfe,
            "nfe_per_step": nfe / num_steps,
            "sampler_method": sampler_method,
            "use_cfg": use_cfg,
            "num_steps": num_steps,
        }


__all__ = ["PipelineAPIMixin"]
