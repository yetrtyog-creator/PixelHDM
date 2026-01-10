"""
Pipeline Output Postprocessing

Handles conversion of generated tensors to various output formats.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Union, Dict, Any, Optional

import torch
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PipelineOutput:
    """
    Pipeline generation output.

    Attributes:
        images: Generated images in requested format
        latents: Initial noise latents
        intermediates: Intermediate sampling results (if requested)
        metadata: Generation metadata (prompts, settings, etc.)
    """
    images: Union[List[Image.Image], torch.Tensor, np.ndarray]
    latents: Optional[torch.Tensor] = None
    intermediates: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Postprocessor:
    """
    Handles output postprocessing for generated images.

    Converts raw tensor output to various formats (PIL, numpy, tensor).
    """

    def __init__(self) -> None:
        """Initialize postprocessor."""
        pass

    def process(
        self,
        images: torch.Tensor,
        output_type: str = "pil",
    ) -> Union[List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Postprocess generated images.

        Args:
            images: (B, H, W, 3) tensor in range [-1, 1]
            output_type: Output format ("pil", "tensor", "numpy")

        Returns:
            Images in requested format
        """
        # Ensure float for processing
        images = images.float()

        # Clamp to valid range
        images = images.clamp(-1, 1)

        if output_type == "tensor":
            return self._to_tensor(images)
        elif output_type == "numpy":
            return self._to_numpy(images)
        else:
            return self._to_pil(images)

    def _to_tensor(self, images: torch.Tensor) -> torch.Tensor:
        """Return clamped tensor."""
        return images

    def _to_numpy(self, images: torch.Tensor) -> np.ndarray:
        """Convert to numpy array in [0, 1] range."""
        images = (images + 1) / 2  # Scale to [0, 1]
        return images.cpu().numpy()

    def _to_pil(self, images: torch.Tensor) -> List[Image.Image]:
        """Convert to list of PIL images."""
        # Scale to [0, 255]
        images = (images + 1) / 2
        images = images.cpu().numpy()
        images = (images * 255).astype(np.uint8)

        pil_images = []
        for img in images:
            pil_images.append(Image.fromarray(img))

        return pil_images

    def create_output(
        self,
        images: Union[List[Image.Image], torch.Tensor, np.ndarray],
        latents: Optional[torch.Tensor] = None,
        intermediates: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PipelineOutput:
        """
        Create PipelineOutput from processed images.

        Args:
            images: Processed images
            latents: Initial noise latents
            intermediates: Intermediate results
            metadata: Generation metadata

        Returns:
            PipelineOutput instance
        """
        return PipelineOutput(
            images=images,
            latents=latents,
            intermediates=intermediates,
            metadata=metadata or {},
        )


__all__ = [
    "PipelineOutput",
    "Postprocessor",
]
