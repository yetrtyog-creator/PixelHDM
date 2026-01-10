"""
Pipeline Optimization Methods

Mixin class providing optimization and device management methods.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from typing_extensions import Self

logger = logging.getLogger(__name__)


class PipelineOptimizationMixin:
    """
    Mixin providing optimization and device management methods for pipelines.

    Requires the host class to have:
        - self.device: torch.device
        - self.model: nn.Module
        - self._text_encoder: Optional[nn.Module]
        - self._preprocessor: Preprocessor with device attribute
        - self.dtype: torch.dtype
    """

    def to(self, device: torch.device) -> "Self":
        """Move pipeline to device."""
        self.device = device
        self.model = self.model.to(device)
        if self._text_encoder is not None:
            self._text_encoder = self._text_encoder.to(device)
        self._preprocessor.device = device
        self._preprocessor.clear_cache()
        return self

    def set_text_encoder_precision(
        self,
        dtype: torch.dtype = torch.float16,
    ) -> "Self":
        """Set text encoder precision."""
        if self._text_encoder is not None:
            self._text_encoder = self._text_encoder.to(dtype)
            logger.info(f"Text encoder precision set to {dtype}")
        elif hasattr(self.model, "text_encoder") and self.model.text_encoder:
            self.model.text_encoder = self.model.text_encoder.to(dtype)
            logger.info(f"Text encoder precision set to {dtype}")
        else:
            logger.warning("No text encoder found to set precision")
        return self

    def enable_model_cpu_offload(self) -> "Self":
        """Enable CPU offload (placeholder)."""
        logger.warning("CPU offload not yet implemented")
        return self

    def enable_attention_slicing(
        self,
        slice_size: Optional[int] = None,
    ) -> "Self":
        """Enable attention slicing (placeholder)."""
        logger.warning("Attention slicing not yet implemented")
        return self

    def enable_torch_compile(
        self,
        mode: str = "reduce-overhead",
        fullgraph: bool = False,
        dynamic: bool = True,
    ) -> "Self":
        """Enable torch.compile optimization."""
        if not hasattr(torch, "compile"):
            logger.warning("torch.compile requires PyTorch 2.0+")
            return self

        try:
            logger.info(f"Enabling torch.compile with mode={mode}")
            self.model = torch.compile(
                self.model, mode=mode, fullgraph=fullgraph, dynamic=dynamic
            )
            self._compiled = True
            logger.info("torch.compile enabled successfully")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
            self._compiled = False

        return self

    def use_mock_text_encoder(
        self,
        hidden_size: int = 1024,
        max_length: int = 512,
    ) -> "Self":
        """Use mock text encoder for testing."""
        from .mock_encoder import MockTextEncoder

        self._text_encoder = MockTextEncoder(
            hidden_size=hidden_size,
            max_length=max_length,
            device=self.device,
            dtype=self.dtype,
        )
        self._preprocessor.text_encoder = self._text_encoder
        self._preprocessor.clear_cache()
        # Clear backward-compatible cache attributes
        self._null_text_embed = None
        self._null_text_mask = None
        logger.info(f"Using MockTextEncoder (hidden_size={hidden_size})")
        return self


__all__ = ["PipelineOptimizationMixin"]
