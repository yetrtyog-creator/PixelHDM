"""
Pipeline Factory Functions

Convenience functions for creating pipelines.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import torch

from .core import PixelHDMPipeline

if TYPE_CHECKING:
    from ...config.model_config import PixelHDMConfig
    from ...models.pixelhdm import PixelHDM, PixelHDMForT2I
    from ...models.encoders.text_encoder import Qwen3TextEncoder


def create_pipeline(
    model: Union["PixelHDM", "PixelHDMForT2I"],
    text_encoder: Optional["Qwen3TextEncoder"] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> PixelHDMPipeline:
    """
    Create inference pipeline.

    Args:
        model: PixelHDM model
        text_encoder: Optional text encoder
        device: Target device
        dtype: Data type

    Returns:
        PixelHDMPipeline instance
    """
    return PixelHDMPipeline(
        model=model,
        text_encoder=text_encoder,
        device=device,
        dtype=dtype,
    )


def create_pipeline_from_pretrained(
    model_path: str,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
    text_encoder: Optional["Qwen3TextEncoder"] = None,
    load_text_encoder: bool = True,
) -> PixelHDMPipeline:
    """
    Create pipeline from pretrained model.

    Args:
        model_path: Path to saved model
        device: Target device
        dtype: Data type
        text_encoder: Optional text encoder (if provided, will be used instead of lazy-loading)
        load_text_encoder: Whether to load text encoder (only used if text_encoder is None)

    Returns:
        PixelHDMPipeline instance
    """
    from ...models.pixelhdm import create_pixelhdm_from_pretrained

    model = create_pixelhdm_from_pretrained(model_path)

    if device is not None:
        model = model.to(device)

    # Use provided text_encoder if available
    return PixelHDMPipeline(
        model=model,
        text_encoder=text_encoder,
        device=device,
        dtype=dtype,
    )


def create_pipeline_from_config(
    config: "PixelHDMConfig",
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
    load_text_encoder: bool = True,
    load_dino_encoder: bool = False,
    text_encoder: Optional["Qwen3TextEncoder"] = None,
) -> PixelHDMPipeline:
    """
    Create pipeline from configuration.

    Args:
        config: PixelHDMConfig instance
        device: Target device
        dtype: Data type
        load_text_encoder: Load text encoder (ignored if text_encoder is provided)
        load_dino_encoder: Load DINO encoder
        text_encoder: Optional text encoder (if provided, will be used instead of lazy-loading)

    Returns:
        PixelHDMPipeline instance

    Example:
        >>> from src.config import PixelHDMConfig
        >>> config = PixelHDMConfig.default()
        >>> pipeline = create_pipeline_from_config(config)
        >>> output = pipeline("a beautiful sunset")
    """
    from ...models.pixelhdm import create_pixelhdm_for_t2i_from_config

    # If text_encoder is provided, don't load from model
    model = create_pixelhdm_for_t2i_from_config(
        config,
        load_text_encoder=load_text_encoder and text_encoder is None,
        load_dino_encoder=load_dino_encoder,
    )

    if device is not None:
        model = model.to(device)
    if dtype is not None:
        model = model.to(dtype)

    # Use provided text_encoder if available
    return PixelHDMPipeline(
        model=model,
        text_encoder=text_encoder,
        device=device,
        dtype=dtype,
    )


__all__ = [
    "create_pipeline",
    "create_pipeline_from_pretrained",
    "create_pipeline_from_config",
]
