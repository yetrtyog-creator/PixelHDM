"""
PixelHDM-RPEA-DinoV3 Main Model Package

This package provides the core PixelHDM dual-path Diffusion Transformer.

Public API:
    - PixelHDM: Main model class
    - PixelHDMForT2I: Text-to-Image wrapper with encoders
    - create_pixelhdm: Factory function for PixelHDM
    - create_pixelhdm_for_t2i: Factory function for PixelHDMForT2I
    - create_pixelhdm_from_pretrained: Load from pretrained weights
    - create_pixelhdm_from_config: Create from config
    - create_pixelhdm_for_t2i_from_config: Create T2I from config

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from .core import PixelHDM
from .t2i import (
    PixelHDMForT2I,
    create_pixelhdm,
    create_pixelhdm_for_t2i,
    create_pixelhdm_from_pretrained,
    create_pixelhdm_from_config,
    create_pixelhdm_for_t2i_from_config,
)

__all__ = [
    "PixelHDM",
    "PixelHDMForT2I",
    "create_pixelhdm",
    "create_pixelhdm_for_t2i",
    "create_pixelhdm_from_pretrained",
    "create_pixelhdm_from_config",
    "create_pixelhdm_for_t2i_from_config",
]
