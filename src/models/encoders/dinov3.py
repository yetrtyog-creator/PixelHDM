"""
DINOv3 Feature Extractor for REPA Loss

This module re-exports from the dinov3 package for backward compatibility.

For new code, import directly from:
    from src.models.encoders.dinov3 import DINOv3Encoder

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

# Re-export everything from the package
from .dinov3 import (
    DINOv3Encoder,
    DINOv3FeatureProjector,
    DINOFeatureProjector,
    create_dinov3_encoder,
    create_dinov3_encoder_from_config,
    create_feature_projector,
    create_feature_projector_from_config,
)

__all__ = [
    "DINOv3Encoder",
    "DINOv3FeatureProjector",
    "DINOFeatureProjector",
    "create_dinov3_encoder",
    "create_dinov3_encoder_from_config",
    "create_feature_projector",
    "create_feature_projector_from_config",
]
