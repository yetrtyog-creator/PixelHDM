"""
DINOv3 Feature Extractor for REPA Loss

DINOv3 (Meta 2025):
    - 7B parameter Teacher model, distillation-trained student models
    - Uses RoPE (Rotary Position Embedding)
    - patch_size=16, perfectly matches this project
    - Supports variable resolution (256x256 to 4096x4096)

Usage:
    - REPA Loss target feature extraction
    - Frozen model, inference only

Loading methods:
    - Local .pth file (recommended, avoids HuggingFace permission issues)
    - HuggingFace transformers (requires gated permission)
    - torch.hub (requires network)

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from typing import TYPE_CHECKING, Optional

from .encoder import DINOv3Encoder
from .projector import DINOv3FeatureProjector, DINOFeatureProjector

if TYPE_CHECKING:
    from ....config.model_config import PixelHDMConfig


# Factory functions
def create_dinov3_encoder(
    model_name: str = "dinov3-vitb16",
    pretrained: bool = True,
    local_path: Optional[str] = None,
    use_bf16: bool = True,
) -> DINOv3Encoder:
    """
    Create DINOv3 encoder

    Args:
        model_name: Model name (dinov3-vitb16, dinov3-vitl16, etc.)
        pretrained: Load pretrained weights
        local_path: Local .pth weight file path (recommended)
        use_bf16: Use bf16 for inference
    """
    return DINOv3Encoder(
        config=None,
        model_name=model_name,
        pretrained=pretrained,
        local_path=local_path,
        use_bf16=use_bf16,
    )


def create_dinov3_encoder_from_config(
    config: "PixelHDMConfig",
) -> DINOv3Encoder:
    """Create DINOv3 encoder from config"""
    return DINOv3Encoder(config=config)


def create_feature_projector(
    input_dim: int = 1024,
    output_dim: int = 768,
) -> DINOv3FeatureProjector:
    """Create feature projector"""
    return DINOv3FeatureProjector(
        config=None,
        input_dim=input_dim,
        output_dim=output_dim,
    )


def create_feature_projector_from_config(
    config: "PixelHDMConfig",
) -> DINOv3FeatureProjector:
    """Create feature projector from config"""
    return DINOv3FeatureProjector(config=config)


__all__ = [
    # Main classes
    "DINOv3Encoder",
    "DINOv3FeatureProjector",
    "DINOFeatureProjector",
    # Factory functions
    "create_dinov3_encoder",
    "create_dinov3_encoder_from_config",
    "create_feature_projector",
    "create_feature_projector_from_config",
]
