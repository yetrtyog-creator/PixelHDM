"""
PixelHDM-RPEA-DinoV3 外部編碼器

包含:
    - DINOv3Encoder: DINOv3 視覺編碼器 (REPA Loss 用)
    - Qwen3TextEncoder: Qwen3 文本編碼器
"""

from .dinov3 import (
    DINOv3Encoder,
    DINOv3FeatureProjector,
    DINOFeatureProjector,
    create_dinov3_encoder,
    create_dinov3_encoder_from_config,
    create_feature_projector,
    create_feature_projector_from_config,
)

from .text_encoder import (
    Qwen3TextEncoder,
    TextProjector,
    CaptionEmbedder,
    NullTextEncoder,
    create_text_encoder,
    create_text_encoder_from_config,
    create_caption_embedder,
    create_null_text_encoder,
)


__all__ = [
    # DINOv3
    "DINOv3Encoder",
    "DINOv3FeatureProjector",
    "DINOFeatureProjector",
    "create_dinov3_encoder",
    "create_dinov3_encoder_from_config",
    "create_feature_projector",
    "create_feature_projector_from_config",
    # Text Encoder
    "Qwen3TextEncoder",
    "TextProjector",
    "CaptionEmbedder",
    "NullTextEncoder",
    "create_text_encoder",
    "create_text_encoder_from_config",
    "create_caption_embedder",
    "create_null_text_encoder",
]
