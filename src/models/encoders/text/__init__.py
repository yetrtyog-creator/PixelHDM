"""
Text Encoder Module

文本編碼器模組，包含 Qwen3 編碼器及相關組件。

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from .encoder import (
    Qwen3TextEncoder,
    create_text_encoder,
    create_text_encoder_from_config,
    SUPPORTED_MODELS,
)

from .auxiliary import (
    TextProjector,
    CaptionEmbedder,
    NullTextEncoder,
    create_caption_embedder,
    create_null_text_encoder,
)

from .tokenizer import TokenizerWrapper
from .pooling import LastTokenPooling, MeanPooling, get_pooler


__all__ = [
    # 主要類別
    "Qwen3TextEncoder",
    "TextProjector",
    "CaptionEmbedder",
    "NullTextEncoder",
    # 工廠函數
    "create_text_encoder",
    "create_text_encoder_from_config",
    "create_caption_embedder",
    "create_null_text_encoder",
    # 輔助類別
    "TokenizerWrapper",
    "LastTokenPooling",
    "MeanPooling",
    "get_pooler",
    # 常量
    "SUPPORTED_MODELS",
]
