"""
Text Encoder for PixelHDM-RPEA-DinoV3

向後兼容模組 - 重新導出 text/ 子模組的所有公開 API。

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
Updated: 2026-01-02 (重構為向後兼容導入)
"""

# 向後兼容: 從重構後的模組重新導出
from .text import (
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
    "Qwen3TextEncoder",
    "TextProjector",
    "CaptionEmbedder",
    "NullTextEncoder",
    "create_text_encoder",
    "create_text_encoder_from_config",
    "create_caption_embedder",
    "create_null_text_encoder",
]
