"""
PixelHDM-RPEA-DinoV3 模型模組

包含:
    - PixelHDM: 主模型類
    - attention: 注意力機制 (GQA + Gating + Token Compaction)
    - layers: 核心層 (RMSNorm, SwiGLU, RoPE, AdaLN, Embedding)
    - blocks: Transformer 塊 (Patch-Level, Pixel-Level)
    - encoders: 外部編碼器 (DINOv3, Qwen3)
"""

from . import attention
from . import layers
from . import blocks
from . import encoders

from .pixelhdm import (
    PixelHDM,
    PixelHDMForT2I,
    create_pixelhdm,
    create_pixelhdm_for_t2i,
    create_pixelhdm_from_pretrained,
    create_pixelhdm_from_config,
    create_pixelhdm_for_t2i_from_config,
)


__all__ = [
    # 子模組
    "attention",
    "layers",
    "blocks",
    "encoders",
    # 主模型
    "PixelHDM",
    "PixelHDMForT2I",
    "create_pixelhdm",
    "create_pixelhdm_for_t2i",
    "create_pixelhdm_from_pretrained",
    "create_pixelhdm_from_config",
    "create_pixelhdm_for_t2i_from_config",
]
