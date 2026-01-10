"""
PixelHDM-RPEA-DinoV3: Pixel-Level Diffusion Transformer for T2I

Architecture:
    - PixelHDM: Dual-pathway (Patch-Level + Pixel-Level) DiT
    - RPEA: 2D Rotary Position Embedding Attention
    - DINOv3: REPA Loss for training acceleration (NO DINOv2 fallback)
    - Qwen3-0.6B: Text encoder (Lumina2-style concat)

Configuration:
    - hidden_dim: 1024
    - patch_size: 16 (matches DINOv3)
    - patch_layers (N): 16
    - pixel_layers (M): 4
    - trainable params: ~362M
    - frozen params: ~686M (Qwen3 + DINOv3)

Modules:
    - config: Model and training configuration
    - models: PixelHDM model, encoders, layers, blocks
    - training: Losses, optimization, flow matching, trainer
    - inference: Sampler, pipeline, CFG

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

__version__ = "0.1.0"

# 延遲導入以避免循環依賴
from . import config
from . import models
from . import training
from . import inference

__all__ = [
    "__version__",
    "config",
    "models",
    "training",
    "inference",
]
