"""
Frequency-aware Loss Module

Based on DeCo paper: Frequency-Decoupled Pixel Diffusion.
Uses 8×8 DCT + JPEG quantization weights.
"""

from .config import FreqLossConfig
from .dct import BlockDCT2D, create_dct_matrix
from .color import rgb_to_ycbcr
from .core import FrequencyLoss
from .factory import create_freq_loss, create_freq_loss_from_config

__all__ = [
    "FreqLossConfig",
    "FrequencyLoss",
    "BlockDCT2D",
    "create_dct_matrix",
    "rgb_to_ycbcr",
    "create_freq_loss",
    "create_freq_loss_from_config",
]
