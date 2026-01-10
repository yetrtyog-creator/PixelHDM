"""
Frequency-aware Loss (Legacy Compatibility)

This module re-exports from the new modular structure.
For new code, import directly from training.losses.freq.
"""

from .freq import (
    FreqLossConfig,
    FrequencyLoss,
    BlockDCT2D,
    rgb_to_ycbcr,
    create_freq_loss,
    create_freq_loss_from_config,
)
from .freq.config import (
    JPEG_LUMINANCE_QUANTIZATION_TABLE,
    JPEG_CHROMINANCE_QUANTIZATION_TABLE,
)
from .freq.dct import create_dct_matrix

__all__ = [
    "FreqLossConfig",
    "FrequencyLoss",
    "BlockDCT2D",
    "rgb_to_ycbcr",
    "create_freq_loss",
    "create_freq_loss_from_config",
    "JPEG_LUMINANCE_QUANTIZATION_TABLE",
    "JPEG_CHROMINANCE_QUANTIZATION_TABLE",
    "create_dct_matrix",
]
