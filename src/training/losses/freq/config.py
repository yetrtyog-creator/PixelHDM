"""
Frequency Loss Configuration and Constants
"""

from dataclasses import dataclass

import torch


@dataclass
class FreqLossConfig:
    """Frequency-aware loss configuration."""
    enabled: bool = True
    quality: int = 90  # JPEG quality factor [1, 100]
    weight: float = 1.0
    use_ycbcr: bool = True
    block_size: int = 8
    only_y_channel: bool = False


# JPEG Standard Luminance Quantization Table
JPEG_LUMINANCE_QUANTIZATION_TABLE = torch.tensor([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=torch.float32)


# JPEG Standard Chrominance Quantization Table
JPEG_CHROMINANCE_QUANTIZATION_TABLE = torch.tensor([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=torch.float32)


__all__ = [
    "FreqLossConfig",
    "JPEG_LUMINANCE_QUANTIZATION_TABLE",
    "JPEG_CHROMINANCE_QUANTIZATION_TABLE",
]
