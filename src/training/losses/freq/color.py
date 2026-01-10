"""
Color Space Conversion Utilities
"""

import torch


def rgb_to_ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    """
    RGB to YCbCr conversion (ITU-R BT.601).

    Args:
        rgb: (B, 3, H, W) RGB tensor

    Returns:
        ycbcr: (B, 3, H, W) YCbCr tensor
    """
    r = rgb[:, 0:1, :, :]
    g = rgb[:, 1:2, :, :]
    b = rgb[:, 2:3, :, :]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b

    return torch.cat([y, cb, cr], dim=1)


__all__ = ["rgb_to_ycbcr"]
