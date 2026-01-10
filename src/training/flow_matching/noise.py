"""
Noise Handling for PixelHDM Flow Matching (V-Prediction)

V-Prediction 不需要噪聲縮放，使用標準單位方差噪聲。
此模組僅保留 interpolate 函數。

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

import torch


def interpolate(
    x: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    PixelHDM interpolation: z_t = t * x + (1 - t) * noise

    Args:
        x: Clean image, shape [B, C, H, W] or [B, H, W, C]
        noise: Noise (unit variance), same shape as x
        t: Timesteps, shape [B]

    Returns:
        z_t: Interpolated (noisy) image
    """
    t_expanded = t.view(-1, *([1] * (x.dim() - 1)))
    return t_expanded * x + (1 - t_expanded) * noise


__all__ = [
    "interpolate",
]
