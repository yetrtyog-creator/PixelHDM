"""
PixelHDM-RPEA-DinoV3 Bucket Configuration

分桶系統配置類。

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BucketConfig:
    """
    分桶配置

    Attributes:
        min_resolution: 最小解析度 (最小邊長)
        max_resolution: 最大解析度 (最大邊長)
        patch_size: Patch 大小 (16, 與 DINOv3 匹配)
        max_aspect_ratio: 最大長寬比 (2:1 或 1:2)
        target_pixels: 目標總像素數 (用於計算桶解析度)
        bucket_max_resolution: 分桶最大解析度約束
        follow_max_resolution: 是否遵循最大解析度約束
    """
    min_resolution: int = 256
    max_resolution: int = 1024
    patch_size: int = 16  # 與 DINOv3 匹配
    max_aspect_ratio: float = 2.0
    target_pixels: int = 512 * 512  # 262144 pixels
    bucket_max_resolution: int = 1024  # 分桶最大解析度約束
    follow_max_resolution: bool = True  # 是否遵循最大解析度約束


__all__ = ["BucketConfig"]
