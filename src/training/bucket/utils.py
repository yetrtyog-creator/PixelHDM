"""
PixelHDM-RPEA-DinoV3 Bucket Utilities

分桶工具函數和管理器。

包含:
    - AdaptiveBufferManager: 自適應緩衝區管理器
    - scan_images_for_buckets: 圖片掃描與桶分配
    - create_bucket_manager: 分桶管理器工廠函數

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from PIL import Image

from .generator import AspectRatioBucket

if TYPE_CHECKING:
    from ...config.model_config import PixelHDMConfig, DataConfig

logger = logging.getLogger(__name__)


# 支援的圖片副檔名 (與 dataset 模塊保持一致)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"}


class AdaptiveBufferManager:
    """
    自適應緩衝區管理器

    根據當前處理的桶動態調整預取策略:
    - 小分辨率桶: 增加預取批次數
    - 大分辨率桶: 減少預取批次數
    - 保持總緩衝區大小在限制內

    Args:
        max_buffer_gb: 最大緩衝區大小 (GB)
        min_prefetch: 最小預取批次數
        max_prefetch: 最大預取批次數
        bytes_per_pixel: 每像素位元組數
    """

    def __init__(
        self,
        max_buffer_gb: float = 4.0,
        min_prefetch: int = 1,
        max_prefetch: int = 8,
        bytes_per_pixel: float = 12.0,
    ) -> None:
        self.max_buffer_bytes = max_buffer_gb * 1024 * 1024 * 1024
        self.min_prefetch = min_prefetch
        self.max_prefetch = max_prefetch
        self.bytes_per_pixel = bytes_per_pixel

    def compute_prefetch_factor(
        self,
        max_pixels: int,
        batch_size: int,
        num_workers: int,
    ) -> int:
        """
        計算最優預取因子

        Args:
            max_pixels: 當前超級塊的最大像素數
            batch_size: 批次大小
            num_workers: 工作進程數

        Returns:
            最優預取因子
        """
        if num_workers <= 0:
            return 2  # 默認值

        # 每個批次的內存需求
        batch_bytes = max_pixels * self.bytes_per_pixel * batch_size

        # 總預取批次數 = prefetch_factor * num_workers
        # 總內存需求 = batch_bytes * prefetch_factor * num_workers
        # 確保: batch_bytes * prefetch_factor * num_workers <= max_buffer_bytes

        if batch_bytes <= 0:
            return self.max_prefetch

        max_total_prefetch = int(self.max_buffer_bytes / batch_bytes)
        optimal_prefetch = max(1, max_total_prefetch // num_workers)

        # 限制在 [min_prefetch, max_prefetch] 範圍內
        optimal_prefetch = max(self.min_prefetch, min(optimal_prefetch, self.max_prefetch))

        return optimal_prefetch

    def get_buffer_stats(
        self,
        max_pixels: int,
        batch_size: int,
        num_workers: int,
        prefetch_factor: int,
    ) -> Dict[str, float]:
        """
        獲取緩衝區統計信息

        Returns:
            包含 batch_mb, total_mb, utilization 的字典
        """
        batch_bytes = max_pixels * self.bytes_per_pixel * batch_size
        batch_mb = batch_bytes / (1024 * 1024)

        total_bytes = batch_bytes * prefetch_factor * num_workers
        total_mb = total_bytes / (1024 * 1024)

        utilization = total_bytes / self.max_buffer_bytes if self.max_buffer_bytes > 0 else 0

        return {
            "batch_mb": batch_mb,
            "total_mb": total_mb,
            "utilization": utilization,
            "max_buffer_mb": self.max_buffer_bytes / (1024 * 1024),
        }


# 支援的圖片副檔名
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def scan_images_for_buckets(
    root_dir: Union[str, Path],
    bucket_manager: AspectRatioBucket,
    min_resolution: int = 256,
) -> Tuple[List[Path], List[int]]:
    """
    掃描圖片目錄，為每張圖片分配桶 ID

    Args:
        root_dir: 圖片根目錄
        bucket_manager: 分桶管理器
        min_resolution: 最小解析度 (小於此值的圖片會被跳過)

    Returns:
        image_paths: 有效圖片路徑列表
        bucket_ids: 對應的桶 ID 列表
    """
    root_dir = Path(root_dir)
    image_paths = []
    bucket_ids = []
    skipped = 0

    # 遞迴搜尋所有圖片
    all_images = []
    for ext in IMAGE_EXTENSIONS:
        all_images.extend(root_dir.rglob(f"*{ext}"))
        all_images.extend(root_dir.rglob(f"*{ext.upper()}"))
    all_images = sorted(set(all_images))

    logger.info(f"Scanning {len(all_images)} images for bucket assignment...")

    for img_path in all_images:
        try:
            with Image.open(img_path) as img:
                width, height = img.size

            # 跳過太小的圖片
            if min(width, height) < min_resolution:
                skipped += 1
                continue

            # 分配桶
            bucket_id = bucket_manager.get_bucket_id(width, height)
            image_paths.append(img_path)
            bucket_ids.append(bucket_id)

        except Exception as e:
            logger.warning(f"Failed to read image {img_path}: {e}")
            skipped += 1

    logger.info(
        f"Bucket assignment complete: {len(image_paths)} valid, {skipped} skipped"
    )

    # 統計各桶數量
    bucket_counts = defaultdict(int)
    for bid in bucket_ids:
        bucket_counts[bid] += 1

    for bid, count in sorted(bucket_counts.items()):
        res = bucket_manager.get_bucket_resolution(bid)
        logger.info(f"  Bucket {bid} ({res[0]}x{res[1]}): {count} images")

    return image_paths, bucket_ids


def create_bucket_manager(
    model_config: Optional["PixelHDMConfig"] = None,
    data_config: Optional["DataConfig"] = None,
    min_resolution: Optional[int] = None,
    max_resolution: Optional[int] = None,
    patch_size: Optional[int] = None,
    max_aspect_ratio: Optional[float] = None,
    target_pixels: Optional[int] = None,
    bucket_max_resolution: Optional[int] = None,
    follow_max_resolution: Optional[bool] = None,
) -> AspectRatioBucket:
    """
    創建分桶管理器

    優先級: 顯式參數 > DataConfig > PixelHDMConfig > 默認值

    Args:
        model_config: PixelHDMConfig 配置 (可選，用於獲取 patch_size)
        data_config: DataConfig 配置 (可選，用於獲取分桶參數)
        min_resolution: 最小解析度 (最小邊長)
        max_resolution: 最大解析度 (最大邊長)
        patch_size: Patch 大小 (16, 與 DINOv3 匹配)
        max_aspect_ratio: 最大長寬比
        target_pixels: 目標像素數
        bucket_max_resolution: 分桶最大解析度約束
        follow_max_resolution: 是否遵循最大解析度約束

    Returns:
        AspectRatioBucket 實例
    """
    # 從 DataConfig 獲取默認值
    if data_config is not None:
        min_resolution = min_resolution if min_resolution is not None else data_config.min_bucket_size
        max_resolution = max_resolution if max_resolution is not None else data_config.max_bucket_size
        max_aspect_ratio = max_aspect_ratio if max_aspect_ratio is not None else data_config.max_aspect_ratio
        target_pixels = target_pixels if target_pixels is not None else data_config.target_pixels
        bucket_max_resolution = bucket_max_resolution if bucket_max_resolution is not None else data_config.bucket_max_resolution
        follow_max_resolution = follow_max_resolution if follow_max_resolution is not None else data_config.bucket_follow_max_resolution

    # 從 PixelHDMConfig 獲取 patch_size
    if model_config is not None:
        patch_size = patch_size if patch_size is not None else model_config.patch_size

    # 使用默認值填充
    min_resolution = min_resolution if min_resolution is not None else 256
    max_resolution = max_resolution if max_resolution is not None else 1024
    patch_size = patch_size if patch_size is not None else 16
    max_aspect_ratio = max_aspect_ratio if max_aspect_ratio is not None else 2.0
    target_pixels = target_pixels if target_pixels is not None else 512 * 512
    bucket_max_resolution = bucket_max_resolution if bucket_max_resolution is not None else max_resolution
    follow_max_resolution = follow_max_resolution if follow_max_resolution is not None else True

    return AspectRatioBucket(
        min_resolution=min_resolution,
        max_resolution=max_resolution,
        patch_size=patch_size,
        max_aspect_ratio=max_aspect_ratio,
        target_pixels=target_pixels,
        bucket_max_resolution=bucket_max_resolution,
        follow_max_resolution=follow_max_resolution,
    )


def create_bucket_manager_from_config(
    model_config: "PixelHDMConfig",
    data_config: "DataConfig",
) -> AspectRatioBucket:
    """
    從配置創建分桶管理器 (便捷函數)

    Args:
        model_config: PixelHDMConfig 配置 (用於 patch_size)
        data_config: DataConfig 配置 (用於分桶參數)

    Returns:
        AspectRatioBucket 實例
    """
    return create_bucket_manager(
        model_config=model_config,
        data_config=data_config,
    )


__all__ = [
    "AdaptiveBufferManager",
    "IMAGE_EXTENSIONS",
    "scan_images_for_buckets",
    "create_bucket_manager",
    "create_bucket_manager_from_config",
]
