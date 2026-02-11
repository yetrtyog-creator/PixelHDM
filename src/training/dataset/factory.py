"""
PixelHDM-RPEA-DinoV3 Dataset Factory Functions

數據集和 DataLoader 工廠函數。

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from torch.utils.data import DataLoader

from .fixed import ImageTextDataset, collate_fn
from .bucketed import BucketImageTextDataset, bucket_collate_fn
from ..bucket import (
    AspectRatioBucket,
    BucketSampler,
    SequentialBucketSampler,
    BufferedShuffleBucketSampler,
    AdaptiveBufferManager,
    scan_images_for_buckets,
)

if TYPE_CHECKING:
    from ...config.model_config import PixelHDMConfig, DataConfig

logger = logging.getLogger(__name__)


def _create_sequential_sampler(
    bucket_ids: List[int],
    bucket_manager: AspectRatioBucket,
    batch_size: int,
    bucket_order: str,
    num_workers: int,
    buffer_manager: AdaptiveBufferManager,
    drop_last: bool = True,
) -> Tuple[SequentialBucketSampler, int]:
    """Create SequentialBucketSampler with optimal prefetch_factor."""
    logger.info(f"Using SequentialBucketSampler (RAM optimization mode, order={bucket_order}, drop_last={drop_last})")
    sampler = SequentialBucketSampler(
        bucket_ids=bucket_ids,
        bucket_manager=bucket_manager,
        batch_size=batch_size,
        drop_last=drop_last,
        order=bucket_order,
    )
    sampler.estimate_ram_usage()  # Log RAM usage

    prefetch_factor = 2
    if num_workers > 0:
        max_pixels = max(
            bucket_manager.get_bucket_resolution(bid)[0] * bucket_manager.get_bucket_resolution(bid)[1]
            for bid in bucket_ids
        )
        prefetch_factor = buffer_manager.compute_prefetch_factor(max_pixels, batch_size, num_workers)
        stats = buffer_manager.get_buffer_stats(max_pixels, batch_size, num_workers, prefetch_factor)
        logger.info(
            f"Buffer: {stats['total_mb']:.1f}/{stats['max_buffer_mb']:.1f} MB "
            f"({stats['utilization']*100:.1f}%), prefetch={prefetch_factor}"
        )
    return sampler, prefetch_factor


def _create_buffered_shuffle_sampler(
    bucket_ids: List[int],
    bucket_manager: AspectRatioBucket,
    batch_size: int,
    chunk_size: int,
    shuffle_chunks: bool,
    shuffle_within_bucket: bool,
    num_workers: int,
    buffer_manager: AdaptiveBufferManager,
    drop_last: bool = True,
) -> Tuple[BufferedShuffleBucketSampler, int]:
    """Create BufferedShuffleBucketSampler with optimal prefetch_factor."""
    logger.info(
        f"Using BufferedShuffleBucketSampler (chunk_size={chunk_size}, "
        f"shuffle_chunks={shuffle_chunks}, shuffle_within_bucket={shuffle_within_bucket}, drop_last={drop_last})"
    )
    sampler = BufferedShuffleBucketSampler(
        bucket_ids=bucket_ids,
        bucket_manager=bucket_manager,
        batch_size=batch_size,
        drop_last=drop_last,
        chunk_size=chunk_size,
        shuffle_chunks=shuffle_chunks,
        shuffle_within_bucket=shuffle_within_bucket,
    )
    buffer_info = sampler.estimate_buffer_size()
    logger.info(
        f"Buffer estimate: current={buffer_info['current_mb']:.1f} MB, "
        f"prefetch={buffer_info['prefetch_mb']:.1f} MB, "
        f"total={buffer_info['total_mb']:.1f} MB"
    )

    prefetch_factor = 2
    if num_workers > 0:
        chunk_info = sampler.get_current_chunk_info()
        if chunk_info:
            prefetch_factor = buffer_manager.compute_prefetch_factor(
                chunk_info['max_pixels'], batch_size, num_workers
            )
    return sampler, prefetch_factor


def _create_random_sampler(
    bucket_ids: List[int],
    batch_size: int,
    shuffle: bool,
    drop_last: bool = True,
) -> Tuple[BucketSampler, int]:
    """Create BucketSampler (random mode)."""
    logger.info(f"Using BucketSampler (random mode, shuffle={shuffle}, drop_last={drop_last})")
    sampler = BucketSampler(
        bucket_ids=bucket_ids,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
    )
    return sampler, 2


def create_dataloader(
    root_dir: Union[str, Path],
    batch_size: int = 4,
    target_resolution: int = 256,
    patch_size: int = 16,  # 與 DINOv3 匹配
    max_resolution: int = 1024,
    min_resolution: int = 256,
    caption_dropout: float = 0.1,
    default_caption: str = "",
    use_random_crop: bool = True,
    use_random_flip: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for fixed resolution training.

    Args:
        root_dir: Root directory
        batch_size: Batch size
        target_resolution: Target resolution (must be multiple of patch_size)
        patch_size: Patch size (default 16, matches DINOv3)
        max_resolution: Maximum resolution
        min_resolution: Minimum resolution
        caption_dropout: Caption dropout rate
        default_caption: Default caption
        use_random_crop: Use random crop
        use_random_flip: Use random horizontal flip
        num_workers: Number of workers
        pin_memory: Pin memory
        prefetch_factor: Prefetch factor
        shuffle: Shuffle data
        drop_last: Drop incomplete batches

    Returns:
        DataLoader instance
    """
    dataset = ImageTextDataset(
        root_dir=root_dir,
        target_resolution=target_resolution,
        patch_size=patch_size,
        max_resolution=max_resolution,
        min_resolution=min_resolution,
        caption_dropout=caption_dropout,
        default_caption=default_caption,
        use_random_crop=use_random_crop,
        use_random_flip=use_random_flip,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )

    return dataloader


def create_dataloader_from_config(
    root_dir: Union[str, Path],
    config: "PixelHDMConfig",
    batch_size: int = 4,
    num_workers: int = 4,
) -> DataLoader:
    """Create DataLoader from PixelHDMConfig."""
    return create_dataloader(
        root_dir=root_dir,
        batch_size=batch_size,
        target_resolution=256,  # 預設 256x256
        patch_size=config.patch_size,  # 從配置獲取 (16)
        num_workers=num_workers,
    )


def create_bucket_dataloader(
    root_dir: Union[str, Path],
    batch_size: int = 4,
    min_resolution: int = 256,
    max_resolution: int = 1024,
    patch_size: int = 16,  # 與 DINOv3 匹配
    target_pixels: int = 512 * 512,
    max_aspect_ratio: float = 2.0,
    caption_dropout: float = 0.1,
    default_caption: str = "",
    use_random_crop: bool = True,
    use_random_flip: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    shuffle: bool = True,
    persistent_workers: bool = False,
    sampler_mode: str = "buffered_shuffle",  # "random", "sequential", "buffered_shuffle"
    bucket_order: str = "ascending",
    buffer_size_gb: float = 4.0,
    chunk_size: int = 4,  # BufferedShuffleBucketSampler 超級塊大小
    shuffle_chunks: bool = True,  # 是否在 epoch 間隨機打亂超級塊順序
    shuffle_within_bucket: bool = True,  # 是否在桶內隨機打亂樣本順序
    # 最大解析度約束 (嚴禁硬編碼)
    bucket_max_resolution: Optional[int] = None,  # 分桶最大解析度約束
    follow_max_resolution: bool = True,  # 是否遵循最大解析度約束
    # Batch handling
    drop_last: bool = True,  # 丟棄未滿 batch_size 的批次
    # 向後兼容
    optimization_mode: bool = False,
) -> DataLoader:
    """
    創建支援分桶的 DataLoader

    Args:
        root_dir: 圖片根目錄
        batch_size: 批次大小
        min_resolution: 最小解析度 (最小邊長)
        max_resolution: 最大解析度 (最大邊長)
        patch_size: Patch 大小 (16, 與 DINOv3 匹配)
        target_pixels: 目標像素數 (用於計算桶解析度)
        max_aspect_ratio: 最大長寬比
        caption_dropout: 標題丟棄率
        default_caption: 預設標題
        use_random_crop: 隨機裁切
        use_random_flip: 隨機翻轉
        num_workers: 工作進程數
        pin_memory: 固定記憶體
        prefetch_factor: 預取因子
        shuffle: 是否打亂 (用於 random 模式)
        persistent_workers: 保持工作進程
        sampler_mode: 採樣器模式
            - "buffered_shuffle": 緩衝區優化的隨機洗牌 (推薦，默認)
            - "random": 完全隨機 (原 BucketSampler)
            - "sequential": 順序處理 (RAM 最優化)
        bucket_order: 桶處理順序 "ascending" 或 "descending"
        buffer_size_gb: 緩衝區大小限制 (GB)
        chunk_size: BufferedShuffleBucketSampler 超級塊大小
        shuffle_chunks: 是否在 epoch 間隨機打亂超級塊順序
        shuffle_within_bucket: 是否在桶內隨機打亂樣本順序
        bucket_max_resolution: 分桶最大解析度約束 (用於約束生成的桶)
        follow_max_resolution: 是否遵循最大解析度約束
        drop_last: 丟棄未滿 batch_size 的批次 (建議保持 True)
        optimization_mode: 向後兼容參數，設為 True 等同於 sampler_mode="sequential"

    Returns:
        DataLoader with bucket-based sampling
    """
    # 設置 bucket_max_resolution 默認值
    if bucket_max_resolution is None:
        bucket_max_resolution = max_resolution

    # 創建分桶管理器
    bucket_manager = AspectRatioBucket(
        min_resolution=min_resolution,
        max_resolution=max_resolution,
        patch_size=patch_size,
        max_aspect_ratio=max_aspect_ratio,
        target_pixels=target_pixels,
        bucket_max_resolution=bucket_max_resolution,
        follow_max_resolution=follow_max_resolution,
    )

    # 掃描圖片並分配桶
    image_paths, bucket_ids = scan_images_for_buckets(
        root_dir=root_dir,
        bucket_manager=bucket_manager,
        min_resolution=min_resolution,
    )

    if len(image_paths) == 0:
        raise RuntimeError(f"No valid images found in {root_dir}")

    # 創建數據集
    dataset = BucketImageTextDataset(
        image_paths=image_paths,
        bucket_ids=bucket_ids,
        bucket_manager=bucket_manager,
        caption_dropout=caption_dropout,
        default_caption=default_caption,
        use_random_crop=use_random_crop,
        use_random_flip=use_random_flip,
    )

    # 向後兼容: optimization_mode=True 等同於 sampler_mode="sequential"
    if optimization_mode:
        sampler_mode = "sequential"

    # 創建採樣器
    buffer_manager = AdaptiveBufferManager(max_buffer_gb=buffer_size_gb, min_prefetch=1, max_prefetch=8)

    if sampler_mode == "sequential":
        sampler, prefetch_factor = _create_sequential_sampler(
            bucket_ids, bucket_manager, batch_size, bucket_order, num_workers, buffer_manager, drop_last
        )
    elif sampler_mode == "buffered_shuffle":
        sampler, prefetch_factor = _create_buffered_shuffle_sampler(
            bucket_ids, bucket_manager, batch_size, chunk_size,
            shuffle_chunks, shuffle_within_bucket, num_workers, buffer_manager, drop_last
        )
    else:  # "random" 模式
        sampler, prefetch_factor = _create_random_sampler(bucket_ids, batch_size, shuffle, drop_last)

    # 創建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers and num_workers > 0,
        collate_fn=bucket_collate_fn,
    )

    return dataloader


def create_dataloader_from_config_v2(
    root_dir: Union[str, Path],
    model_config: "PixelHDMConfig",
    data_config: Optional["DataConfig"] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    use_bucketing: Optional[bool] = None,
    sampler_mode: Optional[str] = None,
    shuffle: bool = True,
) -> DataLoader:
    """
    從配置創建 DataLoader (支援分桶模式)

    所有參數優先從 DataConfig 讀取，嚴禁硬編碼

    Args:
        root_dir: 圖片根目錄
        model_config: PixelHDMConfig 配置 (用於 patch_size)
        data_config: DataConfig 配置 (用於分桶參數，可選)
        batch_size: 批次大小 (覆蓋 data_config)
        num_workers: 工作進程數 (覆蓋 data_config)
        use_bucketing: 是否使用分桶 (覆蓋 data_config)
        sampler_mode: 採樣器模式 (覆蓋 data_config)
        shuffle: 是否打亂 (用於 random 模式)

    Returns:
        DataLoader 實例
    """
    # 從 DataConfig 獲取默認值
    if data_config is not None:
        use_bucketing = use_bucketing if use_bucketing is not None else data_config.use_bucketing
        batch_size = batch_size if batch_size is not None else 4  # TrainingConfig.batch_size
        num_workers = num_workers if num_workers is not None else data_config.num_workers
        sampler_mode = sampler_mode if sampler_mode is not None else data_config.sampler_mode
        min_resolution = data_config.min_bucket_size
        max_resolution = data_config.max_bucket_size
        target_pixels = data_config.target_pixels
        max_aspect_ratio = data_config.max_aspect_ratio
        bucket_max_resolution = data_config.bucket_max_resolution
        follow_max_resolution = data_config.bucket_follow_max_resolution
        buffer_size_gb = data_config.buffer_size_gb
        bucket_order = data_config.bucket_order
        chunk_size = data_config.chunk_size
        shuffle_chunks = data_config.shuffle_chunks
        shuffle_within_bucket = data_config.shuffle_within_bucket
        drop_last = data_config.drop_last
        caption_dropout = data_config.caption_dropout
        default_caption = data_config.default_caption
        # center_crop=True takes priority: disable random_crop
        use_random_crop = data_config.use_random_crop and not data_config.center_crop
        use_random_flip = data_config.random_flip
        pin_memory = data_config.pin_memory
        prefetch_factor = data_config.prefetch_factor
        persistent_workers = data_config.persistent_workers
    else:
        # 使用默認值 (向後兼容)
        use_bucketing = use_bucketing if use_bucketing is not None else True
        batch_size = batch_size if batch_size is not None else 4
        num_workers = num_workers if num_workers is not None else 4
        sampler_mode = sampler_mode if sampler_mode is not None else "buffered_shuffle"
        min_resolution = 256
        max_resolution = 1024
        target_pixels = 512 * 512
        max_aspect_ratio = 2.0
        bucket_max_resolution = 1024
        follow_max_resolution = True
        buffer_size_gb = 4.0
        bucket_order = "ascending"
        chunk_size = 4
        shuffle_chunks = True
        shuffle_within_bucket = True
        drop_last = True
        caption_dropout = 0.1
        default_caption = ""
        use_random_crop = True
        use_random_flip = True
        pin_memory = True
        prefetch_factor = 2
        persistent_workers = False

    if use_bucketing:
        logger.info(
            f"Using multi-resolution bucketing: "
            f"min={min_resolution}, max={max_resolution}, "
            f"bucket_max={bucket_max_resolution}, follow={follow_max_resolution}"
        )
        return create_bucket_dataloader(
            root_dir=root_dir,
            batch_size=batch_size,
            min_resolution=min_resolution,
            max_resolution=max_resolution,
            patch_size=model_config.patch_size,  # 16, 與 DINOv3 匹配
            target_pixels=target_pixels,
            max_aspect_ratio=max_aspect_ratio,
            caption_dropout=caption_dropout,
            default_caption=default_caption,
            use_random_crop=use_random_crop,
            use_random_flip=use_random_flip,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            shuffle=shuffle,
            sampler_mode=sampler_mode,
            bucket_order=bucket_order,
            buffer_size_gb=buffer_size_gb,
            chunk_size=chunk_size,
            shuffle_chunks=shuffle_chunks,
            shuffle_within_bucket=shuffle_within_bucket,
            bucket_max_resolution=bucket_max_resolution,
            follow_max_resolution=follow_max_resolution,
            drop_last=drop_last,
        )
    else:
        logger.info("Using fixed resolution training")
        return create_dataloader(
            root_dir=root_dir,
            batch_size=batch_size,
            target_resolution=data_config.image_size if data_config else 512,
            patch_size=model_config.patch_size,
            max_resolution=max_resolution,
            min_resolution=min_resolution,
            caption_dropout=caption_dropout,
            default_caption=default_caption,
            use_random_crop=use_random_crop,
            use_random_flip=use_random_flip,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            shuffle=shuffle,
            drop_last=drop_last,
        )


__all__ = [
    "create_dataloader",
    "create_dataloader_from_config",
    "create_bucket_dataloader",
    "create_dataloader_from_config_v2",
]
