"""
PixelHDM-RPEA-DinoV3 Dataset

Image-Text Dataset for PixelHDM training.

Features:
    - Recursive subfolder search
    - Multiple image formats
    - .txt caption support
    - Preprocessing (scale, crop to patch_size multiples)
    - Caption dropout for CFG training
    - patch_size=16 匹配 (與 DINOv3 一致)
    - Aspect Ratio Bucketing (多分辨率訓練)
    - RAM 優化 (序列化路徑、動態緩衝區)

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from .base import (
    IMAGE_EXTENSIONS,
    BaseImageTextDataset,
    find_images,
)

from .fixed import (
    ImageTextDataset,
    collate_fn,
)

from .bucketed import (
    BucketImageTextDataset,
    bucket_collate_fn,
)

from .factory import (
    create_dataloader,
    create_dataloader_from_config,
    create_bucket_dataloader,
    create_dataloader_from_config_v2,
)


__all__ = [
    # Base
    "IMAGE_EXTENSIONS",
    "BaseImageTextDataset",
    "find_images",
    # Fixed resolution
    "ImageTextDataset",
    "collate_fn",
    # Bucketed (multi-resolution)
    "BucketImageTextDataset",
    "bucket_collate_fn",
    # Factory functions
    "create_dataloader",
    "create_dataloader_from_config",
    "create_bucket_dataloader",
    "create_dataloader_from_config_v2",
]
