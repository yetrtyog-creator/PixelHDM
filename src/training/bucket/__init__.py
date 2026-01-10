"""
PixelHDM-RPEA-DinoV3 Aspect Ratio Bucketing

Bucket-based data system ensuring same-resolution samples per batch.

Features:
    - Predefined aspect ratio buckets (1:1, 4:3, 3:4, 16:9, 9:16, etc.)
    - All resolutions are multiples of patch_size (16) for DINOv3
    - Automatic image assignment to nearest bucket
    - BucketSampler for random bucket sampling
    - SequentialBucketSampler for RAM-optimized sequential processing
    - BufferedShuffleBucketSampler for buffered shuffle (recommended)
    - RAM estimation and checkpoint resume support

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from .config import BucketConfig
from .generator import AspectRatioBucket
from .sampler import (
    BucketSampler,
    SequentialBucketSampler,
    BufferedShuffleBucketSampler,
)
from .batch_builder import (
    BatchCounter,
    BatchGenerator,
    BucketGrouper,
    BucketSorter,
    ChunkBuilder,
    RAMEstimator,
)
from .state import (
    SamplerState,
    BucketSamplerState,
    SequentialSamplerState,
    BufferedShuffleSamplerState,
)
from .info import (
    SequentialSamplerInfo,
    BufferedShuffleSamplerInfo,
)
from .utils import (
    AdaptiveBufferManager,
    IMAGE_EXTENSIONS,
    scan_images_for_buckets,
    create_bucket_manager,
    create_bucket_manager_from_config,
)


__all__ = [
    # Config
    "BucketConfig",
    # Generator
    "AspectRatioBucket",
    # Samplers
    "BucketSampler",
    "SequentialBucketSampler",
    "BufferedShuffleBucketSampler",
    # Batch Building (advanced usage)
    "BatchCounter",
    "BatchGenerator",
    "BucketGrouper",
    "BucketSorter",
    "ChunkBuilder",
    "RAMEstimator",
    # State Management (advanced usage)
    "SamplerState",
    "BucketSamplerState",
    "SequentialSamplerState",
    "BufferedShuffleSamplerState",
    # Info (advanced usage)
    "SequentialSamplerInfo",
    "BufferedShuffleSamplerInfo",
    # Utils
    "AdaptiveBufferManager",
    "IMAGE_EXTENSIONS",
    "scan_images_for_buckets",
    "create_bucket_manager",
    "create_bucket_manager_from_config",
]
