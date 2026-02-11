"""
Data Configuration for PixelHDM-RPEA-DinoV3.

This module defines the DataConfig dataclass for dataset and dataloader settings.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class DataConfig:
    """Data configuration for dataset loading and augmentation.

    Attributes:
        data_dir: Root directory containing training data.
        image_size: Default training resolution (512 recommended).

    Bucketing Configuration:
        use_bucketing: Enable multi-resolution bucket training.
        min_bucket_size: Minimum bucket resolution (min edge length).
        max_bucket_size: Maximum bucket resolution (max edge length).
        bucket_step: Resolution step size (must be patch_size multiple).
        bucket_max_resolution: Maximum resolution constraint.
        bucket_follow_max_resolution: Enforce max resolution constraint.
        max_aspect_ratio: Maximum aspect ratio (e.g., 2.0 for 2:1).
        target_pixels: Target pixel count per image (512*512=262144).

    RAM Optimization:
        optimization_mode: Legacy RAM optimization flag.
        bucket_order: Bucket processing order ('ascending' or 'descending').
        buffer_size_gb: Buffer size limit in GB.

    Sampler Configuration:
        sampler_mode: Sampling strategy ('random', 'sequential', 'buffered_shuffle').
        chunk_size: Super-chunk size for buffered shuffle.
        shuffle_chunks: Shuffle chunk order between epochs.
        shuffle_within_bucket: Shuffle samples within each bucket.

    Augmentation:
        random_flip: Enable random horizontal flip.
        center_crop: Use center crop (vs random crop).
        use_random_crop: Enable random cropping.

    DataLoader:
        num_workers: Number of data loading workers.
        pin_memory: Pin memory for faster GPU transfer.
        prefetch_factor: Batches to prefetch per worker.
        persistent_workers: Keep workers alive between epochs.

    Caption:
        caption_dropout: Caption dropout rate for CFG training.
        default_caption: Default caption for images without text.
    """

    # Dataset
    data_dir: str = "datasets"
    image_size: int = 512

    # Bucketing Configuration
    use_bucketing: bool = True
    min_bucket_size: int = 256
    max_bucket_size: int = 1024
    bucket_step: int = 64
    bucket_max_resolution: int = 1024
    bucket_follow_max_resolution: bool = True
    max_aspect_ratio: float = 2.0
    target_pixels: int = 262144

    # RAM Optimization
    optimization_mode: bool = False
    bucket_order: Literal["ascending", "descending"] = "ascending"
    buffer_size_gb: float = 4.0

    # Sampler Configuration
    sampler_mode: Literal["random", "sequential", "buffered_shuffle"] = "buffered_shuffle"
    chunk_size: int = 4
    shuffle_chunks: bool = True
    shuffle_within_bucket: bool = True
    drop_last: bool = True  # Drop incomplete batches (buckets with < batch_size samples)

    # Augmentation
    random_flip: bool = True
    center_crop: bool = True
    use_random_crop: bool = True

    # DataLoader
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = False

    # Caption
    caption_dropout: float = 0.1
    default_caption: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "DataConfig":
        """Create DataConfig from dictionary."""
        return cls(**data) if data else cls()
