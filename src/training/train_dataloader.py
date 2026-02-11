"""
Training DataLoader Factory

Creates DataLoader from configuration.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..config.model_config import PixelHDMConfig, TrainingConfig, DataConfig


def create_dataloader_from_config(
    data_config: "DataConfig",
    model_config: "PixelHDMConfig",
    training_config: "TrainingConfig",
) -> torch.utils.data.DataLoader:
    """
    Create dataloader from DataConfig.

    This function delegates to create_dataloader_from_config_v2 which fully
    respects all DataConfig fields including:
    - Bucketing: target_pixels, max_aspect_ratio, bucket_max_resolution,
                 bucket_follow_max_resolution
    - Sampler: sampler_mode, chunk_size, shuffle_chunks, shuffle_within_bucket
    - RAM: buffer_size_gb, bucket_order, optimization_mode
    - DataLoader: prefetch_factor, persistent_workers
    - Augmentation: center_crop (overrides use_random_crop when True)

    Args:
        data_config: Data configuration
        model_config: Model configuration (for patch_size)
        training_config: Training configuration (for batch_size)

    Returns:
        DataLoader instance
    """
    from .dataset import create_dataloader_from_config_v2

    batch_size = training_config.batch_size

    return create_dataloader_from_config_v2(
        root_dir=data_config.data_dir,
        model_config=model_config,
        data_config=data_config,
        batch_size=batch_size,
        num_workers=data_config.num_workers,
        use_bucketing=data_config.use_bucketing,
        sampler_mode=data_config.sampler_mode,
        shuffle=True,
    )


__all__ = ["create_dataloader_from_config"]
