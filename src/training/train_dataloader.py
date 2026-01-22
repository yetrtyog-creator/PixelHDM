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

    Args:
        data_config: Data configuration
        model_config: Model configuration (for patch_size)
        training_config: Training configuration (for batch_size)

    Returns:
        DataLoader instance
    """
    from .dataset import create_bucket_dataloader

    batch_size = training_config.batch_size

    if data_config.use_bucketing:
        dataloader = create_bucket_dataloader(
            root_dir=data_config.data_dir,
            batch_size=batch_size,
            min_resolution=data_config.min_bucket_size,
            max_resolution=data_config.max_bucket_size,
            patch_size=model_config.patch_size,
            caption_dropout=data_config.caption_dropout,
            use_random_crop=data_config.use_random_crop,
            use_random_flip=data_config.random_flip,
            num_workers=data_config.num_workers,
            pin_memory=data_config.pin_memory,
            drop_last=data_config.drop_last,
        )
    else:
        from .dataset import ImageTextDataset, collate_fn
        from torch.utils.data import DataLoader

        dataset = ImageTextDataset(
            root_dir=data_config.data_dir,
            target_resolution=data_config.image_size,
            use_random_flip=data_config.random_flip,
            use_random_crop=data_config.use_random_crop,
            caption_dropout=data_config.caption_dropout,
            default_caption=data_config.default_caption,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=data_config.num_workers,
            pin_memory=data_config.pin_memory,
            collate_fn=collate_fn,
            drop_last=data_config.drop_last,
        )

    return dataloader


__all__ = ["create_dataloader_from_config"]
