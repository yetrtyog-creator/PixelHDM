"""
Scheduler Factory Functions

Factory functions for creating learning rate schedulers.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .warmup import CosineWarmupScheduler, LinearWarmupScheduler, ConstantWarmupScheduler

if TYPE_CHECKING:
    from ....config.model_config import TrainingConfig


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine",
    warmup_steps: int = 1000,
    total_steps: int = 100000,
    min_lr: float = 0.0,
) -> LRScheduler:
    """
    Create a learning rate scheduler.

    Args:
        optimizer: Optimizer instance
        scheduler_type: Scheduler type ("cosine", "linear", "constant")
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr: Minimum learning rate

    Returns:
        Learning rate scheduler

    Example:
        >>> scheduler = create_scheduler(
        ...     optimizer, scheduler_type="cosine", warmup_steps=1000, total_steps=100000,
        ... )
    """
    scheduler_type = scheduler_type.lower()

    if scheduler_type == "cosine":
        return CosineWarmupScheduler(
            optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr=min_lr,
        )
    elif scheduler_type == "linear":
        return LinearWarmupScheduler(
            optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr=min_lr,
        )
    elif scheduler_type == "constant":
        return ConstantWarmupScheduler(optimizer, warmup_steps=warmup_steps)
    else:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}. Valid options: 'cosine', 'linear', 'constant'"
        )


def create_scheduler_from_config(
    optimizer: Optimizer,
    config: "TrainingConfig",
) -> LRScheduler:
    """
    Create a learning rate scheduler from config.

    Args:
        optimizer: Optimizer instance
        config: Training config (TrainingConfig instance)

    Returns:
        Learning rate scheduler

    Note:
        TrainingConfig uses these attribute names:
        - lr_scheduler (not scheduler_type)
        - max_steps (not total_steps)
    """
    return create_scheduler(
        optimizer,
        scheduler_type=config.lr_scheduler,
        warmup_steps=config.warmup_steps,
        total_steps=config.max_steps,
        min_lr=config.min_lr,
    )


__all__ = ["create_scheduler", "create_scheduler_from_config"]
