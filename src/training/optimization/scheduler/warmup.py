"""
Warmup Schedulers

Contains schedulers with linear warmup phase:
    - CosineWarmupScheduler
    - LinearWarmupScheduler
    - ConstantWarmupScheduler

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class CosineWarmupScheduler(LRScheduler):
    """
    Cosine annealing scheduler with linear warmup.

    Learning rate schedule:
        1. Warmup (0 ~ warmup_steps): Linear increase from 0 to base_lr
        2. Decay (warmup_steps ~ total_steps): Cosine decay to min_lr

    Formula:
        - Warmup: lr = base_lr * (step / warmup_steps)
        - Decay: lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))

    Args:
        optimizer: Optimizer instance
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr: Minimum learning rate (default: 0.0)
        last_epoch: Last epoch for resuming
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if warmup_steps >= total_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) must be less than total_steps ({total_steps})"
            )
        if min_lr < 0:
            raise ValueError(f"min_lr must be non-negative, got {min_lr}")

        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Calculate current learning rate."""
        step = self.last_epoch
        if step < self.warmup_steps:
            warmup_factor = step / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_factor for base_lr in self.base_lrs]


class LinearWarmupScheduler(LRScheduler):
    """
    Linear scheduler with linear warmup.

    Learning rate schedule:
        1. Warmup: Linear increase from 0 to base_lr
        2. Decay: Linear decay from base_lr to min_lr

    Args:
        optimizer: Optimizer instance
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr: Minimum learning rate
        last_epoch: Last epoch for resuming
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if warmup_steps >= total_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) must be less than total_steps ({total_steps})"
            )

        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Calculate current learning rate."""
        step = self.last_epoch
        if step < self.warmup_steps:
            warmup_factor = step / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            decay_factor = 1.0 - progress
            return [self.min_lr + (base_lr - self.min_lr) * decay_factor for base_lr in self.base_lrs]


class ConstantWarmupScheduler(LRScheduler):
    """
    Constant scheduler with linear warmup.

    Learning rate schedule:
        1. Warmup: Linear increase from 0 to base_lr
        2. Constant: Maintain base_lr

    Args:
        optimizer: Optimizer instance
        warmup_steps: Number of warmup steps
        last_epoch: Last epoch for resuming
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")

        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Calculate current learning rate."""
        step = self.last_epoch
        if step < self.warmup_steps:
            warmup_factor = step / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            return list(self.base_lrs)


__all__ = ["CosineWarmupScheduler", "LinearWarmupScheduler", "ConstantWarmupScheduler"]
