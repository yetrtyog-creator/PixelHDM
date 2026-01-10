"""
PixelHDM-RPEA-DinoV3 Learning Rate Schedulers

Contains:
    - CosineWarmupScheduler: Cosine annealing with linear warmup
    - LinearWarmupScheduler: Linear decay with linear warmup
    - ConstantWarmupScheduler: Constant LR with linear warmup
    - CosineAnnealingWarmRestartsWithDecay: Periodic restarts with decay
    - SteppedCosineRestartScheduler: Stepped decay for both peak and trough

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-04
"""

from .warmup import CosineWarmupScheduler, LinearWarmupScheduler, ConstantWarmupScheduler
from .restart import CosineAnnealingWarmRestartsWithDecay
from .stepped_restart import SteppedCosineRestartScheduler
from .factory import create_scheduler, create_scheduler_from_config

__all__ = [
    "CosineWarmupScheduler",
    "LinearWarmupScheduler",
    "ConstantWarmupScheduler",
    "CosineAnnealingWarmRestartsWithDecay",
    "SteppedCosineRestartScheduler",
    "create_scheduler",
    "create_scheduler_from_config",
]
