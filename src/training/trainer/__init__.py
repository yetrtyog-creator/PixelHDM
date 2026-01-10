"""
PixelHDM Trainer Module

Refactored trainer with modular components:
    - core: Main Trainer class
    - metrics: TrainerState, TrainMetrics, MetricsLogger
    - scheduler_factory: LR scheduler creation
    - step: StepExecutor for training steps
    - checkpoint: CheckpointManager for save/load
    - loop: TrainingLoop for main loop
    - init_helpers: Initialization helper functions

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from .core import Trainer, create_trainer
from .metrics import TrainerState, TrainMetrics, MetricsLogger
from .scheduler_factory import create_lr_scheduler, apply_warmup_lr
from .step import StepExecutor
from .checkpoint import CheckpointManager
from .loop import TrainingLoop

__all__ = [
    # Core
    "Trainer",
    "create_trainer",
    # Metrics
    "TrainerState",
    "TrainMetrics",
    "MetricsLogger",
    # Scheduler
    "create_lr_scheduler",
    "apply_warmup_lr",
    # Step
    "StepExecutor",
    # Checkpoint
    "CheckpointManager",
    # Loop
    "TrainingLoop",
]
