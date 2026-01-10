"""
Training Configuration for PixelHDM-RPEA-DinoV3.

This module defines the TrainingConfig dataclass for training hyperparameters.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-04
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class SteppedCosineRestartConfig:
    """Configuration for Stepped Cosine Restart learning rate scheduler.

    This scheduler decays both peak and trough learning rates together,
    providing more aggressive and controlled learning rate reduction.

    When enabled, this takes priority over standard lr_scheduler settings.
    The following TrainingConfig fields will be IGNORED when enabled:
        - learning_rate (use base_lr instead)
        - min_lr (use cycle_min_lr and global_min_lr instead)
        - lr_decay_per_cycle (use decay_rate instead)

    The following fields are still respected:
        - restart_epochs (cycle length)
        - restart_period (explicit cycle length in steps)
        - warmup_steps (from this config or TrainingConfig)

    Example decay curve (base_lr=2e-4, cycle_min_lr=1.2e-4, decay_rate=0.75):
        Cycle 0: 2e-4 -> 1.2e-4
        Cycle 1: 1.5e-4 -> 0.9e-4    (* 0.75)
        Cycle 2: 1.125e-4 -> 0.675e-4
        ...until reaching global_min_lr

    Attributes:
        enabled: Whether to use this scheduler (default False).
        base_lr: Initial peak learning rate.
        cycle_min_lr: Initial trough learning rate (per-cycle minimum).
        decay_rate: Decay multiplier applied to both peak and trough each cycle.
        global_min_lr: Absolute minimum learning rate (lower bound).
        warmup_steps: Linear warmup steps (0 = no warmup).
    """

    enabled: bool = False
    base_lr: float = 2e-4
    cycle_min_lr: float = 1.2e-4
    decay_rate: float = 0.75
    global_min_lr: float = 1e-5
    warmup_steps: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "SteppedCosineRestartConfig":
        """Create config from dictionary."""
        if not data:
            return cls()
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k) or k in cls.__dataclass_fields__})


@dataclass
class TrainingConfig:
    """Training configuration for optimizer, scheduler, and training loop.

    Optimizer Settings:
        optimizer: Optimizer type ('adamw' or 'adamw_8bit').
        learning_rate: Base learning rate.
        weight_decay: Weight decay coefficient.
        betas: Adam beta parameters as tuple.
        eps: Adam epsilon for numerical stability.

    Learning Rate Schedule:
        lr_scheduler: Scheduler type ('cosine', 'cosine_restart', etc.).
        warmup_steps: Number of warmup steps.
        min_lr: Minimum learning rate.
        restart_epochs: Epochs per cosine restart cycle.
        restart_period: Steps per restart (0=use restart_epochs).
        lr_decay_per_cycle: Peak LR decay factor per cycle.
        stepped_cosine_restart: Stepped cosine restart config (takes priority).

    Batch Configuration:
        batch_size: Training batch size.
        gradient_accumulation_steps: Steps to accumulate gradients.

    Training Mode:
        training_mode: 'steps' or 'epochs' based training.
        max_steps: Maximum training steps (steps mode).
        num_epochs: Number of epochs (epochs mode).

    Logging and Checkpointing (Step-based):
        log_interval: Log every N steps (0=disabled).
        save_interval: Save every N steps (0=disabled).
        eval_interval: Evaluate every N steps.

    Logging and Checkpointing (Epoch-based):
        save_every_epochs: Save every N epochs (0=disabled).
        log_every_epochs: Log every N epochs (0=disabled).
        max_checkpoints: Maximum checkpoints to keep.
        checkpoint_dir: Directory for checkpoint storage.

    EMA:
        ema_enabled: Enable exponential moving average.
        ema_decay: EMA decay coefficient.

    Mixed Precision:
        precision: Precision mode ('no', 'fp16', 'bf16').
        mixed_precision: Alias for precision (YAML compatibility).

    Gradient Control:
        max_grad_norm: Maximum gradient norm for clipping.
        zclip_threshold: Z-score threshold for adaptive clipping.
        zclip_ema_decay: EMA decay for gradient statistics.

    VRAM Optimization:
        cpu_checkpoint_interval: CPU checkpointing interval (0=disabled).
        cpu_checkpoint_spike_threshold: VRAM spike threshold for CPU offload.
    """

    # Optimizer
    optimizer: str = "adamw_8bit"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8

    # Learning rate schedule
    lr_scheduler: str = "cosine"
    warmup_steps: int = 0
    min_lr: float = 1e-5
    restart_epochs: int = 4
    restart_period: int = 0
    lr_decay_per_cycle: float = 0.5

    # Stepped Cosine Restart (takes priority over lr_scheduler when enabled)
    stepped_cosine_restart: Optional[SteppedCosineRestartConfig] = None

    # Batch size
    batch_size: int = 16
    gradient_accumulation_steps: int = 1

    # Training mode
    training_mode: Literal["steps", "epochs"] = "epochs"
    max_steps: int = 500000
    num_epochs: int = 16

    # Logging & Checkpointing (step-based)
    log_interval: int = 0
    save_interval: int = 0
    eval_interval: int = 1000

    # Logging & Checkpointing (epoch-based)
    save_every_epochs: int = 0
    log_every_epochs: int = 0
    max_checkpoints: int = 1
    checkpoint_dir: str = "./checkpoints"

    # EMA
    ema_enabled: bool = True
    ema_decay: float = 0.99

    # Mixed precision
    precision: Literal["no", "fp16", "bf16"] = "bf16"
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"

    # Gradient clipping
    max_grad_norm: float = 1.0

    # ZClip (adaptive gradient clipping)
    zclip_threshold: float = 2.5
    zclip_ema_decay: float = 0.99

    # VRAM optimization
    cpu_checkpoint_interval: int = 100
    cpu_checkpoint_spike_threshold: float = 5.0

    def __post_init__(self):
        """Sync precision and mixed_precision fields."""
        if self.precision != "bf16":
            self.mixed_precision = self.precision

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingConfig":
        """Create TrainingConfig from dictionary."""
        if not data:
            return cls()

        # Handle stepped_cosine_restart nested config
        stepped_config = data.pop('stepped_cosine_restart', None)
        if stepped_config and isinstance(stepped_config, dict):
            data['stepped_cosine_restart'] = SteppedCosineRestartConfig.from_dict(stepped_config)

        # Filter out unknown keys
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}

        return cls(**filtered_data)
