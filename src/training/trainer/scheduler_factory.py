"""
Learning Rate Scheduler Factory

Contains factory functions for creating LR schedulers.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-04
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional
import logging

import torch

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from ...config.training_config import TrainingConfig, SteppedCosineRestartConfig

from ..optimization import CosineAnnealingWarmRestartsWithDecay, SteppedCosineRestartScheduler

logger = logging.getLogger(__name__)


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    training_config: Optional["TrainingConfig"],
    dataloader: Optional["DataLoader"] = None,
    gradient_accumulation_steps: int = 1,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create learning rate scheduler.

    Priority order:
        1. stepped_cosine_restart (if enabled)
        2. lr_scheduler type (cosine, cosine_restart, etc.)

    Args:
        optimizer: Optimizer instance
        training_config: Training configuration
        dataloader: DataLoader for calculating steps per epoch
        gradient_accumulation_steps: Gradient accumulation steps

    Returns:
        LR scheduler instance
    """
    if training_config is None:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    # Check for stepped_cosine_restart first (takes priority)
    stepped_config = getattr(training_config, 'stepped_cosine_restart', None)
    if stepped_config and getattr(stepped_config, 'enabled', False):
        return _create_stepped_cosine_restart_scheduler(
            optimizer, stepped_config, training_config, dataloader,
            gradient_accumulation_steps
        )

    # Fall back to standard scheduler types
    scheduler_type = training_config.lr_scheduler
    warmup_steps = training_config.warmup_steps
    max_steps = training_config.max_steps
    min_lr = training_config.min_lr

    if scheduler_type == "cosine":
        return _create_cosine_scheduler(optimizer, max_steps, warmup_steps, min_lr)
    elif scheduler_type == "cosine_restart":
        return _create_cosine_restart_scheduler(
            optimizer, training_config, dataloader, gradient_accumulation_steps
        )
    else:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)


def _create_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    max_steps: int,
    warmup_steps: int,
    min_lr: float,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """Create cosine annealing scheduler."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, max_steps - warmup_steps),
        eta_min=min_lr,
    )


def _create_cosine_restart_scheduler(
    optimizer: torch.optim.Optimizer,
    training_config: "TrainingConfig",
    dataloader: Optional["DataLoader"],
    gradient_accumulation_steps: int,
) -> CosineAnnealingWarmRestartsWithDecay:
    """Create cosine annealing with warm restarts scheduler."""
    restart_epochs = getattr(training_config, 'restart_epochs', 4)
    restart_period = getattr(training_config, 'restart_period', 0)
    lr_decay_per_cycle = getattr(training_config, 'lr_decay_per_cycle', 1.0)
    min_lr = training_config.min_lr

    t_0 = _calculate_t0(
        restart_period, restart_epochs, training_config, dataloader,
        gradient_accumulation_steps
    )

    logger.info(
        f"CosineAnnealingWarmRestartsWithDecay: T_0={t_0}, "
        f"lr_decay={lr_decay_per_cycle}"
    )

    return CosineAnnealingWarmRestartsWithDecay(
        optimizer,
        T_0=max(1, t_0),
        T_mult=1,
        eta_min=min_lr,
        lr_decay_per_cycle=lr_decay_per_cycle,
    )


def _create_stepped_cosine_restart_scheduler(
    optimizer: torch.optim.Optimizer,
    stepped_config: "SteppedCosineRestartConfig",
    training_config: "TrainingConfig",
    dataloader: Optional["DataLoader"],
    gradient_accumulation_steps: int,
) -> SteppedCosineRestartScheduler:
    """Create stepped cosine restart scheduler.

    Both peak and trough learning rates decay together after each cycle.

    Args:
        optimizer: Optimizer instance
        stepped_config: Stepped cosine restart configuration
        training_config: Training configuration (for restart_epochs, etc.)
        dataloader: DataLoader for calculating steps per epoch
        gradient_accumulation_steps: Gradient accumulation steps

    Returns:
        SteppedCosineRestartScheduler instance
    """
    restart_epochs = getattr(training_config, 'restart_epochs', 4)
    restart_period = getattr(training_config, 'restart_period', 0)

    t_0 = _calculate_t0(
        restart_period, restart_epochs, training_config, dataloader,
        gradient_accumulation_steps
    )

    logger.info(
        f"SteppedCosineRestartScheduler: T_0={t_0}, "
        f"base_lr={stepped_config.base_lr}, "
        f"cycle_min_lr={stepped_config.cycle_min_lr}, "
        f"decay_rate={stepped_config.decay_rate}, "
        f"global_min_lr={stepped_config.global_min_lr}"
    )

    # Warn about potential double warmup
    external_warmup = training_config.warmup_steps if training_config else 0
    internal_warmup = stepped_config.warmup_steps
    if external_warmup > 0 and internal_warmup > 0:
        logger.warning(
            f"Double warmup detected: external warmup_steps={external_warmup}, "
            f"scheduler warmup_steps={internal_warmup}. This will cause LR to "
            f"warmup twice. Recommend setting one of them to 0."
        )

    return SteppedCosineRestartScheduler(
        optimizer,
        T_0=max(1, t_0),
        base_lr=stepped_config.base_lr,
        cycle_min_lr=stepped_config.cycle_min_lr,
        decay_rate=stepped_config.decay_rate,
        global_min_lr=stepped_config.global_min_lr,
        warmup_steps=stepped_config.warmup_steps,
        T_mult=1,
    )


def _calculate_t0(
    restart_period: int,
    restart_epochs: int,
    training_config: "TrainingConfig",
    dataloader: Optional["DataLoader"],
    gradient_accumulation_steps: int,
) -> int:
    """Calculate T_0 for cosine restart scheduler.

    Priority:
        1. restart_period > 0: Use hardcoded step count
        2. num_cycles > 0: T_0 = total_optimizer_steps / num_cycles
        3. Otherwise: T_0 = steps_per_epoch * restart_epochs

    Args:
        restart_period: Hardcoded steps per cycle (overrides everything if > 0)
        restart_epochs: Epochs per cycle (legacy, used if num_cycles == 0)
        training_config: Training config with num_cycles, num_epochs, etc.
        dataloader: DataLoader for calculating steps_per_epoch
        gradient_accumulation_steps: Kept for API compatibility. T_0 no longer
            scales by accumulation; epoch step count is fixed to dataloader length.
    """
    if restart_period > 0:
        logger.info(f"T_0 calculation: using hardcoded restart_period={restart_period}")
        return restart_period

    num_epochs = getattr(training_config, 'num_epochs', 16)
    num_cycles = getattr(training_config, 'num_cycles', 0)
    max_steps = training_config.max_steps

    if dataloader is not None:
        steps_per_epoch = len(dataloader)
    else:
        steps_per_epoch = max(1, max_steps // num_epochs) if num_epochs > 0 else max_steps

    optimizer_steps_per_epoch = max(1, steps_per_epoch)

    # Priority: num_cycles > restart_epochs
    if num_cycles > 0:
        # Calculate total optimizer steps, then divide by num_cycles
        total_optimizer_steps = optimizer_steps_per_epoch * num_epochs
        t_0 = total_optimizer_steps // num_cycles
        if t_0 == 0:
            t_0 = 1
            logger.warning("T_0 was 0 after num_cycles division, set to 1")

        logger.info(
            f"T_0 calculation (num_cycles mode): "
            f"total_opt_steps={total_optimizer_steps}, "
            f"num_cycles={num_cycles} "
            f"=> T_0={t_0} (cycle length), "
            f"epochs_per_cycle={num_epochs // num_cycles}"
        )
    else:
        # Legacy mode: restart_epochs = epochs per cycle
        t_0 = optimizer_steps_per_epoch * restart_epochs

        logger.info(
            f"T_0 calculation (restart_epochs mode): "
            f"len(dataloader)={steps_per_epoch}, "
            f"opt_steps/epoch={optimizer_steps_per_epoch}, "
            f"restart_epochs={restart_epochs} "
            f"=> T_0={t_0} (cycle length)"
        )

    return t_0


def apply_warmup_lr(
    optimizer: torch.optim.Optimizer,
    step: int,
    warmup_steps: int,
    base_lr: float,
) -> None:
    """Apply linear warmup to learning rate.

    Respects per-group lr_scale by using each group's initial_lr as the
    warmup target instead of overwriting with a global base_lr.

    Args:
        optimizer: Optimizer instance
        step: Current step (should be optimizer steps, not batch steps)
        warmup_steps: Total warmup steps (in optimizer steps)
        base_lr: Base learning rate (used only if initial_lr not set)
    """
    if warmup_steps <= 0:
        return

    if step < warmup_steps:
        warmup_factor = (step + 1) / warmup_steps
        for param_group in optimizer.param_groups:
            # Use group's initial_lr if set (preserves lr_scale), else use base_lr
            target_lr = param_group.get("initial_lr", base_lr)
            param_group["lr"] = target_lr * warmup_factor
    elif step == warmup_steps:
        for param_group in optimizer.param_groups:
            # Use group's initial_lr if set, else use base_lr
            target_lr = param_group.get("initial_lr", base_lr)
            param_group["lr"] = target_lr
            if "initial_lr" not in param_group:
                param_group["initial_lr"] = base_lr


__all__ = [
    "create_lr_scheduler",
    "apply_warmup_lr",
]
