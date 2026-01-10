"""
Trainer Initialization Helpers

Helper functions for initializing Trainer components.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

if TYPE_CHECKING:
    from ...config.model_config import PixelHDMConfig, TrainingConfig

from ..flow_matching import PixelHDMFlowMatching
from ..losses import CombinedLoss
from ..optimization import create_optimizer, ZClip, EMA, CPUMemoryCheckpoint


def init_optimizer(
    model: nn.Module,
    training_config: Optional["TrainingConfig"],
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> torch.optim.Optimizer:
    """Initialize optimizer."""
    if optimizer is not None:
        return optimizer

    lr, weight_decay, betas, eps = get_optimizer_params(training_config)
    optimizer_type, use_8bit = get_optimizer_type(training_config, device)

    return create_optimizer(
        model, optimizer_type=optimizer_type, lr=lr,
        weight_decay=weight_decay, betas=betas, eps=eps, use_8bit=use_8bit,
    )


def get_optimizer_params(config: Optional["TrainingConfig"]):
    """Get optimizer parameters from config."""
    lr, weight_decay = 1e-4, 0.01
    betas, eps = (0.9, 0.999), 1e-8

    if config is not None:
        lr = config.learning_rate
        weight_decay = config.weight_decay
        betas = getattr(config, 'betas', betas)
        eps = getattr(config, 'eps', eps)

    return lr, weight_decay, betas, eps


def get_optimizer_type(config: Optional["TrainingConfig"], device: torch.device):
    """Get optimizer type from config."""
    optimizer_type, use_8bit = "adamw", False

    if config is not None and hasattr(config, 'optimizer'):
        opt_type = config.optimizer
        if opt_type == "adamw_8bit":
            optimizer_type = "adamw"
            use_8bit = (device.type == "cuda")
        else:
            optimizer_type = opt_type

    return optimizer_type, use_8bit


def init_training_components(
    config: Optional["PixelHDMConfig"],
    training_config: Optional["TrainingConfig"],
    model: nn.Module,
    device: torch.device,
) -> Tuple[PixelHDMFlowMatching, CombinedLoss, ZClip, Optional[EMA], Optional[CPUMemoryCheckpoint]]:
    """Initialize all training components."""
    flow_matching = PixelHDMFlowMatching(config=config).to(device)
    combined_loss = CombinedLoss(config=config).to(device)

    threshold, ema_decay = get_zclip_params(training_config)
    zclip = ZClip(threshold=threshold, ema_decay=ema_decay)

    ema = create_ema(training_config, model)
    cpu_checkpoint = create_cpu_checkpoint(training_config)

    return flow_matching, combined_loss, zclip, ema, cpu_checkpoint


def get_zclip_params(config: Optional["TrainingConfig"]) -> Tuple[float, float]:
    """Get ZClip parameters from config."""
    threshold, ema_decay = 2.5, 0.99
    if config is not None:
        threshold = getattr(config, 'zclip_threshold', 2.5)
        ema_decay = getattr(config, 'zclip_ema_decay', 0.99)
    return threshold, ema_decay


def create_ema(config: Optional["TrainingConfig"], model: nn.Module) -> Optional[EMA]:
    """Create EMA if enabled."""
    ema_decay = 0.9999
    ema_enabled = True

    if config is not None:
        ema_decay = config.ema_decay
        ema_enabled = getattr(config, 'ema_enabled', True)

    if ema_decay > 0 and ema_enabled:
        return EMA(model, decay=ema_decay)
    return None


def create_cpu_checkpoint(config: Optional["TrainingConfig"]) -> Optional[CPUMemoryCheckpoint]:
    """Create CPU checkpoint if enabled."""
    interval, threshold = 100, 5.0

    if config is not None:
        interval = getattr(config, 'cpu_checkpoint_interval', 100)
        threshold = getattr(config, 'cpu_checkpoint_spike_threshold', 5.0)

    if interval > 0:
        return CPUMemoryCheckpoint(save_interval=interval, spike_threshold=threshold)
    return None


def init_amp(config: Optional["TrainingConfig"]) -> Tuple[bool, torch.dtype, Optional[GradScaler]]:
    """Initialize automatic mixed precision."""
    precision = "bf16"
    if config is not None:
        precision = config.mixed_precision

    use_amp = precision in ["bf16", "fp16"]
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = GradScaler() if precision == "fp16" else None

    return use_amp, amp_dtype, scaler


def get_training_params(config: Optional["TrainingConfig"]) -> Tuple[float, int]:
    """Get training parameters from config."""
    max_grad_norm = 1.0
    gradient_accumulation_steps = 1

    if config is not None:
        max_grad_norm = config.max_grad_norm
        gradient_accumulation_steps = config.gradient_accumulation_steps

    return max_grad_norm, gradient_accumulation_steps


__all__ = [
    "init_optimizer",
    "get_optimizer_params",
    "get_optimizer_type",
    "init_training_components",
    "get_zclip_params",
    "create_ema",
    "create_cpu_checkpoint",
    "init_amp",
    "get_training_params",
]
