"""
Optimizer Step Logic

Contains optimizer update and gradient management methods.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import torch

if TYPE_CHECKING:
    from torch.cuda.amp import GradScaler
    from ..optimization import ZClip, EMA, CPUMemoryCheckpoint


class OptimizerStepMixin:
    """
    Mixin for optimizer step logic.

    Handles gradient clipping, optimizer updates, and learning rate scheduling.

    After Phase C refactor, step is always an optimizer step (not batch step).
    The accumulation check has been moved to the loop/core level.

    Requires host class to have:
        - self.optimizer: torch.optim.Optimizer
        - self.scaler: Optional[GradScaler]
        - self.model: nn.Module
        - self.zclip: ZClip
        - self.cpu_checkpoint: Optional[CPUMemoryCheckpoint]
        - self.ema: Optional[EMA]
        - self.max_grad_norm: float
    """

    def _optimizer_step(
        self,
        step: int,
        loss_value: float,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        warmup_fn: Optional[Callable[[int], None]],
        warmup_steps: int,
    ) -> float:
        """Execute optimizer step. Always steps (no accumulation check).

        Args:
            step: Current optimizer step (NOT batch step).
            loss_value: Average loss for the accumulation window.
            lr_scheduler: LR scheduler (expects optimizer steps).
            warmup_fn: Warmup function (called with optimizer step).
            warmup_steps: Number of warmup steps (in optimizer steps).

        Returns:
            Gradient norm after clipping.
        """
        from ..optimization import clip_grad_norm_with_zclip

        spike_restored = self._check_loss_spike(loss_value, step)
        if spike_restored:
            self.optimizer.zero_grad()
            return 0.0

        grad_norm = self._clip_and_step(clip_grad_norm_with_zclip)
        self._update_lr(step, lr_scheduler, warmup_fn, warmup_steps)

        return grad_norm

    def _check_loss_spike(self, loss_value: float, step: int) -> bool:
        """Check for loss spike and restore if needed.

        Also saves/restores GradScaler state for fp16 training stability.
        """
        if self.cpu_checkpoint is None:
            return False
        return self.cpu_checkpoint.step(
            self.model, self.optimizer, self.ema, loss_value, step,
            scaler=self.scaler
        )

    def _clip_and_step(self, clip_fn: callable) -> float:
        """Clip gradients and take optimizer step."""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        grad_norm = clip_fn(self.model, self.zclip, max_norm=self.max_grad_norm)

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad()
        return grad_norm

    def _update_lr(
        self,
        step: int,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        warmup_fn: Optional[Callable[[int], None]],
        warmup_steps: int,
    ) -> None:
        """Update learning rate.

        Args:
            step: Current optimizer step (already in optimizer-step units).
            lr_scheduler: LR scheduler.
            warmup_fn: Warmup function.
            warmup_steps: Number of warmup steps (in optimizer steps).
        """
        if step < warmup_steps and warmup_fn is not None:
            warmup_fn(step)
        elif lr_scheduler is not None:
            lr_scheduler.step()

    def _update_ema(self, step: int) -> None:
        """Update EMA weights. Always updates (called once per optimizer step)."""
        if self.ema is not None:
            self.ema.update(self.model, step)


__all__ = ["OptimizerStepMixin"]
