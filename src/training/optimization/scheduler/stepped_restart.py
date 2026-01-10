"""
Stepped Cosine Restart Scheduler

A learning rate scheduler where both peak and trough values decay together
with each restart cycle, providing more aggressive and controlled decay.

Key difference from CosineAnnealingWarmRestartsWithDecay:
- Standard: Only peak decays, trough (eta_min) is fixed
- Stepped: Both peak AND trough decay by the same rate

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-04
"""

from __future__ import annotations

import math
from typing import Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class SteppedCosineRestartScheduler(LRScheduler):
    """
    Stepped Cosine Restart Learning Rate Scheduler.

    Both peak and trough learning rates decay together after each cycle,
    providing more aggressive and controlled learning rate reduction.

    Learning rate schedule:
        - Within each cycle: Cosine decay from cycle_peak to cycle_min
        - After each cycle:
            cycle_peak *= decay_rate
            cycle_min *= decay_rate
        - Both values respect global_min_lr as absolute lower bound

    Formula:
        In cycle n (n=0,1,2...):
            cycle_peak = max(global_min_lr, base_lr * (decay_rate ^ n))
            cycle_min = max(global_min_lr, cycle_min_lr * (decay_rate ^ n))
            progress = T_cur / T_i
            lr = cycle_min + (cycle_peak - cycle_min) * 0.5 * (1 + cos(pi * progress))

    Example:
        >>> scheduler = SteppedCosineRestartScheduler(
        ...     optimizer, T_0=1000,
        ...     base_lr=2e-4, cycle_min_lr=1.2e-4,
        ...     decay_rate=0.75, global_min_lr=1e-5,
        ... )
        >>> # Cycle 0: 2e-4 -> 1.2e-4 -> restart
        >>> # Cycle 1: 1.5e-4 -> 0.9e-4 -> restart
        >>> # Cycle 2: 1.125e-4 -> 0.675e-4 -> restart

    Args:
        optimizer: Optimizer instance
        T_0: Steps in first cycle
        base_lr: Initial peak learning rate
        cycle_min_lr: Initial trough learning rate (per-cycle minimum)
        decay_rate: Decay multiplier for both peak and trough per cycle (0 < decay <= 1)
        global_min_lr: Absolute minimum learning rate (lower bound)
        warmup_steps: Number of warmup steps (linear warmup)
        T_mult: Cycle length multiplier (default 1 = fixed length)
        last_epoch: Last epoch for resuming
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        base_lr: float,
        cycle_min_lr: float,
        decay_rate: float = 0.75,
        global_min_lr: float = 1e-5,
        warmup_steps: int = 0,
        T_mult: int = 1,
        last_epoch: int = -1,
    ) -> None:
        # Validate parameters
        if T_0 <= 0:
            raise ValueError(f"T_0 must be positive, got {T_0}")
        if T_mult < 1:
            raise ValueError(f"T_mult must be >= 1, got {T_mult}")
        if base_lr <= 0:
            raise ValueError(f"base_lr must be positive, got {base_lr}")
        if cycle_min_lr <= 0:
            raise ValueError(f"cycle_min_lr must be positive, got {cycle_min_lr}")
        if cycle_min_lr > base_lr:
            raise ValueError(
                f"cycle_min_lr ({cycle_min_lr}) must be <= base_lr ({base_lr})"
            )
        if not 0 < decay_rate <= 1:
            raise ValueError(f"decay_rate must be in (0, 1], got {decay_rate}")
        if global_min_lr < 0:
            raise ValueError(f"global_min_lr must be non-negative, got {global_min_lr}")
        if global_min_lr > cycle_min_lr:
            raise ValueError(
                f"global_min_lr ({global_min_lr}) must be <= cycle_min_lr ({cycle_min_lr})"
            )
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")

        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_lr = base_lr
        self.cycle_min_lr = cycle_min_lr
        self.decay_rate = decay_rate
        self.global_min_lr = global_min_lr
        self.warmup_steps = warmup_steps

        # Track current cycle state
        self.T_cur = -1  # Initialize to -1, parent __init__ calls step() making it 0
        self.T_i = T_0   # Current cycle total steps
        self.cycle = 0   # Current cycle number
        self.total_steps = 0  # Total steps including warmup

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Calculate current learning rate."""
        # During warmup: linear increase from 0 to base_lr
        if self.total_steps < self.warmup_steps:
            warmup_factor = (self.total_steps + 1) / self.warmup_steps
            return [self.base_lr * warmup_factor for _ in self.base_lrs]

        # After warmup: stepped cosine decay
        decay_factor = self.decay_rate ** self.cycle
        cycle_peak = max(self.global_min_lr, self.base_lr * decay_factor)
        cycle_min = max(self.global_min_lr, self.cycle_min_lr * decay_factor)

        # Ensure cycle_min <= cycle_peak after decay
        if cycle_min > cycle_peak:
            cycle_min = cycle_peak

        progress = max(0, self.T_cur) / self.T_i
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = cycle_min + (cycle_peak - cycle_min) * cosine_factor

        return [lr for _ in self.base_lrs]

    def step(self, epoch: Optional[int] = None) -> None:
        """Update learning rate (override to track cycles)."""
        if epoch is None:
            epoch = self.last_epoch + 1
            self.total_steps += 1

            # Only advance cycle counter after warmup completes
            if self.total_steps > self.warmup_steps:
                self.T_cur = self.T_cur + 1
                if self.T_cur >= self.T_i:
                    self.cycle += 1
                    self.T_cur = 0
                    self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError(f"epoch must be non-negative, got {epoch}")
            self._compute_state_from_epoch(epoch)

        self.last_epoch = epoch
        self._last_lr = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, self._last_lr):
            param_group['lr'] = lr

    def _compute_state_from_epoch(self, epoch: int) -> None:
        """Compute cycle state from given epoch."""
        self.total_steps = epoch

        if epoch <= self.warmup_steps:
            self.cycle = 0
            self.T_cur = 0
            self.T_i = self.T_0
            return

        # Steps after warmup
        steps_after_warmup = epoch - self.warmup_steps

        if self.T_mult == 1:
            self.cycle = steps_after_warmup // self.T_0
            self.T_cur = steps_after_warmup % self.T_0
            self.T_i = self.T_0
        else:
            total = 0
            self.cycle = 0
            self.T_i = self.T_0
            while total + self.T_i <= steps_after_warmup:
                total += self.T_i
                self.cycle += 1
                self.T_i *= self.T_mult
            self.T_cur = steps_after_warmup - total

    def get_cycle_lrs(self) -> tuple[float, float]:
        """Get current cycle's peak and trough learning rates.

        Returns:
            Tuple of (cycle_peak, cycle_min) after decay
        """
        decay_factor = self.decay_rate ** self.cycle
        cycle_peak = max(self.global_min_lr, self.base_lr * decay_factor)
        cycle_min = max(self.global_min_lr, self.cycle_min_lr * decay_factor)
        if cycle_min > cycle_peak:
            cycle_min = cycle_peak
        return cycle_peak, cycle_min

    def state_dict(self) -> dict:
        """Save scheduler state."""
        state = super().state_dict()
        state.update({
            'T_cur': self.T_cur,
            'T_i': self.T_i,
            'cycle': self.cycle,
            'total_steps': self.total_steps,
        })
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        """Load scheduler state."""
        self.T_cur = state_dict.pop('T_cur', 0)
        self.T_i = state_dict.pop('T_i', self.T_0)
        self.cycle = state_dict.pop('cycle', 0)
        self.total_steps = state_dict.pop('total_steps', 0)
        super().load_state_dict(state_dict)

        # Update optimizer LR after loading state
        self._last_lr = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, self._last_lr):
            param_group['lr'] = lr


__all__ = ["SteppedCosineRestartScheduler"]
