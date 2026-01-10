"""
Cosine Annealing Warm Restarts with Decay

Scheduler with periodic restarts and optional learning rate decay per cycle.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

import math
from typing import Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class CosineAnnealingWarmRestartsWithDecay(LRScheduler):
    """
    Cosine annealing scheduler with warm restarts and cycle decay.

    Learning rate schedule:
        - Within each cycle: Cosine decay from peak_lr to eta_min
        - After each cycle: peak_lr *= lr_decay_per_cycle
        - Decayed peak_lr never goes below eta_min

    Formula:
        - In cycle n (n=0,1,2...):
          peak_lr = max(eta_min, base_lr * (lr_decay_per_cycle ^ n))
        - Within cycle: lr = eta_min + 0.5 * (peak_lr - eta_min) * (1 + cos(pi * progress))

    Args:
        optimizer: Optimizer instance
        T_0: Steps in first cycle
        T_mult: Cycle length multiplier (default 1 = fixed length)
        eta_min: Minimum learning rate
        lr_decay_per_cycle: Peak LR decay multiplier per cycle (0 < decay <= 1)
        last_epoch: Last epoch for resuming

    Example:
        >>> scheduler = CosineAnnealingWarmRestartsWithDecay(
        ...     optimizer, T_0=1000, lr_decay_per_cycle=0.9, eta_min=1e-6,
        ... )
        >>> # Cycle 0: 1e-4 -> 1e-6 -> restart
        >>> # Cycle 1: 9e-5 -> 1e-6 -> restart
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        lr_decay_per_cycle: float = 1.0,
        last_epoch: int = -1,
    ) -> None:
        if T_0 <= 0:
            raise ValueError(f"T_0 must be positive, got {T_0}")
        if T_mult < 1:
            raise ValueError(f"T_mult must be >= 1, got {T_mult}")
        if eta_min < 0:
            raise ValueError(f"eta_min must be non-negative, got {eta_min}")
        if not 0 < lr_decay_per_cycle <= 1:
            raise ValueError(f"lr_decay_per_cycle must be in (0, 1], got {lr_decay_per_cycle}")

        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.lr_decay_per_cycle = lr_decay_per_cycle

        # Track current cycle state (note: parent __init__ calls step(), so T_cur becomes 1)
        self.T_cur = -1  # Initialize to -1, parent __init__ calls step() making it 0
        self.T_i = T_0   # Current cycle total steps
        self.cycle = 0   # Current cycle number

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Calculate current learning rate."""
        progress = max(0, self.T_cur) / self.T_i
        decay_factor = self.lr_decay_per_cycle ** self.cycle

        lrs = []
        for base_lr in self.base_lrs:
            peak_lr = max(self.eta_min, base_lr * decay_factor)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = self.eta_min + (peak_lr - self.eta_min) * cosine_factor
            lrs.append(lr)

        return lrs

    def step(self, epoch: Optional[int] = None) -> None:
        """Update learning rate (override to track cycles)."""
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = 0
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError(f"epoch must be non-negative, got {epoch}")
            self._compute_cycle_from_epoch(epoch)

        self.last_epoch = epoch
        self._last_lr = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, self._last_lr):
            param_group['lr'] = lr

    def _compute_cycle_from_epoch(self, epoch: int) -> None:
        """Compute cycle and T_cur from given epoch."""
        if self.T_mult == 1:
            self.cycle = epoch // self.T_0
            self.T_cur = epoch % self.T_0
            self.T_i = self.T_0
        else:
            total = 0
            self.cycle = 0
            self.T_i = self.T_0
            while total + self.T_i <= epoch:
                total += self.T_i
                self.cycle += 1
                self.T_i *= self.T_mult
            self.T_cur = epoch - total

    def state_dict(self) -> dict:
        """Save scheduler state."""
        state = super().state_dict()
        state.update({'T_cur': self.T_cur, 'T_i': self.T_i, 'cycle': self.cycle})
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        """Load scheduler state."""
        self.T_cur = state_dict.pop('T_cur', 0)
        self.T_i = state_dict.pop('T_i', self.T_0)
        self.cycle = state_dict.pop('cycle', 0)
        super().load_state_dict(state_dict)


__all__ = ["CosineAnnealingWarmRestartsWithDecay"]
