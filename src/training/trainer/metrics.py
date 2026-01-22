"""
Trainer Metrics and State

Contains:
    - TrainerState: Training state dataclass
    - TrainMetrics: Training metrics dataclass
    - MetricsLogger: Metrics logging utility

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainerState:
    """Training state."""
    step: int = 0
    epoch: int = 0
    batch_idx: int = 0
    best_loss: float = float("inf")
    total_samples: int = 0


@dataclass
class TrainMetrics:
    """Training metrics."""
    loss: float
    loss_vloss: float = 0.0
    loss_freq: float = 0.0
    loss_repa: float = 0.0
    loss_gamma_l2: float = 0.0
    grad_norm: float = 0.0
    learning_rate: float = 0.0
    samples_per_sec: float = 0.0
    step_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "loss": self.loss,
            "loss_vloss": self.loss_vloss,
            "loss_freq": self.loss_freq,
            "loss_repa": self.loss_repa,
            "loss_gamma_l2": self.loss_gamma_l2,
            "grad_norm": self.grad_norm,
            "learning_rate": self.learning_rate,
            "samples_per_sec": self.samples_per_sec,
            "step_time": self.step_time,
        }


class MetricsLogger:
    """Utility for logging training metrics."""

    def __init__(self, log_interval: int = 100):
        """Initialize metrics logger.

        Args:
            log_interval: Steps between log outputs
        """
        self.log_interval = log_interval
        self._history: List[TrainMetrics] = []

    def log(
        self,
        step: int,
        total_steps: int,
        metrics: TrainMetrics,
        force: bool = False,
    ) -> None:
        """Log metrics if at log interval.

        Args:
            step: Current step
            total_steps: Total steps
            metrics: Training metrics
            force: Force log regardless of interval
        """
        self._history.append(metrics)

        if not force and self.log_interval <= 0:
            return
        if not force and step % self.log_interval != 0:
            return

        logger.info(
            f"Step {step}/{total_steps} | "
            f"Loss: {metrics.loss:.4f} "
            f"(v:{metrics.loss_vloss:.3f} f:{metrics.loss_freq:.3f} "
            f"r:{metrics.loss_repa:.3f} g:{metrics.loss_gamma_l2:.4f}) | "
            f"LR: {metrics.learning_rate:.2e}"
        )

    def get_history(self) -> List[TrainMetrics]:
        """Get metrics history."""
        return self._history.copy()

    def clear_history(self) -> None:
        """Clear metrics history."""
        self._history.clear()


__all__ = [
    "TrainerState",
    "TrainMetrics",
    "MetricsLogger",
]
