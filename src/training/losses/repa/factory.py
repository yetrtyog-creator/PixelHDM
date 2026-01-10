"""
Factory Functions for REPA Loss
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .core import REPALoss

if TYPE_CHECKING:
    from ....config.model_config import PixelHDMConfig


def create_repa_loss(
    hidden_dim: int = 1024,
    dino_dim: int = 768,
    lambda_repa: float = 0.5,
    early_stop_step: int = 250000,
) -> REPALoss:
    """Create REPA Loss."""
    return REPALoss(
        config=None,
        hidden_dim=hidden_dim,
        dino_dim=dino_dim,
        lambda_repa=lambda_repa,
        early_stop_step=early_stop_step,
    )


def create_repa_loss_from_config(
    config: "PixelHDMConfig",
) -> REPALoss:
    """Create REPA Loss from config."""
    return REPALoss(config=config)


__all__ = [
    "create_repa_loss",
    "create_repa_loss_from_config",
]
