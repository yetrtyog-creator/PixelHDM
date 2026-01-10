"""
iREPA Loss (Legacy Compatibility)

This module re-exports from the new modular structure.
For new code, import directly from training.losses.repa.
"""

from .repa import (
    REPALoss,
    REPALossWithProjector,
    create_repa_loss,
    create_repa_loss_from_config,
)

__all__ = [
    "REPALoss",
    "REPALossWithProjector",
    "create_repa_loss",
    "create_repa_loss_from_config",
]
