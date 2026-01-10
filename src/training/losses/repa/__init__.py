"""
iREPA: Improved Representation Alignment Loss Module

Uses DINOv3 feature alignment to accelerate training.
"""

from .core import REPALoss
from .with_projector import REPALossWithProjector
from .factory import create_repa_loss, create_repa_loss_from_config

__all__ = [
    "REPALoss",
    "REPALossWithProjector",
    "create_repa_loss",
    "create_repa_loss_from_config",
]
