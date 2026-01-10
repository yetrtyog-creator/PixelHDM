"""
Factory Functions for Frequency Loss
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .core import FrequencyLoss

if TYPE_CHECKING:
    from ....config.model_config import PixelHDMConfig


def create_freq_loss(
    quality: int = 90,
    weight: float = 1.0,
    use_ycbcr: bool = True,
) -> FrequencyLoss:
    """Create frequency-aware loss."""
    return FrequencyLoss(
        config=None,
        quality=quality,
        weight=weight,
        use_ycbcr=use_ycbcr,
    )


def create_freq_loss_from_config(
    config: "PixelHDMConfig",
) -> FrequencyLoss:
    """Create frequency-aware loss from config."""
    return FrequencyLoss(config=config)


__all__ = [
    "create_freq_loss",
    "create_freq_loss_from_config",
]
