"""
Resume Configuration for PixelHDM-RPEA-DinoV3.

This module defines the ResumeConfig dataclass for training resumption settings.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ResumeConfig:
    """Resume configuration for training checkpoint restoration.

    Attributes:
        enabled: Whether to resume from checkpoint.
        checkpoint_path: Path to checkpoint file, or "auto" for latest.
        reset_optimizer: Reset optimizer state when resuming.
        reset_scheduler: Reset learning rate scheduler when resuming.
    """

    enabled: bool = False
    checkpoint_path: Optional[str] = None
    reset_optimizer: bool = False
    reset_scheduler: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "ResumeConfig":
        """Create ResumeConfig from dictionary."""
        return cls(**data) if data else cls()
