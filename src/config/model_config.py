"""
PixelHDM-RPEA-DinoV3 Configuration (Backward Compatibility Module)

This module re-exports all configuration classes from the refactored modules
to maintain backward compatibility with existing code.

DEPRECATED: Import directly from src.config instead.

    # Old (still works):
    from src.config.model_config import Config, PixelHDMConfig

    # New (recommended):
    from src.config import Config, PixelHDMConfig

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

# Re-export all configuration classes for backward compatibility
from .data_config import DataConfig
from .loader import Config
from .pixelhdm_config import PixelHDMConfig
from .resume_config import ResumeConfig
from .training_config import TrainingConfig

__all__ = [
    "Config",
    "DataConfig",
    "PixelHDMConfig",
    "ResumeConfig",
    "TrainingConfig",
]
