"""
Configuration module for PixelHDM-RPEA-DinoV3.

This module provides unified configuration management for the complete T2I system.

Usage:
    from src.config import Config, PixelHDMConfig
    config = Config.from_yaml("configs/train_config.yaml")

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

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
