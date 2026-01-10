"""
Configuration Loader for PixelHDM-RPEA-DinoV3.

This module provides the Config class and YAML parsing utilities.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .data_config import DataConfig
from .pixelhdm_config import PixelHDMConfig
from .resume_config import ResumeConfig
from .training_config import TrainingConfig
from .parsers import parse_model_config, parse_training_config, parse_data_config


@dataclass
class Config:
    """Complete configuration container.

    Attributes:
        model: PixelHDM model architecture configuration.
        training: Training hyperparameters configuration.
        data: Dataset and dataloader configuration.
        resume: Checkpoint resume configuration.
    """

    model: PixelHDMConfig = field(default_factory=PixelHDMConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    resume: ResumeConfig = field(default_factory=ResumeConfig)

    @classmethod
    def default(cls) -> "Config":
        """Create default configuration."""
        return cls()

    @classmethod
    def for_testing(cls) -> "Config":
        """Create minimal configuration for testing."""
        return cls(
            model=PixelHDMConfig.for_testing(),
            training=TrainingConfig(batch_size=2, max_steps=10, log_interval=1),
            data=DataConfig(image_size=64, num_workers=0),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file.

        Handles nested YAML structure and maps to flat dataclass fields.
        Supports `data_config: "path/to/data.yaml"` reference.

        Args:
            path: Path to YAML configuration file.

        Returns:
            Config instance with all settings applied.

        Raises:
            FileNotFoundError: If config file does not exist.
        """
        import yaml

        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Handle external data_config reference
        data = _load_external_data_config(data, config_path)

        return _build_config_from_dict(data)


def _load_external_data_config(data: dict, config_path: Path) -> dict:
    """Load external data configuration if referenced.

    Args:
        data: Parsed YAML data dictionary.
        config_path: Path to main config file (for relative path resolution).

    Returns:
        Updated data dictionary with merged external config.
    """
    import yaml

    if "data_config" not in data:
        return data

    data_config_path = Path(data["data_config"])

    # Resolve relative paths based on main config directory
    if not data_config_path.is_absolute():
        data_config_path = config_path.parent / data_config_path

    if not data_config_path.exists():
        return data

    with open(data_config_path, "r", encoding="utf-8") as f:
        external_data = yaml.safe_load(f) or {}

    # Merge external sections (external file takes precedence)
    data["dataset"] = external_data.get("dataset", {})
    data["multi_resolution"] = external_data.get("multi_resolution", {})
    data["optimization"] = external_data.get("optimization", {})
    data.pop("data", None)  # Remove inline data section

    return data


def _build_config_from_dict(data: dict) -> Config:
    """Build Config from dictionary, handling nested structure.

    Args:
        data: Parsed YAML data dictionary.

    Returns:
        Fully constructed Config instance.
    """
    model = parse_model_config(data)
    training = parse_training_config(data)
    data_config = parse_data_config(data)
    resume = ResumeConfig.from_dict(data.get("resume", {}))

    return Config(model=model, training=training, data=data_config, resume=resume)


__all__ = ["Config"]
