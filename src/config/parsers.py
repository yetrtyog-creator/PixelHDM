"""
Configuration Parsers

YAML parsing utilities for configuration loading.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .data_config import DataConfig
from .pixelhdm_config import PixelHDMConfig
from .training_config import TrainingConfig, SteppedCosineRestartConfig


def parse_model_config(data: dict) -> PixelHDMConfig:
    """Parse model configuration from YAML data."""
    model_data = dict(data.get("model", {}))

    # Merge flow_matching section into model config
    flow_matching = data.get("flow_matching", {})
    if flow_matching:
        if "P_mean" in flow_matching:
            model_data["time_p_mean"] = flow_matching["P_mean"]
        if "P_std" in flow_matching:
            model_data["time_p_std"] = flow_matching["P_std"]

    return PixelHDMConfig(**model_data) if model_data else PixelHDMConfig()


def parse_training_config(data: dict) -> TrainingConfig:
    """Parse training configuration from YAML data."""
    train_data = data.get("training", {})
    output_data = data.get("output", {})

    if not train_data:
        train_data = {}

    flat: dict[str, Any] = {}

    # Parse direct fields
    _parse_direct_fields(train_data, flat)

    # Parse nested sections
    _parse_optimizer_section(train_data, flat)
    _parse_gradient_section(train_data, flat)
    _parse_ema_section(train_data, flat)
    _parse_lr_schedule_section(train_data, flat)
    _parse_output_section(output_data, flat)

    return TrainingConfig(**flat)


def _parse_direct_fields(train_data: dict, flat: dict) -> None:
    """Parse top-level training fields."""
    direct_fields = [
        "learning_rate", "weight_decay", "precision", "mixed_precision",
        "batch_size", "max_steps", "log_interval", "save_interval",
        "eval_interval", "max_grad_norm",
    ]
    for key in direct_fields:
        if key in train_data:
            flat[key] = train_data[key]


def _parse_optimizer_section(train_data: dict, flat: dict) -> None:
    """Parse optimizer configuration section."""
    opt = train_data.get("optimizer", {})
    if isinstance(opt, dict):
        if "type" in opt:
            flat["optimizer"] = opt["type"]
        if "betas" in opt:
            flat["betas"] = tuple(opt["betas"])
        if "eps" in opt:
            flat["eps"] = opt["eps"]
    elif isinstance(opt, str):
        flat["optimizer"] = opt


def _parse_gradient_section(train_data: dict, flat: dict) -> None:
    """Parse gradient configuration section."""
    grad = train_data.get("gradient", {})
    if not isinstance(grad, dict):
        return

    if "accumulation_steps" in grad:
        flat["gradient_accumulation_steps"] = grad["accumulation_steps"]
    if "max_norm" in grad:
        flat["max_grad_norm"] = grad["max_norm"]
    if "zclip_threshold" in grad:
        flat["zclip_threshold"] = grad["zclip_threshold"]
    if "zclip_ema_decay" in grad:
        flat["zclip_ema_decay"] = grad["zclip_ema_decay"]


def _parse_ema_section(train_data: dict, flat: dict) -> None:
    """Parse EMA configuration section."""
    ema = train_data.get("ema", {})
    if not isinstance(ema, dict):
        return

    if "enabled" in ema:
        flat["ema_enabled"] = ema["enabled"]
    if "decay" in ema:
        flat["ema_decay"] = ema["decay"]


def _parse_lr_schedule_section(train_data: dict, flat: dict) -> None:
    """Parse learning rate schedule section."""
    lr = train_data.get("lr_schedule", {})
    if not isinstance(lr, dict):
        return

    field_mapping = {
        "schedule_type": "lr_scheduler",
        "warmup_steps": "warmup_steps",
        "min_lr": "min_lr",
        "restart_epochs": "restart_epochs",
        "restart_period": "restart_period",
        "lr_decay_per_cycle": "lr_decay_per_cycle",
        "num_epochs": "num_epochs",
    }

    for yaml_key, config_key in field_mapping.items():
        if yaml_key in lr:
            flat[config_key] = lr[yaml_key]

    # Handle training_mode
    training_mode = lr.get("training_mode", "epochs")
    flat["training_mode"] = training_mode

    # Only use total_steps in steps mode
    if training_mode == "steps" and "total_steps" in lr:
        flat["max_steps"] = lr["total_steps"]

    # Parse stepped_cosine_restart configuration
    stepped = lr.get("stepped_cosine_restart", {})
    if isinstance(stepped, dict) and stepped:
        flat["stepped_cosine_restart"] = SteppedCosineRestartConfig.from_dict(stepped)


def _parse_output_section(output_data: dict, flat: dict) -> None:
    """Parse output section for checkpoint/logging settings."""
    if not output_data:
        return

    output_mapping = {
        "save_interval": "save_interval",
        "log_interval": "log_interval",
        "save_every": "save_interval",  # Legacy alias
        "log_every": "log_interval",    # Legacy alias
        "save_every_epochs": "save_every_epochs",
        "log_every_epochs": "log_every_epochs",
        "max_checkpoints": "max_checkpoints",
        "checkpoint_dir": "checkpoint_dir",
    }

    for yaml_key, config_key in output_mapping.items():
        if yaml_key in output_data:
            flat[config_key] = output_data[yaml_key]


def parse_data_config(data: dict) -> DataConfig:
    """Parse data configuration from YAML data."""
    data_cfg: dict[str, Any] = dict(data.get("data", {}))
    dataset_cfg = data.get("dataset", {})
    multi_res_cfg = data.get("multi_resolution", {})

    # Merge dataset section
    _merge_dataset_section(dataset_cfg, data_cfg)

    # Merge multi_resolution section
    _merge_multi_resolution_section(multi_res_cfg, data_cfg)

    return DataConfig(**data_cfg) if data_cfg else DataConfig()


def _merge_dataset_section(dataset_cfg: dict, data_cfg: dict) -> None:
    """Merge dataset YAML section into data config."""
    if not dataset_cfg:
        return

    # Map train_dir to data_dir
    if "train_dir" in dataset_cfg:
        data_cfg.setdefault("data_dir", dataset_cfg["train_dir"])

    # Field mappings (DataConfig fields only, NOT batch_size)
    field_mapping = {
        "max_resolution": "max_bucket_size",
        "min_resolution": "min_bucket_size",
        "target_resolution": "image_size",
        "num_workers": "num_workers",
        "pin_memory": "pin_memory",
        "prefetch_factor": "prefetch_factor",
        "persistent_workers": "persistent_workers",
    }

    for src, dst in field_mapping.items():
        if src in dataset_cfg:
            data_cfg.setdefault(dst, dataset_cfg[src])

    # Handle augmentation sub-section
    aug = dataset_cfg.get("augmentation", {})
    if aug:
        if "use_random_crop" in aug:
            data_cfg.setdefault("use_random_crop", aug["use_random_crop"])
        if "use_random_flip" in aug:
            data_cfg.setdefault("random_flip", aug["use_random_flip"])


def _merge_multi_resolution_section(multi_res_cfg: dict, data_cfg: dict) -> None:
    """Merge multi_resolution YAML section into data config."""
    if not multi_res_cfg:
        return

    if multi_res_cfg.get("enabled", False):
        data_cfg.setdefault("use_bucketing", True)

    field_mapping = {
        "min_size": "min_bucket_size",
        "max_size": "max_bucket_size",
        "step_size": "bucket_step",
        "max_aspect_ratio": "max_aspect_ratio",
        "target_pixels": "target_pixels",
        "sampler_mode": "sampler_mode",
        "chunk_size": "chunk_size",
        "shuffle_chunks": "shuffle_chunks",
        "shuffle_within_bucket": "shuffle_within_bucket",
    }

    for src, dst in field_mapping.items():
        if src in multi_res_cfg:
            data_cfg.setdefault(dst, multi_res_cfg[src])


__all__ = [
    "parse_model_config",
    "parse_training_config",
    "parse_data_config",
]
