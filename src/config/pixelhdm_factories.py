"""
PixelHDM Configuration Factory Methods and Serialization.

This module provides factory methods and serialization utilities for PixelHDMConfig.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pixelhdm_config import PixelHDMConfig


def create_default_config() -> "PixelHDMConfig":
    """Create default configuration (~332M trainable params)."""
    from .pixelhdm_config import PixelHDMConfig
    return PixelHDMConfig()


def create_small_config() -> "PixelHDMConfig":
    """Create small configuration for testing."""
    from .pixelhdm_config import PixelHDMConfig
    return PixelHDMConfig(
        hidden_dim=512,
        patch_layers=8,
        pixel_layers=1,
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
        text_hidden_size=512,
    )


def create_large_config() -> "PixelHDMConfig":
    """Create large configuration (~500M+ trainable params)."""
    from .pixelhdm_config import PixelHDMConfig
    return PixelHDMConfig(
        hidden_dim=1152,
        patch_layers=26,
        pixel_layers=4,
        num_heads=16,
        num_kv_heads=4,
        head_dim=72,
        text_hidden_size=1152,
        mrope_text_dim=16,
        mrope_img_h_dim=28,
        mrope_img_w_dim=28,
    )


def create_testing_config() -> "PixelHDMConfig":
    """Create minimal configuration for unit tests."""
    from .pixelhdm_config import PixelHDMConfig
    return PixelHDMConfig(
        hidden_dim=256,
        pixel_dim=8,
        patch_size=16,
        patch_layers=2,
        pixel_layers=1,
        num_heads=4,
        num_kv_heads=2,
        head_dim=64,
        mlp_ratio=2.0,
        bottleneck_dim=64,
        text_hidden_size=256,
        repa_enabled=False,
        freq_loss_enabled=False,
        use_flash_attention=False,
        use_gradient_checkpointing=False,
        repa_align_layer=1,
        zero_init_output=False,
    )


def config_to_dict(config: "PixelHDMConfig") -> dict:
    """Convert PixelHDMConfig to dictionary."""
    return {
        "hidden_dim": config.hidden_dim,
        "pixel_dim": config.pixel_dim,
        "patch_size": config.patch_size,
        "patch_layers": config.patch_layers,
        "pixel_layers": config.pixel_layers,
        "num_heads": config.num_heads,
        "num_kv_heads": config.num_kv_heads,
        "head_dim": config.head_dim,
        "mlp_ratio": config.mlp_ratio,
        "mlp_type": config.mlp_type,
        "bottleneck_dim": config.bottleneck_dim,
        "in_channels": config.in_channels,
        "out_channels": config.out_channels,
        "max_resolution": config.max_resolution,
        "time_convention": config.time_convention,
        "prediction_type": config.prediction_type,
        "time_p_mean": config.time_p_mean,
        "time_p_std": config.time_p_std,
        "time_eps": config.time_eps,
        "repa_enabled": config.repa_enabled,
        "repa_encoder": config.repa_encoder,
        "repa_local_path": config.repa_local_path,
        "repa_use_bf16": config.repa_use_bf16,
        "repa_hidden_size": config.repa_hidden_size,
        "repa_patch_size": config.repa_patch_size,
        "repa_align_layer": config.repa_align_layer,
        "repa_lambda": config.repa_lambda,
        "repa_early_stop": config.repa_early_stop,
        "freq_loss_enabled": config.freq_loss_enabled,
        "freq_loss_quality": config.freq_loss_quality,
        "freq_loss_lambda": config.freq_loss_lambda,
        "freq_loss_block_size": config.freq_loss_block_size,
        "freq_loss_use_ycbcr": config.freq_loss_use_ycbcr,
        "text_encoder_name": config.text_encoder_name,
        "text_encoder_frozen": config.text_encoder_frozen,
        "text_max_length": config.text_max_length,
        "text_hidden_size": config.text_hidden_size,
        "norm_type": config.norm_type,
        "norm_eps": config.norm_eps,
        "dropout": config.dropout,
        "attention_dropout": config.attention_dropout,
        "cfg_dropout": config.cfg_dropout,
        "use_flash_attention": config.use_flash_attention,
        "use_gradient_checkpointing": config.use_gradient_checkpointing,
        "zero_init_output": config.zero_init_output,
        "mrope_text_dim": config.mrope_text_dim,
        "mrope_img_h_dim": config.mrope_img_h_dim,
        "mrope_img_w_dim": config.mrope_img_w_dim,
        "mrope_text_max_len": config.mrope_text_max_len,
        "mrope_img_max_len": config.mrope_img_max_len,
        "mrope_theta": config.mrope_theta,
        "time_embed_dim": config.time_embed_dim,
        "max_patches": config.max_patches,
        "token_compaction_expand_gain": config.token_compaction_expand_gain,
        "gate_type": config.gate_type,
        "gate_activation": config.gate_activation,
        "gate_bias": config.gate_bias,
        "adaln_num_params": config.adaln_num_params,
        "adaln_init_gain": config.adaln_init_gain,
        "default_num_steps": config.default_num_steps,
        "default_guidance_scale": config.default_guidance_scale,
        "default_sampler_method": config.default_sampler_method,
    }


def config_from_dict(config_dict: dict) -> "PixelHDMConfig":
    """Create PixelHDMConfig from dictionary."""
    from .pixelhdm_config import PixelHDMConfig
    return PixelHDMConfig(**config_dict)


def config_from_json(json_path: str) -> "PixelHDMConfig":
    """Create PixelHDMConfig from JSON file."""
    import json
    with open(json_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    return config_from_dict(config_dict)


def config_to_json(config: "PixelHDMConfig", json_path: str) -> None:
    """Save PixelHDMConfig to JSON file."""
    import json
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(config_to_dict(config), f, indent=2, ensure_ascii=False)
