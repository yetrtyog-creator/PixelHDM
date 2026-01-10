"""
Configuration Validators for PixelHDM-RPEA-DinoV3.

This module provides validation functions for configuration dataclasses.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pixelhdm_config import PixelHDMConfig


def validate_pixelhdm_config(config: "PixelHDMConfig") -> None:
    """Validate PixelHDMConfig fields and update computed values.

    Args:
        config: PixelHDMConfig instance to validate.

    Raises:
        AssertionError: If validation fails.
        ValueError: If positive value constraints are violated.
    """
    _validate_positive_values(config)
    _validate_head_dim(config)
    _validate_gqa_ratio(config)
    _validate_text_hidden_size(config)
    _validate_mrope_dimensions(config)
    _validate_pixel_dimensions(config)


def _validate_positive_values(config: "PixelHDMConfig") -> None:
    """Validate that critical parameters have positive values."""
    positive_int_fields = [
        ("hidden_dim", config.hidden_dim),
        ("pixel_dim", config.pixel_dim),
        ("patch_size", config.patch_size),
        ("patch_layers", config.patch_layers),
        ("pixel_layers", config.pixel_layers),
        ("num_heads", config.num_heads),
        ("num_kv_heads", config.num_kv_heads),
        ("head_dim", config.head_dim),
        ("in_channels", config.in_channels),
        ("out_channels", config.out_channels),
        ("max_resolution", config.max_resolution),
        ("text_max_length", config.text_max_length),
        ("text_hidden_size", config.text_hidden_size),
        ("time_embed_dim", config.time_embed_dim),
        ("max_patches", config.max_patches),
        ("default_num_steps", config.default_num_steps),
    ]

    for name, value in positive_int_fields:
        if value <= 0:
            raise ValueError(
                f"Configuration error: {name} must be positive, got {value}"
            )

    positive_float_fields = [
        ("mlp_ratio", config.mlp_ratio),
        ("norm_eps", config.norm_eps),
        ("time_eps", config.time_eps),
        ("default_guidance_scale", config.default_guidance_scale),
        ("mrope_theta", config.mrope_theta),
    ]

    for name, value in positive_float_fields:
        if value <= 0:
            raise ValueError(
                f"Configuration error: {name} must be positive, got {value}"
            )

    non_negative_float_fields = [
        ("dropout", config.dropout),
        ("attention_dropout", config.attention_dropout),
        ("cfg_dropout", config.cfg_dropout),
    ]

    for name, value in non_negative_float_fields:
        if value < 0:
            raise ValueError(
                f"Configuration error: {name} must be non-negative, got {value}"
            )


def _validate_head_dim(config: "PixelHDMConfig") -> None:
    """Validate and update head_dim based on hidden_dim and num_heads."""
    assert config.hidden_dim % config.num_heads == 0, (
        f"hidden_dim ({config.hidden_dim}) must be divisible by "
        f"num_heads ({config.num_heads})"
    )

    computed_head_dim = config.hidden_dim // config.num_heads
    if config.head_dim != computed_head_dim:
        # Update head_dim to match computed value
        object.__setattr__(config, "head_dim", computed_head_dim)


def _validate_gqa_ratio(config: "PixelHDMConfig") -> None:
    """Validate GQA ratio (num_heads must be divisible by num_kv_heads)."""
    assert config.num_heads % config.num_kv_heads == 0, (
        f"num_heads ({config.num_heads}) must be divisible by "
        f"num_kv_heads ({config.num_kv_heads})"
    )


def _validate_text_hidden_size(config: "PixelHDMConfig") -> None:
    """Validate text_hidden_size configuration.

    Note:
        移除了強制 text_hidden_size == hidden_dim 的約束。
        當維度不匹配時，TextProjector 會自動進行維度對齊。
        這允許使用不同的 Qwen3 模型（如 1.7B 的 2048 維度）。
    """
    # 不再強制相等，TextProjector 會處理維度對齊
    # 只記錄警告信息供調試參考
    if config.text_hidden_size != config.hidden_dim:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(
            f"text_hidden_size ({config.text_hidden_size}) differs from "
            f"hidden_dim ({config.hidden_dim}). TextProjector will handle alignment."
        )


def _validate_mrope_dimensions(config: "PixelHDMConfig") -> None:
    """Validate mRoPE dimensions sum to head_dim."""
    mrope_total = (
        config.mrope_text_dim +
        config.mrope_img_h_dim +
        config.mrope_img_w_dim
    )
    assert mrope_total == config.head_dim, (
        f"mRoPE dimensions ({config.mrope_text_dim}+"
        f"{config.mrope_img_h_dim}+{config.mrope_img_w_dim}={mrope_total}) "
        f"must equal head_dim ({config.head_dim})"
    )


def _validate_pixel_dimensions(config: "PixelHDMConfig") -> None:
    """Validate and document pixel dimension relationship.

    Note:
        PixelHDM 設計允許 hidden_dim ≠ p² × pixel_dim。
        PixelUnpatchify 使用 Linear(hidden_dim → p² × pixel_dim) 進行特徵擴展。
        這是有意的設計選擇，不是約束違反。

        常見配置:
        - hidden_dim=1024, pixel_dim=16, patch_size=16
        - p² × pixel_dim = 256 × 16 = 4096
        - PixelUnpatchify: Linear(1024 → 4096) 進行 4× 特徵擴展
    """
    p2_times_dpix = config.patch_size ** 2 * config.pixel_dim

    if config.hidden_dim != p2_times_dpix:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(
            f"hidden_dim ({config.hidden_dim}) differs from "
            f"p² × pixel_dim ({config.patch_size}² × {config.pixel_dim} = {p2_times_dpix}). "
            f"PixelUnpatchify will use Linear projection for feature expansion."
        )
