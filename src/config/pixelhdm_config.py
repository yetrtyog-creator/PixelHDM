"""
PixelHDM Model Configuration for PixelHDM-RPEA-DinoV3.

This module defines the PixelHDMConfig dataclass for model architecture settings.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class PixelHDMConfig:
    """Complete configuration for PixelHDM-RPEA-DinoV3 model architecture.

    See module docstring and CLAUDE.md for full parameter documentation.
    """

    # Core dimensions
    hidden_dim: int = 1024
    pixel_dim: int = 16
    patch_size: int = 16

    # Layer counts
    patch_layers: int = 16
    pixel_layers: int = 4

    # Attention configuration (GQA)
    num_heads: int = 16
    num_kv_heads: int = 4
    head_dim: int = 64

    # MLP configuration
    mlp_ratio: float = 3.0
    mlp_type: Literal["swiglu", "gelu"] = "swiglu"
    # Bottleneck dimension for PatchEmbedding
    # Default: patch_size^2 // 4 (e.g., 16^2 // 4 = 64 for patch_size=16)
    # Set explicitly to override automatic calculation
    bottleneck_dim: Optional[int] = None

    # Input/Output
    in_channels: int = 3
    out_channels: int = 3
    max_resolution: int = 1024

    # Time convention (PixelHDM style / Rectified Flow)
    time_convention: Literal["pixelhdm", "rectified"] = "pixelhdm"
    prediction_type: Literal["x", "v", "eps"] = "v"  # V-Prediction: v = x - noise
    time_p_mean: float = 0.0
    time_p_std: float = 1.0
    time_eps: float = 0.05

    # REPA Configuration
    repa_enabled: bool = True
    repa_encoder: str = "dinov3-vit-b"
    repa_local_path: Optional[str] = None
    repa_use_bf16: bool = True
    repa_hidden_size: int = 768
    repa_patch_size: int = 16
    repa_align_layer: int = 8
    repa_lambda: float = 0.5
    repa_early_stop: int = 250000

    # Frequency Loss Configuration
    freq_loss_enabled: bool = True
    freq_loss_quality: int = 90
    freq_loss_lambda: float = 1.0
    freq_loss_block_size: int = 8
    freq_loss_use_ycbcr: bool = True

    # Text Encoder Configuration
    text_encoder_name: str = "Qwen/Qwen3-0.6B"
    text_encoder_frozen: bool = True
    text_max_length: int = 511  # 512-1: image tokens use axis0=text_len, so max_seq_len=512
    text_hidden_size: int = 1024

    # Normalization
    norm_type: Literal["rmsnorm", "layernorm"] = "rmsnorm"
    norm_eps: float = 1e-6

    # Dropout
    dropout: float = 0.0
    attention_dropout: float = 0.0
    cfg_dropout: float = 0.1

    # Optimization
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = True
    zero_init_output: bool = True

    # mRoPE Configuration
    mrope_text_dim: int = 16
    mrope_img_h_dim: int = 24
    mrope_img_w_dim: int = 24
    mrope_text_max_len: int = 511  # 512-1: +1 in factory.py gives max_seq_len=512
    mrope_img_max_len: int = 65536  # Legacy: used for fallback calculation
    mrope_img_max_height: int = 128  # Max patches in height (explicit, no sqrt)
    mrope_img_max_width: int = 128   # Max patches in width (explicit, no sqrt)
    mrope_theta: float = 10000.0

    # Embedding Configuration
    time_embed_dim: int = 256
    max_patches: int = 4096

    # Token Compaction
    token_compaction_expand_gain: float = 0.1  # TODO: needs proper fix, see plan

    # Gated Attention
    gate_type: Literal["headwise", "elementwise"] = "headwise"
    gate_activation: Literal["sigmoid", "silu"] = "sigmoid"
    gate_bias: bool = False

    # AdaLN
    adaln_num_params: int = 6
    adaln_init_gain: float = 0.0001  # 1e-4: small init for gradual modulation learning

    # Gamma L2 Penalty (only for pixelwise AdaLN)
    pixel_gamma_l2_lambda: float = 1e-4  # Lambda for pixel gamma L2 penalty

    # Inference defaults
    default_num_steps: int = 50
    default_guidance_scale: float = 7.5
    default_sampler_method: Literal["euler", "heun", "dpm_pp", "dpm_pp_2s"] = "heun"

    def __post_init__(self):
        """Compute defaults and validate configuration after initialization."""
        # Compute bottleneck_dim default if not provided
        # Formula: patch_size^2 // 4 (e.g., 16^2 // 4 = 64)
        if self.bottleneck_dim is None:
            object.__setattr__(self, 'bottleneck_dim', self.patch_size ** 2 // 4)

        from .validators import validate_pixelhdm_config
        validate_pixelhdm_config(self)

    @property
    def gqa_ratio(self) -> int:
        """GQA ratio (Q heads per KV head)."""
        return self.num_heads // self.num_kv_heads

    @property
    def mlp_hidden_dim(self) -> int:
        """MLP hidden dimension."""
        return int(self.hidden_dim * self.mlp_ratio)

    @property
    def pixels_per_patch(self) -> int:
        """Number of pixels per patch (p^2)."""
        return self.patch_size ** 2

    @property
    def patch_pixel_dim(self) -> int:
        """Total dimension for pixel features per patch (p^2 x D_pix)."""
        return self.pixels_per_patch * self.pixel_dim

    @property
    def num_patches(self) -> int:
        """Number of patches for 256x256 image."""
        return (256 // self.patch_size) ** 2

    def get_num_patches(self, height: int, width: int) -> int:
        """Calculate number of patches for given resolution."""
        assert height % self.patch_size == 0 and width % self.patch_size == 0, \
            f"Resolution ({height}x{width}) must be divisible by patch_size ({self.patch_size})"
        return (height // self.patch_size) * (width // self.patch_size)

    @classmethod
    def default(cls) -> "PixelHDMConfig":
        """Create default configuration (~332M trainable params)."""
        from .pixelhdm_factories import create_default_config
        return create_default_config()

    @classmethod
    def small(cls) -> "PixelHDMConfig":
        """Create small configuration for testing."""
        from .pixelhdm_factories import create_small_config
        return create_small_config()

    @classmethod
    def large(cls) -> "PixelHDMConfig":
        """Create large configuration (~500M+ trainable params)."""
        from .pixelhdm_factories import create_large_config
        return create_large_config()

    @classmethod
    def for_testing(cls) -> "PixelHDMConfig":
        """Create minimal configuration for unit tests."""
        from .pixelhdm_factories import create_testing_config
        return create_testing_config()

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        from .pixelhdm_factories import config_to_dict
        return config_to_dict(self)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PixelHDMConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_path: str) -> "PixelHDMConfig":
        """Create config from JSON file."""
        from .pixelhdm_factories import config_from_json
        return config_from_json(json_path)

    def to_json(self, json_path: str) -> None:
        """Save config to JSON file."""
        from .pixelhdm_factories import config_to_json
        config_to_json(self, json_path)
