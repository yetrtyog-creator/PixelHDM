"""
Attention Module Factory Functions

Author: PixelHDM-RPEA-DinoV3 Project
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .gated_attention import GatedMultiHeadAttention
from .token_compaction import TokenCompaction

if TYPE_CHECKING:
    from ...config.model_config import PixelHDMConfig


def create_attention(
    hidden_dim: int = 1024,
    num_heads: int = 16,
    num_kv_heads: int = 4,
    use_gated_attention: bool = True,
    gate_type: Literal["elementwise", "headwise"] = "headwise",
    gate_activation: Literal["sigmoid", "silu"] = "sigmoid",
    gate_bias: bool = False,
    use_qk_norm: bool = True,
    use_flash_attention: bool = True,
    use_checkpoint: bool = True,
    dropout: float = 0.0,
) -> GatedMultiHeadAttention:
    """Create gated attention module (with GQA and QK Norm)."""
    return GatedMultiHeadAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        dropout=dropout,
        use_qk_norm=use_qk_norm,
        use_gated_attention=use_gated_attention,
        gate_type=gate_type,
        gate_activation=gate_activation,
        gate_bias=gate_bias,
        use_flash_attention=use_flash_attention,
        use_checkpoint=use_checkpoint,
    )


def create_attention_from_config(config: "PixelHDMConfig") -> GatedMultiHeadAttention:
    """Create gated attention module from config."""
    return create_attention(
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_kv_heads=config.num_kv_heads,
        use_flash_attention=config.use_flash_attention,
        use_checkpoint=config.use_gradient_checkpointing,
        dropout=config.attention_dropout,
        gate_type=config.gate_type,
        gate_activation=config.gate_activation,
        gate_bias=config.gate_bias,
        use_gated_attention=True,
        use_qk_norm=True,
    )


def create_token_compaction(
    hidden_dim: int = 1024,
    pixel_dim: int = 16,
    patch_size: int = 16,
    num_heads: int = 16,
    num_kv_heads: int = 4,
    use_checkpoint: bool = True,
) -> TokenCompaction:
    """Create Token Compaction module."""
    return TokenCompaction(
        config=None,
        hidden_dim=hidden_dim,
        pixel_dim=pixel_dim,
        patch_size=patch_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        use_checkpoint=use_checkpoint,
    )


def create_token_compaction_from_config(config: "PixelHDMConfig") -> TokenCompaction:
    """Create Token Compaction module from config."""
    return TokenCompaction(config=config)


__all__ = [
    "create_attention",
    "create_attention_from_config",
    "create_token_compaction",
    "create_token_compaction_from_config",
]
