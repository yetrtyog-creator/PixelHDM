"""
Text Processor Block (Pre-joint, Modality-Internal)

A simple Transformer block for processing text tokens BEFORE joint text-image attention.
Key properties:
- NO RoPE: Position encoding only in joint mRoPE (to avoid double-RoPE)
- NO timestep/AdaLN: Text embeddings are time-independent at this stage
- Self-attention only over text tokens
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ...layers.normalization import RMSNorm
from ...layers.feedforward import SwiGLU
from ...attention.gated_attention import GatedMultiHeadAttention

if TYPE_CHECKING:
    from ....config.model_config import PixelHDMConfig


_RESIDUAL_SCALE_K = 2


class TextProcessorBlock(nn.Module):
    """
    Text Processor Block (Modality-Internal, Pre-Joint)

    A simple Pre-Norm Transformer block for text tokens.
    Called BEFORE concatenation with image tokens.

    Key Design:
        - NO RoPE: Position encoding only in joint 3D mRoPE
        - NO timestep/AdaLN: Text embeddings are time-independent here
        - Self-attention only over text tokens (modality-internal alignment)

    Shape:
        - x: (B, T, D) - text tokens
        - text_mask: (B, T) - optional attention mask
        - Output: (B, T, D)
    """

    def __init__(
        self,
        config: Optional["PixelHDMConfig"] = None,
        hidden_dim: int = 1024,
        num_heads: int = 16,
        num_kv_heads: int = 4,
        mlp_ratio: float = 3.0,
        dropout: float = 0.0,
        use_checkpoint: bool = True,
        num_layers: int = 1,
    ) -> None:
        super().__init__()

        # Gate and flash attention configuration
        gate_type = "headwise"
        gate_activation = "sigmoid"
        gate_bias = False
        use_flash_attention = True
        use_qk_norm = True

        if config is not None:
            hidden_dim = config.hidden_dim
            num_heads = config.num_heads
            num_kv_heads = config.num_kv_heads
            mlp_ratio = getattr(config, "text_processor_mlp_ratio", config.mlp_ratio)
            dropout = config.attention_dropout
            use_checkpoint = config.use_gradient_checkpointing
            gate_type = config.gate_type
            gate_activation = config.gate_activation
            gate_bias = config.gate_bias
            use_flash_attention = config.use_flash_attention
            use_qk_norm = getattr(config, "text_processor_use_qk_norm", True)
            num_layers = getattr(config, "text_processor_layers", 1)

        self.hidden_dim = hidden_dim
        self.use_checkpoint = use_checkpoint
        self.num_layers = num_layers

        # Depth scaling (k=2 residual branches per block)
        self.residual_scale = 1.0 / math.sqrt(_RESIDUAL_SCALE_K * max(num_layers, 1))

        # Attention Block (NO RoPE - position encoding in joint mRoPE only)
        self.pre_attn_norm = RMSNorm(hidden_dim)
        self.attention = GatedMultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            use_qk_norm=use_qk_norm,
            use_gated_attention=True,
            gate_type=gate_type,
            gate_activation=gate_activation,
            gate_bias=gate_bias,
            use_flash_attention=use_flash_attention,
            use_checkpoint=False,
        )
        self.post_attn_norm = RMSNorm(hidden_dim)

        # MLP Block
        self.pre_mlp_norm = RMSNorm(hidden_dim)
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.mlp = SwiGLU(
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        self.post_mlp_norm = RMSNorm(hidden_dim)

    def _forward_impl(
        self,
        x: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Internal forward implementation."""
        # Attention Block (NO AdaLN, NO RoPE)
        h = self.pre_attn_norm(x)
        h = self.attention(
            h,
            rope_fn=None,  # NO RoPE - position encoding in joint mRoPE only
            position_ids=None,
            attention_mask=text_mask,
        )
        h = self.post_attn_norm(h)
        x = x + self.residual_scale * h

        # MLP Block
        h = self.pre_mlp_norm(x)
        h = self.mlp(h)
        h = self.post_mlp_norm(h)
        x = x + self.residual_scale * h

        return x

    def forward(
        self,
        x: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        if self.training and self.use_checkpoint:
            return checkpoint(
                self._forward_impl,
                x, text_mask,
                use_reentrant=False,
            )
        else:
            return self._forward_impl(x, text_mask)

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"sandwich_norm=True, "
            f"no_rope=True, no_adaln=True"
        )


class TextProcessorStack(nn.Module):
    """
    Stack of TextProcessorBlocks.

    For processing text tokens through multiple layers before joint attention.
    """

    def __init__(
        self,
        config: Optional["PixelHDMConfig"] = None,
        num_layers: Optional[int] = None,
    ) -> None:
        super().__init__()

        if config is not None:
            num_layers = num_layers or getattr(config, "text_processor_layers", 1)
        else:
            num_layers = num_layers or 1

        self.num_layers = num_layers
        self.blocks = nn.ModuleList([
            TextProcessorBlock(config=config, num_layers=num_layers)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through all blocks."""
        for block in self.blocks:
            x = block(x, text_mask=text_mask)
        return x

    def extra_repr(self) -> str:
        return f"num_layers={self.num_layers}"


__all__ = ["TextProcessorBlock", "TextProcessorStack"]
