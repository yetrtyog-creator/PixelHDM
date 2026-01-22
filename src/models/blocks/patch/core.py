"""
Patch-Level Transformer Block (Core)

Standard Transformer block with Token-Independent AdaLN.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Callable

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ...layers.normalization import RMSNorm
from ...layers.feedforward import SwiGLU
from ...layers.adaln import TokenAdaLN
from ...attention.gated_attention import GatedMultiHeadAttention

if TYPE_CHECKING:
    from ....config.model_config import PixelHDMConfig




_RESIDUAL_SCALE_K = 2


class PatchTransformerBlock(nn.Module):
    """
    Patch-Level DiT Transformer Block

    Standard Transformer block with Token-Independent AdaLN.

    Shape:
        - x: (B, L, D)
        - t_embed: (B, D) - time embedding
        - Output: (B, L, D)
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
    ) -> None:
        super().__init__()

        # Gate and flash attention configuration
        gate_type = "headwise"
        gate_activation = "sigmoid"
        gate_bias = False
        use_flash_attention = True

        if config is not None:
            hidden_dim = config.hidden_dim
            num_heads = config.num_heads
            num_kv_heads = config.num_kv_heads
            mlp_ratio = config.mlp_ratio
            dropout = config.attention_dropout
            use_checkpoint = config.use_gradient_checkpointing
            gate_type = config.gate_type
            gate_activation = config.gate_activation
            gate_bias = config.gate_bias
            use_flash_attention = config.use_flash_attention

        self.hidden_dim = hidden_dim
        self.use_checkpoint = use_checkpoint

        # Depth scaling (k=2 residual branches per block)
        patch_layers = config.patch_layers if config is not None else 1
        self.residual_scale = 1.0 / math.sqrt(_RESIDUAL_SCALE_K * patch_layers)

        # Attention Block
        self.pre_attn_norm = RMSNorm(hidden_dim)
        self.attention = GatedMultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            use_qk_norm=True,
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

        # AdaLN
        self.adaln = TokenAdaLN(hidden_dim=hidden_dim, num_params=6)

    def _forward_impl(
        self,
        x: torch.Tensor,
        t_embed: torch.Tensor,
        rope_fn: Optional[Callable] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Internal forward implementation.

        Args:
            x: Input tensor (B, L, D)
            t_embed: Time embedding (B, D)
            rope_fn: RoPE function (Lumina2 style)
            position_ids: Lumina2-style position IDs (B, L, 3)
            attention_mask: Attention mask (B, L)

        Returns:
            Output tensor (B, L, D)
        """
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaln(t_embed)

        # Attention Block
        h = self.pre_attn_norm(x)
        h = gamma1 * h + beta1
        h = self.attention(
            h,
            rope_fn=rope_fn,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        h = self.post_attn_norm(h)
        x = x + (alpha1 * self.residual_scale) * h

        # MLP Block
        h = self.pre_mlp_norm(x)
        h = gamma2 * h + beta2
        h = self.mlp(h)
        h = self.post_mlp_norm(h)
        x = x + (alpha2 * self.residual_scale) * h

        return x

    def forward(
        self,
        x: torch.Tensor,
        t_embed: torch.Tensor,
        rope_fn: Optional[Callable] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, L, D)
            t_embed: Time embedding (B, D)
            rope_fn: RoPE function (Lumina2 style)
            position_ids: Lumina2-style position IDs (B, L, 3)
            attention_mask: Attention mask (B, L)

        Returns:
            Output tensor (B, L, D)
        """
        if self.training and self.use_checkpoint:
            return checkpoint(
                self._forward_impl,
                x, t_embed, rope_fn, position_ids, attention_mask,
                use_reentrant=False,
            )
        else:
            return self._forward_impl(x, t_embed, rope_fn, position_ids, attention_mask)

    def extra_repr(self) -> str:
        return f"hidden_dim={self.hidden_dim}, sandwich_norm=True, adaln=token_independent"


__all__ = ["PatchTransformerBlock"]
