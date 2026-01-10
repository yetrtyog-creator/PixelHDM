"""
門控注意力 (Gated Attention) 實現

基於 NeurIPS 2025 最佳論文: Gated Attention for LLMs
arXiv: 2505.06708

整合設計:
    - GQA (Grouped Query Attention): 16 Q heads, 4 KV heads (4:1 ratio)
    - QK Norm: per-head RMSNorm, 應用在 RoPE 前
    - Gated Attention: 16 gates (per-Q-head), sigmoid 激活

核心公式:
    Y' = Y ⊙ σ(XW_g)

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

from typing import Callable, Optional, Literal

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..layers.normalization import QKNorm
from .gating import create_gate, GateProjection
from .projections import repeat_kv, QKVProjection, OutputProjection
from .attention_ops import (
    compute_flash_attention, compute_manual_attention, prepare_attention_mask
)


class GatedMultiHeadAttention(nn.Module):
    """
    門控多頭自注意力 (含 GQA 和 QK Norm)

    特性:
    - GQA: 16 Q heads, 4 KV heads (4:1 比例)
    - QK Norm: per-head RMSNorm, 應用在 RoPE 前
    - Gated Attention: 16 gates (per-Q-head), sigmoid 激活
    - Flash Attention 2 支援
    - 梯度檢查點支援
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_heads: int = 16,
        num_kv_heads: int = 4,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_qk_norm: bool = True,
        use_gated_attention: bool = True,
        gate_type: Literal["elementwise", "headwise"] = "headwise",
        gate_activation: Literal["sigmoid", "silu"] = "sigmoid",
        gate_bias: bool = False,
        use_flash_attention: bool = True,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()

        self._validate_config(hidden_dim, num_heads, num_kv_heads)
        self._init_config(hidden_dim, num_heads, num_kv_heads, head_dim, dropout)
        self._init_projections(hidden_dim, dropout)
        self._init_qk_norm(use_qk_norm)
        self._init_gate(
            use_gated_attention, hidden_dim, gate_type, gate_bias, gate_activation
        )
        self._init_flags(use_flash_attention, use_checkpoint, dropout)

    def _validate_config(
        self, hidden_dim: int, num_heads: int, num_kv_heads: int
    ) -> None:
        """驗證配置參數"""
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) 必須能被 num_heads ({num_heads}) 整除"
        assert num_heads % num_kv_heads == 0, \
            f"num_heads ({num_heads}) 必須能被 num_kv_heads ({num_kv_heads}) 整除"

    def _init_config(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int],
        dropout: float,
    ) -> None:
        """初始化配置參數"""
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim if head_dim is not None else hidden_dim // num_heads
        self.n_rep = num_heads // num_kv_heads
        self.scale = self.head_dim ** -0.5

    def _init_projections(self, hidden_dim: int, dropout: float) -> None:
        """初始化投影層"""
        self.qkv_proj = QKVProjection(
            hidden_dim, self.num_heads, self.num_kv_heads, self.head_dim
        )
        self.out_proj_layer = OutputProjection(
            hidden_dim, self.num_heads, self.head_dim, dropout
        )
        # 為了向後兼容，保留這些引用
        self.q_proj = self.qkv_proj.q_proj
        self.k_proj = self.qkv_proj.k_proj
        self.v_proj = self.qkv_proj.v_proj
        self.out_proj = self.out_proj_layer.out_proj
        self.dropout = self.out_proj_layer.dropout

    def _init_qk_norm(self, use_qk_norm: bool) -> None:
        """初始化 QK Norm"""
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = QKNorm(self.head_dim)
            self.k_norm = QKNorm(self.head_dim)

    def _init_gate(
        self,
        use_gated_attention: bool,
        hidden_dim: int,
        gate_type: str,
        gate_bias: bool,
        gate_activation: str,
    ) -> None:
        """初始化門控"""
        self.use_gated_attention = use_gated_attention
        self.gate_type = gate_type
        self.gate = create_gate(
            use_gated_attention=use_gated_attention,
            hidden_dim=hidden_dim,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            gate_type=gate_type,
            gate_bias=gate_bias,
            gate_activation=gate_activation,
        )
        # 向後兼容：保留 gate_proj 引用
        if use_gated_attention and isinstance(self.gate, GateProjection):
            self.gate_proj = self.gate.proj
            self.gate_activation = self.gate.activation

    def _init_flags(
        self, use_flash_attention: bool, use_checkpoint: bool, dropout: float
    ) -> None:
        """初始化標記"""
        self.use_flash_attention = use_flash_attention
        self.use_checkpoint = use_checkpoint
        self.dropout_p = dropout

    def _compute_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """計算注意力"""
        if self.use_flash_attention:
            return compute_flash_attention(
                q, k, v, attention_mask, self.dropout_p, self.training
            )
        return compute_manual_attention(
            q, k, v, attention_mask, self.scale, self.dropout
        )

    def _forward_impl(
        self,
        x: torch.Tensor,
        rope_fn: Optional[Callable] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """內部前向實現

        Args:
            x: Input tensor (B, seq_len, hidden_dim)
            rope_fn: RoPE function that takes (q, k, position_ids)
            position_ids: Lumina2-style position IDs (B, seq_len, 3)
            attention_mask: Attention mask (B, seq_len)
        """
        # QKV 投影
        q, k, v = self.qkv_proj(x)

        # QK Norm
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # GQA: 擴展 KV
        k, v = repeat_kv(k, v, self.n_rep)

        # 應用 mRoPE (Lumina2 style with position_ids)
        if rope_fn is not None:
            q, k = rope_fn(q, k, position_ids)

        # 準備 mask 並計算注意力
        attention_mask = prepare_attention_mask(attention_mask)
        attn_output = self._compute_attention(q, k, v, attention_mask)

        # 應用門控
        attn_output = self.gate(x, attn_output)

        # 輸出投影
        return self.out_proj_layer(attn_output)

    def forward(
        self,
        x: torch.Tensor,
        rope_fn: Optional[Callable] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向傳播

        Args:
            x: Input tensor (B, seq_len, hidden_dim)
            rope_fn: RoPE function that takes (q, k, position_ids)
            position_ids: Lumina2-style position IDs (B, seq_len, 3)
            attention_mask: Attention mask (B, seq_len)

        Returns:
            Output tensor (B, seq_len, hidden_dim)
        """
        if x.dim() != 3 or x.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"輸入形狀應為 (B, seq_len, {self.hidden_dim}), 當前: {x.shape}"
            )

        if self.training and self.use_checkpoint:
            return checkpoint(
                self._forward_impl,
                x, rope_fn, position_ids, attention_mask,
                use_reentrant=False,
            )
        return self._forward_impl(x, rope_fn, position_ids, attention_mask)

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, "
            f"gated={self.use_gated_attention}, "
            f"gate_type={self.gate_type}, "
            f"qk_norm={self.use_qk_norm}, "
            f"flash_attn={self.use_flash_attention}"
        )


# 為了向後兼容，保留舊的 _apply_gate 方法
GatedMultiHeadAttention._apply_gate = lambda self, x, attn_output: self.gate(x, attn_output)


__all__ = [
    "GatedMultiHeadAttention",
    "repeat_kv",
]
