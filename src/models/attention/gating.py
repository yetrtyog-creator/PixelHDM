"""
門控機制 (Gating Mechanism) 實現

基於 NeurIPS 2025 最佳論文: Gated Attention for LLMs
arXiv: 2505.06708

核心公式:
    Y' = Y ⊙ σ(XW_g)

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_gate_activation(activation_type: Literal["sigmoid", "silu"]) -> Callable:
    """
    獲取門控激活函數

    Args:
        activation_type: 激活函數類型 ('sigmoid' 或 'silu')

    Returns:
        激活函數
    """
    if activation_type == "sigmoid":
        return torch.sigmoid
    else:  # silu
        return F.silu


class GateProjection(nn.Module):
    """
    門控投影層

    支援兩種門控類型:
    - elementwise: 逐元素門控，每個元素獨立調製
    - headwise: 逐頭門控，每個 Q head 一個標量

    Args:
        hidden_dim: 隱藏維度
        num_heads: 注意力頭數
        head_dim: 每個頭的維度
        gate_type: 門控類型
        gate_bias: 是否使用偏置
        gate_activation: 門控激活函數類型
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        gate_type: Literal["elementwise", "headwise"] = "headwise",
        gate_bias: bool = False,
        gate_activation: Literal["sigmoid", "silu"] = "sigmoid",
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.gate_type = gate_type

        if gate_type == "elementwise":
            self.proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=gate_bias)
        else:  # headwise
            self.proj = nn.Linear(hidden_dim, num_heads, bias=gate_bias)

        # 零初始化門控權重
        nn.init.zeros_(self.proj.weight)
        if gate_bias and self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

        self.activation = get_gate_activation(gate_activation)

    def forward(
        self,
        x: torch.Tensor,
        attn_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        應用門控機制

        Args:
            x: 原始輸入 (B, seq_len, hidden_dim)
            attn_output: 注意力輸出 (B, num_heads, seq_len, head_dim)

        Returns:
            門控後的輸出 (B, num_heads, seq_len, head_dim)
        """
        B, num_heads, seq_len, head_dim = attn_output.shape

        if self.gate_type == "elementwise":
            gate_scores = self.proj(x)  # (B, seq_len, num_heads * head_dim)
            gate_scores = gate_scores.view(B, seq_len, self.num_heads, self.head_dim)
            gate_scores = gate_scores.transpose(1, 2)  # (B, num_heads, seq_len, head_dim)
            gate = self.activation(gate_scores)
            return attn_output * gate
        else:  # headwise
            gate_scores = self.proj(x)  # (B, seq_len, num_heads)
            gate = self.activation(gate_scores)  # (B, seq_len, num_heads)
            gate = gate.transpose(1, 2).unsqueeze(-1)  # (B, num_heads, seq_len, 1)
            return attn_output * gate


class IdentityGate(nn.Module):
    """
    恆等門控 (無操作)

    當 use_gated_attention=False 時使用
    """

    def forward(
        self,
        x: torch.Tensor,
        attn_output: torch.Tensor,
    ) -> torch.Tensor:
        """直接返回注意力輸出"""
        return attn_output


def create_gate(
    use_gated_attention: bool,
    hidden_dim: int,
    num_heads: int,
    head_dim: int,
    gate_type: Literal["elementwise", "headwise"] = "headwise",
    gate_bias: bool = False,
    gate_activation: Literal["sigmoid", "silu"] = "sigmoid",
) -> nn.Module:
    """
    創建門控模組

    Args:
        use_gated_attention: 是否使用門控
        hidden_dim: 隱藏維度
        num_heads: 注意力頭數
        head_dim: 每個頭的維度
        gate_type: 門控類型
        gate_bias: 是否使用偏置
        gate_activation: 門控激活函數類型

    Returns:
        門控模組 (GateProjection 或 IdentityGate)
    """
    if not use_gated_attention:
        return IdentityGate()

    return GateProjection(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        gate_type=gate_type,
        gate_bias=gate_bias,
        gate_activation=gate_activation,
    )


__all__ = [
    "get_gate_activation",
    "GateProjection",
    "IdentityGate",
    "create_gate",
]
