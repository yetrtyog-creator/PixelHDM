"""
QKV 投影與 GQA 工具函數

包含:
    - repeat_kv: GQA KV 擴展函數
    - QKVProjection: QKV 投影模組

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def repeat_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    n_rep: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    擴展 KV heads 以匹配 Q heads (GQA)

    Args:
        k: (B, num_kv_heads, seq_len, head_dim)
        v: (B, num_kv_heads, seq_len, head_dim)
        n_rep: 重複次數 = num_heads // num_kv_heads

    Returns:
        k: (B, num_heads, seq_len, head_dim)
        v: (B, num_heads, seq_len, head_dim)
    """
    if n_rep == 1:
        return k, v

    B, num_kv_heads, seq_len, head_dim = k.shape

    # Expand and reshape
    k = k.unsqueeze(2).expand(B, num_kv_heads, n_rep, seq_len, head_dim)
    v = v.unsqueeze(2).expand(B, num_kv_heads, n_rep, seq_len, head_dim)

    k = k.reshape(B, num_kv_heads * n_rep, seq_len, head_dim)
    v = v.reshape(B, num_kv_heads * n_rep, seq_len, head_dim)

    return k, v


class QKVProjection(nn.Module):
    """
    QKV 投影模組 (支援 GQA)

    Args:
        hidden_dim: 隱藏維度
        num_heads: Q 注意力頭數
        num_kv_heads: KV 注意力頭數 (GQA)
        head_dim: 每個頭的維度
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim if head_dim is not None else hidden_dim // num_heads
        self.n_rep = num_heads // num_kv_heads

        # Q 投影: hidden_dim -> num_heads * head_dim
        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)

        # KV 投影: hidden_dim -> num_kv_heads * head_dim (GQA)
        self.kv_dim = num_kv_heads * self.head_dim
        self.k_proj = nn.Linear(hidden_dim, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, self.kv_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        投影輸入為 Q, K, V

        Args:
            x: (B, seq_len, hidden_dim)

        Returns:
            q: (B, num_heads, seq_len, head_dim)
            k: (B, num_kv_heads, seq_len, head_dim)
            v: (B, num_kv_heads, seq_len, head_dim)
        """
        B, seq_len, _ = x.shape

        q = self.q_proj(x).view(B, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, seq_len, self.num_kv_heads, self.head_dim)

        # 轉置為 (B, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        return q, k, v


class OutputProjection(nn.Module):
    """
    輸出投影模組

    Args:
        hidden_dim: 隱藏維度
        num_heads: 注意力頭數
        head_dim: 每個頭的維度
        dropout: Dropout 率
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim

        self.out_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, attn_output: torch.Tensor) -> torch.Tensor:
        """
        投影注意力輸出

        Args:
            attn_output: (B, num_heads, seq_len, head_dim)

        Returns:
            (B, seq_len, hidden_dim)
        """
        B, num_heads, seq_len, head_dim = attn_output.shape

        # 轉置回 (B, seq_len, hidden_dim)
        output = attn_output.transpose(1, 2).contiguous()
        output = output.view(B, seq_len, self.num_heads * self.head_dim)

        output = self.out_proj(output)
        output = self.dropout(output)

        return output


__all__ = [
    "repeat_kv",
    "QKVProjection",
    "OutputProjection",
]
