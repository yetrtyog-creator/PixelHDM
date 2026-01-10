"""
Text Encoder Pooling Strategies

池化策略模組，用於從序列輸出中提取固定維度的表示。

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LastTokenPooling(nn.Module):
    """
    使用最後一個非 padding token 的隱藏狀態作為池化輸出。

    Shape:
        - Input: hidden_states (B, T, D), attention_mask (B, T)
        - Output: (B, D)
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        提取最後一個非 padding token 的隱藏狀態。

        Args:
            hidden_states: 序列隱藏狀態 (B, T, D)
            attention_mask: 注意力掩碼 (B, T)，1 表示有效 token

        Returns:
            池化輸出 (B, D)
        """
        seq_lens = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(
            hidden_states.size(0),
            device=hidden_states.device
        )
        return hidden_states[batch_indices, seq_lens]


class MeanPooling(nn.Module):
    """
    使用所有有效 token 的平均值作為池化輸出。

    Shape:
        - Input: hidden_states (B, T, D), attention_mask (B, T)
        - Output: (B, D)
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        計算有效 token 的平均隱藏狀態。

        Args:
            hidden_states: 序列隱藏狀態 (B, T, D)
            attention_mask: 注意力掩碼 (B, T)

        Returns:
            池化輸出 (B, D)
        """
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_hidden / sum_mask


def get_pooler(strategy: str = "last_token") -> nn.Module:
    """
    根據策略名稱獲取池化器。

    Args:
        strategy: 池化策略 ("last_token" 或 "mean")

    Returns:
        池化器模組

    Raises:
        ValueError: 不支援的策略
    """
    poolers = {
        "last_token": LastTokenPooling,
        "mean": MeanPooling,
    }

    if strategy not in poolers:
        supported = ", ".join(poolers.keys())
        raise ValueError(f"不支援的池化策略: {strategy}。支援: {supported}")

    return poolers[strategy]()


__all__ = [
    "LastTokenPooling",
    "MeanPooling",
    "get_pooler",
]
