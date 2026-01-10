"""
正規化層實現

包含:
    - RMSNorm: Root Mean Square Layer Normalization
    - AdaptiveRMSNorm: 條件自適應 RMSNorm (用於 DiT AdaLN)

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    相比 LayerNorm:
    - 不計算均值，只使用 RMS 進行縮放
    - 計算效率更高
    - 被現代 LLM (如 LLaMA, Qwen) 廣泛採用

    公式:
        RMSNorm(x) = x / RMS(x) * γ
        RMS(x) = sqrt(mean(x²) + eps)

    Args:
        dim: 正規化維度
        eps: 數值穩定性參數

    Shape:
        - Input: (..., dim)
        - Output: (..., dim)

    Example:
        >>> norm = RMSNorm(1024)
        >>> x = torch.randn(2, 512, 1024)
        >>> out = norm(x)  # (2, 512, 1024)
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """計算 RMS 正規化"""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播

        Args:
            x: 輸入張量 (..., dim)

        Returns:
            正規化後的張量 (..., dim)
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class AdaptiveRMSNorm(nn.Module):
    """
    自適應 RMSNorm (用於 DiT AdaLN)

    根據條件信號動態調整 scale 和 shift

    Args:
        dim: 正規化維度
        cond_dim: 條件維度
        eps: 數值穩定性參數

    Example:
        >>> norm = AdaptiveRMSNorm(1024, 512)
        >>> x = torch.randn(2, 512, 1024)
        >>> cond = torch.randn(2, 512)
        >>> out = norm(x, cond)
    """

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

        # 基礎權重
        self.weight = nn.Parameter(torch.ones(dim))

        # 條件調製 (輸出 scale 和 shift)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim * 2, bias=True),
        )

        # 初始化調製層為零
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """計算 RMS 正規化"""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向傳播

        Args:
            x: 輸入張量 (B, seq_len, dim)
            cond: 條件張量 (B, cond_dim) 或 (B, seq_len, cond_dim)

        Returns:
            條件正規化後的張量 (B, seq_len, dim)
        """
        # RMS 正規化
        normed = self._norm(x.float()).type_as(x)
        normed = normed * self.weight

        # 計算調製參數
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)  # (B, 1, cond_dim)

        modulation = self.modulation(cond)  # (B, *, dim*2)
        scale, shift = modulation.chunk(2, dim=-1)

        # 應用調製: (1 + scale) * x + shift
        return normed * (1 + scale) + shift


class QKNorm(nn.Module):
    """
    Per-head RMSNorm for Q and K

    在每個 head 維度上獨立進行 RMSNorm
    用於 QK Norm (在 RoPE 前應用)
    """

    def __init__(self, head_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_heads, seq_len, head_dim)
        Returns:
            normalized: (B, num_heads, seq_len, head_dim)
        """
        # RMS norm on last dimension (head_dim)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def create_norm(
    dim: int,
    norm_type: Literal["rmsnorm", "layernorm"] = "rmsnorm",
    eps: float = 1e-6,
) -> nn.Module:
    """
    創建正規化層

    Args:
        dim: 正規化維度
        norm_type: 正規化類型
        eps: 數值穩定性參數

    Returns:
        正規化層
    """
    if norm_type == "rmsnorm":
        return RMSNorm(dim, eps=eps)
    elif norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    else:
        raise ValueError(f"不支援的正規化類型: {norm_type}")


__all__ = [
    "RMSNorm",
    "AdaptiveRMSNorm",
    "QKNorm",
    "create_norm",
]
