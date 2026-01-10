"""
前饋網路層實現

包含:
    - SwiGLU: Swish-Gated Linear Unit MLP
    - AdaptiveSwiGLU: 帶條件調製的 SwiGLU

SwiGLU 公式:
    y = (x @ W_up) ⊙ SiLU(x @ W_gate) @ W_down

配置:
    - mlp_ratio = 3.0 (降低計算量)
    - 無偏置項 (bias=False)

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from ...config.model_config import PixelHDMConfig


class SwiGLU(nn.Module):
    """
    SwiGLU (Swish-Gated Linear Unit) FFN

    公式:
        SwiGLU(x) = (x @ W_up ⊙ SiLU(x @ W_gate)) @ W_down

    相比標準 MLP:
    - 使用門控機制提升表達能力
    - 被 LLaMA, Qwen 等現代 LLM 採用
    - 使用 3x ratio (而非傳統的 4x) 以降低計算量

    Args:
        hidden_dim: 輸入/輸出維度
        mlp_dim: 中間隱藏維度 (預設為 hidden_dim * 3)
        dropout: Dropout 率
        bias: 是否使用偏置

    Shape:
        - Input: (B, seq_len, hidden_dim)
        - Output: (B, seq_len, hidden_dim)

    Example:
        >>> ffn = SwiGLU(1024, 3072)  # 3x ratio
        >>> x = torch.randn(2, 512, 1024)
        >>> out = ffn(x)  # (2, 512, 1024)
    """

    def __init__(
        self,
        hidden_dim: int,
        mlp_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim or int(hidden_dim * 3)

        # SwiGLU: 需要兩個投影 (gate 和 up)
        self.gate_proj = nn.Linear(hidden_dim, self.mlp_dim, bias=bias)
        self.up_proj = nn.Linear(hidden_dim, self.mlp_dim, bias=bias)
        self.down_proj = nn.Linear(self.mlp_dim, hidden_dim, bias=bias)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播

        Args:
            x: 輸入張量 (B, seq_len, hidden_dim)

        Returns:
            輸出張量 (B, seq_len, hidden_dim)
        """
        # SwiGLU: gate * up
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up

        # Down projection
        output = self.down_proj(hidden)
        output = self.dropout(output)

        return output

    def extra_repr(self) -> str:
        return f"hidden_dim={self.hidden_dim}, mlp_dim={self.mlp_dim}"


class AdaptiveSwiGLU(nn.Module):
    """
    自適應 SwiGLU (用於 DiT AdaLN)

    根據條件信號動態調整 scale

    Args:
        hidden_dim: 輸入/輸出維度
        cond_dim: 條件維度
        mlp_dim: 中間隱藏維度
        dropout: Dropout 率

    Example:
        >>> ffn = AdaptiveSwiGLU(1024, 512, 3072)
        >>> x = torch.randn(2, 512, 1024)
        >>> cond = torch.randn(2, 512)
        >>> out = ffn(x, cond)
    """

    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        mlp_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim or int(hidden_dim * 3)
        self.cond_dim = cond_dim

        # SwiGLU 層
        self.swiglu = SwiGLU(hidden_dim, self.mlp_dim, dropout=dropout, bias=False)

        # 條件調製 (輸出 scale)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim, bias=True),
        )

        # 初始化調製層為零
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向傳播

        Args:
            x: 輸入張量 (B, seq_len, hidden_dim)
            cond: 條件張量 (B, cond_dim) 或 (B, seq_len, cond_dim)

        Returns:
            條件調製後的輸出 (B, seq_len, hidden_dim)
        """
        # FFN
        output = self.swiglu(x)

        # 計算調製參數
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)  # (B, 1, cond_dim)

        scale = self.modulation(cond)  # (B, *, hidden_dim)

        # 應用調製: (1 + scale) * output
        return output * (1 + scale)

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"mlp_dim={self.mlp_dim}, "
            f"cond_dim={self.cond_dim}"
        )


def create_ffn(
    hidden_dim: int,
    mlp_ratio: float = 3.0,
    dropout: float = 0.0,
    bias: bool = False,
) -> SwiGLU:
    """
    創建 SwiGLU FFN 層

    Args:
        hidden_dim: 輸入/輸出維度
        mlp_ratio: MLP 維度倍數 (預設 3.0)
        dropout: Dropout 率
        bias: 是否使用偏置

    Returns:
        SwiGLU FFN 層
    """
    mlp_dim = int(hidden_dim * mlp_ratio)
    return SwiGLU(hidden_dim, mlp_dim, dropout=dropout, bias=bias)


def create_ffn_from_config(config: "PixelHDMConfig") -> SwiGLU:
    """
    從配置創建 SwiGLU FFN 層

    Args:
        config: PixelHDMConfig 配置

    Returns:
        SwiGLU FFN 層
    """
    return create_ffn(
        hidden_dim=config.hidden_dim,
        mlp_ratio=config.mlp_ratio,
        dropout=config.dropout,
        bias=False,
    )


__all__ = [
    "SwiGLU",
    "AdaptiveSwiGLU",
    "create_ffn",
    "create_ffn_from_config",
]
