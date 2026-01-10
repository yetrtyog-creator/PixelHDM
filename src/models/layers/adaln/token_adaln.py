"""
Token-Independent Adaptive Layer Normalization

Used for Patch-Level DiT: one condition applies the same modulation to all tokens.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class TokenAdaLN(nn.Module):
    """
    Token-Independent Adaptive Layer Normalization

    Used for Patch-Level DiT: one condition applies the same modulation to all tokens.

    Args:
        hidden_dim: Hidden dimension
        condition_dim: Condition dimension (usually same as hidden_dim)
        num_params: Number of modulation parameters (6 = gamma1, beta1, alpha1, gamma2, beta2, alpha2)

    Shape:
        - condition: (B, D) - time embedding or combined condition
        - output: 6 tensors of shape (B, 1, D)
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        condition_dim: Optional[int] = None,
        num_params: int = 6,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim or hidden_dim
        self.num_params = num_params

        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.condition_dim, num_params * hidden_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize weights for time conditioning.

        CRITICAL FIX (2026-01-04):
        原本使用零初始化 (nn.init.zeros_) 導致時間條件化完全失效：
        - params = SiLU(t_embed) @ 0 + bias = bias (與 t 無關！)
        - 模型對所有時間步輸出相同的值
        - 無法學習去噪，損失永遠停在 ~1

        修復：使用小的非零初始化，讓時間信息從訓練開始就能影響輸出。
        """
        # 使用小的非零初始化而非零初始化，確保時間條件化有效
        nn.init.trunc_normal_(self.proj[-1].weight, std=0.02)
        nn.init.zeros_(self.proj[-1].bias)

        # bias 仍然初始化為接近身份變換，確保訓練穩定
        with torch.no_grad():
            bias = self.proj[-1].bias.view(self.num_params, self.hidden_dim)
            bias[0].fill_(1.0)  # gamma1 = 1 (scale)
            bias[2].fill_(1.0)  # alpha1 = 1 (residual gate)
            bias[3].fill_(1.0)  # gamma2 = 1 (scale)
            bias[5].fill_(1.0)  # alpha2 = 1 (residual gate)
            # beta1, beta2 保持為 0 (shift)

    def forward(
        self,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.

        Args:
            condition: (B, D) condition tensor

        Returns:
            num_params tensors of shape (B, 1, D)
        """
        params = self.proj(condition)
        params = params.unsqueeze(1)
        params = params.chunk(self.num_params, dim=-1)

        return params

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"condition_dim={self.condition_dim}, "
            f"num_params={self.num_params}"
        )


__all__ = ["TokenAdaLN"]
