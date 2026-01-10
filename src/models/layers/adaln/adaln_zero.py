"""
AdaLN-Zero: Zero-initialized AdaLN

Same as standard AdaLN, but alpha parameters are initialized to 0.
Makes the model's initial behavior close to skipping this layer.
Commonly used for DiT's final layer.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..normalization import RMSNorm


class AdaLNZero(nn.Module):
    """
    AdaLN-Zero: Zero-initialized AdaLN

    Same as standard AdaLN, but alpha parameters are initialized to 0.
    Makes the model's initial behavior close to skipping this layer.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        condition_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim or hidden_dim

        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.condition_dim, 3 * hidden_dim),
        )

        self.norm = RMSNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Zero initialization."""
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

        with torch.no_grad():
            bias = self.proj[-1].bias.view(3, self.hidden_dim)
            bias[0].fill_(1.0)  # gamma

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (B, L, D) input (used for getting batch size)
            condition: (B, D) condition

        Returns:
            normalized: Normalized x after modulation
            alpha: Residual gate parameter
        """
        params = self.proj(condition)
        gamma, beta, alpha = params.chunk(3, dim=-1)

        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        alpha = alpha.unsqueeze(1)

        normalized = gamma * self.norm(x) + beta

        return normalized, alpha


__all__ = ["AdaLNZero"]
