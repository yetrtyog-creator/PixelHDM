"""
Time Embedding Module

Contains:
    - TimeEmbedding: Timestep -> Time embedding
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ....config.model_config import PixelHDMConfig


class TimeEmbedding(nn.Module):
    """
    Time Step Embedding using sinusoidal positional encoding + MLP.

    Flow: t in [0, 1] -> Sinusoidal(256) -> MLP(256 -> D -> D) -> (B, D)
    """

    def __init__(
        self,
        config: Optional["PixelHDMConfig"] = None,
        hidden_dim: int = 1024,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()

        if config is not None:
            hidden_dim = config.hidden_dim
            embed_dim = config.time_embed_dim

        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self._register_frequency_buffer(embed_dim)
        self._init_weights()

    def _register_frequency_buffer(self, embed_dim: int) -> None:
        half_dim = embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def _init_weights(self) -> None:
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = self._normalize_input(t)
        embed = self._compute_sinusoidal_embedding(t)
        embed = self.mlp(embed)
        return embed

    def _normalize_input(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 2:
            t = t.squeeze(-1)
        return t

    def _compute_sinusoidal_embedding(self, t: torch.Tensor) -> torch.Tensor:
        t_scaled = t[:, None] * self.freqs[None, :] * 1000.0
        embed = torch.cat([torch.sin(t_scaled), torch.cos(t_scaled)], dim=-1)
        return embed

    def extra_repr(self) -> str:
        return f"hidden_dim={self.hidden_dim}, embed_dim={self.embed_dim}"


# === Factory Functions ===

def create_time_embedding(
    hidden_dim: int = 1024,
    embed_dim: int = 256,
) -> TimeEmbedding:
    """Create TimeEmbedding with explicit parameters."""
    return TimeEmbedding(
        config=None,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
    )


def create_time_embedding_from_config(config: "PixelHDMConfig") -> TimeEmbedding:
    """Create TimeEmbedding from config."""
    return TimeEmbedding(config=config)


__all__ = [
    "TimeEmbedding",
    "create_time_embedding",
    "create_time_embedding_from_config",
]
