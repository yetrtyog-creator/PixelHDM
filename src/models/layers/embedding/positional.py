"""
Positional Embedding Module

Contains:
    - LearnedPositionalEmbedding: Learnable position embeddings for patches
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ....config.model_config import PixelHDMConfig


class LearnedPositionalEmbedding(nn.Module):
    """
    Learnable Positional Embedding for Patch-Level positions.

    Used as a complement or alternative to RoPE.
    """

    def __init__(
        self,
        config: Optional["PixelHDMConfig"] = None,
        max_patches: int = 4096,
        hidden_dim: int = 1024,
    ) -> None:
        super().__init__()

        if config is not None:
            hidden_dim = config.hidden_dim
            max_patches = config.max_patches

        self.max_patches = max_patches
        self.hidden_dim = hidden_dim

        self.embedding = nn.Parameter(torch.zeros(1, max_patches, hidden_dim))
        nn.init.trunc_normal_(self.embedding, std=0.02)

    def forward(self, num_patches: int) -> torch.Tensor:
        """
        Get positional embedding for given number of patches.

        Args:
            num_patches: Current number of patches

        Returns:
            Positional embedding (1, L, D)
        """
        assert num_patches <= self.max_patches, (
            f"num_patches={num_patches} > max_patches={self.max_patches}"
        )
        return self.embedding[:, :num_patches, :]

    def extra_repr(self) -> str:
        return f"max_patches={self.max_patches}, hidden_dim={self.hidden_dim}"


__all__ = [
    "LearnedPositionalEmbedding",
]
