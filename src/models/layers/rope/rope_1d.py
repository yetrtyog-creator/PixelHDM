"""
1D Rotary Position Embedding

Contains:
    - RoPE1D: Standard 1D RoPE for text sequences

Author: PixelHDM-RPEA-DinoV3 Project
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .utils import precompute_freqs_cis, apply_rotary_emb


class RoPE1D(nn.Module):
    """
    1D Rotary Position Embedding.

    Used for standard RoPE on text sequences.

    Args:
        dim: Encoding dimension
        max_seq_len: Maximum sequence length
        theta: Frequency base

    Example:
        >>> rope = RoPE1D(64, 8192)
        >>> q = torch.randn(2, 16, 512, 64)
        >>> k = torch.randn(2, 16, 512, 64)
        >>> positions = torch.arange(512).unsqueeze(0).expand(2, -1)
        >>> q_rot, k_rot = rope(q, k, positions)
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        freqs = precompute_freqs_cis(dim, max_seq_len, theta)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 1D RoPE.

        Args:
            q: Query (B, num_heads, seq_len, head_dim)
            k: Key (B, num_heads, seq_len, head_dim)
            positions: Position indices (B, seq_len)

        Returns:
            (q_rotated, k_rotated)
        """
        freqs = self.freqs[positions]
        freqs = freqs.unsqueeze(1)

        freqs_cos = torch.cos(freqs)
        freqs_sin = torch.sin(freqs)

        q_rot = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k_rot = apply_rotary_emb(k, freqs_cos, freqs_sin)

        return q_rot, k_rot


__all__ = ["RoPE1D"]
