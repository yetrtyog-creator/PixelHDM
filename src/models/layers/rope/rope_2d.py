"""
2D Rotary Position Embedding

Contains:
    - RoPE2D: 2D RoPE for image patches

Author: PixelHDM-RPEA-DinoV3 Project
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .utils import precompute_freqs_cis, apply_rotary_emb


class RoPE2D(nn.Module):
    """
    2D Rotary Position Embedding for image patches.

    Splits head_dim in half for H and W position encoding.

    Args:
        dim: Encoding dimension (head_dim)
        max_size: Maximum grid size (e.g., 1024 // 16 = 64)
        theta: Frequency base
    """

    def __init__(
        self,
        dim: int,
        max_size: int = 128,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_size = max_size
        self.theta = theta

        self.h_dim = dim // 2
        self.w_dim = dim - self.h_dim

        h_freqs = precompute_freqs_cis(self.h_dim, max_size, theta)
        w_freqs = precompute_freqs_cis(self.w_dim, max_size, theta)

        self.register_buffer("h_freqs", h_freqs, persistent=False)
        self.register_buffer("w_freqs", w_freqs, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        h_positions: torch.Tensor,
        w_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 2D RoPE.

        Args:
            q: Query (B, num_heads, seq_len, head_dim)
            k: Key (B, num_heads, seq_len, head_dim)
            h_positions: Height positions (B, seq_len) or (seq_len,)
            w_positions: Width positions (B, seq_len) or (seq_len,)

        Returns:
            (q_rotated, k_rotated)
        """
        h_freqs = self.h_freqs[h_positions]
        w_freqs = self.w_freqs[w_positions]

        if h_freqs.dim() == 2:
            h_freqs = h_freqs.unsqueeze(0).unsqueeze(0)
            w_freqs = w_freqs.unsqueeze(0).unsqueeze(0)
        else:
            h_freqs = h_freqs.unsqueeze(1)
            w_freqs = w_freqs.unsqueeze(1)

        h_cos, h_sin = torch.cos(h_freqs), torch.sin(h_freqs)
        w_cos, w_sin = torch.cos(w_freqs), torch.sin(w_freqs)

        q_h, q_w = q[..., :self.h_dim], q[..., self.h_dim:]
        k_h, k_w = k[..., :self.h_dim], k[..., self.h_dim:]

        q_h_rot = apply_rotary_emb(q_h, h_cos, h_sin)
        q_w_rot = apply_rotary_emb(q_w, w_cos, w_sin)
        k_h_rot = apply_rotary_emb(k_h, h_cos, h_sin)
        k_w_rot = apply_rotary_emb(k_w, w_cos, w_sin)

        q_rot = torch.cat([q_h_rot, q_w_rot], dim=-1)
        k_rot = torch.cat([k_h_rot, k_w_rot], dim=-1)

        return q_rot, k_rot


# Alias for backward compatibility
RotaryPositionEmbedding2D = RoPE2D


__all__ = ["RoPE2D", "RotaryPositionEmbedding2D"]
