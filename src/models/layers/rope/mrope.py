"""
Multi-axis Rotary Position Embedding (mRoPE) - Lumina2 Style

Unified 3-axis position encoding for joint text-image sequences:
- axis0 (16 dims): sequence position (text=0..L-1, image=L)
- axis1 (24 dims): height position (text=0, image=0..H-1)
- axis2 (24 dims): width position (text=0, image=0..W-1)

Key differences from original design:
- All tokens use unified position_ids (B, seq_len, 3) format
- Image tokens have fixed axis0=text_len (provides text/image boundary)
- Text tokens have axis1=axis2=0 (cos(0)=1, sin(0)=0 = no rotation)

Author: PixelHDM-RPEA-DinoV3 Project
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .utils import precompute_freqs_cis, apply_rotary_emb

logger = logging.getLogger(__name__)


class MRoPE(nn.Module):
    """
    Lumina2-style Multi-axis Rotary Position Embedding.

    Unified processing for text + image tokens using 3-axis position encoding:
    - axis0: sequence position (text=0..L-1, image=L fixed)
    - axis1: height position (text=0, image=0..H-1)
    - axis2: width position (text=0, image=0..W-1)

    Args:
        head_dim: Attention head dimension (default: 64)
        axes_dims: Tuple of (axis0_dim, axis1_dim, axis2_dim) (default: (16, 24, 24))
        max_seq_len: Maximum sequence length for axis0 (default: 512)
        max_height: Maximum height in patches for axis1 (default: 256)
        max_width: Maximum width in patches for axis2 (default: 256)
        theta: Frequency base (default: 10000.0)
    """

    def __init__(
        self,
        head_dim: int = 64,
        axes_dims: Tuple[int, int, int] = (16, 24, 24),
        max_seq_len: int = 512,
        max_height: int = 256,
        max_width: int = 256,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()

        # Validate dimensions
        if sum(axes_dims) != head_dim:
            raise ValueError(
                f"axes_dims sum must equal head_dim: "
                f"{axes_dims[0]} + {axes_dims[1]} + {axes_dims[2]} = {sum(axes_dims)} != {head_dim}"
            )

        self.head_dim = head_dim
        self.axes_dims = axes_dims
        self.axis0_dim, self.axis1_dim, self.axis2_dim = axes_dims

        # Precompute frequencies for each axis
        self.register_buffer(
            "freqs_axis0",
            precompute_freqs_cis(self.axis0_dim, max_seq_len, theta),
            persistent=False
        )
        self.register_buffer(
            "freqs_axis1",
            precompute_freqs_cis(self.axis1_dim, max_height, theta),
            persistent=False
        )
        self.register_buffer(
            "freqs_axis2",
            precompute_freqs_cis(self.axis2_dim, max_width, theta),
            persistent=False
        )

    def _split_dims(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split tensor by axes dimensions."""
        return (
            x[..., :self.axis0_dim],
            x[..., self.axis0_dim:self.axis0_dim + self.axis1_dim],
            x[..., self.axis0_dim + self.axis1_dim:],
        )

    def _get_freqs(
        self, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get cos/sin frequencies for all 3 axes.

        Args:
            position_ids: (B, seq_len, 3) - position IDs for each axis

        Returns:
            (cos0, sin0, cos1, sin1, cos2, sin2) - frequencies for each axis
        """
        # Extract positions for each axis
        pos0 = position_ids[..., 0]  # (B, seq_len)
        pos1 = position_ids[..., 1]  # (B, seq_len)
        pos2 = position_ids[..., 2]  # (B, seq_len)

        # Gather frequencies using index_select for better performance
        # freqs_axis*: (max_len, dim/2) -> select by positions -> (B, seq_len, dim/2)
        freqs0 = self.freqs_axis0[pos0]
        freqs1 = self.freqs_axis1[pos1]
        freqs2 = self.freqs_axis2[pos2]

        # Return cos/sin pairs
        return (
            torch.cos(freqs0), torch.sin(freqs0),
            torch.cos(freqs1), torch.sin(freqs1),
            torch.cos(freqs2), torch.sin(freqs2),
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply mRoPE uniformly to all tokens.

        Args:
            q: Query tensor (B, num_heads, seq_len, head_dim)
            k: Key tensor (B, num_heads, seq_len, head_dim)
            position_ids: Position IDs (B, seq_len, 3)

        Returns:
            (q_rotated, k_rotated): Rotated query and key tensors
        """
        # Get frequencies for all axes
        cos0, sin0, cos1, sin1, cos2, sin2 = self._get_freqs(position_ids)

        # Add head dimension for broadcasting: (B, seq_len, dim/2) -> (B, 1, seq_len, dim/2)
        cos0, sin0 = cos0.unsqueeze(1), sin0.unsqueeze(1)
        cos1, sin1 = cos1.unsqueeze(1), sin1.unsqueeze(1)
        cos2, sin2 = cos2.unsqueeze(1), sin2.unsqueeze(1)

        # Split by axes
        q0, q1, q2 = self._split_dims(q)
        k0, k1, k2 = self._split_dims(k)

        # Apply rotary embedding to each axis
        q0_rot = apply_rotary_emb(q0, cos0, sin0)
        k0_rot = apply_rotary_emb(k0, cos0, sin0)
        q1_rot = apply_rotary_emb(q1, cos1, sin1)
        k1_rot = apply_rotary_emb(k1, cos1, sin1)
        q2_rot = apply_rotary_emb(q2, cos2, sin2)
        k2_rot = apply_rotary_emb(k2, cos2, sin2)

        # Concatenate
        q_out = torch.cat([q0_rot, q1_rot, q2_rot], dim=-1)
        k_out = torch.cat([k0_rot, k1_rot, k2_rot], dim=-1)

        return q_out, k_out

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, "
            f"axes_dims={self.axes_dims}"
        )


# Backward compatibility: factory function for old config format
def create_mrope(
    head_dim: int = 64,
    text_dim: int = 16,
    img_h_dim: int = 24,
    img_w_dim: int = 24,
    text_max_len: int = 512,
    img_max_len: int = 256,
    theta: float = 10000.0,
) -> MRoPE:
    """
    Create MRoPE with old-style parameters for backward compatibility.

    Maps old parameter names to new axes_dims format:
    - text_dim -> axis0_dim
    - img_h_dim -> axis1_dim
    - img_w_dim -> axis2_dim
    """
    return MRoPE(
        head_dim=head_dim,
        axes_dims=(text_dim, img_h_dim, img_w_dim),
        max_seq_len=text_max_len,
        max_height=img_max_len,
        max_width=img_max_len,
        theta=theta,
    )


__all__ = ["MRoPE", "create_mrope"]
