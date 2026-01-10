"""
Attention Operations

Low-level attention computation utilities.

Author: PixelHDM-RPEA-DinoV3 Project
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_flash_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout_p: float, training: bool,
) -> torch.Tensor:
    """Compute attention using Flash Attention."""
    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attention_mask,
        dropout_p=dropout_p if training else 0.0,
        is_causal=False,
    )


def compute_manual_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scale: float, dropout: nn.Module,
) -> torch.Tensor:
    """Compute attention manually (non-Flash Attention)."""
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            attn_weights = attn_weights.masked_fill(~attention_mask, float('-inf'))
        else:
            attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_weights = dropout(attn_weights)
    return torch.matmul(attn_weights, v)


def prepare_attention_mask(
    attention_mask: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Prepare attention mask for computation."""
    if attention_mask is None:
        return None
    if attention_mask.dim() == 2:
        attention_mask = attention_mask.bool()
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    return attention_mask


__all__ = [
    "compute_flash_attention",
    "compute_manual_attention",
    "prepare_attention_mask",
]
