"""
DCT (Discrete Cosine Transform) Utilities
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_dct_matrix(n: int = 8) -> torch.Tensor:
    """Create N×N DCT-II transform matrix."""
    dct_matrix = torch.zeros(n, n)

    for k in range(n):
        for i in range(n):
            if k == 0:
                dct_matrix[k, i] = 1.0 / math.sqrt(n)
            else:
                dct_matrix[k, i] = math.sqrt(2.0 / n) * math.cos(
                    math.pi * k * (2 * i + 1) / (2 * n)
                )

    return dct_matrix


class BlockDCT2D(nn.Module):
    """2D Block-wise DCT Transform."""

    def __init__(self, block_size: int = 8):
        super().__init__()
        self.block_size = block_size

        dct_matrix = create_dct_matrix(block_size)
        self.register_buffer('dct_matrix', dct_matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward DCT transform."""
        B, C, H, W = x.shape
        bs = self.block_size

        # Padding
        H_pad = (bs - H % bs) % bs
        W_pad = (bs - W % bs) % bs

        if H_pad > 0 or W_pad > 0:
            x = F.pad(x, (0, W_pad, 0, H_pad), mode='reflect')

        _, _, H_new, W_new = x.shape

        # Reshape into blocks
        x = x.view(B, C, H_new // bs, bs, W_new // bs, bs)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()

        # Apply 2D DCT: Y = D @ X @ D^T
        dct = self.dct_matrix.to(x.dtype)
        x_dct = torch.einsum('ij,bcmnjk,kl->bcmnil', dct, x, dct)

        # Restore shape
        x_dct = x_dct.permute(0, 1, 2, 4, 3, 5).contiguous()
        x_dct = x_dct.view(B, C, H_new, W_new)

        if H_pad > 0 or W_pad > 0:
            x_dct = x_dct[:, :, :H, :W]

        return x_dct


__all__ = ["create_dct_matrix", "BlockDCT2D"]
