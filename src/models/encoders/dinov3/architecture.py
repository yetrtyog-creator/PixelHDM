"""
DINOv3 Architecture Components

Contains the core building blocks:
    - LayerScale: Learnable per-channel scaling
    - DINOv3Attention: Multi-Head Self-Attention with SDPA
    - DINOv3MLP: Feed-Forward MLP
    - DINOv3Block: Transformer block with LayerScale
    - PatchEmbed: Patch embedding layer

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerScale(nn.Module):
    """LayerScale (DINOv3 style)"""

    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class DINOv3Attention(nn.Module):
    """DINOv3 Multi-Head Self-Attention with Flash Attention (SDPA)"""

    def __init__(self, dim: int, num_heads: int = 12, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Use F.scaled_dot_product_attention (auto Flash Attention 2 backend)
        x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class DINOv3MLP(nn.Module):
    """DINOv3 Feed-Forward MLP"""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DINOv3Block(nn.Module):
    """DINOv3 Transformer Block with LayerScale"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        init_values: float = 1e-5,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DINOv3Attention(dim, num_heads)
        self.ls1 = LayerScale(dim, init_values)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = DINOv3MLP(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim, init_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Patch Embedding (matches DINOv3 structure)"""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


__all__ = [
    "LayerScale",
    "DINOv3Attention",
    "DINOv3MLP",
    "DINOv3Block",
    "PatchEmbed",
]
