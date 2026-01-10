"""
DINOv3 Vision Transformer (ViT)

Contains the complete ViT architecture for DINOv3.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from .architecture import DINOv3Block, PatchEmbed


@dataclass
class DINOv3Config:
    """DINOv3 ViT configuration"""

    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    patch_size: int = 16
    img_size: int = 224
    in_chans: int = 3
    mlp_ratio: float = 4.0

    # Predefined ViT configurations
    VIT_CONFIGS: Dict[str, Dict] = None

    def __post_init__(self):
        if DINOv3Config.VIT_CONFIGS is None:
            DINOv3Config.VIT_CONFIGS = {
                "vit_small": {"embed_dim": 384, "depth": 12, "num_heads": 6},
                "vit_base": {"embed_dim": 768, "depth": 12, "num_heads": 12},
                "vit_large": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
                "vit_giant": {"embed_dim": 4096, "depth": 40, "num_heads": 32},
            }

    @classmethod
    def from_model_name(cls, model_name: str) -> "DINOv3Config":
        """Create config from model name"""
        configs = {
            "dinov3-vit7b16": {"embed_dim": 4096, "depth": 40, "num_heads": 32},
            "dinov3-vitl16": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
            "dinov3-vitb16": {"embed_dim": 768, "depth": 12, "num_heads": 12},
            "dinov3-vits16": {"embed_dim": 384, "depth": 12, "num_heads": 6},
        }
        cfg = configs.get(model_name, configs["dinov3-vitb16"])
        return cls(**cfg)


class DINOv3ViT(nn.Module):
    """
    DINOv3 Vision Transformer (ViT)

    Used for loading weights from local .pth files.
    - Supports LayerScale
    - Supports RoPE (position info embedded in attention)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer blocks with LayerScale
        self.blocks = nn.ModuleList([
            DINOv3Block(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # Final norm
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Note: DINOv3 uses RoPE, position info is embedded in attention
        # For simplicity, we skip explicit pos_embed (RoPE handles it)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    @classmethod
    def from_config(cls, config: DINOv3Config) -> "DINOv3ViT":
        """Create ViT from config"""
        return cls(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
        )


__all__ = [
    "DINOv3Config",
    "DINOv3ViT",
]
