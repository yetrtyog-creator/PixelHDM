"""
Patch-Level Transformer Block Stack

N-layer stack of PatchTransformerBlocks with optional REPA feature extraction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Callable, Tuple

import torch
import torch.nn as nn

from .core import PatchTransformerBlock

if TYPE_CHECKING:
    from ....config.model_config import PixelHDMConfig


class PatchTransformerBlockStack(nn.Module):
    """
    Patch-Level Transformer Block Stack

    N layers of PatchTransformerBlock, optionally returning intermediate features for REPA.

    Args:
        config: PixelHDMConfig configuration
        num_layers: Number of layers (N)
        repa_layer: REPA alignment layer index (0-indexed)
    """

    def __init__(
        self,
        config: Optional["PixelHDMConfig"] = None,
        num_layers: int = 16,
        repa_layer: int = 8,
        hidden_dim: int = 1024,
        num_heads: int = 16,
        num_kv_heads: int = 4,
        mlp_ratio: float = 3.0,
    ) -> None:
        super().__init__()

        if config is not None:
            num_layers = config.patch_layers
            repa_layer = config.repa_align_layer
            hidden_dim = config.hidden_dim
            num_heads = config.num_heads
            num_kv_heads = config.num_kv_heads
            mlp_ratio = config.mlp_ratio

        self.num_layers = num_layers
        self.repa_layer = repa_layer

        self.blocks = nn.ModuleList([
            PatchTransformerBlock(
                config=config,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        t_embed: torch.Tensor,
        rope_fn: Optional[Callable] = None,
        text_positions: Optional[torch.Tensor] = None,
        img_positions: Optional[torch.Tensor] = None,
        text_len: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_repa_features: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: (B, L, D) input features
            t_embed: (B, D) time embedding
            rope_fn: mRoPE function
            text_positions: text positions
            img_positions: image positions
            text_len: text length
            attention_mask: attention mask
            return_repa_features: whether to return REPA alignment layer features

        Returns:
            output: (B, L, D) output features
            repa_features: Optional[(B, L, D)] REPA features if return_repa_features=True
        """
        repa_features = None

        for i, block in enumerate(self.blocks):
            x = block(
                x, t_embed, rope_fn,
                text_positions, img_positions, text_len, attention_mask
            )

            if return_repa_features and i == self.repa_layer:
                repa_features = x.clone()

        return x, repa_features


__all__ = ["PatchTransformerBlockStack"]
