"""
Pixel-Level Transformer Block (Core)

Pixel-Level DiT Block with Pixel-wise AdaLN and Token Compaction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Callable

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ...layers.normalization import RMSNorm
from ...layers.feedforward import SwiGLU
from ...layers.adaln import PixelwiseAdaLN
from ...attention.token_compaction import TokenCompactionNoResidual

if TYPE_CHECKING:
    from ....config.model_config import PixelHDMConfig


class PixelTransformerBlock(nn.Module):
    """
    Pixel-Level DiT Transformer Block

    Uses Pixel-wise AdaLN and Token Compaction for fine-grained processing.

    Shape:
        - x: (B, L, p^2, D_pix)
        - s_cond: (B, L, D) - semantic+time condition from Patch-Level
        - Output: (B, L, p^2, D_pix)
    """

    def __init__(
        self,
        config: Optional["PixelHDMConfig"] = None,
        hidden_dim: int = 1024,
        pixel_dim: int = 16,
        patch_size: int = 16,
        mlp_ratio: float = 3.0,
        num_heads: int = 16,
        num_kv_heads: int = 4,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()

        if config is not None:
            hidden_dim = config.hidden_dim
            pixel_dim = config.pixel_dim
            patch_size = config.patch_size
            mlp_ratio = config.mlp_ratio
            num_heads = config.num_heads
            num_kv_heads = config.num_kv_heads
            use_checkpoint = config.use_gradient_checkpointing

        self.hidden_dim = hidden_dim
        self.pixel_dim = pixel_dim
        self.patch_size = patch_size
        self.p2 = patch_size ** 2
        self.use_checkpoint = use_checkpoint

        self.adaln = PixelwiseAdaLN(
            config=config,
            hidden_dim=hidden_dim,
            pixel_dim=pixel_dim,
            patch_size=patch_size,
            num_params=6,
        )

        # Use NoResidual version - residual is handled by outer block with alpha gating
        # This matches the architecture diagram: x + α₁ × expand(attn(compress(modulate(x))))
        self.compaction = TokenCompactionNoResidual(
            config=config,
            hidden_dim=hidden_dim,
            pixel_dim=pixel_dim,
            patch_size=patch_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            use_checkpoint=False,
        )

        mlp_dim = int(pixel_dim * mlp_ratio)
        self.mlp = SwiGLU(
            hidden_dim=pixel_dim,
            mlp_dim=mlp_dim,
            dropout=0.0,
        )

    def _forward_impl(
        self,
        x: torch.Tensor,
        s_cond: torch.Tensor,
        rope_fn: Optional[Callable] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Internal forward implementation.

        Args:
            x: Input tensor (B, L, p^2, D_pix)
            s_cond: Semantic+time condition from Patch-Level (B, L, D)
            rope_fn: RoPE function (Lumina2 style)
            position_ids: Lumina2-style position IDs (B, L, 3)

        Returns:
            Output tensor (B, L, p^2, D_pix)
        """
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaln(s_cond)

        h = self.adaln.modulate(x, gamma1, beta1)
        h = self.compaction(h, rope_fn, position_ids)
        x = x + alpha1 * h

        h = self.adaln.modulate(x, gamma2, beta2)
        h = self.mlp(h)
        x = x + alpha2 * h

        return x

    def forward(
        self,
        x: torch.Tensor,
        s_cond: torch.Tensor,
        rope_fn: Optional[Callable] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, L, p^2, D_pix)
            s_cond: Semantic+time condition from Patch-Level (B, L, D)
            rope_fn: RoPE function (Lumina2 style)
            position_ids: Lumina2-style position IDs (B, L, 3)

        Returns:
            Output tensor (B, L, p^2, D_pix)
        """
        if x.dim() != 4:
            raise ValueError(
                f"Input should be 4D tensor (B, L, p^2, D_pix), got: {x.dim()}"
            )

        if self.training and self.use_checkpoint:
            return checkpoint(
                self._forward_impl,
                x, s_cond, rope_fn, position_ids,
                use_reentrant=False,
            )
        else:
            return self._forward_impl(x, s_cond, rope_fn, position_ids)

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"pixel_dim={self.pixel_dim}, "
            f"p^2={self.p2}, "
            f"adaln=pixel_wise"
        )


__all__ = ["PixelTransformerBlock"]
