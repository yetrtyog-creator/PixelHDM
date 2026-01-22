"""
Pixel-Level Transformer Block (Core)

Pixel-Level DiT Block with Pixel-wise AdaLN and Token Compaction.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Callable, Tuple, Union

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

        # Alpha depth scaling: 1/sqrt(L) for stable residual updates
        # config=None defaults to 1.0 (no scaling) for backward compatibility
        pixel_layers = config.pixel_layers if config is not None else 1
        self.residual_scale = 1.0 / math.sqrt(pixel_layers)

        # Gamma L2 lambda for penalty (0 = disabled)
        self.gamma_l2_lambda = config.pixel_gamma_l2_lambda if config is not None else 0.0

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
        return_gamma_l2: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Internal forward implementation.

        Args:
            x: Input tensor (B, L, p^2, D_pix)
            s_cond: Semantic+time condition from Patch-Level (B, L, D)
            rope_fn: RoPE function (Lumina2 style)
            position_ids: Lumina2-style position IDs (B, L, 3)
            return_gamma_l2: Whether to return gamma L2 penalty

        Returns:
            Output tensor (B, L, p^2, D_pix), optionally with gamma_l2
        """
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaln(s_cond)

        h = self.adaln.modulate(x, gamma1, beta1)
        h = self.compaction(h, rope_fn, position_ids)
        x = x + (alpha1 * self.residual_scale) * h

        h = self.adaln.modulate(x, gamma2, beta2)
        h = self.mlp(h)
        x = x + (alpha2 * self.residual_scale) * h

        if return_gamma_l2 and self.gamma_l2_lambda > 0:
            # Compute gamma L2: mean(gamma1^2) + mean(gamma2^2)
            gamma_l2 = gamma1.pow(2).mean() + gamma2.pow(2).mean()
            return x, gamma_l2

        return x

    def forward(
        self,
        x: torch.Tensor,
        s_cond: torch.Tensor,
        rope_fn: Optional[Callable] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_gamma_l2: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input tensor (B, L, p^2, D_pix)
            s_cond: Semantic+time condition from Patch-Level (B, L, D)
            rope_fn: RoPE function (Lumina2 style)
            position_ids: Lumina2-style position IDs (B, L, 3)
            return_gamma_l2: Whether to return gamma L2 penalty

        Returns:
            Output tensor (B, L, p^2, D_pix), optionally with gamma_l2
        """
        if x.dim() != 4:
            raise ValueError(
                f"Input should be 4D tensor (B, L, p^2, D_pix), got: {x.dim()}"
            )

        if self.training and self.use_checkpoint:
            # Use checkpoint even when returning gamma_l2 to avoid VRAM spikes.
            if return_gamma_l2 and self.gamma_l2_lambda > 0:
                return checkpoint(
                    self._forward_impl,
                    x, s_cond, rope_fn, position_ids, True,
                    use_reentrant=False,
                )
            return checkpoint(
                self._forward_impl,
                x, s_cond, rope_fn, position_ids, False,
                use_reentrant=False,
            )
        else:
            return self._forward_impl(x, s_cond, rope_fn, position_ids, return_gamma_l2)

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"pixel_dim={self.pixel_dim}, "
            f"p^2={self.p2}, "
            f"adaln=pixel_wise"
        )


__all__ = ["PixelTransformerBlock"]
