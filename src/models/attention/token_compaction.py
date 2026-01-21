"""
Token Compaction (Compress-Attend-Expand Pipeline)

PixelHDM core: p^4 = 65,536x attention cost reduction.
Flow: Compress -> Attention -> Expand -> Residual

Author: PixelHDM-RPEA-DinoV3 Project
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Callable

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..layers.normalization import RMSNorm
from .gated_attention import GatedMultiHeadAttention

if TYPE_CHECKING:
    from ...config.model_config import PixelHDMConfig


class TokenCompaction(nn.Module):
    """
    Token Compaction: Compress-Attend-Expand Pipeline.
    Reduces complexity from O((L*p^2)^2) to O(L^2).
    Input/Output: (B, L, p^2, D_pix)
    """

    def __init__(
        self,
        config: Optional["PixelHDMConfig"] = None,
        hidden_dim: int = 1024,
        pixel_dim: int = 16,
        patch_size: int = 16,
        num_heads: int = 16,
        num_kv_heads: int = 4,
        use_checkpoint: bool = True,
        expand_gain: float = 0.1,
    ) -> None:
        super().__init__()

        # 從 config 獲取參數 (如果提供)
        if config is not None:
            hidden_dim = config.hidden_dim
            pixel_dim = config.pixel_dim
            patch_size = config.patch_size
            num_heads = config.num_heads
            num_kv_heads = config.num_kv_heads
            use_checkpoint = config.use_gradient_checkpointing
            expand_gain = config.token_compaction_expand_gain

        self.hidden_dim = hidden_dim
        self.pixel_dim = pixel_dim
        self.patch_size = patch_size
        self.p2 = patch_size ** 2  # 256
        self.p2_d_pix = self.p2 * pixel_dim  # 4096
        self.use_checkpoint = use_checkpoint
        self.expand_gain = expand_gain

        # Compress: (p^2 * D_pix) -> D
        # 4096 -> 1024
        self.compress = nn.Linear(self.p2_d_pix, hidden_dim, bias=False)

        # Pre-attention normalization
        self.norm = RMSNorm(hidden_dim)

        # 獲取門控類型和 flash attention 設置
        gate_type = "headwise"
        gate_activation = "sigmoid"
        gate_bias = False
        use_flash_attention = True
        if config is not None:
            gate_type = config.gate_type
            gate_activation = config.gate_activation
            gate_bias = config.gate_bias
            use_flash_attention = config.use_flash_attention

        # Self-Attention at patch level (使用 GQA)
        self.attention = GatedMultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            use_qk_norm=True,
            use_gated_attention=True,
            gate_type=gate_type,
            gate_activation=gate_activation,
            gate_bias=gate_bias,
            use_flash_attention=use_flash_attention,
            use_checkpoint=False,  # 外層已有 checkpoint
        )

        # Post-attention normalization
        self.post_norm = RMSNorm(hidden_dim)

        # Expand: D -> (p^2 * D_pix)
        # 1024 -> 4096
        self.expand = nn.Linear(hidden_dim, self.p2_d_pix, bias=False)

        # 初始化 (確保殘差連接初始為恆等)
        self._init_weights()

    def _init_weights(self) -> None:
        """初始化權重，expand 層使用小值初始化"""
        nn.init.xavier_uniform_(self.compress.weight)
        # 擴展層使用較小的初始化，讓初始輸出接近零
        nn.init.xavier_uniform_(self.expand.weight, gain=self.expand_gain)

    def _forward_impl(
        self,
        x: torch.Tensor,
        rope_fn: Optional[Callable] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        內部前向實現

        Args:
            x: (B, L, p^2, D_pix) - 像素特徵
            rope_fn: 可選的 RoPE 函數 (Lumina2 style)
            position_ids: Lumina2-style position IDs (B, L, 3)

        Returns:
            輸出張量 (B, L, p^2, D_pix)
        """
        B, L, P2, D = x.shape

        # 驗證維度
        assert P2 == self.p2, f"Expected p^2={self.p2}, got {P2}"
        assert D == self.pixel_dim, f"Expected D_pix={self.pixel_dim}, got {D}"

        # 保存殘差
        residual = x

        # Compress: (B, L, p^2, D_pix) -> (B, L, D)
        x_flat = x.reshape(B, L, self.p2_d_pix)  # (B, L, 4096)
        x_comp = self.compress(x_flat)  # (B, L, 1024)

        # Pre-norm
        x_comp = self.norm(x_comp)

        # Self-Attention at patch level (Lumina2 style with position_ids)
        # NOTE: No internal residual around MHSA - matches architecture diagram
        # Residual is handled at block-level via alpha gating (x + α₁ * h)
        x_comp = self.attention(
            x_comp,
            rope_fn=rope_fn,
            position_ids=position_ids,
        )

        # Post-norm
        x_comp = self.post_norm(x_comp)

        # Expand: (B, L, D) -> (B, L, p^2 * D_pix)
        x_exp = self.expand(x_comp)  # (B, L, 4096)
        x_exp = x_exp.reshape(B, L, P2, D)  # (B, L, p^2, D_pix)

        # Residual connection
        return residual + x_exp

    def forward(
        self,
        x: torch.Tensor,
        rope_fn: Optional[Callable] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向傳播

        Args:
            x: (B, L, p^2, D_pix) - 像素特徵
            rope_fn: 可選的 RoPE 函數 (Lumina2 style)
            position_ids: Lumina2-style position IDs (B, L, 3)

        Returns:
            輸出張量 (B, L, p^2, D_pix)
        """
        if x.dim() != 4:
            raise ValueError(
                f"輸入應為 4D 張量 (B, L, p^2, D_pix), 當前維度: {x.dim()}"
            )

        if self.training and self.use_checkpoint:
            return checkpoint(
                self._forward_impl,
                x, rope_fn, position_ids,
                use_reentrant=False,
            )
        else:
            return self._forward_impl(x, rope_fn, position_ids)

    def extra_repr(self) -> str:
        return (
            f"compress={self.p2_d_pix}→{self.hidden_dim}, "
            f"expand={self.hidden_dim}→{self.p2_d_pix}, "
            f"patch_size={self.patch_size}, "
            f"compression_ratio={self.p2_d_pix // self.hidden_dim}x"
        )


class TokenCompactionNoResidual(TokenCompaction):
    """
    Token Compaction 變體: 無殘差連接

    用於需要完全替換特徵的場景
    """

    def _forward_impl(
        self,
        x: torch.Tensor,
        rope_fn: Optional[Callable] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """無殘差的前向實現"""
        B, L, P2, D = x.shape

        # Compress
        x_flat = x.reshape(B, L, self.p2_d_pix)
        x_comp = self.compress(x_flat)

        # Norm + Attention (Lumina2 style with position_ids)
        # NOTE: No internal residual around MHSA - matches architecture diagram
        # Residual is handled at block-level via alpha gating (x + α₁ * h)
        x_comp = self.norm(x_comp)
        x_comp = self.attention(
            x_comp,
            rope_fn=rope_fn,
            position_ids=position_ids,
        )
        x_comp = self.post_norm(x_comp)

        # Expand (無殘差)
        x_exp = self.expand(x_comp)
        x_exp = x_exp.reshape(B, L, P2, D)

        return x_exp


__all__ = [
    "TokenCompaction",
    "TokenCompactionNoResidual",
]
