"""
PixelHDM Core Model

Contains the main PixelHDM dual-path Diffusion Transformer class.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict, Tuple, Union

import torch
import torch.nn as nn

from ..layers.embedding import (
    PixelEmbedding,
    PatchEmbedding,
    TimeEmbedding,
    PixelPatchify,
)
from ..layers.normalization import RMSNorm
from ..layers.rope import (
    create_rope_from_config,
    create_position_ids_batched,
    create_image_only_position_ids_batched,
)
from ..blocks.patch_block import PatchTransformerBlock
from ..blocks.pixel_block import PixelTransformerBlock

if TYPE_CHECKING:
    from ...config.model_config import PixelHDMConfig


class PixelHDM(nn.Module):
    """
    PixelHDM-RPEA-DinoV3 Main Model

    Dual-path Diffusion Transformer for high-quality image generation.

    Architecture (雙輸入路徑):
        Noised Image x_t
            │
            ├─→ 16×16 Patchify ─→ DiT Blocks ─→ Semantic Tokens ─→ s_cond (AdaLN)
            │                                                          ↓
            └─→ 1×1 Patchify ──→ Pixel Tokens ─────────────────→ PiT Blocks ─→ Output
                (per-pixel)      (high-freq)                      (Token Compaction)

    Key Components:
        - 16×16 Patchify (PatchEmbedding): 語義理解路徑
        - 1×1 Patchify (PixelEmbedding): 高頻細節保留路徑
        - Token Compaction: O((L×p²)²) → O(L²) 複雜度降維

    Prediction Type: V-Prediction (velocity)
        - 網路直接輸出 velocity v = x - ε
        - 符合 PixelHDM 論文的 velocity-matching loss 設計
        - 避免 X-Prediction 在 t→1 時的數值不穩定 (1/(1-t) 放大誤差)

    Args:
        config: PixelHDMConfig configuration

    Shape:
        - x_t: (B, H, W, 3) or (B, C, H, W) - noisy image
        - t: (B,) - timestep [0, 1]
        - text_embed: (B, T, D) - text embedding (optional)
        - text_mask: (B, T) - text attention mask (optional)
        - Output: (B, H, W, 3) - predicted velocity v
    """

    def __init__(self, config: "PixelHDMConfig") -> None:
        super().__init__()
        self.config = config
        self._init_dimensions(config)
        self._init_embeddings(config)
        self._init_blocks(config)
        self._init_output(config)
        self._init_weights()

    def _init_dimensions(self, config: "PixelHDMConfig") -> None:
        """Initialize dimension parameters."""
        self.hidden_dim = config.hidden_dim
        self.pixel_dim = config.pixel_dim
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.patch_layers = config.patch_layers
        self.pixel_layers = config.pixel_layers
        self.repa_align_layer = config.repa_align_layer - 1

    def _init_embeddings(self, config: "PixelHDMConfig") -> None:
        """Initialize embedding layers."""
        self.patch_embed = PatchEmbedding(config=config)  # 16×16 Patchify
        self.pixel_embed = PixelEmbedding(config=config)  # 1×1 Patchify (per-pixel)
        self.time_embed = TimeEmbedding(config=config)
        self.rope_2d = create_rope_from_config(config)

    def _init_blocks(self, config: "PixelHDMConfig") -> None:
        """Initialize transformer blocks."""
        self.patch_blocks = nn.ModuleList([
            PatchTransformerBlock(config=config)
            for _ in range(self.patch_layers)
        ])
        self.pixel_blocks = nn.ModuleList([
            PixelTransformerBlock(config=config)
            for _ in range(self.pixel_layers)
        ])

    def _init_output(self, config: "PixelHDMConfig") -> None:
        """Initialize output layers."""
        # Note: PixelUnpatchify removed in 2026-01-06 architecture update
        # Pixel tokens now come from pixel_embed (1×1 Patchify) instead of
        # being expanded from semantic tokens via unpatchify
        self.output_norm = RMSNorm(self.pixel_dim)
        self.output_proj = PixelPatchify(config=config)

    def _init_weights(self) -> None:
        """Initialize weights."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

        self.apply(_basic_init)
        self._init_output_proj()
        self._reinit_adaln()

    def _reinit_adaln(self) -> None:
        """Re-initialize AdaLN after _basic_init overwrites bias to zero."""
        for block in self.patch_blocks:
            block.adaln._init_weights()
        for block in self.pixel_blocks:
            block.adaln._init_weights()

    def _init_output_proj(self) -> None:
        # Removed: small std init caused output to be ~17x too small
        # Xavier init from _basic_init is sufficient
        pass

    def _create_joint_sequence(
        self,
        img_tokens: torch.Tensor,
        text_embed: Optional[torch.Tensor],
        text_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
        """Create joint text-image sequence (Lumina2 style)."""
        if text_embed is None:
            return img_tokens, None, 0

        B, L, D = img_tokens.shape
        T = text_embed.shape[1]

        joint_tokens = torch.cat([text_embed, img_tokens], dim=1)
        img_mask = torch.ones(B, L, dtype=torch.bool, device=img_tokens.device)

        if text_mask is None:
            text_mask = torch.ones(B, T, dtype=torch.bool, device=img_tokens.device)
        else:
            text_mask = text_mask.bool()

        joint_mask = torch.cat([text_mask, img_mask], dim=1)
        return joint_tokens, joint_mask, T

    def _extract_image_tokens(
        self, joint_tokens: torch.Tensor, text_len: int
    ) -> torch.Tensor:
        """Extract image tokens from joint sequence."""
        if text_len == 0:
            return joint_tokens
        return joint_tokens[:, text_len:]

    def _process_input(self, x_t: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Process input tensor to (B, H, W, C) format."""
        if x_t.dim() != 4:
            raise ValueError(
                f"Input should be 4D tensor, expected (B, H, W, 3) or (B, 3, H, W), "
                f"got {x_t.dim()}D"
            )
        if x_t.shape[1] == self.in_channels:
            x_t = x_t.permute(0, 2, 3, 1)
        return x_t, (x_t.shape[1], x_t.shape[2])

    def _create_rope_fn(self, position_ids: torch.Tensor):
        """Create RoPE function for attention with Lumina2-style position IDs.

        Args:
            position_ids: Pre-computed position IDs (B, seq_len, 3)
        """
        rope_2d = self.rope_2d

        def rope_fn(q, k, pos_ids=None):
            # Use provided position_ids or default
            ids = pos_ids if pos_ids is not None else position_ids
            return rope_2d(q, k, ids)

        return rope_fn

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        pooled_text_embed: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Args:
            x_t: Noisy image (B, H, W, 3) or (B, 3, H, W)
            t: Timestep (B,)
            text_embed: Text embedding sequence (B, T, D)
            text_mask: Text attention mask (B, T)
            pooled_text_embed: Pooled text embedding (B, D) for AdaLN conditioning
            return_features: Whether to return REPA features

        Returns:
            Predicted velocity v (B, H, W, 3), optionally with REPA features

        Note:
            V-Prediction: 網路直接輸出 velocity v = x - ε
            用於 ODE 積分: z_{t+dt} = z_t + dt * v
        """
        x_t, (H, W) = self._process_input(x_t)
        B = x_t.shape[0]

        x = self.patch_embed(x_t)
        t_embed = self.time_embed(t)

        x, joint_mask, text_len = self._create_joint_sequence(x, text_embed, text_mask)

        # Create Lumina2-style position IDs for joint text+image sequence
        position_ids = create_position_ids_batched(
            batch_size=B,
            text_len=text_len,
            img_height=H,
            img_width=W,
            patch_size=self.patch_size,
            device=x.device,
        )
        rope_fn = self._create_rope_fn(position_ids)

        repa_features = None
        for i, block in enumerate(self.patch_blocks):
            x = block(x, t_embed=t_embed, rope_fn=rope_fn, position_ids=position_ids, attention_mask=joint_mask)
            if return_features and i == self.repa_align_layer:
                repa_features = self._extract_image_tokens(x, text_len).clone()

        semantic_tokens = self._extract_image_tokens(x, text_len)

        # s_cond: 結合語義 tokens、時間嵌入和池化文本嵌入 (SD3/Lumina style)
        # 這使得文本和語義信息直接參與 PixelTransformer 的 AdaLN 調制
        s_cond = semantic_tokens + t_embed.unsqueeze(1)
        if pooled_text_embed is not None:
            s_cond = s_cond + pooled_text_embed.unsqueeze(1)

        # 1×1 Patchify: 直接從輸入圖像獲取像素級特徵 (保留高頻細節)
        # 這是 PixelHDM 論文的設計 - 雙輸入路徑
        x = self.pixel_embed(x_t)

        # 創建像素級位置編碼 (Lumina2 style, image-only)
        # Token Compaction 內的 Attention 需要位置信息
        # 注意: 像素級沒有文本 token，所以使用 image-only position IDs
        pixel_position_ids = create_image_only_position_ids_batched(
            batch_size=B,
            img_height=H,
            img_width=W,
            patch_size=self.patch_size,
            device=x.device,
        )
        pixel_rope_fn = self._create_rope_fn(pixel_position_ids)

        for block in self.pixel_blocks:
            x = block(x, s_cond=s_cond, rope_fn=pixel_rope_fn, position_ids=pixel_position_ids)

        x = self.output_norm(x)
        x = self.output_proj(x, image_size=(H, W))

        if return_features:
            return x, repa_features
        return x

    def forward_with_cfg(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        pooled_text_embed: Optional[torch.Tensor] = None,
        cfg_scale: float = 7.5,
        null_text_embed: Optional[torch.Tensor] = None,
        null_text_mask: Optional[torch.Tensor] = None,
        null_pooled_text_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with Classifier-Free Guidance."""
        x_cond = self.forward(x_t, t, text_embed, text_mask, pooled_text_embed)

        if cfg_scale == 1.0 or null_text_embed is None:
            return x_cond

        # Use null_text_mask for unconditional branch (fallback to text_mask if None)
        uncond_mask = null_text_mask if null_text_mask is not None else text_mask
        x_uncond = self.forward(x_t, t, null_text_embed, uncond_mask, null_pooled_text_embed)
        return x_uncond + cfg_scale * (x_cond - x_uncond)

    def get_repa_features(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        pooled_text_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get REPA alignment layer features."""
        _, features = self.forward(
            x_t, t, text_embed, text_mask, pooled_text_embed, return_features=True
        )
        return features

    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "patch_level": sum(p.numel() for p in self.patch_blocks.parameters()),
            "pixel_level": sum(p.numel() for p in self.pixel_blocks.parameters()),
            "embeddings": sum(p.numel() for p in self.patch_embed.parameters())
                        + sum(p.numel() for p in self.pixel_embed.parameters())
                        + sum(p.numel() for p in self.time_embed.parameters()),
        }

    def extra_repr(self) -> str:
        params = self.count_parameters()
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"pixel_dim={self.pixel_dim}, "
            f"patch_size={self.patch_size}, "
            f"patch_layers={self.patch_layers}, "
            f"pixel_layers={self.pixel_layers}, "
            f"params={params['total']/1e6:.1f}M"
        )
