"""
Frequency-aware Loss (Core)

Based on DeCo paper: Frequency-Decoupled Pixel Diffusion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    JPEG_LUMINANCE_QUANTIZATION_TABLE,
    JPEG_CHROMINANCE_QUANTIZATION_TABLE,
)
from .dct import BlockDCT2D
from .color import rgb_to_ycbcr

if TYPE_CHECKING:
    from ....config.model_config import PixelHDMConfig


class FrequencyLoss(nn.Module):
    """
    Frequency-aware Loss

    Uses 8×8 DCT + JPEG quantization weights.

    Args:
        config: FreqLossConfig or PixelHDMConfig
        quality: JPEG quality factor
        weight: Loss weight

    Shape:
        - v_pred: (B, C, H, W)
        - v_target: (B, C, H, W)
        - Output: scalar
    """

    def __init__(
        self,
        config: "PixelHDMConfig" = None,
        quality: int = 90,
        weight: float = 1.0,
        use_ycbcr: bool = True,
    ) -> None:
        super().__init__()

        if config is not None:
            quality = config.freq_loss_quality
            weight = config.freq_loss_lambda
            use_ycbcr = config.freq_loss_use_ycbcr
            enabled = config.freq_loss_enabled
        else:
            enabled = True

        self._enabled = enabled
        self.quality = quality
        self.weight = weight
        self.use_ycbcr = use_ycbcr
        self.block_size = 8

        if not self._enabled:
            return

        self.dct = BlockDCT2D(block_size=self.block_size)

        weights_y, weights_c = self._compute_jpeg_weights(quality)
        self.register_buffer('weights_y', weights_y)
        self.register_buffer('weights_c', weights_c)

    def _compute_jpeg_weights(
        self,
        quality: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute frequency weights based on JPEG quality factor."""
        quality = max(1, min(100, quality))

        if quality < 50:
            scale = 5000 / quality
        else:
            scale = 200 - 2 * quality

        def scale_table(q_base):
            q_scaled = torch.floor((q_base * scale + 50) / 100)
            return torch.clamp(q_scaled, min=1, max=255)

        q_y = scale_table(JPEG_LUMINANCE_QUANTIZATION_TABLE)
        q_c = scale_table(JPEG_CHROMINANCE_QUANTIZATION_TABLE)

        def to_weights(q):
            w = 1.0 / q
            return w / w.mean()

        return to_weights(q_y), to_weights(q_c)

    def _apply_block_weights(
        self,
        dct_coeffs: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Apply weights to 8×8 block DCT coefficients."""
        B, C, H, W = dct_coeffs.shape
        bs = self.block_size

        H_pad = (bs - H % bs) % bs
        W_pad = (bs - W % bs) % bs

        if H_pad > 0 or W_pad > 0:
            dct_coeffs = F.pad(dct_coeffs, (0, W_pad, 0, H_pad), mode='constant', value=0)

        _, _, H_new, W_new = dct_coeffs.shape
        H_blocks = H_new // bs
        W_blocks = W_new // bs

        coeffs_blocks = dct_coeffs.view(B, C, H_blocks, bs, W_blocks, bs)
        coeffs_blocks = coeffs_blocks.permute(0, 1, 2, 4, 3, 5)

        weights_exp = weights.view(1, 1, 1, 1, bs, bs).to(dct_coeffs.dtype)
        weighted_blocks = coeffs_blocks * weights_exp.sqrt()

        weighted_blocks = weighted_blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
        result = weighted_blocks.view(B, C, H_new, W_new)

        if H_pad > 0 or W_pad > 0:
            result = result[:, :, :H, :W]

        return result

    def forward(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute frequency-aware loss.

        Args:
            v_pred: Predicted velocity (B, C, H, W)
            v_target: Target velocity (B, C, H, W)

        Returns:
            Loss value
        """
        if not self._enabled:
            return torch.tensor(0.0, device=v_pred.device, dtype=v_pred.dtype)

        original_dtype = v_pred.dtype
        v_pred = v_pred.float()
        v_target = v_target.float()

        B, C, H, W = v_pred.shape

        if self.use_ycbcr and C == 3:
            v_pred = rgb_to_ycbcr(v_pred)
            v_target = rgb_to_ycbcr(v_target)
            use_chroma = True
        else:
            use_chroma = False

        V_pred = self.dct(v_pred)
        V_target = self.dct(v_target)

        _, C_new, _, _ = V_pred.shape

        if use_chroma and C_new == 3:
            V_pred_y = self._apply_block_weights(V_pred[:, 0:1], self.weights_y)
            V_pred_c = self._apply_block_weights(V_pred[:, 1:3], self.weights_c)
            V_pred_weighted = torch.cat([V_pred_y, V_pred_c], dim=1)

            V_target_y = self._apply_block_weights(V_target[:, 0:1], self.weights_y)
            V_target_c = self._apply_block_weights(V_target[:, 1:3], self.weights_c)
            V_target_weighted = torch.cat([V_target_y, V_target_c], dim=1)
        else:
            V_pred_weighted = self._apply_block_weights(V_pred, self.weights_y)
            V_target_weighted = self._apply_block_weights(V_target, self.weights_y)

        diff_sq = (V_pred_weighted - V_target_weighted).pow(2)
        loss = diff_sq.mean()

        return (loss * self.weight).to(original_dtype)

    @property
    def enabled(self) -> bool:
        return self._enabled


__all__ = ["FrequencyLoss"]
