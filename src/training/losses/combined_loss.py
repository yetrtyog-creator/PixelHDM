"""
PixelHDM-RPEA-DinoV3 Combined Loss (Triple Loss System)

整合三個損失函數:
    - V-Loss: velocity-matching loss (V-Prediction)
    - Frequency Loss: DeCo 風格的頻率感知損失
    - REPA Loss: DINOv3 特徵對齊損失

損失公式:
    L = L_vloss + λ_freq × L_freq + λ_repa × L_REPA

V-Prediction 設計:
    - 網路直接輸出 velocity v = x - ε
    - 無需 X→V 轉換，避免 1/(1-t) 誤差放大

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict, Any

import torch
import torch.nn as nn

from .vloss import VLossWithVelocity
from .freq_loss import FrequencyLoss
from .repa_loss import REPALoss

if TYPE_CHECKING:
    from ...config.model_config import PixelHDMConfig


class CombinedLoss(nn.Module):
    """
    Triple Loss System: V-Loss + Frequency Loss + REPA Loss

    整合三個損失函數，提供統一的訓練接口

    損失組成:
        - V-Loss: 主要損失，velocity space MSE
        - Freq Loss: 頻率感知損失，強調低頻視覺重要成分
        - REPA Loss: DINOv3 特徵對齊，加速訓練 (250K 步後停用)

    Args:
        config: PixelHDMConfig 配置

    Shape:
        - v_pred: (B, H, W, 3) 或 (B, 3, H, W) 網路預測的 velocity
        - x_clean: (B, H, W, 3) 或 (B, 3, H, W) 乾淨圖像
        - noise: (B, H, W, 3) 或 (B, 3, H, W) 噪聲
        - h_t: (B, L, D) 模型中間層特徵 (REPA 用，可選)
        - Output: Dict with total loss and breakdown
    """

    def __init__(
        self,
        config: Optional["PixelHDMConfig"] = None,
        lambda_freq: float = 1.0,
        lambda_repa: float = 0.5,
        freq_quality: int = 90,
        repa_early_stop: int = 250000,
        lambda_gamma_l2: float = 1e-4,
    ) -> None:
        super().__init__()

        if config is not None:
            # 直接使用配置參數 (與 PixelHDMConfig 保持一致)
            lambda_freq = config.freq_loss_lambda
            freq_quality = config.freq_loss_quality
            lambda_repa = config.repa_lambda
            repa_early_stop = config.repa_early_stop
            lambda_gamma_l2 = config.pixel_gamma_l2_lambda

        self.lambda_freq = lambda_freq
        self.lambda_repa = lambda_repa
        self.lambda_gamma_l2 = lambda_gamma_l2

        # V-Loss (with velocity output for freq loss)
        self.vloss = VLossWithVelocity(config=config)

        # Frequency Loss
        self.freq_loss = FrequencyLoss(
            config=config,
            quality=freq_quality,
            weight=lambda_freq,
        )

        # REPA Loss
        self.repa_loss = REPALoss(
            config=config,
            lambda_repa=lambda_repa,
            early_stop_step=repa_early_stop,
        )

        self._repa_enabled = True

    def set_dino_encoder(self, encoder: nn.Module) -> None:
        """設置 DINOv3 編碼器 (用於 REPA Loss)"""
        self.repa_loss.set_dino_encoder(encoder)

    def disable_repa(self) -> None:
        """禁用 REPA Loss"""
        self._repa_enabled = False

    def enable_repa(self) -> None:
        """啟用 REPA Loss"""
        self._repa_enabled = True

    def forward(
        self,
        v_pred: torch.Tensor,
        x_clean: torch.Tensor,
        noise: torch.Tensor,
        h_t: Optional[torch.Tensor] = None,
        step: int = 0,
        dino_features: Optional[torch.Tensor] = None,
        gamma_l2: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        計算 Combined Loss

        Args:
            v_pred: 網路預測的 velocity (V-Prediction)
            x_clean: 乾淨圖像 (目標)
            noise: 噪聲
            h_t: 模型中間層特徵 (REPA 用，可選)
            step: 當前訓練步數
            dino_features: 預計算的 DINOv3 特徵 (可選)
            gamma_l2: Pixel gamma L2 penalty from model (可選)

        Returns:
            Dict containing:
                - total: 總損失
                - vloss: V-Loss
                - freq_loss: Frequency Loss
                - repa_loss: REPA Loss
                - gamma_l2: Gamma L2 penalty
        """
        # 處理輸入格式 (確保是 BCHW)
        if v_pred.dim() == 4 and v_pred.shape[-1] == 3:
            # (B, H, W, C) → (B, C, H, W)
            v_pred = v_pred.permute(0, 3, 1, 2)
            x_clean = x_clean.permute(0, 3, 1, 2)
            noise = noise.permute(0, 3, 1, 2)

        # 1. V-Loss (V-Prediction: 網路直接輸出 velocity)
        loss_vloss, v_pred_out, v_target = self.vloss.forward_with_velocity(
            v_pred, x_clean, noise
        )

        # 2. Frequency Loss
        if self.freq_loss.enabled:
            loss_freq = self.freq_loss(v_pred_out, v_target)
        else:
            loss_freq = torch.tensor(0.0, device=v_pred.device, dtype=v_pred.dtype)

        # 3. REPA Loss
        if self._repa_enabled and h_t is not None:
            loss_repa = self.repa_loss(h_t, x_clean, step, dino_features)
        else:
            loss_repa = torch.tensor(0.0, device=v_pred.device, dtype=v_pred.dtype)

        # 4. Gamma L2 Penalty
        if gamma_l2 is not None and self.lambda_gamma_l2 > 0:
            loss_gamma_l2 = self.lambda_gamma_l2 * gamma_l2
        else:
            loss_gamma_l2 = torch.tensor(0.0, device=v_pred.device, dtype=v_pred.dtype)

        # 總損失
        total = loss_vloss + loss_freq + loss_repa + loss_gamma_l2

        return {
            "total": total,
            "vloss": loss_vloss.detach(),
            "freq_loss": loss_freq.detach(),
            "repa_loss": loss_repa.detach(),
            "gamma_l2": loss_gamma_l2.detach(),
        }


class CombinedLossSimple(nn.Module):
    """
    簡化版 Combined Loss (無 REPA)

    只使用 V-Loss + Frequency Loss
    適用於不需要 REPA 對齊的場景
    """

    def __init__(
        self,
        config: Optional["PixelHDMConfig"] = None,
        lambda_freq: float = 1.0,
        freq_quality: int = 90,
    ) -> None:
        super().__init__()

        if config is not None:
            # 直接使用配置參數 (與 PixelHDMConfig 保持一致)
            lambda_freq = config.freq_loss_lambda
            freq_quality = config.freq_loss_quality

        self.lambda_freq = lambda_freq

        self.vloss = VLossWithVelocity(config=config)
        self.freq_loss = FrequencyLoss(
            config=config,
            quality=freq_quality,
            weight=lambda_freq,
        )

    def forward(
        self,
        v_pred: torch.Tensor,
        x_clean: torch.Tensor,
        noise: torch.Tensor,
        gamma_l2: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        計算簡化版 Combined Loss

        Args:
            v_pred: 網路預測的 velocity (V-Prediction)
            x_clean: 乾淨圖像 (目標)
            noise: 噪聲
            gamma_l2: Pixel gamma L2 penalty (ignored in simple version)

        Returns:
            Dict containing:
                - total: 總損失
                - vloss: V-Loss
                - freq_loss: Frequency Loss
                - gamma_l2: Always 0 (for compatibility)
        """
        # 處理輸入格式
        if v_pred.dim() == 4 and v_pred.shape[-1] == 3:
            v_pred = v_pred.permute(0, 3, 1, 2)
            x_clean = x_clean.permute(0, 3, 1, 2)
            noise = noise.permute(0, 3, 1, 2)

        # V-Loss (V-Prediction)
        loss_vloss, v_pred_out, v_target = self.vloss.forward_with_velocity(
            v_pred, x_clean, noise
        )

        # Frequency Loss
        if self.freq_loss.enabled:
            loss_freq = self.freq_loss(v_pred_out, v_target)
        else:
            loss_freq = torch.tensor(0.0, device=v_pred.device, dtype=v_pred.dtype)

        total = loss_vloss + loss_freq

        return {
            "total": total,
            "vloss": loss_vloss.detach(),
            "freq_loss": loss_freq.detach(),
            "gamma_l2": torch.tensor(0.0, device=v_pred.device, dtype=v_pred.dtype),
        }


# === 工廠函數 ===

def create_combined_loss(
    lambda_freq: float = 1.0,
    lambda_repa: float = 0.5,
    freq_quality: int = 90,
    repa_early_stop: int = 250000,
    lambda_gamma_l2: float = 1e-4,
) -> CombinedLoss:
    """創建 Combined Loss"""
    return CombinedLoss(
        config=None,
        lambda_freq=lambda_freq,
        lambda_repa=lambda_repa,
        freq_quality=freq_quality,
        repa_early_stop=repa_early_stop,
        lambda_gamma_l2=lambda_gamma_l2,
    )


def create_combined_loss_from_config(
    config: "PixelHDMConfig",
) -> CombinedLoss:
    """從配置創建 Combined Loss"""
    return CombinedLoss(config=config)


def create_combined_loss_simple(
    lambda_freq: float = 1.0,
    freq_quality: int = 90,
) -> CombinedLossSimple:
    """創建簡化版 Combined Loss"""
    return CombinedLossSimple(
        config=None,
        lambda_freq=lambda_freq,
        freq_quality=freq_quality,
    )


__all__ = [
    "CombinedLoss",
    "CombinedLossSimple",
    "create_combined_loss",
    "create_combined_loss_from_config",
    "create_combined_loss_simple",
]
