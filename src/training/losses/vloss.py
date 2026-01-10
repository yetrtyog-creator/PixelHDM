"""
PixelHDM-RPEA-DinoV3 V-Loss

基於 PixelHDM 論文設計 (arXiv:2511.20645):
    - 訓練目標: velocity-matching loss
    - 網路直接輸出 velocity v

核心設計:
    - 時間方向: t=0 噪聲, t=1 乾淨圖像
    - 插值公式: z_t = t * x + (1 - t) * ε
    - 網路輸出: v_pred (直接預測 velocity)
    - 損失函數: ||v_pred - v_target||²

V-Prediction 優勢:
    - 避免 X-Prediction 的 1/(1-t) 誤差放大問題
    - 在像素層細微預測時更穩定
    - 直接對應 ODE 積分: dz/dt = v

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ...config.model_config import PixelHDMConfig


class VLoss(nn.Module):
    """
    V-Loss 損失函數 (V-Prediction)

    公式:
        v_pred = model(z_t, t)  # 網路直接輸出 velocity
        v_target = x - ε
        L = E[||v_pred - v_target||²]

    Args:
        config: PixelHDMConfig 配置

    Shape:
        - v_pred: (B, H, W, C) 或 (B, C, H, W) - 網路預測的 velocity
        - x_clean: (B, H, W, C) 或 (B, C, H, W) - 乾淨圖像
        - noise: (B, H, W, C) 或 (B, C, H, W) - 噪聲
        - Output: scalar
    """

    def __init__(
        self,
        config: Optional["PixelHDMConfig"] = None,
        t_eps: float = 0.05,
    ) -> None:
        super().__init__()
        # t_eps 保留用於兼容性，但 V-Prediction 不需要除法
        self.t_eps = t_eps if config is None else config.time_eps

    def compute_v_target(
        self,
        x_clean: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        計算目標 velocity

        公式: v_target = x - ε

        Args:
            x_clean: 乾淨圖像
            noise: 噪聲

        Returns:
            v_target: 目標 velocity
        """
        return x_clean - noise

    def forward(
        self,
        v_pred: torch.Tensor,
        x_clean: torch.Tensor,
        noise: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        計算 V-Loss

        Args:
            v_pred: 網路預測的 velocity (直接從模型輸出)
            x_clean: 乾淨圖像 (目標)
            noise: 噪聲
            reduction: "mean", "sum", "none"

        Returns:
            loss: V-Loss 值
        """
        original_dtype = v_pred.dtype

        # 計算目標 velocity
        v_target = self.compute_v_target(x_clean, noise)

        # MSE 損失 (float32 計算)
        loss = (v_pred.float() - v_target.float()).pow(2)

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss.to(original_dtype)


class VLossWithVelocity(VLoss):
    """
    V-Loss 並返回 velocity (供其他損失函數使用)

    用於與 Frequency Loss 配合使用
    """

    def forward_with_velocity(
        self,
        v_pred: torch.Tensor,
        x_clean: torch.Tensor,
        noise: torch.Tensor,
        reduction: str = "mean",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        計算 V-Loss 並返回 velocity

        Args:
            v_pred: 網路預測的 velocity
            x_clean: 乾淨圖像
            noise: 噪聲
            reduction: "mean", "sum", "none"

        Returns:
            loss: V-Loss 值
            v_pred: 預測的 velocity (直接返回輸入)
            v_target: 目標 velocity
        """
        original_dtype = v_pred.dtype

        v_target = self.compute_v_target(x_clean, noise)

        loss = (v_pred.float() - v_target.float()).pow(2)

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss.to(original_dtype), v_pred, v_target


# === 工廠函數 ===

def create_vloss(
    t_eps: float = 0.05,
) -> VLoss:
    """創建 V-Loss"""
    return VLoss(config=None, t_eps=t_eps)


def create_vloss_from_config(
    config: "PixelHDMConfig",
) -> VLoss:
    """從配置創建 V-Loss"""
    return VLoss(config=config)


def create_vloss_with_velocity(
    t_eps: float = 0.05,
) -> VLossWithVelocity:
    """創建帶 velocity 輸出的 V-Loss"""
    return VLossWithVelocity(config=None, t_eps=t_eps)


__all__ = [
    "VLoss",
    "VLossWithVelocity",
    "create_vloss",
    "create_vloss_from_config",
    "create_vloss_with_velocity",
]
