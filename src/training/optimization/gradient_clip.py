"""
PixelHDM-RPEA-DinoV3 ZClip - Z-Score 自適應梯度剪裁

使用滑動統計量 (EMA) 追蹤梯度範數分布，動態檢測異常梯度尖峰。

原理:
    - 維護梯度 norm 的滑動平均值 (mean) 和方差 (variance)
    - 計算當前梯度的 z-score = (grad_norm - mean) / std
    - 當 z-score > threshold 時視為異常，進行裁剪

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union
import math

import torch
import torch.nn as nn


@dataclass
class ZClipStats:
    """ZClip 統計信息"""
    mean: float
    std: float
    steps: int
    clips: int
    clip_rate: float


class ZClip:
    """
    Z-Score 自適應梯度剪裁

    Args:
        threshold: z-score 閾值
        ema_decay: EMA 衰減係數

    Example:
        >>> zclip = ZClip(threshold=2.5, ema_decay=0.99)
        >>> for step in range(num_steps):
        ...     loss.backward()
        ...     grad_norm = clip_grad_norm_with_zclip(model, zclip)
        ...     optimizer.step()
    """

    def __init__(
        self,
        threshold: float = 2.5,
        ema_decay: float = 0.99,
    ) -> None:
        if threshold <= 0:
            raise ValueError(f"threshold 必須為正數")
        if not (0 < ema_decay < 1):
            raise ValueError(f"ema_decay 必須在 (0, 1) 之間")

        self.threshold = threshold
        self.ema_decay = ema_decay

        # 滑動統計量
        self.mean_ema: Optional[float] = None
        self.var_ema: Optional[float] = None

        # 計數器
        self.step_count: int = 0
        self.clip_count: int = 0

    def __call__(self, grad_norm: float) -> float:
        """
        處理梯度範數

        Args:
            grad_norm: 當前步的梯度範數

        Returns:
            調整後的範數
        """
        if not (grad_norm >= 0 and grad_norm != float('inf')):
            raise ValueError(f"grad_norm 必須為非負有限值")

        self.step_count += 1

        # 首次初始化
        if self.mean_ema is None:
            self.mean_ema = grad_norm
            self.var_ema = 0.0
            return grad_norm

        # 更新 EMA 統計量
        old_mean = self.mean_ema
        delta = grad_norm - old_mean
        self.mean_ema = self.ema_decay * self.mean_ema + (1 - self.ema_decay) * grad_norm
        delta2 = grad_norm - self.mean_ema
        var_update = delta * delta2
        self.var_ema = max(0.0, self.ema_decay * self.var_ema + (1 - self.ema_decay) * var_update)

        # 計算標準差
        std = max((self.var_ema + 1e-8) ** 0.5, 0.01)

        # 計算 z-score (修正: 使用新均值 delta2，確保使用同一時刻的統計量)
        z_score = delta2 / std

        # 異常檢測與裁剪
        if z_score > self.threshold:
            clipped_norm = self.mean_ema + self.threshold * std
            self.clip_count += 1
            return clipped_norm

        return grad_norm

    def get_stats(self) -> ZClipStats:
        """獲取統計信息"""
        std = (self.var_ema + 1e-8) ** 0.5 if self.var_ema is not None else 0.0
        clip_rate = self.clip_count / self.step_count if self.step_count > 0 else 0.0

        return ZClipStats(
            mean=self.mean_ema if self.mean_ema is not None else 0.0,
            std=std,
            steps=self.step_count,
            clips=self.clip_count,
            clip_rate=clip_rate,
        )

    def reset(self) -> None:
        """重置所有統計量"""
        self.mean_ema = None
        self.var_ema = None
        self.step_count = 0
        self.clip_count = 0

    def state_dict(self) -> Dict[str, Union[float, int, None]]:
        """獲取狀態字典"""
        return {
            "mean_ema": self.mean_ema,
            "var_ema": self.var_ema,
            "step_count": self.step_count,
            "clip_count": self.clip_count,
            "threshold": self.threshold,
            "ema_decay": self.ema_decay,
        }

    def load_state_dict(self, state: Dict[str, Union[float, int, None]]) -> None:
        """載入狀態字典"""
        self.mean_ema = state.get("mean_ema")
        self.var_ema = state.get("var_ema")
        self.step_count = state.get("step_count", 0)
        self.clip_count = state.get("clip_count", 0)


def clip_grad_norm_with_zclip(
    model: nn.Module,
    zclip: ZClip,
    max_norm: float = 1.0,
) -> float:
    """
    結合 ZClip 和標準梯度裁剪

    Args:
        model: 模型
        zclip: ZClip 實例
        max_norm: 最大梯度範數

    Returns:
        原始梯度範數
    """
    # 計算原始梯度範數
    parameters = [p for p in model.parameters() if p.grad is not None]
    if len(parameters) == 0:
        return 0.0

    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]),
        2,
    ).item()

    # 檢測 NaN/Inf 梯度
    if not math.isfinite(total_norm):
        raise RuntimeError(f"檢測到無效梯度範數: {total_norm}")

    # ZClip 計算自適應上限
    adaptive_norm = zclip(total_norm)

    # 執行裁剪
    effective_max_norm = min(max_norm, adaptive_norm)
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=effective_max_norm,
    )

    return total_norm


__all__ = [
    "ZClip",
    "ZClipStats",
    "clip_grad_norm_with_zclip",
]
