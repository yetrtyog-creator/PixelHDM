"""
PixelHDM-RPEA-DinoV3 CPU Memory Checkpoint

定期將模型、優化器、EMA 狀態保存到 CPU RAM，實現:
    - 毫秒級恢復 (vs. 磁碟 checkpoint 的秒級)
    - Loss spike 自動檢測與回滾
    - 零磁碟 I/O，不影響訓練速度

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .ema import EMA


@dataclass
class CheckpointStats:
    """檢查點統計信息"""
    restore_count: int
    last_save_step: int
    has_checkpoint: bool


class CPUMemoryCheckpoint:
    """
    CPU 記憶體快照檢查點

    Args:
        save_interval: 保存間隔 (訓練步數)
        spike_threshold: Loss spike 倍數閾值

    Example:
        >>> cpu_ckpt = CPUMemoryCheckpoint(save_interval=100)
        >>> for step, batch in enumerate(dataloader):
        ...     loss = train_step(batch)
        ...     restored = cpu_ckpt.step(model, optimizer, ema, loss.item(), step)
        ...     if restored:
        ...         print(f"Recovered from loss spike at step {step}")
    """

    def __init__(
        self,
        save_interval: int = 100,
        spike_threshold: float = 5.0,
    ) -> None:
        if save_interval <= 0:
            raise ValueError(f"save_interval 必須為正數")
        if spike_threshold <= 1.0:
            raise ValueError(f"spike_threshold 必須 > 1.0")

        self.save_interval = save_interval
        self.spike_threshold = spike_threshold

        # CPU 狀態快照
        self._cpu_model_state: Optional[Dict[str, torch.Tensor]] = None
        self._cpu_optimizer_state: Optional[Dict[str, Any]] = None
        self._cpu_ema_state: Optional[Dict[str, torch.Tensor]] = None

        # 追蹤信息
        self._last_loss: Optional[float] = None
        self._last_save_step: int = 0
        self._restore_count: int = 0

    def step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        ema: Optional["EMA"],
        loss: float,
        global_step: int,
    ) -> bool:
        """
        每訓練步調用一次

        Args:
            model: 訓練模型
            optimizer: 優化器
            ema: EMA 實例
            loss: 當前步的 loss 值
            global_step: 全局步數

        Returns:
            是否執行了恢復操作
        """
        # 驗證 loss
        if not (loss == loss and loss != float('inf') and loss != float('-inf')):
            raise ValueError(f"無效的 loss 值: {loss}")

        restored = False

        # 定期保存
        if global_step % self.save_interval == 0:
            self._save(model, optimizer, ema)
            self._last_save_step = global_step

        # Loss spike 檢測
        if self._last_loss is not None:
            if loss > self._last_loss * self.spike_threshold:
                warnings.warn(
                    f"Loss spike 檢測於 step {global_step}: "
                    f"{self._last_loss:.6f} → {loss:.6f}",
                    category=UserWarning,
                )

                if self._restore(model, optimizer, ema):
                    self._restore_count += 1
                    restored = True

        self._last_loss = loss
        return restored

    def _save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        ema: Optional["EMA"],
    ) -> None:
        """保存狀態到 CPU RAM"""
        self._cpu_model_state = {
            k: v.cpu().clone() for k, v in model.state_dict().items()
        }

        self._cpu_optimizer_state = self._copy_optimizer_state(optimizer)

        if ema is not None:
            self._cpu_ema_state = {
                k: v.cpu().clone() for k, v in ema.shadow.items()
            }

    def _restore(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        ema: Optional["EMA"],
    ) -> bool:
        """從 CPU RAM 恢復狀態"""
        if self._cpu_model_state is None:
            return False

        device = next(model.parameters()).device

        model.load_state_dict({
            k: v.to(device) for k, v in self._cpu_model_state.items()
        })

        if self._cpu_optimizer_state is not None:
            optimizer.load_state_dict(self._cpu_optimizer_state)

        if ema is not None and self._cpu_ema_state is not None:
            ema.shadow = {
                k: v.to(device) for k, v in self._cpu_ema_state.items()
            }

        return True

    def _copy_optimizer_state(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, Any]:
        """深拷貝優化器狀態到 CPU"""
        state = optimizer.state_dict()
        cpu_state: Dict[str, Any] = {
            "state": {},
            "param_groups": copy.deepcopy(state["param_groups"]),
        }

        for param_id, param_state in state["state"].items():
            cpu_state["state"][param_id] = {
                k: v.cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v)
                for k, v in param_state.items()
            }

        return cpu_state

    def get_stats(self) -> CheckpointStats:
        """獲取統計信息"""
        return CheckpointStats(
            restore_count=self._restore_count,
            last_save_step=self._last_save_step,
            has_checkpoint=self._cpu_model_state is not None,
        )

    def clear(self) -> None:
        """清除所有 CPU 快照"""
        self._cpu_model_state = None
        self._cpu_optimizer_state = None
        self._cpu_ema_state = None

    def state_dict(self) -> Dict[str, Any]:
        """獲取狀態字典"""
        return {
            "save_interval": self.save_interval,
            "spike_threshold": self.spike_threshold,
            "last_save_step": self._last_save_step,
            "restore_count": self._restore_count,
            "last_loss": self._last_loss,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """載入狀態字典"""
        self.save_interval = state.get("save_interval", self.save_interval)
        self.spike_threshold = state.get("spike_threshold", self.spike_threshold)
        self._last_save_step = state.get("last_save_step", 0)
        self._restore_count = state.get("restore_count", 0)
        self._last_loss = state.get("last_loss")


__all__ = [
    "CPUMemoryCheckpoint",
    "CheckpointStats",
]
