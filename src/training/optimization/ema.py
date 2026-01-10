"""
PixelHDM-RPEA-DinoV3 EMA (Exponential Moving Average)

維護模型參數的滑動平均版本，用於推理時提高穩定性。

公式:
    shadow = decay * shadow + (1 - decay) * param

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

import torch
import torch.nn as nn


class EMA:
    """
    指數移動平均

    Args:
        model: 要追蹤的模型
        decay: EMA 衰減係數，建議範圍 [0.999, 0.9999]
        update_after_step: 在多少步之後開始更新 EMA
        device: 存儲設備

    Example:
        >>> ema = EMA(model, decay=0.9999)
        >>> for step, batch in enumerate(dataloader):
        ...     loss = train_step(batch)
        ...     optimizer.step()
        ...     ema.update(model, step)
        ...
        >>> # 推理時使用 EMA 權重
        >>> with ema.apply_to(model):
        ...     output = model(input)
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        update_after_step: int = 0,
        device: Optional[torch.device] = None,
    ) -> None:
        if not (0 < decay < 1):
            raise ValueError(f"decay 必須在 (0, 1) 之間, 當前: {decay}")

        self.decay = decay
        self.update_after_step = update_after_step
        self.device = device or next(model.parameters()).device

        # 創建 shadow 參數
        self.shadow: Dict[str, torch.Tensor] = {}
        self._register_model(model)

        # 備份 (用於臨時切換)
        self._backup: Dict[str, torch.Tensor] = {}

        self.num_updates: int = 0

    def _register_model(self, model: nn.Module) -> None:
        """註冊模型並初始化 shadow 參數"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(self.device)

    def update(self, model: nn.Module, step: Optional[int] = None) -> None:
        """
        更新 EMA 參數

        Args:
            model: 當前訓練的模型
            step: 當前訓練步數
        """
        if step is not None and step < self.update_after_step:
            return

        self.num_updates += 1

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name not in self.shadow:
                        self.shadow[name] = param.data.clone().to(self.device)
                    else:
                        self.shadow[name].mul_(self.decay).add_(
                            param.data.to(self.device), alpha=1 - self.decay
                        )

    def copy_to(self, model: nn.Module) -> None:
        """將 EMA 權重複製到模型"""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    param.data.copy_(self.shadow[name].to(param.device))

    def store(self, model: nn.Module) -> None:
        """備份當前模型權重"""
        self._backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._backup[name] = param.data.clone()

    def restore(self, model: nn.Module) -> None:
        """恢復備份的模型權重"""
        if not self._backup:
            raise RuntimeError("沒有備份可恢復，請先調用 store()")

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self._backup:
                    param.data.copy_(self._backup[name])

        self._backup.clear()

    @contextmanager
    def apply_to(self, model: nn.Module) -> Iterator[None]:
        """
        臨時應用 EMA 權重的上下文管理器

        Example:
            >>> with ema.apply_to(model):
            ...     output = model(input)  # 使用 EMA 權重
            >>> # 自動恢復原始權重
        """
        self.store(model)
        self.copy_to(model)
        try:
            yield
        finally:
            self.restore(model)

    def state_dict(self) -> Dict[str, Any]:
        """獲取狀態字典"""
        return {
            "shadow": {k: v.cpu() for k, v in self.shadow.items()},
            "decay": self.decay,
            "num_updates": self.num_updates,
            "update_after_step": self.update_after_step,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """載入狀態字典"""
        self.shadow = {
            k: v.to(self.device) for k, v in state["shadow"].items()
        }
        self.decay = state.get("decay", self.decay)
        self.num_updates = state.get("num_updates", 0)
        self.update_after_step = state.get("update_after_step", 0)


class EMAModelWrapper(nn.Module):
    """
    EMA 模型包裝器

    在推理時自動使用 EMA 權重
    """

    def __init__(self, model: nn.Module, ema: EMA) -> None:
        super().__init__()
        self.model = model
        self.ema = ema

    def forward(self, *args, **kwargs):
        with self.ema.apply_to(self.model):
            return self.model(*args, **kwargs)


__all__ = [
    "EMA",
    "EMAModelWrapper",
]
