"""
Qwen3 Text Encoder

Qwen3 文本編碼器主類。

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .tokenizer import TokenizerWrapper
from .pooling import LastTokenPooling

if TYPE_CHECKING:
    from ....config.model_config import PixelHDMConfig


# 支援的 Qwen3 模型配置
SUPPORTED_MODELS = {
    "Qwen/Qwen3-0.6B": {"hidden_size": 1024, "num_layers": 28},
    "Qwen/Qwen3-1.7B": {"hidden_size": 2048, "num_layers": 28},
    "Qwen/Qwen3-4B": {"hidden_size": 2560, "num_layers": 36},
}


class Qwen3TextEncoder(nn.Module):
    """
    Qwen3 Text Encoder

    用於 Text-to-Image 生成的文本編碼器。

    特點:
        - hidden_size=1024 (0.6B)，與 DiT hidden_dim 匹配
        - 凍結參數，不參與訓練
        - 輸出最後一層隱藏狀態

    Args:
        config: PixelHDMConfig 配置
        model_name: Hugging Face 模型名稱
        max_length: 最大文本長度
        use_pooler: 是否使用池化輸出
        freeze: 是否凍結參數
        device_map: 設備映射

    Shape:
        - Input: input_ids (B, T), attention_mask (B, T)
        - Output: hidden_states (B, T, D), attention_mask (B, T)
    """

    SUPPORTED_MODELS = SUPPORTED_MODELS

    def __init__(
        self,
        config: Optional["PixelHDMConfig"] = None,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_length: int = 256,
        use_pooler: bool = False,
        freeze: bool = True,
        device_map: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._init_from_config_or_args(config, model_name, max_length, freeze)
        self.use_pooler = use_pooler
        self.device_map = device_map

        self.hidden_size = self._get_hidden_size()
        self._tokenizer_wrapper = TokenizerWrapper(self.model_name, self.max_length)
        self._pooler = LastTokenPooling()

        self.model: Optional[nn.Module] = None
        self._loaded = False
        self._target_device: Optional[torch.device] = None  # 記錄目標設備

    def _init_from_config_or_args(
        self,
        config: Optional["PixelHDMConfig"],
        model_name: str,
        max_length: int,
        freeze: bool,
    ) -> None:
        """從配置或參數初始化。"""
        if config is not None:
            self.model_name = config.text_encoder_name
            self.max_length = config.text_max_length
            self.freeze = config.text_encoder_frozen
        else:
            self.model_name = model_name
            self.max_length = max_length
            self.freeze = freeze

    def _get_hidden_size(self) -> int:
        """獲取模型的隱藏層大小。"""
        if self.model_name in SUPPORTED_MODELS:
            return SUPPORTED_MODELS[self.model_name]["hidden_size"]
        return 1024  # 默認

    def to(self, device: Optional[Union[str, torch.device]] = None, *args, **kwargs):
        """重寫 to() 方法，記錄目標設備供懶加載使用。

        解決問題：懶加載時 self.model=None，.to(device) 無效。
        此方法記錄目標設備，在 _load_model() 時使用。
        """
        if device is not None:
            self._target_device = torch.device(device) if isinstance(device, str) else device
        return super().to(device, *args, **kwargs)

    @property
    def tokenizer(self) -> Any:
        """兼容舊 API: 返回底層 tokenizer。"""
        self._ensure_loaded()
        return self._tokenizer_wrapper._tokenizer

    def _load_model(self) -> None:
        """懶加載模型。"""
        if self._loaded:
            return
        try:
            from transformers import AutoModel
            self._tokenizer_wrapper.load()
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                device_map=self.device_map,
            )
            # 移動到目標設備（解決懶加載時 .to(device) 無效的問題）
            if self._target_device is not None:
                self.model = self.model.to(self._target_device)
            self.model.eval()
            self._freeze_if_needed()
            self._loaded = True
        except Exception as e:
            raise RuntimeError(f"無法加載 Qwen3 模型 {self.model_name}: {e}")

    def _freeze_if_needed(self) -> None:
        """根據配置凍結模型參數。"""
        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def _ensure_loaded(self) -> None:
        """確保模型已加載。"""
        if not self._loaded:
            self._load_model()

    def tokenize(
        self,
        texts: Union[str, List[str]],
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """分詞。"""
        self._ensure_loaded()
        return self._tokenizer_wrapper.tokenize(texts, return_tensors)

    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        texts: Optional[Union[str, List[str]]] = None,
        return_dict: bool = False,
        return_pooled: bool = True,
    ) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        """前向傳播。

        Args:
            input_ids: 輸入 token IDs
            attention_mask: 注意力掩碼
            texts: 文本列表（替代 input_ids）
            return_dict: 是否返回字典格式
            return_pooled: 是否返回池化輸出（用於 AdaLN 調制）

        Returns:
            如果 return_dict=False:
                (hidden_states, attention_mask, pooled_output) 當 return_pooled=True
                (hidden_states, attention_mask) 當 return_pooled=False
            如果 return_dict=True:
                包含 hidden_states, attention_mask, pooled_output 的字典
        """
        self._ensure_loaded()
        input_ids, attention_mask = self._prepare_inputs(input_ids, attention_mask, texts)
        hidden_states = self._encode(input_ids, attention_mask)
        return self._format_output(hidden_states, attention_mask, return_dict, return_pooled)

    def _prepare_inputs(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        texts: Optional[Union[str, List[str]]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """準備輸入張量。"""
        if texts is not None:
            tokens = self.tokenize(texts)
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
            input_ids, attention_mask = self._move_to_device(input_ids, attention_mask)

        if input_ids is None:
            raise ValueError("必須提供 input_ids 或 texts")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask

    def _move_to_device(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """將輸入移動到模型設備。"""
        if hasattr(self.model, "device"):
            device = self.model.device
        else:
            device = next(self.model.parameters()).device
        return input_ids.to(device), attention_mask.to(device)

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """編碼輸入序列。"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.last_hidden_state

    def _format_output(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool,
        return_pooled: bool = True,
    ) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        """格式化輸出。

        Args:
            hidden_states: 序列隱藏狀態 (B, T, D)
            attention_mask: 注意力掩碼 (B, T)
            return_dict: 是否返回字典格式
            return_pooled: 是否返回池化輸出

        Returns:
            格式化後的輸出
        """
        pooled_output = self._pooler(hidden_states, attention_mask) if return_pooled else None

        if not return_dict:
            if return_pooled:
                return hidden_states, attention_mask, pooled_output
            return hidden_states, attention_mask

        result: Dict[str, torch.Tensor] = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
        }
        if return_pooled or self.use_pooler:
            result["pooled_output"] = pooled_output if pooled_output is not None else self._pooler(hidden_states, attention_mask)
        return result

    def get_output_dim(self) -> int:
        """獲取輸出維度。"""
        return self.hidden_size

    def extra_repr(self) -> str:
        return (
            f"model={self.model_name}, "
            f"hidden_size={self.hidden_size}, "
            f"max_length={self.max_length}, "
            f"freeze={self.freeze}"
        )


# === 工廠函數 ===

def create_text_encoder(
    model_name: str = "Qwen/Qwen3-0.6B",
    max_length: int = 256,
    freeze: bool = True,
) -> Qwen3TextEncoder:
    """創建文本編碼器。"""
    return Qwen3TextEncoder(
        config=None,
        model_name=model_name,
        max_length=max_length,
        freeze=freeze,
    )


def create_text_encoder_from_config(config: "PixelHDMConfig") -> Qwen3TextEncoder:
    """從配置創建文本編碼器。"""
    return Qwen3TextEncoder(config=config)


__all__ = [
    "Qwen3TextEncoder",
    "create_text_encoder",
    "create_text_encoder_from_config",
    "SUPPORTED_MODELS",
]
