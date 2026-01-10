"""
Text Encoder Tokenizer Wrapper

Tokenizer 封裝模組，提供統一的分詞介面。

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union, Any

import torch


class TokenizerWrapper:
    """
    Tokenizer 封裝類。

    封裝 HuggingFace tokenizer，提供統一的分詞介面。

    Args:
        model_name: Hugging Face 模型名稱
        max_length: 最大序列長度

    Example:
        >>> wrapper = TokenizerWrapper("Qwen/Qwen3-0.6B", max_length=256)
        >>> wrapper.load()
        >>> tokens = wrapper.tokenize(["Hello world"])
        >>> tokens["input_ids"].shape
        torch.Size([1, 256])
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_length: int = 256,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self._tokenizer: Optional[Any] = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """檢查 tokenizer 是否已加載。"""
        return self._loaded

    def load(self) -> None:
        """
        加載 tokenizer。

        Raises:
            RuntimeError: 加載失敗
        """
        if self._loaded:
            return

        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            self._ensure_pad_token()
            self._loaded = True

        except Exception as e:
            raise RuntimeError(f"無法加載 tokenizer {self.model_name}: {e}")

    def _ensure_pad_token(self) -> None:
        """確保 tokenizer 有 pad token。"""
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def _ensure_loaded(self) -> None:
        """確保 tokenizer 已加載。"""
        if not self._loaded:
            self.load()

    def tokenize(
        self,
        texts: Union[str, List[str]],
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """
        分詞。

        Args:
            texts: 輸入文本 (單個或列表)
            return_tensors: 返回張量類型

        Returns:
            {"input_ids": (B, T), "attention_mask": (B, T)}
        """
        self._ensure_loaded()

        if isinstance(texts, str):
            texts = [texts]

        return self._tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors=return_tensors,
        )

    def __call__(
        self,
        texts: Union[str, List[str]],
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """分詞 (與 tokenize 相同)。"""
        return self.tokenize(texts, return_tensors=return_tensors)


__all__ = ["TokenizerWrapper"]
