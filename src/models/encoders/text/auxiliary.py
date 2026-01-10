"""
Text Encoder Auxiliary Classes

輔助類別：TextProjector, CaptionEmbedder, NullTextEncoder

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ....config.model_config import PixelHDMConfig
    from .encoder import Qwen3TextEncoder


class TextProjector(nn.Module):
    """
    文本投影層。

    當文本編碼器輸出維度與 DiT 不匹配時使用。

    Args:
        config: PixelHDMConfig
        input_dim: 輸入維度
        output_dim: 輸出維度
    """

    def __init__(
        self,
        config: Optional["PixelHDMConfig"] = None,
        input_dim: int = 1024,
        output_dim: int = 1024,
    ) -> None:
        super().__init__()

        if config is not None:
            output_dim = config.hidden_dim
            input_dim = config.text_hidden_size

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.proj = self._create_projection()

    def _create_projection(self) -> nn.Module:
        """創建投影層。"""
        if self.input_dim == self.output_dim:
            return nn.Identity()
        proj = nn.Linear(self.input_dim, self.output_dim, bias=False)
        nn.init.xavier_uniform_(proj.weight)
        return proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向傳播。"""
        return self.proj(x)

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}"


class CaptionEmbedder(nn.Module):
    """
    Caption Embedder

    整合文本編碼和投影。
    """

    def __init__(
        self,
        config: Optional["PixelHDMConfig"] = None,
        encoder: Optional["Qwen3TextEncoder"] = None,
    ) -> None:
        super().__init__()
        # 延遲導入避免循環依賴
        if encoder is None:
            from .encoder import Qwen3TextEncoder
            encoder = Qwen3TextEncoder(config=config)
        self.encoder = encoder
        self.projector = TextProjector(config=config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        texts: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向傳播。"""
        hidden_states, mask = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            texts=texts,
        )
        return self.projector(hidden_states), mask

    def tokenize(self, texts: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """分詞。"""
        return self.encoder.tokenize(texts)

    def get_output_dim(self) -> int:
        """獲取輸出維度。"""
        return self.projector.output_dim


class NullTextEncoder(nn.Module):
    """
    空文本編碼器。

    用於無條件生成場景，輸出可學習的 null embedding。
    """

    def __init__(
        self,
        config: Optional["PixelHDMConfig"] = None,
        hidden_dim: int = 1024,
        max_length: int = 1,
    ) -> None:
        super().__init__()

        if config is not None:
            hidden_dim = config.hidden_dim
            max_length = 1

        self.hidden_dim = hidden_dim
        self.max_length = max_length

        self.null_embedding = nn.Parameter(torch.zeros(1, max_length, hidden_dim))
        nn.init.normal_(self.null_embedding, std=0.02)

    def forward(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向傳播。"""
        if device is None:
            device = self.null_embedding.device

        embeddings = self.null_embedding.expand(batch_size, -1, -1)
        attention_mask = torch.ones(
            batch_size, self.max_length,
            dtype=torch.long,
            device=device,
        )
        return embeddings, attention_mask

    def get_output_dim(self) -> int:
        """獲取輸出維度。"""
        return self.hidden_dim


# === 工廠函數 ===

def create_caption_embedder(
    config: Optional["PixelHDMConfig"] = None,
) -> CaptionEmbedder:
    """創建 Caption Embedder。"""
    return CaptionEmbedder(config=config)


def create_null_text_encoder(
    hidden_dim: int = 1024,
    max_length: int = 1,
) -> NullTextEncoder:
    """創建空文本編碼器。"""
    return NullTextEncoder(
        config=None,
        hidden_dim=hidden_dim,
        max_length=max_length,
    )


__all__ = [
    "TextProjector",
    "CaptionEmbedder",
    "NullTextEncoder",
    "create_caption_embedder",
    "create_null_text_encoder",
]
