"""
PixelHDM Text-to-Image Model and Factory Functions

Contains:
    - PixelHDMForT2I: T2I wrapper with text/DINO encoders
    - Factory functions for model creation

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch

from .core import PixelHDM

if TYPE_CHECKING:
    from ...config.model_config import PixelHDMConfig


class PixelHDMForT2I(PixelHDM):
    """
    PixelHDM for Text-to-Image

    Includes:
    - PixelHDM main model
    - Qwen3 text encoder (frozen, lazy-loaded)
    - DINOv3 REPA encoder (frozen, lazy-loaded)

    Args:
        config: PixelHDMConfig configuration
        load_text_encoder: Whether to load text encoder
        load_dino_encoder: Whether to load DINO encoder
    """

    def __init__(
        self,
        config: "PixelHDMConfig",
        load_text_encoder: bool = True,
        load_dino_encoder: bool = True,
    ) -> None:
        super().__init__(config)
        self.load_text_encoder = load_text_encoder
        self.load_dino_encoder = load_dino_encoder
        self._text_encoder = None
        self._text_projector = None
        self._dino_encoder = None
        self._dino_projector = None

    @property
    def text_encoder(self):
        """Lazy-load text encoder."""
        if self._text_encoder is None and self.load_text_encoder:
            from ..encoders.text_encoder import Qwen3TextEncoder, TextProjector
            self._text_encoder = Qwen3TextEncoder(config=self.config)
            self._text_projector = TextProjector(config=self.config)
        return self._text_encoder

    @property
    def dino_encoder(self):
        """Lazy-load DINO encoder."""
        if self._dino_encoder is None and self.load_dino_encoder:
            from ..encoders.dinov3 import DINOv3Encoder, DINOFeatureProjector
            self._dino_encoder = DINOv3Encoder(config=self.config)
            self._dino_projector = DINOFeatureProjector(config=self.config)
        return self._dino_encoder

    def encode_text(self, texts: list[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text to embeddings.

        Args:
            texts: List of text strings

        Returns:
            (text_embed, text_mask)
        """
        if self.text_encoder is None:
            raise RuntimeError("文本編碼器未加載")
        # Use return_pooled=False to get only (hidden_states, mask)
        # The text encoder returns 3 values when return_pooled=True (default)
        hidden_states, mask = self.text_encoder(texts=texts, return_pooled=False)
        text_embed = self._text_projector(hidden_states)
        return text_embed, mask

    def get_dino_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get DINO features for REPA Loss.

        Args:
            x: (B, H, W, 3) - clean image

        Returns:
            features: (B, L, D_dino)
        """
        if self.dino_encoder is None:
            raise RuntimeError("DINO 編碼器未加載")
        return self.dino_encoder(x)

    def forward_t2i(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        texts: list[str],
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Text-to-Image forward pass.

        Args:
            x_t: (B, H, W, 3) - noisy image
            t: (B,) - timestep
            texts: List of text strings
            return_features: Whether to return REPA features

        Returns:
            x_pred or (x_pred, repa_features)
        """
        text_embed, text_mask = self.encode_text(texts)
        return self.forward(x_t, t, text_embed, text_mask, return_features=return_features)


def create_pixelhdm(
    config: Optional["PixelHDMConfig"] = None,
    **kwargs,
) -> PixelHDM:
    """Create PixelHDM model."""
    if config is None:
        from ...config.model_config import PixelHDMConfig
        config = PixelHDMConfig(**kwargs)
    return PixelHDM(config)


def create_pixelhdm_for_t2i(
    config: Optional["PixelHDMConfig"] = None,
    load_text_encoder: bool = True,
    load_dino_encoder: bool = True,
    **kwargs,
) -> PixelHDMForT2I:
    """Create T2I complete model."""
    if config is None:
        from ...config.model_config import PixelHDMConfig
        config = PixelHDMConfig(**kwargs)
    return PixelHDMForT2I(
        config,
        load_text_encoder=load_text_encoder,
        load_dino_encoder=load_dino_encoder,
    )


def create_pixelhdm_from_pretrained(
    path: str,
    config: Optional["PixelHDMConfig"] = None,
) -> PixelHDM:
    """Load model from pretrained weights."""
    if config is None:
        from ...config.model_config import PixelHDMConfig
        config_path = os.path.join(os.path.dirname(path), "config.json")
        if os.path.exists(config_path):
            config = PixelHDMConfig.from_json(config_path)
        else:
            config = PixelHDMConfig.default()

    model = PixelHDM(config)
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def create_pixelhdm_from_config(config: "PixelHDMConfig") -> PixelHDM:
    """
    Create PixelHDM model from config.

    Args:
        config: PixelHDMConfig instance

    Returns:
        PixelHDM model instance
    """
    return PixelHDM(config)


def create_pixelhdm_for_t2i_from_config(
    config: "PixelHDMConfig",
    load_text_encoder: bool = True,
    load_dino_encoder: bool = True,
) -> PixelHDMForT2I:
    """
    Create PixelHDMForT2I model from config.

    Args:
        config: PixelHDMConfig instance
        load_text_encoder: Whether to load text encoder
        load_dino_encoder: Whether to load DINO encoder

    Returns:
        PixelHDMForT2I model instance
    """
    return PixelHDMForT2I(
        config=config,
        load_text_encoder=load_text_encoder,
        load_dino_encoder=load_dino_encoder,
    )
