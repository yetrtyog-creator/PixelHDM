"""
DINOv3 Weight Loader

Handles loading weights from various sources:
    - Local .pth files (recommended)
    - HuggingFace transformers
    - torch.hub

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Set

import torch
import torch.nn as nn


class DINOv3WeightLoader:
    """
    Weight loader for DINOv3 models

    Supports multiple loading sources with key matching.
    """

    # Hugging Face model ID mapping
    HF_MODEL_IDS: Dict[str, str] = {
        "dinov3-vit7b16": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        "dinov3-vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "dinov3-vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "dinov3-vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    }

    def __init__(self, model_name: str):
        self.model_name = model_name

    def load_from_local(self, model: nn.Module, local_path: str) -> None:
        """Load weights from local .pth file"""
        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(
                f"DINOv3 weight file not found: {path}\n"
                f"Please download weights to the specified path"
            )

        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = self._unwrap_state_dict(state_dict)

        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            matched_dict = self._match_state_dict(state_dict, model)
            model.load_state_dict(matched_dict, strict=False)

        print(f"[DINOv3] Loaded weights from: {path}")

    def _unwrap_state_dict(self, state_dict: Dict) -> Dict:
        """Unwrap nested state_dict structure"""
        if "model" in state_dict:
            return state_dict["model"]
        if "state_dict" in state_dict:
            return state_dict["state_dict"]
        return state_dict

    def _match_state_dict(
        self, state_dict: Dict, model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """Match state_dict keys with model keys"""
        new_dict = {}
        model_keys: Set[str] = set(model.state_dict().keys())
        prefixes = ["backbone.", "encoder.", "model.", "module."]

        for key, value in state_dict.items():
            new_key = self._remove_prefixes(key, prefixes)
            if new_key in model_keys:
                new_dict[new_key] = value

        return new_dict

    def _remove_prefixes(self, key: str, prefixes: list) -> str:
        """Remove known prefixes from key"""
        for prefix in prefixes:
            if key.startswith(prefix):
                return key[len(prefix):]
        return key

    def load_from_hf(self) -> nn.Module:
        """Load model from HuggingFace"""
        from transformers import AutoModel

        hf_id = self.HF_MODEL_IDS.get(
            self.model_name,
            f"facebook/{self.model_name.replace('-', '_')}"
        )

        return AutoModel.from_pretrained(hf_id, trust_remote_code=True)

    def load_from_hub(self, pretrained: bool = True) -> nn.Module:
        """Load model from torch.hub"""
        return torch.hub.load(
            "facebookresearch/dinov3",
            self.model_name.replace("-", "_"),
            pretrained=pretrained,
        )


__all__ = ["DINOv3WeightLoader"]
