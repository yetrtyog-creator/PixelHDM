"""
DINOv3 Encoder and Feature Projector

Main interface for DINOv3 feature extraction used in REPA Loss.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

import platform
import sys
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, Optional, Union

import torch
import torch.nn as nn

from .loader import DINOv3WeightLoader
from .transformer import DINOv3ViT
from .cache import FeatureCacheMixin

if TYPE_CHECKING:
    from ....config.model_config import PixelHDMConfig


class DINOv3Encoder(FeatureCacheMixin, nn.Module):
    """
    DINOv3 Feature Extractor for REPA Loss

    Args:
        config: PixelHDMConfig
        model_name: Model variant name
        pretrained: Load pretrained weights
        local_path: Local .pth weight file path (recommended)
        use_bf16: Use bf16 for inference
        use_hf: Use HuggingFace for loading
    """

    MODEL_CONFIGS: Dict[str, Dict] = {
        "dinov3-vit7b16": {"embed_dim": 4096, "depth": 40, "num_heads": 32, "patch_size": 16},
        "dinov3-vitl16": {"embed_dim": 1024, "depth": 24, "num_heads": 16, "patch_size": 16},
        "dinov3-vitb16": {"embed_dim": 768, "depth": 12, "num_heads": 12, "patch_size": 16},
        "dinov3-vits16": {"embed_dim": 384, "depth": 12, "num_heads": 6, "patch_size": 16},
    }

    def __init__(
        self,
        config: Optional["PixelHDMConfig"] = None,
        model_name: str = "dinov3-vitb16",
        pretrained: bool = True,
        local_path: Optional[str] = None,
        use_bf16: bool = True,
        use_hf: bool = True,
    ) -> None:
        super().__init__()

        if config is not None:
            model_name = self._parse_model_name(config.repa_encoder)
            if hasattr(config, 'repa_local_path') and config.repa_local_path:
                local_path = config.repa_local_path
            if hasattr(config, 'repa_use_bf16'):
                use_bf16 = config.repa_use_bf16

        self.model_name = self._validate_model_name(model_name)
        self.local_path = local_path
        self.use_bf16 = use_bf16
        self.use_hf = use_hf
        self.patch_size = 16
        self._use_compile = False

        self._feature_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._cache_max_size = 64
        self._cache_enabled = True

        model_config = self.MODEL_CONFIGS[self.model_name]
        self.embed_dim = model_config["embed_dim"]
        self._model_config = model_config

        self.model: Optional[nn.Module] = None
        self._pretrained = pretrained
        self._loaded = False
        self._target_device: torch.device = torch.device("cpu")

    def _parse_model_name(self, name: str) -> str:
        """Parse model name from various formats"""
        name = name.lower().replace("_", "-").replace(" ", "-")
        if "7b" in name or "giant" in name:
            return "dinov3-vit7b16"
        elif "large" in name or "vit-l" in name or "vitl" in name:
            return "dinov3-vitl16"
        elif "small" in name or "vit-s" in name or "vits" in name:
            return "dinov3-vits16"
        return "dinov3-vitb16"

    def _validate_model_name(self, model_name: str) -> str:
        """Validate and normalize model name"""
        if model_name not in self.MODEL_CONFIGS:
            model_name = self._parse_model_name(model_name)
        return model_name

    def _load_model(self) -> None:
        """Lazy load model"""
        if self._loaded:
            return
        loader = DINOv3WeightLoader(self.model_name)
        if self.local_path is not None:
            self._create_local_model(loader)
        elif self.use_hf:
            self.model = loader.load_from_hf()
        else:
            self.model = loader.load_from_hub(self._pretrained)
        self._finalize_model()

    def _create_local_model(self, loader: DINOv3WeightLoader) -> None:
        """Create model and load local weights"""
        cfg = self._model_config
        self.model = DINOv3ViT(
            img_size=224, patch_size=cfg["patch_size"],
            embed_dim=cfg["embed_dim"], depth=cfg["depth"], num_heads=cfg["num_heads"],
        )
        loader.load_from_local(self.model, self.local_path)

    def _finalize_model(self) -> None:
        """Finalize model setup after loading"""
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model = self.model.to(self._target_device)
        if self.use_bf16 and self._target_device.type == "cuda":
            self.model = self.model.to(torch.bfloat16)
        self._try_compile()
        self._loaded = True

    def _try_compile(self) -> None:
        """Try to compile model with torch.compile"""
        if not self._use_compile or self._target_device.type != "cuda":
            return
        is_windows = platform.system() == "Windows" or sys.platform == "win32"
        if is_windows:
            return
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
            print("[DINOv3] torch.compile enabled")
        except Exception as e:
            print(f"[DINOv3] torch.compile failed: {e}")

    def to(self, device: Union[str, torch.device], *args, **kwargs) -> "DINOv3Encoder":
        """Override to() for lazy loading"""
        if isinstance(device, str):
            device = torch.device(device)
        self._target_device = device
        if self._loaded and self.model is not None:
            self.model = self.model.to(device)
            if self.use_bf16 and device.type == "cuda":
                self.model = self.model.to(torch.bfloat16)
        return super().to(device, *args, **kwargs)

    def _ensure_loaded(self) -> None:
        """Ensure model is loaded"""
        if not self._loaded:
            self._load_model()

    def forward(
        self, x: torch.Tensor, return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Extract DINOv3 features."""
        self._ensure_loaded()
        x = self._prepare_input(x)

        with torch.inference_mode():
            cache_result = self._check_cache(x, return_dict)
            if cache_result is not None:
                return cache_result
            output = self._run_forward(x)
            patch_tokens, cls_token = self._parse_output(output)
            self._update_cache(x, patch_tokens)

        if return_dict:
            return {"patch_tokens": patch_tokens, "cls_token": cls_token}
        return patch_tokens

    def _run_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run model forward pass"""
        if self.use_bf16 and x.device.type == "cuda":
            x_input = x.to(torch.bfloat16)
        else:
            x_input = x.float()
        return self._forward_model(x_input)

    def _forward_model(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through model"""
        if self.local_path is not None:
            return self.model(x)
        return self.model(x, output_hidden_states=True)

    def get_output_dim(self) -> int:
        """Get output dimension"""
        return self.embed_dim

    def extra_repr(self) -> str:
        source = "local" if self.local_path else ("hf" if self.use_hf else "hub")
        dtype = "bf16" if self.use_bf16 else "fp32"
        return f"model={self.model_name}, embed_dim={self.embed_dim}, patch_size={self.patch_size}, source={source}, dtype={dtype}"


__all__ = ["DINOv3Encoder"]
