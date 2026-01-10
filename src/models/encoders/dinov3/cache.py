"""
DINOv3 Feature Caching

LRU cache for DINOv3 feature extraction.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F


class FeatureCacheMixin:
    """
    Mixin providing feature caching for DINOv3 encoder.

    Requires host class to have:
        - self._feature_cache: OrderedDict
        - self._cache_max_size: int
        - self._cache_enabled: bool
        - self.patch_size: int
    """

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Prepare input tensor."""
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got: {x.dim()}D")
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        B, C, H, W = x.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            new_H = ((H + self.patch_size - 1) // self.patch_size) * self.patch_size
            new_W = ((W + self.patch_size - 1) // self.patch_size) * self.patch_size
            x = F.interpolate(x, size=(new_H, new_W), mode="bilinear")
        return x

    def _check_cache(
        self, x: torch.Tensor, return_dict: bool
    ) -> Optional[Union[torch.Tensor, Dict]]:
        """Check feature cache."""
        if not self._cache_enabled or x.shape[0] != 1:
            return None
        cache_key = self._compute_cache_key(x)
        if cache_key in self._feature_cache:
            cached = self._feature_cache.pop(cache_key)
            self._feature_cache[cache_key] = cached
            if return_dict:
                return {"patch_tokens": cached, "cls_token": None}
            return cached
        return None

    def _update_cache(self, x: torch.Tensor, patch_tokens: torch.Tensor) -> None:
        """Update feature cache."""
        if not self._cache_enabled or x.shape[0] != 1:
            return
        cache_key = self._compute_cache_key(x)
        self._feature_cache[cache_key] = patch_tokens.clone()
        while len(self._feature_cache) > self._cache_max_size:
            self._feature_cache.popitem(last=False)

    def _compute_cache_key(self, x: torch.Tensor) -> str:
        """Compute cache key from tensor content."""
        with torch.no_grad():
            sample = x[0, :, ::16, ::16].flatten().cpu().numpy().tobytes()
            return hashlib.md5(sample).hexdigest()

    def clear_cache(self) -> None:
        """Clear feature cache."""
        self._feature_cache.clear()

    def set_cache_enabled(self, enabled: bool) -> None:
        """Set cache enabled."""
        self._cache_enabled = enabled
        if not enabled:
            self.clear_cache()

    def _parse_output(self, output) -> tuple:
        """Parse model output."""
        if hasattr(output, "last_hidden_state"):
            hidden = output.last_hidden_state
            return hidden[:, 1:], hidden[:, 0]
        elif isinstance(output, torch.Tensor):
            return output[:, 1:], output[:, 0]
        raise ValueError(f"Unknown output format: {type(output)}")


__all__ = ["FeatureCacheMixin"]
