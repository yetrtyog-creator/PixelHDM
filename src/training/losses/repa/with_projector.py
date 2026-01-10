"""
iREPA Loss with Projector Output

Variant that returns projected features for debugging and visualization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict

import torch
import torch.nn.functional as F

from .core import REPALoss

if TYPE_CHECKING:
    from ....config.model_config import PixelHDMConfig


class REPALossWithProjector(REPALoss):
    """
    iREPA Loss variant that returns projected features.

    Used for debugging and visualization.
    """

    def forward_with_features(
        self,
        h_t: torch.Tensor,
        x_clean: torch.Tensor,
        step: int = 0,
        dino_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute iREPA Loss and return intermediate features.

        Returns:
            Dict containing:
                - loss: iREPA loss value
                - h_proj: Projected model features
                - dino_features: DINOv3 features (after spatial normalization)
                - cos_sim: Cosine similarity
        """
        if step >= self.early_stop_step:
            return {
                "loss": torch.tensor(0.0, device=h_t.device, dtype=h_t.dtype),
                "h_proj": None,
                "dino_features": None,
                "cos_sim": None,
            }

        if dino_features is not None:
            y = dino_features
        else:
            if self.dino_encoder is None:
                raise RuntimeError("iREPA Loss requires DINOv3 encoder")

            with torch.no_grad():
                y = self.dino_encoder(x_clean)

        y = self._spatial_normalize(y)

        B, L, D = h_t.shape
        H, W = self._compute_hw(L, x_clean)

        h_2d = h_t.permute(0, 2, 1).reshape(B, D, H, W)
        h_proj = self.projector(h_2d)
        h_proj = h_proj.flatten(2).permute(0, 2, 1)

        if h_proj.shape[1] != y.shape[1]:
            h_proj = self._interpolate_features(h_proj, y.shape[1])

        h_norm = F.normalize(h_proj, dim=-1)
        y_norm = F.normalize(y, dim=-1)
        cos_sim = torch.sum(h_norm * y_norm, dim=-1)
        loss = (1.0 - cos_sim).mean() * self.lambda_repa

        return {
            "loss": loss,
            "h_proj": h_proj.detach(),
            "dino_features": y.detach(),
            "cos_sim": cos_sim.detach(),
        }


__all__ = ["REPALossWithProjector"]
