"""
iREPA: Improved Representation Alignment Loss (Core)

Uses DINOv3 feature alignment to accelerate training.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from ....config.model_config import PixelHDMConfig

logger = logging.getLogger(__name__)


class REPALoss(nn.Module):
    """
    iREPA: Improved Representation Alignment Loss

    Uses DINOv3 to extract clean image features and align with model intermediate features.

    iREPA improvements (arXiv:2512.10794):
        1. Conv2d projection replaces MLP - preserves spatial structure
        2. Spatial normalization - enhances patch-wise contrast

    Loss formula:
        L_iREPA = -cos_sim(conv(h_t), spatial_norm(y))

    Shape:
        - h_t: (B, L, D) DiT intermediate features
        - x_clean: (B, H, W, 3) or (B, 3, H, W) clean image
        - Output: scalar
    """

    def __init__(
        self,
        config: Optional["PixelHDMConfig"] = None,
        hidden_dim: int = 1024,
        dino_dim: int = 768,
        lambda_repa: float = 0.5,
        early_stop_step: int = 250000,
        dino_encoder: Optional[nn.Module] = None,
        repa_patch_size: int = 16,
    ) -> None:
        super().__init__()

        if config is not None:
            hidden_dim = config.hidden_dim
            lambda_repa = config.repa_lambda
            early_stop_step = config.repa_early_stop
            dino_dim = config.repa_hidden_size
            repa_patch_size = config.repa_patch_size

        self.hidden_dim = hidden_dim
        self.dino_dim = dino_dim
        self.lambda_repa = lambda_repa
        self.early_stop_step = early_stop_step
        self.repa_patch_size = repa_patch_size

        self.projector = nn.Conv2d(
            hidden_dim, dino_dim,
            kernel_size=3, padding=1
        )

        self._init_weights()

        self.dino_encoder = dino_encoder
        self._dino_loaded = dino_encoder is not None

    def _init_weights(self) -> None:
        """Weight initialization (Conv2d)."""
        nn.init.kaiming_normal_(self.projector.weight, mode='fan_out', nonlinearity='relu')
        if self.projector.bias is not None:
            nn.init.zeros_(self.projector.bias)

    def _spatial_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        iREPA improvement 2: Spatial normalization

        Normalizes along L (spatial) dimension to enhance patch-wise contrast.
        """
        x = x - x.mean(dim=1, keepdim=True)
        x = x / (x.std(dim=1, keepdim=True) + 1e-6)
        return x

    def set_dino_encoder(self, encoder: nn.Module) -> None:
        """Set external DINOv3 encoder."""
        self.dino_encoder = encoder
        self._dino_loaded = True

    def _compute_hw(
        self, L: int, x_clean: Optional[torch.Tensor]
    ) -> tuple[int, int]:
        """Compute H, W from sequence length and clean image using repa_patch_size."""
        if x_clean is not None and x_clean.dim() == 4:
            if x_clean.shape[-1] == 3:
                img_H, img_W = x_clean.shape[1], x_clean.shape[2]
            else:
                img_H, img_W = x_clean.shape[2], x_clean.shape[3]
            H = img_H // self.repa_patch_size
            W = img_W // self.repa_patch_size
        else:
            H = W = int(L ** 0.5)
            if H * W != L:
                for h in range(int(L ** 0.5), 0, -1):
                    if L % h == 0:
                        H = h
                        W = L // h
                        break
        return H, W

    def forward(
        self,
        h_t: torch.Tensor,
        x_clean: torch.Tensor,
        step: int = 0,
        dino_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute iREPA Loss.

        Args:
            h_t: DiT intermediate features (B, L, D)
            x_clean: Clean image (B, H, W, 3) or (B, 3, H, W)
            step: Current training step (for early stopping)
            dino_features: Pre-computed DINOv3 features (optional)

        Returns:
            loss: iREPA loss value
        """
        if step >= self.early_stop_step:
            return torch.tensor(0.0, device=h_t.device, dtype=h_t.dtype)

        if dino_features is not None:
            y = dino_features
        else:
            if self.dino_encoder is None:
                raise RuntimeError(
                    "iREPA Loss requires DINOv3 encoder. Use set_dino_encoder() to set, "
                    "or provide dino_features parameter, "
                    "or set repa_enabled: false in config"
                )
            with torch.no_grad():
                y = self.dino_encoder(x_clean)

        y = self._spatial_normalize(y)

        B, L, D = h_t.shape
        H, W = self._compute_hw(L, x_clean)

        h_2d = h_t.permute(0, 2, 1).reshape(B, D, H, W)
        h_proj = self.projector(h_2d)
        h_proj = h_proj.flatten(2).permute(0, 2, 1)

        if h_proj.shape[1] != y.shape[1]:
            # Compute target H/W from DINO feature length using real image dimensions
            H_target, W_target = self._compute_hw(y.shape[1], x_clean)

            # Debug warning for interpolation mismatch
            logger.warning(
                f"[REPA] Interpolation triggered: h_proj ({H}x{W}={h_proj.shape[1]}) -> "
                f"DINO ({H_target}x{W_target}={y.shape[1]})"
            )

            h_proj = self._interpolate_features(h_proj, H, W, H_target, W_target)

        h_norm = F.normalize(h_proj, dim=-1)
        y_norm = F.normalize(y, dim=-1)

        cos_sim = torch.sum(h_norm * y_norm, dim=-1)
        loss = (1.0 - cos_sim).mean()

        return self.lambda_repa * loss

    def _interpolate_features(
        self,
        features: torch.Tensor,
        H: int,
        W: int,
        H_target: int,
        W_target: int,
    ) -> torch.Tensor:
        """
        Interpolate features to match target grid size.

        Args:
            features: (B, L, D) features to interpolate
            H, W: Source grid dimensions (must satisfy H * W == L)
            H_target, W_target: Target grid dimensions

        Returns:
            Interpolated features (B, H_target * W_target, D)
        """
        B, L, D = features.shape

        features = features.view(B, H, W, D).permute(0, 3, 1, 2)
        features = F.interpolate(
            features,
            size=(H_target, W_target),
            mode="bilinear",
            align_corners=False,
        )
        features = features.permute(0, 2, 3, 1).reshape(B, H_target * W_target, D)

        return features

    def get_projector(self) -> nn.Conv2d:
        """Get projector (for external use)."""
        return self.projector

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"dino_dim={self.dino_dim}, "
            f"lambda={self.lambda_repa}, "
            f"early_stop={self.early_stop_step}"
        )


__all__ = ["REPALoss"]
