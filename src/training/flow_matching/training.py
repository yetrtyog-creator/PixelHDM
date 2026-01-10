"""
PixelHDM Flow Matching Training Module (V-Prediction)

Time Convention (PixelHDM):
    - t=0: noise
    - t=1: clean image
    - Interpolation: z_t = t * x + (1 - t) * noise

V-Prediction:
    - 網路直接輸出 velocity v = x - ε
    - 無需 X→V 轉換，避免 1/(1-t) 誤差放大
    - Loss: L = E[||v_pred - v_target||²]

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Callable, Dict, Any

import torch
import torch.nn as nn

from .noise import interpolate
from .time_sampling import sample_logit_normal

if TYPE_CHECKING:
    from ...config.model_config import PixelHDMConfig


class PixelHDMFlowMatching(nn.Module):
    """
    PixelHDM Flow Matching Training Module (V-Prediction).

    V-Prediction Loss:
        v_pred = model(z_t, t)  # 網路直接輸出 velocity
        v_target = x - noise
        L = E[||v_pred - v_target||²]

    Args:
        config: Optional PixelHDMConfig
        p_mean: Time distribution mean (SD3/PixelHDM: 0.0)
        p_std: Time distribution std (SD3/PixelHDM: 1.0)
        t_eps: Time epsilon
    """

    def __init__(
        self,
        config: Optional["PixelHDMConfig"] = None,
        p_mean: float = 0.0,
        p_std: float = 1.0,
        t_eps: float = 0.05,
    ) -> None:
        super().__init__()
        if config is not None:
            p_mean = config.time_p_mean
            p_std = config.time_p_std
            t_eps = config.time_eps
        self.p_mean = p_mean
        self.p_std = p_std
        self.t_eps = t_eps

    def sample_timesteps(
        self, batch_size: int, device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Sample timesteps using Logit-Normal distribution."""
        return sample_logit_normal(
            batch_size=batch_size, device=device, dtype=dtype,
            p_mean=self.p_mean, p_std=self.p_std, t_eps=self.t_eps,
        )

    def interpolate(
        self, x: torch.Tensor, noise: torch.Tensor, t: torch.Tensor,
    ) -> torch.Tensor:
        """PixelHDM interpolation: z_t = t * x + (1 - t) * noise."""
        return interpolate(x, noise, t)

    def prepare_training(
        self, x: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare training data. Returns: (t, z_t, x, noise)."""
        B, device, dtype = x.shape[0], x.device, x.dtype
        if noise is None:
            noise = torch.randn_like(x)
        if t is None:
            t = self.sample_timesteps(B, device, dtype)
        z_t = self.interpolate(x, noise, t)
        return t, z_t, x, noise

    def compute_v_target(
        self, x: torch.Tensor, noise: torch.Tensor,
    ) -> torch.Tensor:
        """Compute v_target = x - noise."""
        return x.float() - noise.float()

    def compute_loss(
        self, v_pred: torch.Tensor, x: torch.Tensor,
        noise: torch.Tensor, reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute V-Loss (V-Prediction).

        Args:
            v_pred: Network output (velocity)
            x: Clean image
            noise: Noise
            reduction: "mean", "sum", or "none"

        Returns:
            Loss value
        """
        original_dtype = v_pred.dtype
        v_target = self.compute_v_target(x, noise)
        loss = (v_pred.float() - v_target).pow(2)
        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        return loss.to(original_dtype)

    def forward(
        self, model: Callable, x: torch.Tensor,
        text_embeddings: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None, **model_kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Complete forward pass (training step).

        Returns: (loss, metrics)
        """
        t, z_t, x_clean, noise = self.prepare_training(x, noise, t)
        v_pred = model(z_t, t, text_embeddings=text_embeddings, **model_kwargs)
        loss = self.compute_loss(v_pred, x_clean, noise)
        with torch.no_grad():
            metrics = {
                "loss": loss.detach(),
                "v_pred_mean": v_pred.mean().detach(),
                "v_pred_std": v_pred.std().detach(),
                "t_mean": t.mean(),
            }
        return loss, metrics


__all__ = ["PixelHDMFlowMatching"]
