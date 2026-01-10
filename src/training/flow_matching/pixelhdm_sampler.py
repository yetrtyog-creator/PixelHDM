"""
PixelHDM Sampler for Inference (V-Prediction)

Sampling direction: t from 0 to 1 (noise to clean).

V-Prediction:
    - Model outputs velocity v = x - noise
    - ODE integration: z_next = z + dt * v

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from typing import Optional, Callable

import torch

from .time_sampling import get_sampling_timesteps


class PixelHDMSampler:
    """
    PixelHDM Sampler for inference (V-Prediction).

    Args:
        num_steps: Number of sampling steps
        method: Sampling method ("euler" or "heun")
        t_eps: Time epsilon
    """

    def __init__(
        self,
        num_steps: int = 50,
        method: str = "heun",
        t_eps: float = 0.05,
    ) -> None:
        self.num_steps = num_steps
        self.method = method
        self.t_eps = t_eps

    def get_timesteps(
        self, num_steps: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Get sampling timesteps (PixelHDM: from t_eps to 1-t_eps)."""
        num_steps = num_steps or self.num_steps
        return get_sampling_timesteps(num_steps, device, dtype, self.t_eps)

    def v_to_x(
        self, v_pred: torch.Tensor, z: torch.Tensor, t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert velocity to x_pred.

        Formula: x = z + (1-t) * v

        Derivation:
            z_t = t*x + (1-t)*noise
            v = x - noise
            => x = z + (1-t) * v
        """
        original_dtype = v_pred.dtype
        v_f32, z_f32, t_f32 = v_pred.float(), z.float(), t.float()
        t_expanded = t_f32.view(-1, *([1] * (v_f32.dim() - 1)))
        x_pred = z_f32 + (1 - t_expanded) * v_f32
        return x_pred.to(original_dtype)

    def _predict_v(
        self, model: Callable, z: torch.Tensor, t: torch.Tensor,
        text_embeddings: Optional[torch.Tensor],
        guidance_scale: float, null_text_embeddings: Optional[torch.Tensor],
        **model_kwargs,
    ) -> torch.Tensor:
        """Predict velocity with optional CFG."""
        if guidance_scale > 1.0 and null_text_embeddings is not None:
            v_uncond = model(z, t, text_embeddings=null_text_embeddings, **model_kwargs)
            v_cond = model(z, t, text_embeddings=text_embeddings, **model_kwargs)
            return v_uncond + guidance_scale * (v_cond - v_uncond)
        return model(z, t, text_embeddings=text_embeddings, **model_kwargs)

    def _heun_step(
        self, model: Callable, z: torch.Tensor, t: torch.Tensor,
        t_next: torch.Tensor, v_t: torch.Tensor,
        text_embeddings: Optional[torch.Tensor],
        guidance_scale: float, null_text_embeddings: Optional[torch.Tensor],
        **model_kwargs,
    ) -> torch.Tensor:
        """Heun step (second-order ODE solver)."""
        dt = t_next - t
        z_euler = z + dt * v_t
        t_next_batch = t_next.expand(z.shape[0])
        v_next = self._predict_v(
            model, z_euler, t_next_batch, text_embeddings,
            guidance_scale, null_text_embeddings, **model_kwargs
        )
        v_avg = (v_t + v_next) / 2
        return z + dt * v_avg

    @torch.no_grad()
    def sample(
        self, model: Callable, z_0: torch.Tensor,
        text_embeddings: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None, guidance_scale: float = 1.0,
        null_text_embeddings: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[int, torch.Tensor], None]] = None,
        **model_kwargs,
    ) -> torch.Tensor:
        """Generate sample from noise."""
        num_steps = num_steps or self.num_steps
        device, dtype = z_0.device, z_0.dtype
        timesteps = self.get_timesteps(num_steps, device, dtype)
        z = z_0

        for i in range(num_steps):
            t, t_next = timesteps[i], timesteps[i + 1]
            dt = t_next - t
            t_batch = t.expand(z.shape[0])

            v = self._predict_v(
                model, z, t_batch, text_embeddings,
                guidance_scale, null_text_embeddings, **model_kwargs
            )

            if self.method == "heun" and i < num_steps - 1:
                z = self._heun_step(
                    model, z, t, t_next, v, text_embeddings,
                    guidance_scale, null_text_embeddings, **model_kwargs
                )
            else:
                z = z + dt * v

            if callback is not None:
                callback(i, z)

        return z


__all__ = ["PixelHDMSampler"]
