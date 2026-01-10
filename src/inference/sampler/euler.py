"""
PixelHDM-RPEA-DinoV3 Sampler - Euler Solver

First-order ODE solver for Flow Matching (V-Prediction).

Steps:
    1. Model prediction: v = model(z_t, t)  # V-Prediction
    2. Update: z_{t+dt} = z_t + dt * v

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from .base import BaseSampler


class EulerSampler(BaseSampler):
    """
    Euler sampler (first-order ODE solver).

    V-Prediction: 模型直接輸出 velocity v = x - ε

    Steps:
        1. Model prediction: v = model(z_t, t)
        2. Update: z_{t+dt} = z_t + dt * v
    """

    def step(
        self,
        model: Callable,
        z: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        text_embeddings: Optional[torch.Tensor],
        guidance_scale: float = 1.0,
        null_text_embeddings: Optional[torch.Tensor] = None,
        **model_kwargs,
    ) -> torch.Tensor:
        """Euler step."""
        dt = t_next - t
        t_batch = t.expand(z.shape[0])

        # Model prediction - V-Prediction (直接輸出 velocity)
        v = self._predict_v(
            model, z, t_batch, text_embeddings,
            guidance_scale, null_text_embeddings, **model_kwargs
        )

        # Euler update
        z_next = z + dt * v

        return z_next


__all__ = ["EulerSampler"]
