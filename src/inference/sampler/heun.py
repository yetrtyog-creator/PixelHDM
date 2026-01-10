"""
PixelHDM-RPEA-DinoV3 Sampler - Heun Solver

Second-order ODE solver for Flow Matching (V-Prediction).

Steps:
    1. Model prediction: v_t = model(z_t, t)
    2. Euler prediction: z_euler = z_t + dt * v_t
    3. Model at Euler point: v_next = model(z_euler, t_next)
    4. Average velocity update: z_{t+dt} = z_t + dt * (v_t + v_next) / 2

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from .euler import EulerSampler


class HeunSampler(EulerSampler):
    """
    Heun sampler (second-order ODE solver).

    V-Prediction: 模型直接輸出 velocity v = x - ε

    Steps:
        1. Model prediction: v_t = model(z_t, t)
        2. Euler prediction: z_euler = z_t + dt * v_t
        3. Model at Euler point: v_next = model(z_euler, t_next)
        4. Average: z_{t+dt} = z_t + dt * (v_t + v_next) / 2
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
        """Heun step."""
        dt = t_next - t
        t_batch = t.expand(z.shape[0])

        # Model prediction - V-Prediction (直接輸出 velocity)
        v_t = self._predict_v(
            model, z, t_batch, text_embeddings,
            guidance_scale, null_text_embeddings, **model_kwargs
        )

        # Euler prediction
        z_euler = z + dt * v_t

        # Compute at Euler prediction point
        t_next_batch = t_next.expand(z.shape[0])
        v_next = self._predict_v(
            model, z_euler, t_next_batch, text_embeddings,
            guidance_scale, null_text_embeddings, **model_kwargs
        )

        # Average velocity
        v_avg = (v_t + v_next) / 2

        # Heun update
        z_next = z + dt * v_avg

        return z_next


__all__ = ["HeunSampler"]
