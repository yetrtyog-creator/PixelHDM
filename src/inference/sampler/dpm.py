"""
PixelHDM-RPEA-DinoV3 Sampler - DPM++ Solver

Multi-step predictor-corrector solver based on DPM++ paper.
Adapted for V-Prediction: model outputs velocity v = x - ε.

x_pred 計算: x = z + (1-t) * v

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from .base import BaseSampler
from .timesteps import get_lambda


class DPMPPSampler(BaseSampler):
    """
    DPM++ sampler (multi-step predictor-corrector).

    V-Prediction: 模型輸出 velocity v = x - ε
    內部將 v 轉換為 x_pred: x = z + (1-t) * v

    Based on the DPM++ paper's second-order solver.
    """

    def __init__(
        self,
        num_steps: int = 50,
        t_eps: float = 0.05,
    ) -> None:
        super().__init__(num_steps, t_eps)
        self._prev_x_pred = None
        self._prev_t = None

    def reset(self) -> None:
        """Reset state."""
        self._prev_x_pred = None
        self._prev_t = None

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
        """DPM++ step."""
        t_batch = t.expand(z.shape[0])

        # Model prediction (V-Prediction → convert to x_pred)
        v_pred = self._predict_v(
            model, z, t_batch, text_embeddings,
            guidance_scale, null_text_embeddings, **model_kwargs
        )

        # Convert v_pred to x_pred: x = z + (1-t) * v
        x_pred = self._v_to_x(v_pred, z, t_batch)

        # Compute coefficients
        lambda_t = get_lambda(t, self.t_eps)
        lambda_next = get_lambda(t_next, self.t_eps)
        h = lambda_next - lambda_t

        if self._prev_x_pred is None:
            # First order (first step)
            z_next = self._first_order_step(z, x_pred, t, t_next)
        else:
            # Second order
            lambda_prev = get_lambda(self._prev_t, self.t_eps)
            h_prev = lambda_t - lambda_prev
            r = h_prev / (h + 1e-8)
            z_next = self._second_order_step(z, x_pred, self._prev_x_pred, t, t_next, r)

        # Save history
        self._prev_x_pred = x_pred
        self._prev_t = t

        return z_next

    def _v_to_x(
        self,
        v_pred: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert velocity to x_pred.

        Formula: x = z + (1-t) * v

        Derivation:
            z_t = t*x + (1-t)*ε
            v = x - ε
            => x = z_t + (1-t)*v
        """
        t_expanded = t.view(-1, *([1] * (z.dim() - 1)))
        return z + (1 - t_expanded) * v_pred

    def _first_order_step(
        self,
        z: torch.Tensor,
        x_pred: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
    ) -> torch.Tensor:
        """First order step."""
        t_expanded = t.view(-1, *([1] * (z.dim() - 1)))
        t_next_expanded = t_next.view(-1, *([1] * (z.dim() - 1)))

        coef1 = t_next_expanded / (t_expanded + 1e-8)
        coef2 = (1 - t_next_expanded) - (1 - t_expanded) * coef1

        z_next = coef1 * z + coef2 * x_pred
        return z_next

    def _second_order_step(
        self,
        z: torch.Tensor,
        x_pred: torch.Tensor,
        x_pred_prev: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        r: torch.Tensor,
    ) -> torch.Tensor:
        """Second order step."""
        # Second order correction
        D = x_pred + (r / (1 + r)) * (x_pred - x_pred_prev)

        t_expanded = t.view(-1, *([1] * (z.dim() - 1)))
        t_next_expanded = t_next.view(-1, *([1] * (z.dim() - 1)))

        coef1 = t_next_expanded / (t_expanded + 1e-8)
        coef2 = (1 - t_next_expanded) - (1 - t_expanded) * coef1

        z_next = coef1 * z + coef2 * D
        return z_next


__all__ = ["DPMPPSampler"]
