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
        t_eps: float = 0.0001,
        use_logit_normal: bool = True,
        time_p_mean: float = 0.0,
        time_p_std: float = 1.0,
        use_dynamic_timestep_shift: bool = True,
        timestep_shift_config: Optional[dict] = None,
    ) -> None:
        super().__init__(
            num_steps=num_steps,
            t_eps=t_eps,
            use_logit_normal=use_logit_normal,
            time_p_mean=time_p_mean,
            time_p_std=time_p_std,
            use_dynamic_timestep_shift=use_dynamic_timestep_shift,
            timestep_shift_config=timestep_shift_config,
        )
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
        guidance_rescale: float = 0.0,
        step_idx: Optional[int] = None,
        num_steps: Optional[int] = None,
        **model_kwargs,
    ) -> torch.Tensor:
        """DPM++ step.

        Args:
            model: PixelHDM model
            z: Current latent state
            t: Current timestep
            t_next: Next timestep
            text_embeddings: Conditional text embeddings
            guidance_scale: CFG scale
            null_text_embeddings: Unconditional text embeddings
            guidance_rescale: Rescale factor to prevent oversaturation
            step_idx: Current step index (unused by DPM++, accepted to avoid kwarg leak)
            num_steps: Total number of steps (unused by DPM++, accepted to avoid kwarg leak)
            **model_kwargs: Additional model arguments
        """
        t_batch = t.expand(z.shape[0])

        # Model prediction (V-Prediction → convert to x_pred)
        v_pred = self._predict_v(
            model, z, t_batch, text_embeddings,
            guidance_scale, null_text_embeddings,
            guidance_rescale=guidance_rescale,
            **model_kwargs
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
        """First order step.

        Correct coefficients for t: 0→1 (noise→clean):
        z_{t_next} = coef1 * z + coef2 * x_pred
        where coef1 = (1 - t_next)/(1 - t), coef2 = (t_next - t)/(1 - t)

        Derivation:
        - z_t = t*x + (1-t)*noise, v = x - noise
        - x = z_t + (1-t)*v, so x_pred = z + (1-t)*v_pred
        - z_{t_next} = z_t + (t_next - t)*v = z*(1-t_next)/(1-t) + x_pred*(t_next-t)/(1-t)
        """
        t_expanded = t.view(-1, *([1] * (z.dim() - 1)))
        t_next_expanded = t_next.view(-1, *([1] * (z.dim() - 1)))

        # Clamp t to prevent numerical instability when t→1
        # At t=0.99, denom=0.01+eps; max coefficient ~100 (acceptable)
        t_clamped = t_expanded.clamp(max=1 - self.t_eps)

        # Correct coefficients for forward time (t: 0→1)
        denom = 1 - t_clamped + max(self.t_eps, 1e-6)
        coef1 = (1 - t_next_expanded) / denom
        coef2 = (t_next_expanded - t_clamped) / denom

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
        """Second order step.

        Uses the same corrected coefficients as first order step,
        but with second-order extrapolation of x_pred.
        """
        # Second order correction
        D = x_pred + (r / (1 + r)) * (x_pred - x_pred_prev)

        t_expanded = t.view(-1, *([1] * (z.dim() - 1)))
        t_next_expanded = t_next.view(-1, *([1] * (z.dim() - 1)))

        # Clamp t to prevent numerical instability when t→1
        t_clamped = t_expanded.clamp(max=1 - self.t_eps)

        # Correct coefficients for forward time (t: 0→1)
        denom = 1 - t_clamped + max(self.t_eps, 1e-6)
        coef1 = (1 - t_next_expanded) / denom
        coef2 = (t_next_expanded - t_clamped) / denom

        z_next = coef1 * z + coef2 * D
        return z_next


__all__ = ["DPMPPSampler"]
