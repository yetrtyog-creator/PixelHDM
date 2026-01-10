"""
Full Heun CFG Sampling (V-Prediction)

Higher quality CFG sampling that uses Heun integration for both branches.

V-Prediction:
    - Model outputs velocity v = x - noise
    - ODE integration: z_next = z + dt * v

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from typing import Callable, Optional

import torch


class FullHeunCFGMixin:
    """
    Mixin for full Heun CFG sampling (V-Prediction).

    Unlike standard CFG, this method uses Heun integration for both
    conditional and unconditional predictions. Provides higher quality
    but doubles NFE.
    """

    @torch.no_grad()
    def sample_with_full_heun_cfg(
        self,
        model: Callable,
        z_0: torch.Tensor,
        text_embeddings: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        guidance_scale: float = 7.5,
        null_text_embeddings: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        **model_kwargs,
    ) -> torch.Tensor:
        """
        Sample with full Heun CFG (V-Prediction).

        Args:
            model: PixelHDM model (outputs velocity)
            z_0: (B, H, W, 3) Initial noise
            text_embeddings: (B, T, D) Conditional text embeddings
            text_mask: (B, T) Text mask
            num_steps: Number of sampling steps
            guidance_scale: CFG guidance scale
            null_text_embeddings: (B, T, D) Unconditional text embeddings
            callback: Progress callback (step, total_steps, z)
            **model_kwargs: Additional model arguments

        Returns:
            x: (B, H, W, 3) Generated samples
        """
        num_steps = num_steps or self.num_steps
        device = z_0.device
        dtype = z_0.dtype

        timesteps = self._sampler.get_timesteps(num_steps, device, dtype)
        z = z_0

        for i in range(num_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]

            z = self._full_heun_cfg_step(
                model, z, t, t_next, i, num_steps,
                text_embeddings, null_text_embeddings,
                guidance_scale, **model_kwargs
            )

            if callback is not None:
                callback(i, num_steps, z)

        return z

    def _full_heun_cfg_step(
        self,
        model: Callable,
        z: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        step_idx: int,
        num_steps: int,
        text_embeddings: Optional[torch.Tensor],
        null_text_embeddings: Optional[torch.Tensor],
        guidance_scale: float,
        **model_kwargs,
    ) -> torch.Tensor:
        """Single step of full Heun CFG sampling (V-Prediction).

        Note:
            CFG 分支需要區分條件和無條件的 pooled_text_embed 和 text_mask：
            - pooled_text_embed / text_mask: 用於條件分支
            - null_pooled_text_embed / null_text_mask: 用於無條件分支
        """
        dt = t_next - t
        t_batch = t.expand(z.shape[0])

        # 提取 embeddings 和 masks (CFG 需要區分)
        pooled_text_embed = model_kwargs.pop("pooled_text_embed", None)
        null_pooled_text_embed = model_kwargs.pop("null_pooled_text_embed", None)
        text_mask = model_kwargs.pop("text_mask", None)
        null_text_mask = model_kwargs.pop("null_text_mask", None)

        # Conditional branch Heun (使用 pooled_text_embed 和 text_mask)
        v_cond_avg = self._heun_branch(
            model, z, t_batch, t_next, dt, step_idx, num_steps,
            text_embeddings, pooled_text_embed, text_mask, **model_kwargs
        )

        # Unconditional branch Heun (if CFG enabled, 使用 null_pooled_text_embed 和 null_text_mask)
        if guidance_scale > 1.0 and null_text_embeddings is not None:
            uncond_mask = null_text_mask if null_text_mask is not None else text_mask
            v_uncond_avg = self._heun_branch(
                model, z, t_batch, t_next, dt, step_idx, num_steps,
                null_text_embeddings, null_pooled_text_embed, uncond_mask, **model_kwargs
            )
            # CFG combination
            v_final = v_uncond_avg + guidance_scale * (v_cond_avg - v_uncond_avg)
        else:
            v_final = v_cond_avg

        return z + dt * v_final

    def _heun_branch(
        self,
        model: Callable,
        z: torch.Tensor,
        t_batch: torch.Tensor,
        t_next: torch.Tensor,
        dt: torch.Tensor,
        step_idx: int,
        num_steps: int,
        text_embeddings: Optional[torch.Tensor],
        pooled_text_embed: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        **model_kwargs,
    ) -> torch.Tensor:
        """Heun integration for a single branch (V-Prediction)."""
        # Step 1: Model directly outputs velocity
        v = model(
            z, t_batch,
            text_embed=text_embeddings,
            text_mask=text_mask,
            pooled_text_embed=pooled_text_embed,
            **model_kwargs
        )
        z_euler = z + dt * v

        # Step 2: Re-evaluate at Euler point (except last step)
        if step_idx < num_steps - 1:
            t_next_batch = t_next.expand(z.shape[0])
            v_next = model(
                z_euler, t_next_batch,
                text_embed=text_embeddings,
                text_mask=text_mask,
                pooled_text_embed=pooled_text_embed,
                **model_kwargs
            )
            return (v + v_next) / 2
        else:
            return v


__all__ = ["FullHeunCFGMixin"]
