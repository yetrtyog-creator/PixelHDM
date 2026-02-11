"""
Tests for H-1: Heun last-step Euler fallback

HeunSampler.step() should skip the corrector on the last step,
saving 1 NFE. Consistent with full_heun.py and pixelhdm_sampler.py.

Phase 2 of audit fix plan (FIX_PLAN_INDEX_2026-02-06.md).
"""

from __future__ import annotations

import pytest
import torch

from src.inference.sampler.heun import HeunSampler
from src.inference.sampler.unified import UnifiedSampler


class _CallCountModel:
    """Model that counts forward calls."""

    def __init__(self):
        self.call_count = 0

    def __call__(self, z, t, text_embeddings=None, **kwargs):
        self.call_count += 1
        return torch.zeros_like(z)


class TestHeunLastStep:
    """H-1: Heun last-step Euler fallback."""

    def test_heun_non_last_step_calls_twice(self):
        """Non-last step should call model twice (predictor + corrector)."""
        model = _CallCountModel()
        sampler = HeunSampler(num_steps=5)
        z = torch.randn(1, 3, 8, 8)
        t = torch.tensor(0.0)
        t_next = torch.tensor(0.2)

        sampler.step(
            model, z, t, t_next, text_embeddings=None,
            step_idx=0, num_steps=5,
        )
        assert model.call_count == 2

    def test_heun_last_step_calls_once(self):
        """Last step should call model only once (Euler fallback)."""
        model = _CallCountModel()
        sampler = HeunSampler(num_steps=5)
        z = torch.randn(1, 3, 8, 8)
        t = torch.tensor(0.8)
        t_next = torch.tensor(1.0)

        sampler.step(
            model, z, t, t_next, text_embeddings=None,
            step_idx=4, num_steps=5,
        )
        assert model.call_count == 1

    def test_heun_without_step_idx_full_heun(self):
        """Without step_idx/num_steps, all steps do full Heun (backward compat)."""
        model = _CallCountModel()
        sampler = HeunSampler(num_steps=5)
        z = torch.randn(1, 3, 8, 8)
        t = torch.tensor(0.8)
        t_next = torch.tensor(1.0)

        # No step_idx/num_steps → full Heun
        sampler.step(model, z, t, t_next, text_embeddings=None)
        assert model.call_count == 2

    def test_heun_5_steps_total_nfe(self):
        """5-step Heun: 4 full Heun (8 NFE) + 1 Euler (1 NFE) = 9 NFE."""
        model = _CallCountModel()
        sampler = HeunSampler(num_steps=5)
        z = torch.randn(1, 3, 8, 8)

        timesteps = torch.linspace(0, 1, 6)  # 6 boundaries for 5 steps
        for i in range(5):
            sampler.step(
                model, z, timesteps[i], timesteps[i + 1], text_embeddings=None,
                step_idx=i, num_steps=5,
            )
        assert model.call_count == 9  # 4*2 + 1*1

    def test_heun_nfe_count_method(self):
        """UnifiedSampler.count_nfe() should return 2*N-1 for Heun."""
        sampler = UnifiedSampler(method="heun", num_steps=50)
        nfe = sampler.count_nfe(use_cfg=False)
        assert nfe == 99  # 2*50 - 1
