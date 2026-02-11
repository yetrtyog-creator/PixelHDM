"""
Tests for D-1: DPM++ numerical stability with t_eps=0.0001

Verify DPM++ coefficients remain finite and NaN-free near boundaries.

Phase 4 of audit fix plan (FIX_PLAN_INDEX_2026-02-06.md).
"""

from __future__ import annotations

import pytest
import torch

from src.inference.sampler.dpm import DPMPPSampler
from src.inference.sampler.timesteps import get_lambda


class TestDPMNumericalStability:
    """D-1: DPM++ stability with small t_eps."""

    def test_dpm_stability_t_eps_0001(self):
        """DPM++ should produce finite output with t_eps=0.0001."""
        sampler = DPMPPSampler(num_steps=10, t_eps=0.0001)
        z = torch.randn(1, 3, 8, 8)

        model = lambda z, t, **kw: torch.zeros_like(z)

        timesteps = sampler.get_timesteps()
        sampler.reset()

        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = sampler.step(model, z, t, t_next, text_embeddings=None)
            assert torch.isfinite(z).all(), f"NaN/Inf at step {i}"

    def test_dpm_coef_bounds_near_t1(self):
        """Coefficients should stay bounded even near t→1."""
        sampler = DPMPPSampler(num_steps=10, t_eps=0.0001)
        z = torch.ones(1, 3, 4, 4)
        x_pred = torch.ones(1, 3, 4, 4)

        # Near t=1: denom should use max(t_eps, 1e-6) for stability
        t_high = torch.tensor(0.9999)
        t_next = torch.tensor(0.99999)

        result = sampler._first_order_step(z, x_pred, t_high, t_next)
        assert torch.isfinite(result).all(), "Coefficients diverged near t=1"

    def test_get_lambda_stability_at_boundaries(self):
        """get_lambda should be finite at t near 0 and t near 1."""
        t_low = torch.tensor(0.0001)
        t_high = torch.tensor(0.9999)
        t_eps = 0.0001

        lambda_low = get_lambda(t_low, t_eps)
        lambda_high = get_lambda(t_high, t_eps)

        assert torch.isfinite(lambda_low), f"lambda at t={t_low} is not finite"
        assert torch.isfinite(lambda_high), f"lambda at t={t_high} is not finite"
