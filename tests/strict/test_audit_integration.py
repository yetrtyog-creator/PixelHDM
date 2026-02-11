"""
Integration tests for audit fix plan.

Verifies cross-component consistency after all fixes.

Phase 4 of audit fix plan (FIX_PLAN_INDEX_2026-02-06.md).
"""

from __future__ import annotations

import pytest
import torch

from src.inference.sampler.unified import UnifiedSampler
from src.inference.sampler.dpm import DPMPPSampler
from src.inference.sampler.euler import EulerSampler
from src.inference.sampler.heun import HeunSampler


class _IdentityModel:
    """Returns zero velocity (identity transformation)."""
    def __call__(self, z, t, **kwargs):
        return torch.zeros_like(z)


class TestAuditIntegration:
    """Integration tests verifying audit fixes work together."""

    def test_all_samplers_produce_valid_output(self):
        """All sampler methods should produce finite output."""
        model = _IdentityModel()
        z = torch.randn(1, 3, 8, 8)

        for method in ["euler", "heun", "dpm_pp"]:
            sampler = UnifiedSampler(
                method=method, num_steps=5,
                use_dynamic_timestep_shift=True,
                timestep_shift_config={
                    "shift_type": "exponential",
                    "fixed_shift": 3.0,
                },
            )
            result = sampler.sample(model, z, text_embeddings=None)
            assert torch.isfinite(result).all(), f"{method} produced NaN/Inf"

    def test_dts_training_inference_consistency(self):
        """DTS config defaults should be consistent between model and sampler."""
        from src.config.pixelhdm_config import PixelHDMConfig
        from src.inference.sampler.base import SamplerConfig

        model_config = PixelHDMConfig()
        sampler_config = SamplerConfig()

        # Both should have DTS enabled by default
        assert model_config.use_dynamic_timestep_shift is True
        assert sampler_config.use_dynamic_timestep_shift is True

        # fixed_shift should be 3.0 (not 1.0 no-op)
        assert model_config.timestep_shift_fixed == 3.0

    def test_t_eps_consistent_across_stack(self):
        """t_eps should be 0.0001 everywhere."""
        from src.config.pixelhdm_config import PixelHDMConfig
        from src.inference.sampler.base import SamplerConfig

        assert PixelHDMConfig().time_eps == 0.0001
        assert SamplerConfig().t_eps == 0.0001
        assert EulerSampler().t_eps == 0.0001
        assert HeunSampler().t_eps == 0.0001
        assert DPMPPSampler().t_eps == 0.0001

    def test_sampler_config_dts_default_matches_model(self):
        """SamplerConfig.use_dynamic_timestep_shift should match model default."""
        from src.config.pixelhdm_config import PixelHDMConfig
        from src.inference.sampler.base import SamplerConfig

        model_dts = PixelHDMConfig().use_dynamic_timestep_shift
        sampler_dts = SamplerConfig().use_dynamic_timestep_shift
        assert model_dts == sampler_dts == True
