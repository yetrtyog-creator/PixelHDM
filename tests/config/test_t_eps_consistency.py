"""
Tests for H-2: t_eps default consistency (0.0001 across all components)

All components must use t_eps=0.0001 as default, matching the documentation.

Phase 2 of audit fix plan (FIX_PLAN_INDEX_2026-02-06.md).
"""

from __future__ import annotations

import pytest

EXPECTED_T_EPS = 0.0001


class TestTEpsConsistency:
    """H-2: All default t_eps values must be 0.0001."""

    def test_config_default_t_eps(self):
        from src.config.pixelhdm_config import PixelHDMConfig
        config = PixelHDMConfig()
        assert config.time_eps == EXPECTED_T_EPS

    def test_sampler_config_default_t_eps(self):
        from src.inference.sampler.base import SamplerConfig
        config = SamplerConfig()
        assert config.t_eps == EXPECTED_T_EPS

    def test_base_sampler_default_t_eps(self):
        from src.inference.sampler.euler import EulerSampler
        sampler = EulerSampler()
        assert sampler.t_eps == EXPECTED_T_EPS

    def test_dpmpp_default_t_eps(self):
        from src.inference.sampler.dpm import DPMPPSampler
        sampler = DPMPPSampler()
        assert sampler.t_eps == EXPECTED_T_EPS

    def test_unified_sampler_default_t_eps(self):
        from src.inference.sampler.unified import UnifiedSampler
        sampler = UnifiedSampler(method="euler")
        assert sampler.t_eps == EXPECTED_T_EPS

    def test_create_sampler_default_t_eps(self):
        from src.inference.sampler.unified import create_sampler
        sampler = create_sampler(method="euler")
        assert sampler.t_eps == EXPECTED_T_EPS

    def test_get_timesteps_default_t_eps(self):
        import inspect
        from src.inference.sampler.timesteps import get_timesteps
        sig = inspect.signature(get_timesteps)
        assert sig.parameters["t_eps"].default == EXPECTED_T_EPS

    def test_flow_matching_default_t_eps(self):
        from src.training.flow_matching.training import PixelHDMFlowMatching
        fm = PixelHDMFlowMatching()
        assert fm.t_eps == EXPECTED_T_EPS

    def test_generator_fallback_t_eps(self):
        import inspect
        from src.inference.pipeline.generation import Generator
        sig = inspect.signature(Generator._get_t_eps)
        # _get_t_eps reads from self.config; verify the fallback return value
        # by checking generation.py source for the fallback constant
        import src.inference.pipeline.generation as gen_mod
        source = inspect.getsource(gen_mod.Generator._get_t_eps)
        assert "0.0001" in source

    def test_vloss_default_t_eps(self):
        from src.training.losses.vloss import VLoss
        vloss = VLoss()
        assert vloss.t_eps == EXPECTED_T_EPS

    def test_from_config_uses_config_value(self):
        from src.config.pixelhdm_config import PixelHDMConfig
        from src.inference.sampler.unified import UnifiedSampler
        config = PixelHDMConfig()
        sampler = UnifiedSampler(method="euler", t_eps=config.time_eps)
        assert sampler.t_eps == config.time_eps == EXPECTED_T_EPS
