"""
Tests for H-3: Legacy sampler lacks DTS support

PixelHDMSampler uses linspace and has no DTS attributes.
It is deprecated in favor of UnifiedSampler.

Phase 2 of audit fix plan (FIX_PLAN_INDEX_2026-02-06.md).
"""

from __future__ import annotations

import pytest
import torch

from src.training.flow_matching.pixelhdm_sampler import PixelHDMSampler


class TestLegacySamplerDTS:
    """H-3: Legacy sampler documentation and DTS absence."""

    def test_legacy_sampler_get_timesteps(self):
        """Legacy sampler should produce valid ascending timesteps."""
        sampler = PixelHDMSampler(num_steps=10)
        ts = sampler.get_timesteps(device=torch.device("cpu"))
        # Should be ascending (noise→clean)
        assert (ts[1:] - ts[:-1] > 0).all()
        # Should have num_steps + 1 boundaries
        assert ts.shape[0] == 11

    def test_legacy_sampler_no_dts_param(self):
        """Legacy sampler should not have DTS attributes."""
        sampler = PixelHDMSampler(num_steps=10)
        assert not hasattr(sampler, "use_dynamic_timestep_shift")
        assert not hasattr(sampler, "timestep_shift_config")

    def test_legacy_sampler_deprecation_docstring(self):
        """Legacy sampler should have deprecation notice in docstring."""
        assert "deprecated" in PixelHDMSampler.__doc__.lower()
