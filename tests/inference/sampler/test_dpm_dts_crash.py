"""
Tests for C-1: DPM++ DTS TypeError

DPMPPSampler.__init__() did not accept DTS parameters
(use_dynamic_timestep_shift, timestep_shift_config), causing TypeError
when UnifiedSampler._create_sampler() passes them.

Phase 1 of audit fix plan (FIX_PLAN_INDEX_2026-02-06.md).
"""

from __future__ import annotations

import pytest
import torch

from src.inference.sampler.dpm import DPMPPSampler
from src.inference.sampler.unified import UnifiedSampler


class TestDPMPPDTSCrash:
    """C-1: DPM++ should accept DTS parameters without TypeError."""

    def test_dpmpp_with_dts_after_fix(self):
        """DPMPPSampler should accept DTS kwargs after fix."""
        sampler = DPMPPSampler(
            num_steps=10,
            use_dynamic_timestep_shift=True,
            timestep_shift_config={"shift_type": "exponential", "fixed_shift": 3.0},
        )
        assert sampler.use_dynamic_timestep_shift is True
        assert sampler.timestep_shift_config is not None

    def test_dpmpp_dts_params_forwarded(self):
        """DTS params should be stored on BaseSampler via super().__init__()."""
        config = {"shift_type": "exponential", "fixed_shift": 3.0}
        sampler = DPMPPSampler(
            num_steps=10,
            use_dynamic_timestep_shift=True,
            timestep_shift_config=config,
        )
        assert sampler.use_dynamic_timestep_shift is True
        assert sampler.timestep_shift_config == config

    def test_unified_sampler_dpmpp_with_dts(self):
        """UnifiedSampler with method='dpm_pp' + DTS should not raise."""
        sampler = UnifiedSampler(
            method="dpm_pp",
            num_steps=10,
            use_dynamic_timestep_shift=True,
            timestep_shift_config={"shift_type": "exponential", "fixed_shift": 3.0},
        )
        assert isinstance(sampler._sampler, DPMPPSampler)
        assert sampler._sampler.use_dynamic_timestep_shift is True

    def test_unified_sampler_dpmpp_dts_false(self):
        """UnifiedSampler with DTS=False should also work (baseline)."""
        sampler = UnifiedSampler(
            method="dpm_pp",
            num_steps=10,
            use_dynamic_timestep_shift=False,
        )
        assert isinstance(sampler._sampler, DPMPPSampler)
        assert sampler._sampler.use_dynamic_timestep_shift is False
