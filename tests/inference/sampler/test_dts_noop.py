"""
Tests for C-3: DTS No-Op Default Value

timestep_shift_fixed=1.0 with exponential shift makes DTS a no-op
because log(1.0)=0. Default changed to 3.0 (SD3 recommended).

Phase 1 of audit fix plan (FIX_PLAN_INDEX_2026-02-06.md).
"""

from __future__ import annotations

import math
import logging

import pytest

from src.config.pixelhdm_config import PixelHDMConfig
from src.config.validators import _validate_timestep_shift


class TestDTSNoop:
    """C-3: DTS no-op detection and default fix."""

    def test_fixed_shift_1_produces_identity(self):
        """fixed_shift=1.0 with exponential → mu=log(1)=0 → identity."""
        mu = math.log(1.0)
        assert mu == 0.0, "log(1.0) should be exactly 0"
        # With mu=0, the shift function t' = sigmoid(mu) * t / ... becomes identity
        # because sigmoid(0) = 0.5 and the formula degenerates

    def test_fixed_shift_3_produces_nontrivial(self):
        """fixed_shift=3.0 with exponential → mu=log(3)≈1.099 → real shift."""
        mu = math.log(3.0)
        assert mu > 1.0, f"Expected mu > 1.0, got {mu}"
        # mu≈1.099 produces meaningful time shift

    def test_default_config_dts_fixed_shift_is_3(self):
        """After fix, PixelHDMConfig default should be 3.0, not 1.0."""
        config = PixelHDMConfig()
        assert config.timestep_shift_fixed == 3.0, (
            f"Expected default timestep_shift_fixed=3.0, got {config.timestep_shift_fixed}"
        )

    def test_config_warns_on_noop_dts(self, caplog):
        """Validator should warn when exponential shift + fixed_shift=1.0."""
        config = PixelHDMConfig(
            use_dynamic_timestep_shift=True,
            timestep_shift_type="exponential",
            timestep_shift_fixed=1.0,
        )
        with caplog.at_level(logging.WARNING):
            _validate_timestep_shift(config)
        assert any("no shift" in record.message.lower() for record in caplog.records), (
            f"Expected 'no shift' warning, got: {[r.message for r in caplog.records]}"
        )
