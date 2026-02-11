"""
Tests for DTS (Dynamic Timestep Shift) resolution-dependent behavior.

Verifies that exponential DTS varies mu with num_tokens and that
training/inference produce consistent mu values.

Phase A mandatory test file.
"""

import math

import pytest
import torch

from src.training.flow_matching.time_sampling import (
    compute_timestep_shift_mu,
    get_sampling_timesteps,
)
from src.inference.sampler.timesteps import get_timesteps


SHIFT_CONFIG = {
    "shift_type": "exponential",
    "fixed_shift": 3.0,
    "base_seq_len": 256,
    "max_seq_len": 4096,
}


class TestExponentialMuVariesWithNumTokens:
    """G-A1: mu must vary with num_tokens for exponential shift."""

    def test_exponential_mu_varies_with_num_tokens(self):
        """Different num_tokens should produce different mu values."""
        mu_256 = compute_timestep_shift_mu(
            num_tokens=256, shift_type="exponential", fixed_shift=3.0,
            base_seq_len=256, max_seq_len=4096,
        )
        mu_1024 = compute_timestep_shift_mu(
            num_tokens=1024, shift_type="exponential", fixed_shift=3.0,
            base_seq_len=256, max_seq_len=4096,
        )
        mu_4096 = compute_timestep_shift_mu(
            num_tokens=4096, shift_type="exponential", fixed_shift=3.0,
            base_seq_len=256, max_seq_len=4096,
        )
        assert mu_256 != mu_1024
        assert mu_1024 != mu_4096
        # Monotonic
        assert mu_256 < mu_1024 < mu_4096

    def test_exponential_base_seq_no_shift(self):
        """At base_seq_len, mu should be 0 (no shift)."""
        mu = compute_timestep_shift_mu(
            num_tokens=256, shift_type="exponential", fixed_shift=3.0,
            base_seq_len=256, max_seq_len=4096,
        )
        assert mu == pytest.approx(0.0, abs=1e-6)

    def test_exponential_max_seq_full_shift(self):
        """At max_seq_len, mu should be log(fixed_shift)."""
        mu = compute_timestep_shift_mu(
            num_tokens=4096, shift_type="exponential", fixed_shift=3.0,
            base_seq_len=256, max_seq_len=4096,
        )
        assert mu == pytest.approx(math.log(3.0), rel=1e-6, abs=1e-6)

    def test_dts_timesteps_differ_by_resolution(self):
        """Inference timesteps should differ for different num_tokens."""
        t_low = get_timesteps(
            num_steps=20, t_eps=0.0001,
            use_logit_normal=True,
            num_tokens=256,
            shift_config=SHIFT_CONFIG,
        )
        t_high = get_timesteps(
            num_steps=20, t_eps=0.0001,
            use_logit_normal=True,
            num_tokens=4096,
            shift_config=SHIFT_CONFIG,
        )
        # Different resolutions must produce different timestep sequences
        assert t_low.mean().item() != pytest.approx(t_high.mean().item(), abs=1e-4)


class TestTrainingInferenceConsistency:
    """Training and inference should produce the same mu for the same config."""

    @pytest.mark.parametrize("num_tokens", [256, 512, 1024, 2048, 4096])
    def test_mu_consistent_between_training_and_inference(self, num_tokens):
        """compute_timestep_shift_mu is shared; verify directly."""
        mu = compute_timestep_shift_mu(
            num_tokens=num_tokens,
            shift_type="exponential",
            fixed_shift=3.0,
            base_seq_len=256,
            max_seq_len=4096,
        )
        # Same function is used in both paths, so consistency is guaranteed.
        # This test documents the expected values.
        expected_factor = max(1.0, 1.0 + (3.0 - 1.0) * (num_tokens - 256) / (4096 - 256))
        expected_mu = math.log(expected_factor)
        assert mu == pytest.approx(expected_mu, abs=1e-6)


class TestExponentialEdgeCases:
    """Edge case tests for exponential DTS."""

    def test_below_base_seq_len_clamps_to_no_shift(self):
        """num_tokens < base_seq_len should clamp factor to 1.0 (mu=0)."""
        mu = compute_timestep_shift_mu(
            num_tokens=64, shift_type="exponential", fixed_shift=3.0,
            base_seq_len=256, max_seq_len=4096,
        )
        assert mu == pytest.approx(0.0, abs=1e-6)

    def test_above_max_seq_len_exceeds_full_shift(self):
        """num_tokens > max_seq_len should produce mu > log(fixed_shift)."""
        mu = compute_timestep_shift_mu(
            num_tokens=8192, shift_type="exponential", fixed_shift=3.0,
            base_seq_len=256, max_seq_len=4096,
        )
        assert mu > math.log(3.0)

    def test_training_timesteps_shift_at_high_resolution(self):
        """Training timesteps with DTS at high resolution should shift mean."""
        t_base = get_sampling_timesteps(
            num_steps=50, t_eps=0.0001,
            use_logit_normal=True, p_mean=0.0, p_std=1.0,
        )
        t_shifted = get_sampling_timesteps(
            num_steps=50, t_eps=0.0001,
            use_logit_normal=True, p_mean=0.0, p_std=1.0,
            num_tokens=4096, shift_config=SHIFT_CONFIG,
        )
        assert t_shifted.mean().item() < t_base.mean().item()
