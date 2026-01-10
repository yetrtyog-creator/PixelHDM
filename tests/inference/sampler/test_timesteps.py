"""
Timestep Scheduling Unit Tests.

Validates timestep generation and lambda calculation:
- Linear timestep generation
- Boundary handling (t_eps)
- Image-to-image (t_start)
- Lambda function properties

Test Count: 15 test cases

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-08
"""

import pytest
import torch
import math

from src.inference.sampler.timesteps import get_timesteps, get_lambda


# =============================================================================
# Test Class: Timestep Generation
# =============================================================================


class TestTimestepGeneration:
    """Test timestep sequence generation."""

    def test_timesteps_shape(self):
        """Test output shape is (num_steps + 1,)."""
        for num_steps in [10, 20, 50, 100]:
            timesteps = get_timesteps(num_steps)
            assert timesteps.shape == (num_steps + 1,)

    def test_timesteps_increasing(self):
        """Test timesteps are monotonically increasing."""
        timesteps = get_timesteps(50)

        for i in range(len(timesteps) - 1):
            assert timesteps[i] < timesteps[i + 1], f"Not increasing at index {i}"

    def test_timesteps_bounds(self):
        """Test timesteps respect t_eps boundaries."""
        t_eps = 0.05
        timesteps = get_timesteps(50, t_eps=t_eps)

        assert timesteps[0] >= t_eps, f"First timestep {timesteps[0]} < t_eps {t_eps}"
        assert timesteps[-1] <= 1 - t_eps, f"Last timestep {timesteps[-1]} > 1 - t_eps"

    def test_timesteps_uniform_spacing(self):
        """Test timesteps are uniformly spaced."""
        timesteps = get_timesteps(50, t_eps=0.05)

        diffs = timesteps[1:] - timesteps[:-1]
        expected_diff = diffs[0]

        for i, diff in enumerate(diffs):
            assert torch.allclose(diff, expected_diff, atol=1e-5), \
                f"Non-uniform spacing at index {i}"

    def test_timesteps_device(self):
        """Test timesteps on specified device."""
        device = torch.device("cpu")
        timesteps = get_timesteps(50, device=device)
        assert timesteps.device == device

        if torch.cuda.is_available():
            device_cuda = torch.device("cuda")
            timesteps_cuda = get_timesteps(50, device=device_cuda)
            # Compare device type, not exact device (cuda vs cuda:0)
            assert timesteps_cuda.device.type == device_cuda.type

    def test_timesteps_dtype(self):
        """Test timesteps with specified dtype."""
        for dtype in [torch.float32, torch.float64]:
            timesteps = get_timesteps(50, dtype=dtype)
            assert timesteps.dtype == dtype


# =============================================================================
# Test Class: Image-to-Image Timesteps
# =============================================================================


class TestI2ITimesteps:
    """Test timestep generation for image-to-image."""

    def test_t_start_parameter(self):
        """Test t_start shifts starting point."""
        t_start = 0.5
        timesteps = get_timesteps(50, t_eps=0.05, t_start=t_start)

        # First timestep should be at or near t_start
        assert timesteps[0] >= t_start - 0.01

    def test_t_start_reduces_steps(self):
        """Test t_start effectively reduces active steps."""
        timesteps_full = get_timesteps(50, t_eps=0.05)
        timesteps_half = get_timesteps(50, t_eps=0.05, t_start=0.5)

        # With t_start=0.5, range is [0.5, 0.95] vs [0.05, 0.95]
        range_full = timesteps_full[-1] - timesteps_full[0]
        range_half = timesteps_half[-1] - timesteps_half[0]

        assert range_half < range_full

    def test_t_start_boundaries(self):
        """Test t_start at boundary values."""
        # t_start = t_eps (full generation)
        ts_full = get_timesteps(50, t_eps=0.05, t_start=0.05)
        assert ts_full[0] >= 0.05

        # t_start = 0.95 (minimal change)
        ts_minimal = get_timesteps(50, t_eps=0.05, t_start=0.9)
        assert ts_minimal[0] >= 0.9


# =============================================================================
# Test Class: Lambda Function
# =============================================================================


class TestLambdaFunction:
    """Test lambda calculation for DPM++."""

    def test_lambda_at_half(self):
        """Test lambda(0.5) = 0."""
        t = torch.tensor([0.5])
        lambda_val = get_lambda(t)

        # lambda(0.5) = log(0.5 / 0.5) = log(1) = 0
        assert torch.allclose(lambda_val, torch.zeros(1), atol=1e-5)

    def test_lambda_monotonic_increasing(self):
        """Test lambda is monotonically increasing with t."""
        t_values = torch.linspace(0.1, 0.9, 9)
        lambda_values = get_lambda(t_values)

        for i in range(len(lambda_values) - 1):
            assert lambda_values[i + 1] > lambda_values[i], \
                f"Lambda not increasing at t={t_values[i]}"

    def test_lambda_antisymmetry(self):
        """Test lambda(t) = -lambda(1-t)."""
        t_pairs = [(0.2, 0.8), (0.3, 0.7), (0.4, 0.6)]

        for t1, t2 in t_pairs:
            lambda_1 = get_lambda(torch.tensor([t1]))
            lambda_2 = get_lambda(torch.tensor([t2]))

            assert torch.allclose(lambda_1, -lambda_2, atol=1e-4), \
                f"Antisymmetry failed for t={t1}, 1-t={t2}"

    def test_lambda_stability_near_boundaries(self):
        """Test lambda stability near t=0 and t=1."""
        # Near t=0 (should be large negative)
        t_near_0 = torch.tensor([0.01])
        lambda_0 = get_lambda(t_near_0, t_eps=0.01)

        assert not torch.isnan(lambda_0).any()
        assert not torch.isinf(lambda_0).any()
        assert lambda_0.item() < 0  # Should be negative

        # Near t=1 (should be large positive)
        t_near_1 = torch.tensor([0.99])
        lambda_1 = get_lambda(t_near_1, t_eps=0.01)

        assert not torch.isnan(lambda_1).any()
        assert not torch.isinf(lambda_1).any()
        assert lambda_1.item() > 0  # Should be positive

    def test_lambda_t_eps_clamping(self):
        """Test t_eps clamping prevents division by zero."""
        t_eps = 0.05

        # t=0 should be clamped to t_eps
        t_zero = torch.tensor([0.0])
        lambda_zero = get_lambda(t_zero, t_eps=t_eps)

        # Should not be -inf
        assert not torch.isinf(lambda_zero).any()

        # t=1 should be clamped to 1-t_eps
        t_one = torch.tensor([1.0])
        lambda_one = get_lambda(t_one, t_eps=t_eps)

        # Should not be +inf
        assert not torch.isinf(lambda_one).any()


# =============================================================================
# Test Class: Edge Cases
# =============================================================================


class TestTimestepEdgeCases:
    """Test edge cases for timestep utilities."""

    def test_single_step(self):
        """Test with num_steps=1."""
        timesteps = get_timesteps(1, t_eps=0.05)

        assert timesteps.shape == (2,)
        assert timesteps[0] < timesteps[1]

    def test_many_steps(self):
        """Test with large num_steps."""
        timesteps = get_timesteps(1000, t_eps=0.05)

        assert timesteps.shape == (1001,)
        assert torch.all(timesteps[1:] > timesteps[:-1])

    def test_different_t_eps(self):
        """Test different t_eps values."""
        for t_eps in [0.01, 0.05, 0.1, 0.2]:
            timesteps = get_timesteps(50, t_eps=t_eps)

            assert timesteps[0] >= t_eps
            assert timesteps[-1] <= 1 - t_eps

    def test_timesteps_reproducible(self):
        """Test timestep generation is deterministic."""
        ts1 = get_timesteps(50, t_eps=0.05)
        ts2 = get_timesteps(50, t_eps=0.05)

        assert torch.allclose(ts1, ts2)
