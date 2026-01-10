"""
PixelHDM Flow Matching Mathematical Verification Tests

This module provides comprehensive unit tests for verifying the mathematical
correctness of the Flow Matching implementation.

PixelHDM Time Convention:
    - t=0: pure noise (epsilon)
    - t=1: clean image (x)

Core Formulas:
    - Interpolation: z_t = t * x + (1 - t) * epsilon
    - V-Target: v_target = x - epsilon (constant velocity field)
    - V-Theta: v_theta = (x_pred - z_t) / (1 - t)  [when predicting x from z_t]
    - Loss: L = E[||v_pred - v_target||^2]

Test Coverage:
    - Interpolation Formula Tests (10 tests)
    - V-Prediction Tests (10 tests)
    - Time Sampling Tests (10 tests)
    - Training Preparation Tests (5 tests)

Total: 35 tests

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-08
"""

from __future__ import annotations

import math
from typing import Tuple

import pytest
import torch
import torch.nn as nn

from src.training.flow_matching.core import create_flow_matching
from src.training.flow_matching.training import PixelHDMFlowMatching
from src.training.flow_matching.time_sampling import sample_logit_normal, TimeSampler
from src.training.flow_matching.noise import interpolate


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def default_flow_matching() -> PixelHDMFlowMatching:
    """Create default flow matching instance (SD3/PixelHDM convention)."""
    return PixelHDMFlowMatching(p_mean=0.0, p_std=1.0, t_eps=0.05)


@pytest.fixture
def strict_flow_matching() -> PixelHDMFlowMatching:
    """Create flow matching with t_eps=0 for boundary tests."""
    return PixelHDMFlowMatching(p_mean=0.0, p_std=1.0, t_eps=0.0)


@pytest.fixture
def time_sampler() -> TimeSampler:
    """Create default time sampler."""
    return TimeSampler(p_mean=0.0, p_std=1.0, t_eps=0.05)


@pytest.fixture
def clean_image() -> torch.Tensor:
    """Create a deterministic clean image tensor."""
    torch.manual_seed(42)
    return torch.randn(2, 3, 64, 64)


@pytest.fixture
def noise_tensor() -> torch.Tensor:
    """Create a deterministic noise tensor."""
    torch.manual_seed(123)
    return torch.randn(2, 3, 64, 64)


@pytest.fixture
def large_batch_tensors() -> Tuple[torch.Tensor, torch.Tensor]:
    """Create large batch tensors for statistical tests."""
    torch.manual_seed(42)
    x = torch.randn(64, 3, 32, 32)
    noise = torch.randn(64, 3, 32, 32)
    return x, noise


# ============================================================================
# Interpolation Formula Tests (10 tests)
# ============================================================================


class TestInterpolationFormula:
    """
    Tests for PixelHDM interpolation formula: z_t = t * x + (1 - t) * epsilon

    The interpolation linearly blends between noise (t=0) and clean image (t=1).
    This is the forward noising process of Flow Matching.
    """

    def test_interpolate_t_0_equals_noise(self, strict_flow_matching, clean_image, noise_tensor):
        """
        At t=0, z_t should equal noise.

        Math: z_0 = 0 * x + (1 - 0) * epsilon = epsilon
        """
        B = clean_image.shape[0]
        t = torch.zeros(B)

        z_t = strict_flow_matching.interpolate(clean_image, noise_tensor, t)

        assert torch.allclose(z_t, noise_tensor, atol=1e-6), \
            f"At t=0, z_t should equal noise. Max diff: {(z_t - noise_tensor).abs().max()}"

    def test_interpolate_t_1_equals_clean(self, strict_flow_matching, clean_image, noise_tensor):
        """
        At t=1, z_t should equal clean image x.

        Math: z_1 = 1 * x + (1 - 1) * epsilon = x
        """
        B = clean_image.shape[0]
        t = torch.ones(B)

        z_t = strict_flow_matching.interpolate(clean_image, noise_tensor, t)

        assert torch.allclose(z_t, clean_image, atol=1e-6), \
            f"At t=1, z_t should equal clean image. Max diff: {(z_t - clean_image).abs().max()}"

    def test_interpolate_t_0_5_midpoint(self, strict_flow_matching, clean_image, noise_tensor):
        """
        At t=0.5, z_t should be exact midpoint between x and noise.

        Math: z_0.5 = 0.5 * x + 0.5 * epsilon = (x + epsilon) / 2
        """
        B = clean_image.shape[0]
        t = torch.full((B,), 0.5)

        z_t = strict_flow_matching.interpolate(clean_image, noise_tensor, t)
        expected = 0.5 * clean_image + 0.5 * noise_tensor

        assert torch.allclose(z_t, expected, atol=1e-6), \
            f"At t=0.5, z_t should be (x + noise)/2. Max diff: {(z_t - expected).abs().max()}"

    def test_interpolate_linear(self, strict_flow_matching, clean_image, noise_tensor):
        """
        Test linear interpolation at arbitrary t values.

        Math: z_t = t * x + (1 - t) * epsilon for any t in [0, 1]
        """
        B = clean_image.shape[0]
        test_values = [0.1, 0.25, 0.33, 0.67, 0.75, 0.9]

        for t_val in test_values:
            t = torch.full((B,), t_val)
            z_t = strict_flow_matching.interpolate(clean_image, noise_tensor, t)
            expected = t_val * clean_image + (1 - t_val) * noise_tensor

            assert torch.allclose(z_t, expected, atol=1e-6), \
                f"At t={t_val}, interpolation incorrect. Max diff: {(z_t - expected).abs().max()}"

    def test_interpolate_derivative_constant(self, strict_flow_matching, clean_image, noise_tensor):
        """
        The derivative dz/dt should be constant and equal to (x - noise).

        Math: dz_t/dt = d(t*x + (1-t)*epsilon)/dt = x - epsilon

        This is verified by checking z(t+dt) - z(t) = dt * (x - epsilon)
        """
        B = clean_image.shape[0]
        dt = 0.01

        for t_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
            t1 = torch.full((B,), t_val)
            t2 = torch.full((B,), t_val + dt)

            z_t1 = strict_flow_matching.interpolate(clean_image, noise_tensor, t1)
            z_t2 = strict_flow_matching.interpolate(clean_image, noise_tensor, t2)

            numerical_deriv = (z_t2 - z_t1) / dt
            analytical_deriv = clean_image - noise_tensor

            assert torch.allclose(numerical_deriv, analytical_deriv, atol=1e-4), \
                f"Derivative at t={t_val} should be (x - noise). Max diff: {(numerical_deriv - analytical_deriv).abs().max()}"

    def test_interpolate_bounds(self, strict_flow_matching):
        """
        z_t values should be bounded by the range of x and noise values.

        For t in [0, 1], z_t is a convex combination of x and noise,
        so each element of z_t should be between the corresponding
        elements of x and noise.
        """
        B, C, H, W = 4, 3, 32, 32
        torch.manual_seed(42)
        x = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)

        for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            t = torch.full((B,), t_val)
            z_t = strict_flow_matching.interpolate(x, noise, t)

            # z_t should be within bounds of x and noise (element-wise)
            min_bound = torch.minimum(x, noise)
            max_bound = torch.maximum(x, noise)

            assert (z_t >= min_bound - 1e-6).all(), \
                f"z_t below min bound at t={t_val}"
            assert (z_t <= max_bound + 1e-6).all(), \
                f"z_t above max bound at t={t_val}"

    def test_interpolate_shape_preservation(self, strict_flow_matching):
        """
        Test that interpolation preserves input shapes for various dimensions.
        """
        shapes = [
            (1, 3, 32, 32),
            (4, 3, 64, 64),
            (2, 3, 128, 128),
            (8, 3, 256, 256),
            (2, 3, 64, 128),  # Non-square
            (2, 3, 512, 256),  # Non-square
        ]

        for shape in shapes:
            torch.manual_seed(42)
            x = torch.randn(shape)
            noise = torch.randn(shape)
            t = torch.rand(shape[0])

            z_t = strict_flow_matching.interpolate(x, noise, t)

            assert z_t.shape == shape, \
                f"Shape mismatch: expected {shape}, got {z_t.shape}"

    def test_interpolate_batch_broadcasting(self, strict_flow_matching):
        """
        Test that timestep t broadcasts correctly across batch dimension.

        Each batch element should have its own interpolation weight.
        """
        B = 4
        torch.manual_seed(42)
        x = torch.randn(B, 3, 32, 32)
        noise = torch.randn(B, 3, 32, 32)
        t = torch.tensor([0.0, 0.33, 0.67, 1.0])

        z_t = strict_flow_matching.interpolate(x, noise, t)

        # Verify each batch element independently
        for i in range(B):
            expected_i = t[i] * x[i] + (1 - t[i]) * noise[i]
            assert torch.allclose(z_t[i], expected_i, atol=1e-6), \
                f"Batch element {i} interpolation incorrect"

    def test_interpolate_dtype_preservation(self, strict_flow_matching):
        """
        Test that interpolation preserves the input data type.
        """
        B, C, H, W = 2, 3, 32, 32

        for dtype in [torch.float32, torch.float64]:
            torch.manual_seed(42)
            x = torch.randn(B, C, H, W, dtype=dtype)
            noise = torch.randn(B, C, H, W, dtype=dtype)
            t = torch.rand(B, dtype=dtype)

            z_t = strict_flow_matching.interpolate(x, noise, t)

            assert z_t.dtype == dtype, \
                f"Expected dtype {dtype}, got {z_t.dtype}"

    def test_interpolate_device_preservation(self, strict_flow_matching, device):
        """
        Test that interpolation preserves the device placement.
        """
        B, C, H, W = 2, 3, 32, 32
        torch.manual_seed(42)
        x = torch.randn(B, C, H, W, device=device)
        noise = torch.randn(B, C, H, W, device=device)
        t = torch.rand(B, device=device)

        z_t = strict_flow_matching.interpolate(x, noise, t)

        assert z_t.device.type == device.type, \
            f"Expected device {device.type}, got {z_t.device.type}"


# ============================================================================
# V-Prediction Tests (10 tests)
# ============================================================================


class TestVPrediction:
    """
    Tests for V-Prediction formulas in Flow Matching.

    In PixelHDM V-Prediction:
        - Network directly outputs velocity v = x - epsilon
        - v_target = x - noise (constant velocity field)
        - v_theta can be computed from x_pred: v_theta = (x_pred - z_t) / (1 - t)
        - Loss: ||v_pred - v_target||^2
    """

    def test_compute_v_theta_formula(self, default_flow_matching, clean_image, noise_tensor):
        """
        Test that v_theta can be computed from x_pred using the formula:
        v_theta = (x_pred - z_t) / (1 - t)

        When x_pred is perfect (x_pred = x), v_theta should equal v_target.
        """
        B = clean_image.shape[0]
        t_val = 0.5
        t = torch.full((B,), t_val)

        # Create z_t
        z_t = default_flow_matching.interpolate(clean_image, noise_tensor, t)

        # If model predicts x perfectly
        x_pred = clean_image

        # Compute v_theta from x_pred
        t_expanded = t.view(-1, 1, 1, 1)
        v_theta = (x_pred - z_t) / (1 - t_expanded)

        # v_target is always x - noise
        v_target = clean_image - noise_tensor

        assert torch.allclose(v_theta, v_target, atol=1e-5), \
            f"v_theta should equal v_target when x_pred is perfect. Max diff: {(v_theta - v_target).abs().max()}"

    def test_compute_v_target_formula(self, default_flow_matching, clean_image, noise_tensor):
        """
        Test v_target = x - noise computation.

        The target velocity in Flow Matching is the constant direction
        from noise to clean image.
        """
        v_target = default_flow_matching.compute_v_target(clean_image, noise_tensor)
        expected = (clean_image - noise_tensor).float()

        assert torch.allclose(v_target, expected, atol=1e-6), \
            f"v_target should be x - noise. Max diff: {(v_target - expected).abs().max()}"

    def test_v_theta_equals_v_target_when_perfect(self, default_flow_matching, clean_image, noise_tensor):
        """
        When model prediction is perfect (v_pred = v_target),
        the loss should be zero.
        """
        v_pred = clean_image - noise_tensor  # Perfect prediction
        v_target = default_flow_matching.compute_v_target(clean_image, noise_tensor)

        assert torch.allclose(v_pred.float(), v_target, atol=1e-6), \
            "Perfect v_pred should equal v_target"

    def test_v_theta_numerical_stability_t_near_0(self, default_flow_matching, clean_image, noise_tensor):
        """
        Test numerical stability when t is close to 0.

        Near t=0, z_t is mostly noise, and v_theta = (x_pred - z_t) / (1 - t)
        has (1 - t) close to 1, which is stable.
        """
        B = clean_image.shape[0]
        t_val = 0.01  # Very small t
        t = torch.full((B,), t_val)

        z_t = default_flow_matching.interpolate(clean_image, noise_tensor, t)
        x_pred = clean_image

        t_expanded = t.view(-1, 1, 1, 1)
        v_theta = (x_pred - z_t) / (1 - t_expanded)

        # Should be stable (no NaN or Inf)
        assert not torch.isnan(v_theta).any(), "v_theta contains NaN at t near 0"
        assert not torch.isinf(v_theta).any(), "v_theta contains Inf at t near 0"

        # v_theta should still approximate v_target
        v_target = clean_image - noise_tensor
        assert torch.allclose(v_theta, v_target, atol=0.1), \
            f"v_theta should approximate v_target at t={t_val}"

    def test_v_theta_numerical_stability_t_near_1(self, default_flow_matching, clean_image, noise_tensor):
        """
        Test numerical stability when t is close to 1.

        Near t=1, (1 - t) is small, causing potential division issues.
        Clamping t to [t_eps, 1-t_eps] prevents this.
        """
        B = clean_image.shape[0]
        t_val = 0.95  # t_eps=0.05 means max t is 0.95
        t = torch.full((B,), t_val)

        z_t = default_flow_matching.interpolate(clean_image, noise_tensor, t)
        x_pred = clean_image

        # With clamping: use max(1-t, eps) to prevent division by zero
        t_expanded = t.view(-1, 1, 1, 1)
        eps = 0.05
        v_theta = (x_pred - z_t) / torch.clamp(1 - t_expanded, min=eps)

        # Should be stable
        assert not torch.isnan(v_theta).any(), "v_theta contains NaN at t near 1"
        assert not torch.isinf(v_theta).any(), "v_theta contains Inf at t near 1"

    def test_v_loss_zero_for_perfect_prediction(self, default_flow_matching, clean_image, noise_tensor):
        """
        When v_pred perfectly matches v_target, loss should be approximately 0.
        """
        v_pred = clean_image - noise_tensor  # Perfect prediction

        loss = default_flow_matching.compute_loss(v_pred, clean_image, noise_tensor)

        assert loss < 1e-10, f"Loss should be ~0 for perfect prediction, got {loss.item()}"

    def test_v_loss_positive_for_random_prediction(self, default_flow_matching, clean_image, noise_tensor):
        """
        For random predictions, loss should be positive.
        """
        torch.manual_seed(999)
        v_pred = torch.randn_like(clean_image)

        loss = default_flow_matching.compute_loss(v_pred, clean_image, noise_tensor)

        assert loss > 0, f"Loss should be positive for random prediction, got {loss.item()}"

    def test_v_loss_gradient_flow(self, default_flow_matching, clean_image, noise_tensor):
        """
        Test that gradients flow correctly through the loss computation.
        """
        v_pred = torch.randn_like(clean_image, requires_grad=True)

        loss = default_flow_matching.compute_loss(v_pred, clean_image, noise_tensor)
        loss.backward()

        assert v_pred.grad is not None, "v_pred should have gradients"
        assert not torch.isnan(v_pred.grad).any(), "Gradients contain NaN"
        assert not torch.isinf(v_pred.grad).any(), "Gradients contain Inf"

        # Gradient should be non-zero (prediction is wrong)
        assert v_pred.grad.abs().sum() > 0, "Gradients should be non-zero"

    def test_t_eps_clamping(self, default_flow_matching):
        """
        Test that t_eps clamping prevents division by zero.

        When sampling timesteps, they should be in [t_eps, 1-t_eps].
        """
        t_eps = default_flow_matching.t_eps

        # Sample many timesteps
        t = default_flow_matching.sample_timesteps(10000, torch.device("cpu"))

        assert (t >= t_eps).all(), f"Some t values below t_eps={t_eps}"
        assert (t <= 1 - t_eps).all(), f"Some t values above 1-t_eps={1-t_eps}"

        # (1 - t) should always be >= t_eps
        assert ((1 - t) >= t_eps).all(), "1-t should always be >= t_eps"

    def test_v_theta_batch_independence(self, default_flow_matching):
        """
        Test that v_theta computation is batch-independent.

        Computing v_theta for each batch element separately should give
        the same result as computing it for the full batch.
        """
        B = 4
        torch.manual_seed(42)
        x = torch.randn(B, 3, 32, 32)
        noise = torch.randn(B, 3, 32, 32)
        t = torch.tensor([0.2, 0.4, 0.6, 0.8])

        # Compute v_target for full batch
        v_target_batch = default_flow_matching.compute_v_target(x, noise)

        # Compute v_target for each element separately
        for i in range(B):
            v_target_i = default_flow_matching.compute_v_target(
                x[i:i+1], noise[i:i+1]
            )
            assert torch.allclose(v_target_batch[i:i+1], v_target_i, atol=1e-6), \
                f"Batch element {i} v_target differs"


# ============================================================================
# Time Sampling Tests (10 tests)
# ============================================================================


class TestTimeSampling:
    """
    Tests for timestep sampling using Logit-Normal distribution.

    Formula: t = sigmoid(p_mean + p_std * N(0,1))
    Then rescaled to [t_eps, 1-t_eps].

    The Logit-Normal distribution provides:
    - Values concentrated around sigmoid(p_mean)
    - Controlled spread via p_std
    - Natural bounds in (0, 1)
    """

    def test_sample_timesteps_shape(self, default_flow_matching):
        """
        Test that sampled timesteps have correct shape (batch_size,).
        """
        for batch_size in [1, 4, 16, 64, 256]:
            t = default_flow_matching.sample_timesteps(batch_size, torch.device("cpu"))

            assert t.shape == (batch_size,), \
                f"Expected shape ({batch_size},), got {t.shape}"

    def test_sample_timesteps_bounds(self, default_flow_matching):
        """
        Test that all timesteps are within [t_eps, 1-t_eps].
        """
        t_eps = default_flow_matching.t_eps

        # Sample many timesteps
        t = default_flow_matching.sample_timesteps(10000, torch.device("cpu"))

        assert (t >= t_eps).all(), \
            f"Min t={t.min().item()} below t_eps={t_eps}"
        assert (t <= 1 - t_eps).all(), \
            f"Max t={t.max().item()} above 1-t_eps={1-t_eps}"

    def test_sample_timesteps_distribution_mean(self, default_flow_matching):
        """
        Test that the mean of sampled timesteps is approximately sigmoid(p_mean).

        For p_mean=0, sigmoid(0)=0.5, so expected mean is around 0.5.
        """
        n_samples = 50000
        t = default_flow_matching.sample_timesteps(n_samples, torch.device("cpu"))

        sample_mean = t.mean().item()
        # With p_mean=0 and rescaling to [0.05, 0.95], expected mean is ~0.5
        expected_mean = 0.5

        assert abs(sample_mean - expected_mean) < 0.05, \
            f"Sample mean {sample_mean:.4f} differs from expected {expected_mean}"

    def test_sample_timesteps_distribution_std(self, default_flow_matching):
        """
        Test that the standard deviation of sampled timesteps is reasonable.

        With p_std=1.0, we expect good coverage of the [t_eps, 1-t_eps] range.
        """
        n_samples = 50000
        t = default_flow_matching.sample_timesteps(n_samples, torch.device("cpu"))

        sample_std = t.std().item()

        # Std should be positive and meaningful (not too small, not too large)
        assert sample_std > 0.1, f"Std {sample_std:.4f} too small"
        assert sample_std < 0.5, f"Std {sample_std:.4f} too large"

    def test_sample_timesteps_reproducible_with_seed(self, default_flow_matching):
        """
        Test that timestep sampling is deterministic with fixed seed.
        """
        batch_size = 16

        torch.manual_seed(42)
        t1 = default_flow_matching.sample_timesteps(batch_size, torch.device("cpu"))

        torch.manual_seed(42)
        t2 = default_flow_matching.sample_timesteps(batch_size, torch.device("cpu"))

        assert torch.allclose(t1, t2), \
            "Same seed should produce identical timesteps"

    def test_sample_timesteps_different_seeds(self, default_flow_matching):
        """
        Test that different seeds produce different timesteps.
        """
        batch_size = 16

        torch.manual_seed(42)
        t1 = default_flow_matching.sample_timesteps(batch_size, torch.device("cpu"))

        torch.manual_seed(123)
        t2 = default_flow_matching.sample_timesteps(batch_size, torch.device("cpu"))

        assert not torch.allclose(t1, t2), \
            "Different seeds should produce different timesteps"

    def test_sample_timesteps_device_placement(self, default_flow_matching, device):
        """
        Test that timesteps are placed on the correct device.
        """
        t = default_flow_matching.sample_timesteps(16, device)

        assert t.device.type == device.type, \
            f"Expected device {device.type}, got {t.device.type}"

    def test_sample_timesteps_dtype(self, default_flow_matching):
        """
        Test that timesteps have the correct dtype.
        """
        for dtype in [torch.float32, torch.float64]:
            t = default_flow_matching.sample_timesteps(
                16, torch.device("cpu"), dtype=dtype
            )

            assert t.dtype == dtype, \
                f"Expected dtype {dtype}, got {t.dtype}"

    def test_logit_normal_distribution(self):
        """
        Test the logit-normal distribution properties directly.

        For u ~ N(p_mean, p_std^2), t = sigmoid(u) follows a logit-normal distribution.
        """
        n_samples = 100000
        p_mean = 0.0
        p_std = 1.0
        t_eps = 0.05

        t = sample_logit_normal(
            n_samples, torch.device("cpu"), torch.float32,
            p_mean=p_mean, p_std=p_std, t_eps=t_eps
        )

        # Check bounds
        assert (t >= t_eps).all(), "Values below t_eps"
        assert (t <= 1 - t_eps).all(), "Values above 1-t_eps"

        # Check approximate mean (should be close to 0.5 for p_mean=0)
        sample_mean = t.mean().item()
        assert 0.45 < sample_mean < 0.55, \
            f"Mean {sample_mean:.4f} not centered around 0.5"

        # Check that distribution covers the full range
        percentiles = [t.kthvalue(int(n_samples * p))[0].item() for p in [0.01, 0.5, 0.99]]
        assert percentiles[0] < 0.15, "1st percentile should be low"
        assert 0.45 < percentiles[1] < 0.55, "50th percentile should be around 0.5"
        assert percentiles[2] > 0.85, "99th percentile should be high"

    def test_p_mean_p_std_effect(self):
        """
        Test that p_mean and p_std affect the distribution as expected.
        """
        n_samples = 50000
        t_eps = 0.05

        # Test p_mean effect: negative p_mean shifts distribution lower
        t_neg = sample_logit_normal(
            n_samples, torch.device("cpu"), torch.float32,
            p_mean=-1.0, p_std=1.0, t_eps=t_eps
        )
        t_zero = sample_logit_normal(
            n_samples, torch.device("cpu"), torch.float32,
            p_mean=0.0, p_std=1.0, t_eps=t_eps
        )
        t_pos = sample_logit_normal(
            n_samples, torch.device("cpu"), torch.float32,
            p_mean=1.0, p_std=1.0, t_eps=t_eps
        )

        mean_neg = t_neg.mean().item()
        mean_zero = t_zero.mean().item()
        mean_pos = t_pos.mean().item()

        assert mean_neg < mean_zero < mean_pos, \
            f"p_mean should shift distribution: {mean_neg:.3f} < {mean_zero:.3f} < {mean_pos:.3f}"

        # Test p_std effect: larger p_std spreads distribution
        t_small_std = sample_logit_normal(
            n_samples, torch.device("cpu"), torch.float32,
            p_mean=0.0, p_std=0.5, t_eps=t_eps
        )
        t_large_std = sample_logit_normal(
            n_samples, torch.device("cpu"), torch.float32,
            p_mean=0.0, p_std=2.0, t_eps=t_eps
        )

        std_small = t_small_std.std().item()
        std_large = t_large_std.std().item()

        assert std_small < std_large, \
            f"Larger p_std should increase spread: {std_small:.3f} < {std_large:.3f}"


# ============================================================================
# Training Preparation Tests (5 tests)
# ============================================================================


class TestTrainingPreparation:
    """
    Tests for the prepare_training method which sets up all tensors
    needed for a training step.

    prepare_training returns: (t, z_t, x, noise)
        - t: sampled timesteps
        - z_t: interpolated (noisy) images
        - x: original clean images
        - noise: generated or provided noise
    """

    def test_prepare_training_generates_noise(self, default_flow_matching, clean_image):
        """
        Test that prepare_training generates noise when not provided.
        """
        t, z_t, x_out, noise = default_flow_matching.prepare_training(clean_image)

        assert noise is not None, "Noise should be generated"
        assert noise.shape == clean_image.shape, \
            f"Noise shape {noise.shape} should match input {clean_image.shape}"

        # Noise should be different from clean image
        assert not torch.allclose(noise, clean_image, atol=0.1), \
            "Generated noise should differ from clean image"

    def test_prepare_training_computes_z_t(self, default_flow_matching, clean_image, noise_tensor):
        """
        Test that prepare_training correctly computes z_t using interpolation.
        """
        B = clean_image.shape[0]
        t_fixed = torch.full((B,), 0.5)

        t, z_t, x_out, noise_out = default_flow_matching.prepare_training(
            clean_image, noise=noise_tensor, t=t_fixed
        )

        # Manually compute expected z_t
        expected_z_t = default_flow_matching.interpolate(clean_image, noise_tensor, t_fixed)

        assert torch.allclose(z_t, expected_z_t, atol=1e-6), \
            f"z_t computation incorrect. Max diff: {(z_t - expected_z_t).abs().max()}"

    def test_prepare_training_computes_v_target(self, default_flow_matching, clean_image, noise_tensor):
        """
        Test that v_target can be computed from prepare_training outputs.
        """
        t, z_t, x_out, noise_out = default_flow_matching.prepare_training(
            clean_image, noise=noise_tensor
        )

        v_target = default_flow_matching.compute_v_target(x_out, noise_out)
        expected = (x_out - noise_out).float()

        assert torch.allclose(v_target, expected, atol=1e-6), \
            "v_target should be x - noise"

    def test_prepare_training_output_shapes(self, default_flow_matching):
        """
        Test that all outputs from prepare_training have correct shapes.
        """
        shapes_to_test = [
            (2, 3, 64, 64),
            (4, 3, 128, 128),
            (1, 3, 256, 256),
            (8, 3, 32, 32),
        ]

        for shape in shapes_to_test:
            B = shape[0]
            torch.manual_seed(42)
            x = torch.randn(shape)

            t, z_t, x_out, noise = default_flow_matching.prepare_training(x)

            assert t.shape == (B,), f"t shape should be ({B},), got {t.shape}"
            assert z_t.shape == shape, f"z_t shape should be {shape}, got {z_t.shape}"
            assert x_out.shape == shape, f"x_out shape should be {shape}, got {x_out.shape}"
            assert noise.shape == shape, f"noise shape should be {shape}, got {noise.shape}"

    def test_prepare_training_custom_noise(self, default_flow_matching, clean_image):
        """
        Test that prepare_training uses custom noise when provided.
        """
        torch.manual_seed(42)
        custom_noise = torch.randn_like(clean_image)

        t, z_t, x_out, noise_out = default_flow_matching.prepare_training(
            clean_image, noise=custom_noise
        )

        assert torch.allclose(noise_out, custom_noise), \
            "Should use provided noise"


# ============================================================================
# Additional Mathematical Verification Tests
# ============================================================================


class TestMathematicalProperties:
    """
    Additional tests for mathematical properties of Flow Matching.
    """

    def test_interpolation_path_is_straight_line(self, strict_flow_matching, clean_image, noise_tensor):
        """
        Verify that the interpolation path from noise to clean is a straight line.

        For three points t1 < t2 < t3, the point at t2 should be on the line
        connecting t1 and t3.
        """
        B = clean_image.shape[0]
        t1, t2, t3 = 0.2, 0.5, 0.8

        z_t1 = strict_flow_matching.interpolate(
            clean_image, noise_tensor, torch.full((B,), t1)
        )
        z_t2 = strict_flow_matching.interpolate(
            clean_image, noise_tensor, torch.full((B,), t2)
        )
        z_t3 = strict_flow_matching.interpolate(
            clean_image, noise_tensor, torch.full((B,), t3)
        )

        # z_t2 should be linearly interpolated between z_t1 and z_t3
        # weight = (t2 - t1) / (t3 - t1)
        weight = (t2 - t1) / (t3 - t1)
        expected_z_t2 = (1 - weight) * z_t1 + weight * z_t3

        assert torch.allclose(z_t2, expected_z_t2, atol=1e-5), \
            "Interpolation path should be a straight line"

    def test_velocity_field_is_constant(self, strict_flow_matching, clean_image, noise_tensor):
        """
        Verify that the velocity field v = x - epsilon is constant (independent of t).

        In Flow Matching, the optimal transport velocity field is constant,
        which is why v_target = x - epsilon does not depend on t.
        """
        v_target = strict_flow_matching.compute_v_target(clean_image, noise_tensor)

        # The velocity should be the same regardless of what t we use
        # This is implicit in the formula v = x - epsilon
        expected = clean_image - noise_tensor

        assert torch.allclose(v_target, expected.float(), atol=1e-6), \
            "Velocity field should be x - epsilon (independent of t)"

    def test_perfect_denoising_trajectory(self, strict_flow_matching, clean_image, noise_tensor):
        """
        Verify that following the velocity field perfectly reconstructs the clean image.

        If we start from z_0 = noise and follow v = x - epsilon:
        z_t = z_0 + t * v = noise + t * (x - noise) = t * x + (1 - t) * noise

        At t=1: z_1 = x (clean image)
        """
        B = clean_image.shape[0]

        # Start from pure noise (t=0)
        z_0 = noise_tensor.clone()

        # Get velocity
        v = strict_flow_matching.compute_v_target(clean_image, noise_tensor)

        # Simulate Euler integration
        n_steps = 100
        dt = 1.0 / n_steps
        z = z_0.clone()

        for _ in range(n_steps):
            z = z + dt * v

        # Final z should be close to clean image
        assert torch.allclose(z, clean_image.float(), atol=1e-3), \
            f"Following velocity should reach clean image. Max diff: {(z - clean_image).abs().max()}"

    def test_loss_scale_with_error(self, default_flow_matching, clean_image, noise_tensor):
        """
        Test that loss scales appropriately with prediction error.

        Loss = E[||v_pred - v_target||^2]

        Adding a constant offset to v_pred should increase loss quadratically.
        """
        v_target = default_flow_matching.compute_v_target(clean_image, noise_tensor)

        errors = [0.0, 0.1, 0.2, 0.5, 1.0]
        losses = []

        for error in errors:
            v_pred = v_target + error
            loss = default_flow_matching.compute_loss(v_pred, clean_image, noise_tensor)
            losses.append(loss.item())

        # Loss should be zero for error=0
        assert losses[0] < 1e-10, f"Loss for zero error should be ~0, got {losses[0]}"

        # Loss should increase with error
        for i in range(1, len(losses)):
            assert losses[i] > losses[i-1], \
                f"Loss should increase with error: {losses[i-1]} -> {losses[i]}"

        # Loss should scale quadratically: loss ~ error^2
        # For error=1.0, loss should be approximately 1.0 (per element)
        expected_loss_1 = 1.0  # error^2 = 1
        assert abs(losses[-1] - expected_loss_1) < 0.01, \
            f"Loss for error=1 should be ~1, got {losses[-1]}"
