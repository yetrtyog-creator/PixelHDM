"""
PixelHDM-RPEA-DinoV3 Flow Matching Tests (V-Prediction)

Tests for PixelHDM Flow Matching implementation:
    - Time sampling (Logit-Normal distribution)
    - Interpolation formula (z_t = t * x + (1 - t) * epsilon)
    - Velocity computation (v_target)
    - Loss computation (v_pred vs v_target)
    - Training preparation

JiT Time Convention:
    - t=0: pure noise
    - t=1: clean image

V-Prediction:
    - Network outputs velocity v = x - noise
    - Loss: ||v_pred - v_target||^2
    - No noise scaling (standard unit variance)

Test Count: 23 (removed noise scaling tests)

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn as nn

from src.training.flow_matching import (
    PixelHDMFlowMatching,
    PixelHDMSampler,
    create_flow_matching,
    create_sampler,
)


# ============================================================================
# Time Sampling Tests (7 tests)
# ============================================================================


class TestTimeSampling:
    """Tests for timestep sampling using Logit-Normal distribution."""

    @pytest.fixture
    def flow_matching(self):
        """Create flow matching instance."""
        return PixelHDMFlowMatching(p_mean=-0.8, p_std=0.8, t_eps=0.05)

    def test_sample_timesteps_shape(self, flow_matching):
        """Test sampled timesteps have correct shape (B,)."""
        batch_size = 8
        device = torch.device("cpu")

        t = flow_matching.sample_timesteps(batch_size, device)

        assert t.shape == (batch_size,)
        assert t.dtype == torch.float32

    def test_sample_timesteps_bounds_lower(self, flow_matching):
        """Test all timesteps are greater than t_eps (> 0)."""
        batch_size = 1000
        device = torch.device("cpu")

        t = flow_matching.sample_timesteps(batch_size, device)

        # All values should be >= t_eps
        assert (t >= flow_matching.t_eps).all()
        assert (t > 0).all()

    def test_sample_timesteps_bounds_upper(self, flow_matching):
        """Test all timesteps are less than 1 - t_eps (< 1)."""
        batch_size = 1000
        device = torch.device("cpu")

        t = flow_matching.sample_timesteps(batch_size, device)

        # All values should be <= 1 - t_eps
        assert (t <= 1 - flow_matching.t_eps).all()
        assert (t < 1).all()

    def test_sample_timesteps_distribution(self, flow_matching):
        """Test timesteps follow Logit-Normal distribution (centered around 0.5)."""
        batch_size = 10000
        device = torch.device("cpu")

        t = flow_matching.sample_timesteps(batch_size, device)

        # Mean should be roughly around 0.5 (due to sigmoid)
        # With p_mean=-0.8, the distribution is slightly shifted
        mean = t.mean().item()
        std = t.std().item()

        # Mean should be in reasonable range
        assert 0.2 < mean < 0.8
        # Std should be positive
        assert std > 0.0

    def test_sample_timesteps_deterministic_with_seed(self, flow_matching):
        """Test timesteps are deterministic with fixed seed."""
        batch_size = 16
        device = torch.device("cpu")

        torch.manual_seed(42)
        t1 = flow_matching.sample_timesteps(batch_size, device)

        torch.manual_seed(42)
        t2 = flow_matching.sample_timesteps(batch_size, device)

        assert torch.allclose(t1, t2)

    def test_sample_timesteps_different_batch_sizes(self, flow_matching):
        """Test timestep sampling works for different batch sizes."""
        device = torch.device("cpu")

        for batch_size in [1, 4, 16, 64, 256]:
            t = flow_matching.sample_timesteps(batch_size, device)
            assert t.shape == (batch_size,)

    def test_sample_timesteps_device_placement(self, flow_matching, device):
        """Test timesteps are placed on correct device."""
        batch_size = 8

        t = flow_matching.sample_timesteps(batch_size, device)

        # Compare device type (cuda vs cpu), not exact device (cuda:0 vs cuda)
        assert t.device.type == device.type


# ============================================================================
# Interpolation Formula Tests (8 tests) - Core Numerical Verification
# ============================================================================


class TestInterpolation:
    """Tests for JiT interpolation formula: z_t = t * x + (1 - t) * epsilon."""

    @pytest.fixture
    def flow_matching(self):
        """Create flow matching instance."""
        return PixelHDMFlowMatching(t_eps=0.0)  # Allow t=0 and t=1 for testing

    def test_interpolate_at_t_zero(self, flow_matching):
        """Test t=0 returns pure noise (z_0 = epsilon)."""
        B, C, H, W = 2, 3, 256, 256
        x = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)
        t = torch.zeros(B)

        z_t = flow_matching.interpolate(x, noise, t)

        # PixelHDM: t=0 is noise, z_0 = 0*x + (1-0)*noise = noise
        assert torch.allclose(z_t, noise, atol=1e-6)

    def test_interpolate_at_t_one(self, flow_matching):
        """Test t=1 returns clean image (z_1 = x)."""
        B, C, H, W = 2, 3, 256, 256
        x = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)
        t = torch.ones(B)

        z_t = flow_matching.interpolate(x, noise, t)

        # PixelHDM: t=1 is clean, z_1 = 1*x + (1-1)*noise = x
        assert torch.allclose(z_t, x, atol=1e-6)

    def test_interpolate_linear(self, flow_matching):
        """Test linear interpolation at t=0.5."""
        B, C, H, W = 2, 3, 256, 256
        x = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)
        t = torch.full((B,), 0.5)

        z_t = flow_matching.interpolate(x, noise, t)

        # z_0.5 = 0.5*x + 0.5*noise = (x + noise) / 2
        expected = 0.5 * x + 0.5 * noise
        assert torch.allclose(z_t, expected, atol=1e-6)

    def test_interpolate_shape_preservation(self, flow_matching):
        """Test interpolation preserves input shape."""
        shapes = [
            (2, 3, 64, 64),
            (4, 3, 256, 256),
            (1, 3, 512, 512),
            (2, 3, 256, 512),  # Non-square
        ]

        for shape in shapes:
            x = torch.randn(shape)
            noise = torch.randn(shape)
            t = torch.rand(shape[0])

            z_t = flow_matching.interpolate(x, noise, t)

            assert z_t.shape == x.shape

    def test_interpolate_dtype_preservation(self, flow_matching):
        """Test interpolation preserves data type."""
        B, C, H, W = 2, 3, 256, 256

        for dtype in [torch.float32, torch.float16]:
            x = torch.randn(B, C, H, W, dtype=dtype)
            noise = torch.randn(B, C, H, W, dtype=dtype)
            t = torch.rand(B, dtype=dtype)

            z_t = flow_matching.interpolate(x, noise, t)

            assert z_t.dtype == dtype

    def test_interpolate_batch_expansion(self, flow_matching):
        """Test timestep is correctly expanded for batch dimension."""
        B, C, H, W = 4, 3, 256, 256
        x = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)

        # Different t for each batch element
        t = torch.tensor([0.0, 0.25, 0.75, 1.0])

        z_t = flow_matching.interpolate(x, noise, t)

        # Check each batch element individually
        assert torch.allclose(z_t[0], noise[0], atol=1e-6)  # t=0 -> noise
        assert torch.allclose(z_t[3], x[3], atol=1e-6)      # t=1 -> clean

        # Check intermediate values
        expected_1 = 0.25 * x[1] + 0.75 * noise[1]
        expected_2 = 0.75 * x[2] + 0.25 * noise[2]
        assert torch.allclose(z_t[1], expected_1, atol=1e-6)
        assert torch.allclose(z_t[2], expected_2, atol=1e-6)

    def test_interpolate_numerical_stability(self, flow_matching):
        """Test interpolation is numerically stable."""
        B, C, H, W = 2, 3, 256, 256
        x = torch.randn(B, C, H, W) * 1e6  # Large values
        noise = torch.randn(B, C, H, W) * 1e6
        t = torch.rand(B)

        z_t = flow_matching.interpolate(x, noise, t)

        # No NaN or Inf
        assert not torch.isnan(z_t).any()
        assert not torch.isinf(z_t).any()

    def test_pixelhdm_time_direction(self, flow_matching):
        """Test JiT time direction: t=0 noise, t=1 clean."""
        B, C, H, W = 2, 3, 256, 256
        x = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)

        # At t=0, z_t should be noise (NOT clean)
        z_0 = flow_matching.interpolate(x, noise, torch.zeros(B))
        assert torch.allclose(z_0, noise, atol=1e-6), \
            "PixelHDM convention: t=0 should be noise, not clean image"

        # At t=1, z_t should be clean (NOT noise)
        z_1 = flow_matching.interpolate(x, noise, torch.ones(B))
        assert torch.allclose(z_1, x, atol=1e-6), \
            "PixelHDM convention: t=1 should be clean image, not noise"


# ============================================================================
# Velocity Computation Tests (4 tests) - V-Prediction
# ============================================================================


class TestVelocityComputation:
    """Tests for velocity computation in V-Prediction."""

    @pytest.fixture
    def flow_matching(self):
        """Create flow matching instance."""
        return PixelHDMFlowMatching(t_eps=0.05)

    def test_compute_v_target_basic(self, flow_matching):
        """Test v_target = x - noise basic computation."""
        B, C, H, W = 2, 3, 64, 64
        x = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)

        v_target = flow_matching.compute_v_target(x, noise)

        expected = x - noise
        assert torch.allclose(v_target, expected.float(), atol=1e-6)

    def test_compute_v_target_equals_x_minus_noise(self, flow_matching):
        """Test v_target is exactly x - epsilon."""
        B, C, H, W = 2, 3, 64, 64
        x = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)

        v_target = flow_matching.compute_v_target(x, noise)

        # v_target should equal x - noise (in float32)
        assert torch.allclose(v_target, (x - noise).float(), atol=1e-6)

    def test_pixelhdm_velocity_target(self, flow_matching):
        """Test JiT velocity target formula."""
        B, C, H, W = 2, 3, 64, 64
        x = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)

        v_target = flow_matching.compute_v_target(x, noise)

        # In JiT, the velocity field should point from noise to clean
        # v = x - epsilon (direction: noise -> clean)
        direction = x - noise
        assert torch.allclose(v_target, direction.float(), atol=1e-6)

    def test_velocity_target_numerical_stability(self, flow_matching):
        """Test v_target is numerically stable."""
        B, C, H, W = 2, 3, 64, 64
        x = torch.randn(B, C, H, W) * 100
        noise = torch.randn(B, C, H, W) * 100

        v_target = flow_matching.compute_v_target(x, noise)

        assert not torch.isnan(v_target).any()
        assert not torch.isinf(v_target).any()


# ============================================================================
# Loss Computation Tests (8 tests) - V-Prediction
# ============================================================================


class TestLossComputation:
    """Tests for v-loss computation."""

    @pytest.fixture
    def flow_matching(self):
        """Create flow matching instance."""
        return PixelHDMFlowMatching(t_eps=0.05)

    def test_compute_loss_basic(self, flow_matching):
        """Test basic loss computation."""
        B, C, H, W = 2, 3, 64, 64
        x = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)

        # Perfect prediction: v_pred = v_target = x - noise
        v_pred = x - noise
        loss = flow_matching.compute_loss(v_pred, x, noise)

        # Loss should be near zero for perfect prediction
        assert loss < 1e-5

    def test_compute_loss_reduction_mean(self, flow_matching):
        """Test loss with mean reduction."""
        B, C, H, W = 2, 3, 64, 64
        v_pred = torch.randn(B, C, H, W)
        x = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)

        loss = flow_matching.compute_loss(v_pred, x, noise, reduction="mean")

        # Mean reduction should produce scalar
        assert loss.dim() == 0

    def test_compute_loss_reduction_sum(self, flow_matching):
        """Test loss with sum reduction."""
        B, C, H, W = 2, 3, 64, 64
        v_pred = torch.randn(B, C, H, W)
        x = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)

        loss = flow_matching.compute_loss(v_pred, x, noise, reduction="sum")

        # Sum reduction should produce scalar
        assert loss.dim() == 0

    def test_compute_loss_dtype_conversion(self, flow_matching):
        """Test loss preserves original dtype."""
        B, C, H, W = 2, 3, 64, 64

        for dtype in [torch.float32, torch.float16]:
            v_pred = torch.randn(B, C, H, W, dtype=dtype)
            x = torch.randn(B, C, H, W, dtype=dtype)
            noise = torch.randn(B, C, H, W, dtype=dtype)

            loss = flow_matching.compute_loss(v_pred, x, noise)

            assert loss.dtype == dtype

    def test_compute_loss_gradient_flow(self):
        """Test gradients flow through loss computation."""
        flow_matching = PixelHDMFlowMatching(t_eps=0.05)

        B, C, H, W = 2, 3, 64, 64
        v_pred = torch.randn(B, C, H, W, requires_grad=True)
        x = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)

        loss = flow_matching.compute_loss(v_pred, x, noise)
        loss.backward()

        # v_pred should have gradients
        assert v_pred.grad is not None
        assert not torch.isnan(v_pred.grad).any()

    def test_compute_loss_zero_prediction(self, flow_matching):
        """Test loss with zero prediction."""
        B, C, H, W = 2, 3, 64, 64
        v_pred = torch.zeros(B, C, H, W)
        x = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)

        loss = flow_matching.compute_loss(v_pred, x, noise)

        # Loss should be positive (prediction is wrong)
        assert loss > 0
        assert not torch.isnan(loss)

    def test_loss_perfect_prediction(self, flow_matching):
        """Test loss is approximately 0 for perfect prediction."""
        B, C, H, W = 2, 3, 64, 64
        x = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)

        # Perfect prediction: v_pred = v_target = x - noise
        v_pred = x - noise
        loss = flow_matching.compute_loss(v_pred, x, noise)

        # Loss should be approximately 0
        assert loss < 1e-5

    def test_loss_random_prediction(self, flow_matching):
        """Test loss is positive for random prediction."""
        B, C, H, W = 2, 3, 64, 64
        v_pred = torch.randn(B, C, H, W)  # Random prediction
        x = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)

        loss = flow_matching.compute_loss(v_pred, x, noise)

        # Loss should be positive for random prediction
        assert loss > 0


# ============================================================================
# Training Preparation Tests (5 tests)
# ============================================================================


class TestTrainingPreparation:
    """Tests for prepare_training method."""

    @pytest.fixture
    def flow_matching(self):
        """Create flow matching instance."""
        return PixelHDMFlowMatching(t_eps=0.05)

    def test_prepare_training_generates_noise(self, flow_matching):
        """Test prepare_training generates noise when not provided."""
        B, C, H, W = 2, 3, 256, 256
        x = torch.randn(B, C, H, W)

        t, z_t, x_out, noise = flow_matching.prepare_training(x)

        assert noise is not None
        assert noise.shape == x.shape

    def test_prepare_training_samples_time(self, flow_matching):
        """Test prepare_training samples timesteps."""
        B, C, H, W = 2, 3, 256, 256
        x = torch.randn(B, C, H, W)

        t, z_t, x_out, noise = flow_matching.prepare_training(x)

        assert t.shape == (B,)
        assert (t >= flow_matching.t_eps).all()
        assert (t <= 1 - flow_matching.t_eps).all()

    def test_prepare_training_computes_z_t(self, flow_matching):
        """Test prepare_training computes correct z_t."""
        B, C, H, W = 2, 3, 256, 256
        x = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)
        t_fixed = torch.full((B,), 0.5)

        t, z_t, x_out, noise_out = flow_matching.prepare_training(
            x, noise=noise, t=t_fixed
        )

        # Verify interpolation
        expected_z_t = flow_matching.interpolate(x, noise, t_fixed)
        assert torch.allclose(z_t, expected_z_t, atol=1e-6)

    def test_prepare_training_output_shapes(self, flow_matching):
        """Test prepare_training output shapes."""
        B, C, H, W = 4, 3, 512, 512
        x = torch.randn(B, C, H, W)

        t, z_t, x_out, noise = flow_matching.prepare_training(x)

        assert t.shape == (B,)
        assert z_t.shape == (B, C, H, W)
        assert x_out.shape == (B, C, H, W)
        assert noise.shape == (B, C, H, W)

    def test_prepare_training_uses_provided_inputs(self, flow_matching):
        """Test prepare_training uses provided noise and t."""
        B, C, H, W = 2, 3, 256, 256
        x = torch.randn(B, C, H, W)
        noise_in = torch.randn(B, C, H, W)
        t_in = torch.tensor([0.3, 0.7])

        t, z_t, x_out, noise_out = flow_matching.prepare_training(
            x, noise=noise_in, t=t_in
        )

        # Should use provided inputs
        assert torch.allclose(t, t_in)
        assert torch.allclose(noise_out, noise_in)


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_flow_matching_default(self):
        """Test create_flow_matching with defaults (SD3/PixelHDM convention)."""
        fm = create_flow_matching()

        assert isinstance(fm, PixelHDMFlowMatching)
        assert fm.p_mean == 0.0  # SD3/PixelHDM default
        assert fm.p_std == 1.0   # SD3/PixelHDM default
        assert fm.t_eps == 0.05

    def test_create_flow_matching_custom(self):
        """Test create_flow_matching with custom params."""
        fm = create_flow_matching(p_mean=-1.0, p_std=1.0, t_eps=0.1)

        assert fm.p_mean == -1.0
        assert fm.p_std == 1.0
        assert fm.t_eps == 0.1

    def test_create_sampler_default(self):
        """Test create_sampler with defaults."""
        sampler = create_sampler()

        assert isinstance(sampler, PixelHDMSampler)
        assert sampler.num_steps == 50
        assert sampler.method == "heun"
        assert sampler.t_eps == 0.05

    def test_create_sampler_euler(self):
        """Test create_sampler with euler method."""
        sampler = create_sampler(method="euler", num_steps=100)

        assert sampler.method == "euler"
        assert sampler.num_steps == 100


# ============================================================================
# PixelHDMSampler Tests
# ============================================================================


class TestPixelHDMSampler:
    """Tests for PixelHDMSampler."""

    @pytest.fixture
    def sampler(self):
        """Create sampler instance."""
        return PixelHDMSampler(num_steps=10, method="euler", t_eps=0.05)

    def test_get_timesteps_shape(self, sampler):
        """Test timesteps have correct shape."""
        timesteps = sampler.get_timesteps()

        # num_steps + 1 timesteps (including endpoints)
        assert timesteps.shape == (11,)

    def test_get_timesteps_range(self, sampler):
        """Test timesteps are in correct range."""
        timesteps = sampler.get_timesteps()

        assert timesteps[0] >= sampler.t_eps
        assert timesteps[-1] <= 1 - sampler.t_eps

    def test_get_timesteps_monotonic(self, sampler):
        """Test timesteps are monotonically increasing."""
        timesteps = sampler.get_timesteps()

        # PixelHDM: should go from small to large (noise -> clean)
        for i in range(len(timesteps) - 1):
            assert timesteps[i] < timesteps[i + 1]

    def test_v_to_x_computation(self, sampler):
        """Test velocity to x_pred conversion."""
        B, C, H, W = 2, 3, 64, 64
        v_pred = torch.randn(B, C, H, W)
        z = torch.randn(B, C, H, W)
        t = torch.full((B,), 0.5)

        x_pred = sampler.v_to_x(v_pred, z, t)

        # x = z + (1-t) * v
        expected = z + (1 - 0.5) * v_pred
        assert torch.allclose(x_pred, expected, atol=1e-5)
