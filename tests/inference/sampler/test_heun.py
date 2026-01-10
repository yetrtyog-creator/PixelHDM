"""
HeunSampler Unit Tests.

Validates the Heun two-step predictor-corrector ODE solver:
- Step formula correctness (average velocity method)
- NFE count (2 per step, except last)
- Comparison with Euler
- Numerical stability

Test Count: 20 test cases

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-08
"""

import pytest
import torch
import torch.nn as nn
from typing import Optional
from unittest.mock import Mock

from src.inference.sampler.heun import HeunSampler
from src.inference.sampler.euler import EulerSampler


# =============================================================================
# Test Fixtures
# =============================================================================


class MockModel(nn.Module):
    """Mock model for Heun testing."""

    def __init__(self, velocity_mode: str = "constant"):
        super().__init__()
        self.velocity_mode = velocity_mode
        self.call_count = 0
        self.call_history = []

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
        pooled_text_embed: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        self.call_count += 1
        self.call_history.append({
            'z_mean': z.mean().item(),
            't': t.mean().item(),
        })

        if self.velocity_mode == "constant":
            return torch.ones_like(z) * 0.5
        elif self.velocity_mode == "zero":
            return torch.zeros_like(z)
        elif self.velocity_mode == "time_dependent":
            # Velocity depends on t: v = t * 0.5
            return torch.ones_like(z) * t.view(-1, 1, 1, 1) * 0.5
        elif self.velocity_mode == "state_dependent":
            # Velocity depends on z: v = -0.1 * z
            return -0.1 * z
        else:
            return z * 0.1

    def reset(self):
        self.call_count = 0
        self.call_history = []


@pytest.fixture
def heun_sampler():
    """Create Heun sampler with default settings."""
    return HeunSampler(num_steps=50, t_eps=0.05)


@pytest.fixture
def euler_sampler():
    """Create Euler sampler for comparison."""
    return EulerSampler(num_steps=50, t_eps=0.05)


@pytest.fixture
def mock_model():
    """Create mock model."""
    return MockModel(velocity_mode="constant")


@pytest.fixture
def sample_z():
    """Create sample latent tensor."""
    torch.manual_seed(42)
    return torch.randn(2, 32, 32, 3)


@pytest.fixture
def sample_text_embed():
    """Create sample text embeddings."""
    torch.manual_seed(42)
    return torch.randn(2, 77, 1024)


# =============================================================================
# Test Class: Heun Step Formula
# =============================================================================


class TestHeunStepFormula:
    """Test Heun step formula correctness."""

    def test_heun_step_basic(self, heun_sampler, sample_z):
        """Test basic Heun step with constant velocity."""
        model = MockModel(velocity_mode="constant")
        t = torch.tensor([0.3])
        t_next = torch.tensor([0.5])

        z_next = heun_sampler.step(
            model=model,
            z=sample_z,
            t=t,
            t_next=t_next,
            text_embeddings=None,
            guidance_scale=1.0,
        )

        # For constant velocity, Heun should match Euler
        # v = 0.5, dt = 0.2
        # z_next = z + 0.2 * 0.5 = z + 0.1
        expected = sample_z + 0.2 * 0.5

        assert z_next.shape == sample_z.shape
        assert torch.allclose(z_next, expected, atol=1e-5)

    def test_heun_two_evaluations(self, heun_sampler, sample_z):
        """Test that Heun makes two model evaluations per step."""
        model = MockModel(velocity_mode="constant")

        heun_sampler.step(
            model=model,
            z=sample_z,
            t=torch.tensor([0.3]),
            t_next=torch.tensor([0.5]),
            text_embeddings=None,
            guidance_scale=1.0,
        )

        # Heun should make 2 calls: initial prediction + correction
        assert model.call_count == 2

    def test_heun_evaluation_points(self, heun_sampler, sample_z):
        """Test that Heun evaluates at correct time points."""
        model = MockModel(velocity_mode="time_dependent")
        t = torch.tensor([0.3])
        t_next = torch.tensor([0.5])

        heun_sampler.step(
            model=model,
            z=sample_z,
            t=t,
            t_next=t_next,
            text_embeddings=None,
            guidance_scale=1.0,
        )

        # First call should be at t=0.3, second at t=0.5
        assert len(model.call_history) == 2
        assert abs(model.call_history[0]['t'] - 0.3) < 0.01
        assert abs(model.call_history[1]['t'] - 0.5) < 0.01

    def test_heun_average_velocity(self, heun_sampler, sample_z):
        """Test Heun uses average velocity: v_avg = (v_t + v_{t+dt}) / 2."""
        # For time-dependent velocity v = t * 0.5:
        # v_t = 0.3 * 0.5 = 0.15
        # z_euler = z + dt * v_t = z + 0.2 * 0.15 = z + 0.03
        # v_next = 0.5 * 0.5 = 0.25
        # v_avg = (0.15 + 0.25) / 2 = 0.2
        # z_heun = z + dt * v_avg = z + 0.2 * 0.2 = z + 0.04

        model = MockModel(velocity_mode="time_dependent")
        t = torch.tensor([0.3])
        t_next = torch.tensor([0.5])

        z_next = heun_sampler.step(
            model=model,
            z=sample_z,
            t=t,
            t_next=t_next,
            text_embeddings=None,
            guidance_scale=1.0,
        )

        expected = sample_z + 0.2 * 0.2
        assert torch.allclose(z_next, expected, atol=1e-4)

    def test_heun_shape_preservation(self, heun_sampler):
        """Test that Heun preserves input shape."""
        model = MockModel(velocity_mode="constant")

        shapes = [
            (1, 16, 16, 3),
            (2, 32, 32, 3),
            (4, 64, 64, 3),
        ]

        for shape in shapes:
            z = torch.randn(shape)
            z_next = heun_sampler.step(
                model=model,
                z=z,
                t=torch.tensor([0.3]),
                t_next=torch.tensor([0.5]),
                text_embeddings=None,
                guidance_scale=1.0,
            )
            assert z_next.shape == shape


# =============================================================================
# Test Class: Heun vs Euler Comparison
# =============================================================================


class TestHeunVsEuler:
    """Compare Heun and Euler behavior."""

    def test_heun_equals_euler_constant_velocity(
        self, heun_sampler, euler_sampler, sample_z
    ):
        """Heun equals Euler for constant velocity."""
        model_heun = MockModel(velocity_mode="constant")
        model_euler = MockModel(velocity_mode="constant")

        z_heun = heun_sampler.step(
            model=model_heun,
            z=sample_z,
            t=torch.tensor([0.3]),
            t_next=torch.tensor([0.5]),
            text_embeddings=None,
            guidance_scale=1.0,
        )

        z_euler = euler_sampler.step(
            model=model_euler,
            z=sample_z,
            t=torch.tensor([0.3]),
            t_next=torch.tensor([0.5]),
            text_embeddings=None,
            guidance_scale=1.0,
        )

        assert torch.allclose(z_heun, z_euler, atol=1e-5)

    def test_heun_more_accurate_time_dependent(
        self, heun_sampler, euler_sampler, sample_z
    ):
        """Heun is more accurate for time-dependent velocity."""
        # For v = t * 0.5, the exact solution over [0.3, 0.5] is:
        # integral of t*0.5 from 0.3 to 0.5 = 0.5 * (0.5^2 - 0.3^2) / 2 = 0.04

        model_heun = MockModel(velocity_mode="time_dependent")
        model_euler = MockModel(velocity_mode="time_dependent")

        z_heun = heun_sampler.step(
            model=model_heun,
            z=sample_z,
            t=torch.tensor([0.3]),
            t_next=torch.tensor([0.5]),
            text_embeddings=None,
            guidance_scale=1.0,
        )

        z_euler = euler_sampler.step(
            model=model_euler,
            z=sample_z,
            t=torch.tensor([0.3]),
            t_next=torch.tensor([0.5]),
            text_embeddings=None,
            guidance_scale=1.0,
        )

        # Heun: z + 0.04, Euler: z + 0.03
        # Heun should be closer to exact (z + 0.04)
        exact = sample_z + 0.04

        heun_error = (z_heun - exact).abs().mean()
        euler_error = (z_euler - exact).abs().mean()

        assert heun_error < euler_error


# =============================================================================
# Test Class: CFG with Heun
# =============================================================================


class TestHeunCFG:
    """Test CFG application in Heun sampler."""

    def test_heun_cfg_call_count(self, heun_sampler, sample_z, sample_text_embed):
        """Test CFG doubles the number of model calls."""
        model = MockModel(velocity_mode="constant")
        null_embed = torch.randn_like(sample_text_embed)

        heun_sampler.step(
            model=model,
            z=sample_z,
            t=torch.tensor([0.3]),
            t_next=torch.tensor([0.5]),
            text_embeddings=sample_text_embed,
            guidance_scale=7.5,
            null_text_embeddings=null_embed,
        )

        # Heun with CFG: 2 (uncond + cond) * 2 (prediction + correction) = 4
        assert model.call_count == 4

    def test_heun_no_cfg_call_count(self, heun_sampler, sample_z, sample_text_embed):
        """Test no CFG uses 2 calls."""
        model = MockModel(velocity_mode="constant")

        heun_sampler.step(
            model=model,
            z=sample_z,
            t=torch.tensor([0.3]),
            t_next=torch.tensor([0.5]),
            text_embeddings=sample_text_embed,
            guidance_scale=1.0,
        )

        assert model.call_count == 2

    def test_heun_cfg_valid_output(self, heun_sampler, sample_z, sample_text_embed):
        """Test CFG produces valid output."""
        model = MockModel(velocity_mode="constant")
        null_embed = torch.randn_like(sample_text_embed)

        z_next = heun_sampler.step(
            model=model,
            z=sample_z,
            t=torch.tensor([0.3]),
            t_next=torch.tensor([0.5]),
            text_embeddings=sample_text_embed,
            guidance_scale=7.5,
            null_text_embeddings=null_embed,
        )

        assert not torch.isnan(z_next).any()
        assert not torch.isinf(z_next).any()
        assert z_next.shape == sample_z.shape


# =============================================================================
# Test Class: Numerical Stability
# =============================================================================


class TestHeunNumericalStability:
    """Test numerical stability of Heun sampler."""

    def test_stability_near_boundaries(self, heun_sampler, sample_z):
        """Test stability near t=0 and t=1."""
        model = MockModel(velocity_mode="constant")

        # Near t=0
        z_next = heun_sampler.step(
            model=model,
            z=sample_z,
            t=torch.tensor([0.01]),
            t_next=torch.tensor([0.1]),
            text_embeddings=None,
            guidance_scale=1.0,
        )
        assert not torch.isnan(z_next).any()

        # Near t=1
        model.reset()
        z_next = heun_sampler.step(
            model=model,
            z=sample_z,
            t=torch.tensor([0.9]),
            t_next=torch.tensor([0.99]),
            text_embeddings=None,
            guidance_scale=1.0,
        )
        assert not torch.isnan(z_next).any()

    def test_stability_small_dt(self, heun_sampler, sample_z):
        """Test stability with very small dt."""
        model = MockModel(velocity_mode="constant")

        z_next = heun_sampler.step(
            model=model,
            z=sample_z,
            t=torch.tensor([0.5]),
            t_next=torch.tensor([0.5001]),
            text_embeddings=None,
            guidance_scale=1.0,
        )

        assert not torch.isnan(z_next).any()
        assert not torch.isinf(z_next).any()

    def test_stability_state_dependent(self, heun_sampler):
        """Test stability with state-dependent velocity."""
        model = MockModel(velocity_mode="state_dependent")
        z = torch.randn(2, 32, 32, 3) * 10  # Large values

        z_next = heun_sampler.step(
            model=model,
            z=z,
            t=torch.tensor([0.3]),
            t_next=torch.tensor([0.5]),
            text_embeddings=None,
            guidance_scale=1.0,
        )

        assert not torch.isnan(z_next).any()
        assert not torch.isinf(z_next).any()


# =============================================================================
# Test Class: Edge Cases
# =============================================================================


class TestHeunEdgeCases:
    """Test edge cases for Heun sampler."""

    def test_zero_dt(self, heun_sampler, sample_z):
        """Test with dt=0."""
        model = MockModel(velocity_mode="constant")

        z_next = heun_sampler.step(
            model=model,
            z=sample_z,
            t=torch.tensor([0.5]),
            t_next=torch.tensor([0.5]),
            text_embeddings=None,
            guidance_scale=1.0,
        )

        assert torch.allclose(z_next, sample_z, atol=1e-6)

    def test_batch_size_one(self, heun_sampler):
        """Test with single sample."""
        model = MockModel(velocity_mode="constant")
        z = torch.randn(1, 32, 32, 3)

        z_next = heun_sampler.step(
            model=model,
            z=z,
            t=torch.tensor([0.3]),
            t_next=torch.tensor([0.5]),
            text_embeddings=None,
            guidance_scale=1.0,
        )

        assert z_next.shape == (1, 32, 32, 3)

    def test_non_square(self, heun_sampler):
        """Test with non-square resolution."""
        model = MockModel(velocity_mode="constant")
        z = torch.randn(2, 64, 32, 3)

        z_next = heun_sampler.step(
            model=model,
            z=z,
            t=torch.tensor([0.3]),
            t_next=torch.tensor([0.5]),
            text_embeddings=None,
            guidance_scale=1.0,
        )

        assert z_next.shape == (2, 64, 32, 3)

    def test_dtype_float32(self, heun_sampler):
        """Test with float32 dtype (mock model always returns float32)."""
        model = MockModel(velocity_mode="constant")
        z = torch.randn(2, 32, 32, 3, dtype=torch.float32)

        z_next = heun_sampler.step(
            model=model,
            z=z,
            t=torch.tensor([0.3]),
            t_next=torch.tensor([0.5]),
            text_embeddings=None,
            guidance_scale=1.0,
        )

        # Mock returns float32 regardless of input
        assert z_next.dtype == torch.float32
