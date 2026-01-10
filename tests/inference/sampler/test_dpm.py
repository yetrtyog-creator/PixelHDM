"""
DPMPPSampler Unit Tests.

Validates the DPM++ multi-step ODE solver:
- First-order and second-order steps
- Lambda calculation
- V-to-X conversion
- History-based correction
- State management (reset)

Test Count: 20 test cases

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-08
"""

import pytest
import torch
import torch.nn as nn
from typing import Optional
import math

from src.inference.sampler.dpm import DPMPPSampler
from src.inference.sampler.timesteps import get_lambda


# =============================================================================
# Test Fixtures
# =============================================================================


class MockModel(nn.Module):
    """Mock model for DPM++ testing."""

    def __init__(self, velocity_mode: str = "constant"):
        super().__init__()
        self.velocity_mode = velocity_mode
        self.call_count = 0

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
        pooled_text_embed: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        self.call_count += 1

        if self.velocity_mode == "constant":
            return torch.ones_like(z) * 0.5
        elif self.velocity_mode == "zero":
            return torch.zeros_like(z)
        elif self.velocity_mode == "time_dependent":
            return torch.ones_like(z) * t.view(-1, 1, 1, 1)
        elif self.velocity_mode == "state_dependent":
            return -0.1 * z
        else:
            return z * 0.1

    def reset(self):
        self.call_count = 0


@pytest.fixture
def dpm_sampler():
    """Create DPM++ sampler."""
    return DPMPPSampler(num_steps=50, t_eps=0.05)


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
# Test Class: Lambda Calculation
# =============================================================================


class TestLambdaCalculation:
    """Test lambda calculation for DPM++."""

    def test_lambda_formula(self):
        """Test lambda = log(t / (1-t))."""
        t = torch.tensor([0.5])
        lambda_val = get_lambda(t, t_eps=0.05)

        # lambda(0.5) = log(0.5 / 0.5) = log(1) = 0
        expected = torch.tensor([0.0])
        assert torch.allclose(lambda_val, expected, atol=1e-5)

    def test_lambda_increases_with_t(self):
        """Test that lambda increases as t increases."""
        t_values = torch.tensor([0.2, 0.4, 0.6, 0.8])
        lambda_values = get_lambda(t_values, t_eps=0.05)

        for i in range(len(lambda_values) - 1):
            assert lambda_values[i + 1] > lambda_values[i]

    def test_lambda_symmetry(self):
        """Test lambda(t) = -lambda(1-t)."""
        t1 = torch.tensor([0.3])
        t2 = torch.tensor([0.7])

        lambda_1 = get_lambda(t1, t_eps=0.05)
        lambda_2 = get_lambda(t2, t_eps=0.05)

        assert torch.allclose(lambda_1, -lambda_2, atol=1e-4)

    def test_lambda_near_boundaries(self):
        """Test lambda near t=0 and t=1 with clamping."""
        t_near_0 = torch.tensor([0.01])
        t_near_1 = torch.tensor([0.99])

        lambda_0 = get_lambda(t_near_0, t_eps=0.05)
        lambda_1 = get_lambda(t_near_1, t_eps=0.05)

        assert not torch.isnan(lambda_0).any()
        assert not torch.isnan(lambda_1).any()
        assert not torch.isinf(lambda_0).any()
        assert not torch.isinf(lambda_1).any()


# =============================================================================
# Test Class: V-to-X Conversion
# =============================================================================


class TestVtoXConversion:
    """Test V-prediction to X-prediction conversion."""

    def test_v_to_x_formula(self, dpm_sampler, sample_z):
        """Test x = z + (1-t) * v."""
        t = torch.tensor([0.3])
        v = torch.ones_like(sample_z) * 0.5

        # _v_to_x signature: (v_pred, z, t)
        # x = z + (1 - 0.3) * 0.5 = z + 0.35
        x = dpm_sampler._v_to_x(v, sample_z, t)
        expected = sample_z + 0.7 * 0.5

        assert torch.allclose(x, expected, atol=1e-5)

    def test_v_to_x_at_t_zero(self, dpm_sampler, sample_z):
        """Test v_to_x at t=0 (noise state)."""
        t = torch.tensor([0.0])
        v = torch.ones_like(sample_z) * 0.5

        # _v_to_x signature: (v_pred, z, t)
        # x = z + (1 - 0) * 0.5 = z + 0.5
        x = dpm_sampler._v_to_x(v, sample_z, t)
        expected = sample_z + 1.0 * 0.5

        assert torch.allclose(x, expected, atol=1e-5)

    def test_v_to_x_at_t_one(self, dpm_sampler, sample_z):
        """Test v_to_x at t=1 (clean state)."""
        t = torch.tensor([1.0])
        v = torch.ones_like(sample_z) * 0.5

        # _v_to_x signature: (v_pred, z, t)
        # x = z + (1 - 1) * 0.5 = z
        x = dpm_sampler._v_to_x(v, sample_z, t)

        assert torch.allclose(x, sample_z, atol=1e-5)


# =============================================================================
# Test Class: First-Order Step
# =============================================================================


class TestDPMFirstOrder:
    """Test DPM++ first-order step."""

    def test_first_step_is_first_order(self, dpm_sampler, sample_z):
        """Test that first step uses first-order method."""
        model = MockModel(velocity_mode="constant")
        dpm_sampler.reset()

        z_next = dpm_sampler.step(
            model=model,
            z=sample_z,
            t=torch.tensor([0.1]),
            t_next=torch.tensor([0.3]),
            text_embeddings=None,
            guidance_scale=1.0,
        )

        # First step should work without history
        assert z_next.shape == sample_z.shape
        assert not torch.isnan(z_next).any()

    def test_first_order_single_evaluation(self, dpm_sampler, sample_z):
        """Test first-order step uses single model evaluation."""
        model = MockModel(velocity_mode="constant")
        dpm_sampler.reset()

        dpm_sampler.step(
            model=model,
            z=sample_z,
            t=torch.tensor([0.1]),
            t_next=torch.tensor([0.3]),
            text_embeddings=None,
            guidance_scale=1.0,
        )

        assert model.call_count == 1


# =============================================================================
# Test Class: Second-Order Step
# =============================================================================


class TestDPMSecondOrder:
    """Test DPM++ second-order correction."""

    def test_second_step_uses_history(self, dpm_sampler, sample_z):
        """Test that second step uses history for correction."""
        model = MockModel(velocity_mode="constant")
        dpm_sampler.reset()

        # First step
        z1 = dpm_sampler.step(
            model=model,
            z=sample_z,
            t=torch.tensor([0.1]),
            t_next=torch.tensor([0.3]),
            text_embeddings=None,
            guidance_scale=1.0,
        )

        # Second step (should use history)
        z2 = dpm_sampler.step(
            model=model,
            z=z1,
            t=torch.tensor([0.3]),
            t_next=torch.tensor([0.5]),
            text_embeddings=None,
            guidance_scale=1.0,
        )

        assert z2.shape == sample_z.shape
        assert not torch.isnan(z2).any()

    def test_second_order_correction_affects_output(self, dpm_sampler, sample_z):
        """Test that second-order correction changes output."""
        model = MockModel(velocity_mode="time_dependent")

        # Run with reset between steps (first-order only)
        dpm_sampler.reset()
        z1_first = dpm_sampler.step(
            model=model,
            z=sample_z,
            t=torch.tensor([0.1]),
            t_next=torch.tensor([0.3]),
            text_embeddings=None,
            guidance_scale=1.0,
        )

        dpm_sampler.reset()  # Reset to force first-order
        z2_first_only = dpm_sampler.step(
            model=model,
            z=z1_first,
            t=torch.tensor([0.3]),
            t_next=torch.tensor([0.5]),
            text_embeddings=None,
            guidance_scale=1.0,
        )

        # Run without reset (second-order)
        dpm_sampler.reset()
        z1_second = dpm_sampler.step(
            model=model,
            z=sample_z,
            t=torch.tensor([0.1]),
            t_next=torch.tensor([0.3]),
            text_embeddings=None,
            guidance_scale=1.0,
        )

        z2_second_order = dpm_sampler.step(
            model=model,
            z=z1_second,
            t=torch.tensor([0.3]),
            t_next=torch.tensor([0.5]),
            text_embeddings=None,
            guidance_scale=1.0,
        )

        # Results should differ due to second-order correction
        # (only for time-dependent velocity)
        diff = (z2_first_only - z2_second_order).abs().max()
        assert diff > 1e-6  # Should be different


# =============================================================================
# Test Class: State Management
# =============================================================================


class TestDPMStateManagement:
    """Test DPM++ state management."""

    def test_reset_clears_history(self, dpm_sampler, sample_z):
        """Test that reset clears previous prediction history."""
        model = MockModel(velocity_mode="constant")

        # Run two steps
        dpm_sampler.reset()
        z1 = dpm_sampler.step(
            model=model,
            z=sample_z,
            t=torch.tensor([0.1]),
            t_next=torch.tensor([0.3]),
            text_embeddings=None,
            guidance_scale=1.0,
        )

        dpm_sampler.step(
            model=model,
            z=z1,
            t=torch.tensor([0.3]),
            t_next=torch.tensor([0.5]),
            text_embeddings=None,
            guidance_scale=1.0,
        )

        # Reset
        dpm_sampler.reset()

        # Internal state should be cleared
        assert dpm_sampler._prev_x_pred is None
        assert dpm_sampler._prev_t is None

    def test_reset_before_new_sample(self, dpm_sampler, sample_z):
        """Test that reset is required for new samples."""
        model = MockModel(velocity_mode="constant")

        # First sample
        dpm_sampler.reset()
        z1 = dpm_sampler.step(
            model=model,
            z=sample_z,
            t=torch.tensor([0.1]),
            t_next=torch.tensor([0.3]),
            text_embeddings=None,
            guidance_scale=1.0,
        )

        # New sample without reset might use stale history
        # This test ensures behavior is defined
        new_z = torch.randn_like(sample_z)

        dpm_sampler.reset()  # Proper usage: reset before new sample
        z_new = dpm_sampler.step(
            model=model,
            z=new_z,
            t=torch.tensor([0.1]),
            t_next=torch.tensor([0.3]),
            text_embeddings=None,
            guidance_scale=1.0,
        )

        assert z_new.shape == new_z.shape


# =============================================================================
# Test Class: CFG with DPM++
# =============================================================================


class TestDPMCFG:
    """Test CFG application in DPM++ sampler."""

    def test_cfg_doubles_calls(self, dpm_sampler, sample_z, sample_text_embed):
        """Test CFG doubles model evaluations."""
        model = MockModel(velocity_mode="constant")
        null_embed = torch.randn_like(sample_text_embed)

        dpm_sampler.reset()
        dpm_sampler.step(
            model=model,
            z=sample_z,
            t=torch.tensor([0.1]),
            t_next=torch.tensor([0.3]),
            text_embeddings=sample_text_embed,
            guidance_scale=7.5,
            null_text_embeddings=null_embed,
        )

        # First-order + CFG = 2 calls
        assert model.call_count == 2

    def test_cfg_valid_output(self, dpm_sampler, sample_z, sample_text_embed):
        """Test CFG produces valid output."""
        model = MockModel(velocity_mode="constant")
        null_embed = torch.randn_like(sample_text_embed)

        dpm_sampler.reset()
        z_next = dpm_sampler.step(
            model=model,
            z=sample_z,
            t=torch.tensor([0.1]),
            t_next=torch.tensor([0.3]),
            text_embeddings=sample_text_embed,
            guidance_scale=7.5,
            null_text_embeddings=null_embed,
        )

        assert not torch.isnan(z_next).any()
        assert not torch.isinf(z_next).any()


# =============================================================================
# Test Class: Numerical Stability
# =============================================================================


class TestDPMNumericalStability:
    """Test numerical stability of DPM++ sampler."""

    def test_stability_across_full_range(self, dpm_sampler, sample_z):
        """Test stability across full time range."""
        model = MockModel(velocity_mode="constant")
        dpm_sampler.reset()

        timesteps = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
        z = sample_z.clone()

        for i in range(len(timesteps) - 1):
            z = dpm_sampler.step(
                model=model,
                z=z,
                t=torch.tensor([timesteps[i]]),
                t_next=torch.tensor([timesteps[i + 1]]),
                text_embeddings=None,
                guidance_scale=1.0,
            )

            assert not torch.isnan(z).any(), f"NaN at step {i}"
            assert not torch.isinf(z).any(), f"Inf at step {i}"

    def test_stability_small_steps(self, dpm_sampler, sample_z):
        """Test stability with small time steps."""
        model = MockModel(velocity_mode="constant")
        dpm_sampler.reset()

        z = sample_z.clone()
        t = 0.3

        for _ in range(10):
            z = dpm_sampler.step(
                model=model,
                z=z,
                t=torch.tensor([t]),
                t_next=torch.tensor([t + 0.01]),
                text_embeddings=None,
                guidance_scale=1.0,
            )
            t += 0.01

        assert not torch.isnan(z).any()
        assert not torch.isinf(z).any()


# =============================================================================
# Test Class: Edge Cases
# =============================================================================


class TestDPMEdgeCases:
    """Test edge cases for DPM++ sampler."""

    def test_single_batch(self, dpm_sampler):
        """Test with batch size 1."""
        model = MockModel(velocity_mode="constant")
        z = torch.randn(1, 32, 32, 3)

        dpm_sampler.reset()
        z_next = dpm_sampler.step(
            model=model,
            z=z,
            t=torch.tensor([0.3]),
            t_next=torch.tensor([0.5]),
            text_embeddings=None,
            guidance_scale=1.0,
        )

        assert z_next.shape == (1, 32, 32, 3)

    def test_dtype_preservation(self, dpm_sampler):
        """Test dtype preservation (float32 only - mock returns float32)."""
        model = MockModel(velocity_mode="constant")

        # Note: Mock model always returns float32
        z = torch.randn(2, 32, 32, 3, dtype=torch.float32)

        dpm_sampler.reset()
        z_next = dpm_sampler.step(
            model=model,
            z=z,
            t=torch.tensor([0.3]),
            t_next=torch.tensor([0.5]),
            text_embeddings=None,
            guidance_scale=1.0,
        )

        assert z_next.dtype == torch.float32
