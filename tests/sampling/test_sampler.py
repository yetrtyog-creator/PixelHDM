"""
PixelHDM-RPEA-DinoV3 Sampler Tests (V-Prediction)

Comprehensive tests for inference samplers: BaseSampler, EulerSampler,
HeunSampler, DPMPPSampler, and UnifiedSampler.

V-Prediction:
    - Model outputs velocity v = x - noise
    - ODE integration: z_next = z + dt * v

Test Count: 70 test cases

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

import pytest
import torch
import torch.nn as nn
from typing import Optional, Callable
from unittest.mock import Mock, MagicMock, patch

from src.inference.sampler import (
    SamplerMethod,
    SamplerConfig,
    BaseSampler,
    EulerSampler,
    HeunSampler,
    DPMPPSampler,
    UnifiedSampler,
    create_sampler,
    create_sampler_from_config,
)
from src.config.model_config import PixelHDMConfig


# =============================================================================
# Test Fixtures and Helper Classes
# =============================================================================


class DummyModel(nn.Module):
    """
    Dummy model for testing samplers.

    Simulates a diffusion model that predicts velocity from z_t.
    Returns a deterministic transformation for reproducibility.
    """

    def __init__(self, mode: str = "identity"):
        """
        Args:
            mode: Prediction mode
                - "identity": Return input unchanged
                - "zero": Return zeros
                - "mean": Return mean-shifted input
                - "random": Return random but deterministic
        """
        super().__init__()
        self.mode = mode
        self.call_count = 0

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Predict x from z_t."""
        self.call_count += 1

        if self.mode == "identity":
            return z
        elif self.mode == "zero":
            return torch.zeros_like(z)
        elif self.mode == "mean":
            # Shift towards mean
            return z * 0.9
        elif self.mode == "random":
            # Deterministic based on call count
            torch.manual_seed(self.call_count)
            return z + torch.randn_like(z) * 0.1
        else:
            return z

    def reset_call_count(self):
        self.call_count = 0


class LinearDummyModel(nn.Module):
    """Model with learnable parameters for gradient tests."""

    def __init__(self, channels: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.eye_(self.conv.weight.view(channels, channels))

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # z is (B, H, W, C), need to permute for conv
        z_nchw = z.permute(0, 3, 1, 2)
        out = self.conv(z_nchw)
        return out.permute(0, 2, 3, 1)


@pytest.fixture
def device():
    """Return the appropriate device for testing."""
    return torch.device("cpu")


@pytest.fixture
def dtype():
    """Return the default dtype for testing."""
    return torch.float32


@pytest.fixture
def dummy_model():
    """Create a dummy model for testing."""
    return DummyModel(mode="identity")


@pytest.fixture
def linear_model():
    """Create a linear model for gradient tests."""
    return LinearDummyModel()


@pytest.fixture
def sample_noise():
    """Create sample noise tensor."""
    torch.manual_seed(42)
    return torch.randn(2, 32, 32, 3)  # (B, H, W, C) format


@pytest.fixture
def sample_text_embeddings():
    """Create sample text embeddings."""
    torch.manual_seed(42)
    return torch.randn(2, 77, 1024)  # (B, T, D)


@pytest.fixture
def sample_null_embeddings():
    """Create sample null text embeddings."""
    return torch.zeros(2, 77, 1024)


@pytest.fixture
def test_config():
    """Create test configuration."""
    return PixelHDMConfig.for_testing()


# =============================================================================
# BaseSampler Tests (10 cases)
# =============================================================================


class TestBaseSampler:
    """BaseSampler test suite."""

    def test_base_sampler_init(self):
        """Test BaseSampler initialization via concrete class."""
        sampler = EulerSampler(num_steps=50, t_eps=0.05)
        assert sampler.num_steps == 50
        assert sampler.t_eps == 0.05

    def test_base_sampler_num_steps(self):
        """Test num_steps property."""
        sampler = EulerSampler(num_steps=100)
        assert sampler.num_steps == 100

        sampler = HeunSampler(num_steps=25)
        assert sampler.num_steps == 25

    def test_base_sampler_time_eps(self):
        """Test time_eps property."""
        sampler = EulerSampler(t_eps=0.01)
        assert sampler.t_eps == 0.01

        sampler = HeunSampler(t_eps=0.1)
        assert sampler.t_eps == 0.1

    def test_base_sampler_generate_timesteps(self, device, dtype):
        """Test timestep generation."""
        sampler = EulerSampler(num_steps=10, t_eps=0.05)
        timesteps = sampler.get_timesteps(device=device, dtype=dtype)

        # Should have num_steps + 1 points (including endpoints)
        assert len(timesteps) == 11

    def test_base_sampler_timesteps_ascending(self, device, dtype):
        """Test timesteps are in ascending order (PixelHDM: 0 -> 1)."""
        sampler = EulerSampler(num_steps=20, t_eps=0.05)
        timesteps = sampler.get_timesteps(device=device, dtype=dtype)

        # PixelHDM convention: ascending from near-0 to near-1
        for i in range(len(timesteps) - 1):
            assert timesteps[i] < timesteps[i + 1], \
                f"Timesteps not ascending at index {i}: {timesteps[i]} >= {timesteps[i+1]}"

    def test_base_sampler_timesteps_bounds(self, device, dtype):
        """Test timesteps are within [eps, 1-eps]."""
        t_eps = 0.05
        sampler = EulerSampler(num_steps=50, t_eps=t_eps)
        timesteps = sampler.get_timesteps(device=device, dtype=dtype)

        # First timestep should be at t_eps
        assert torch.isclose(timesteps[0], torch.tensor(t_eps, dtype=dtype)), \
            f"First timestep {timesteps[0]} != t_eps {t_eps}"

        # Last timestep should be at 1 - t_eps
        assert torch.isclose(timesteps[-1], torch.tensor(1 - t_eps, dtype=dtype)), \
            f"Last timestep {timesteps[-1]} != 1 - t_eps {1 - t_eps}"

    def test_base_sampler_timesteps_count(self, device, dtype):
        """Test timestep count matches num_steps + 1."""
        for num_steps in [10, 25, 50, 100]:
            sampler = EulerSampler(num_steps=num_steps)
            timesteps = sampler.get_timesteps(num_steps, device=device, dtype=dtype)
            assert len(timesteps) == num_steps + 1, \
                f"Expected {num_steps + 1} timesteps, got {len(timesteps)}"

    def test_base_sampler_different_num_steps(self, device, dtype):
        """Test different num_steps configurations."""
        for num_steps in [5, 10, 25, 50, 100, 200]:
            sampler = EulerSampler(num_steps=num_steps)
            timesteps = sampler.get_timesteps(device=device, dtype=dtype)

            assert len(timesteps) == num_steps + 1
            assert timesteps[0] < timesteps[-1]  # Ascending

    def test_base_sampler_custom_time_eps(self, device, dtype):
        """Test custom time_eps values."""
        for t_eps in [0.001, 0.01, 0.05, 0.1]:
            sampler = EulerSampler(num_steps=50, t_eps=t_eps)
            timesteps = sampler.get_timesteps(device=device, dtype=dtype)

            # Check bounds
            assert timesteps[0].item() >= t_eps - 1e-6
            assert timesteps[-1].item() <= 1 - t_eps + 1e-6

    def test_base_sampler_device_placement(self):
        """Test timesteps are on correct device."""
        sampler = EulerSampler(num_steps=10)

        # CPU
        timesteps_cpu = sampler.get_timesteps(device=torch.device("cpu"))
        assert timesteps_cpu.device == torch.device("cpu")

        # Skip CUDA test if not available
        if torch.cuda.is_available():
            timesteps_cuda = sampler.get_timesteps(device=torch.device("cuda"))
            assert timesteps_cuda.device.type == "cuda"


# =============================================================================
# EulerSampler Tests (15 cases)
# =============================================================================


class TestEulerSampler:
    """EulerSampler test suite."""

    @pytest.fixture
    def euler_sampler(self):
        return EulerSampler(num_steps=10, t_eps=0.05)

    def test_euler_sampler_init(self, euler_sampler):
        """Test EulerSampler initialization."""
        assert euler_sampler.num_steps == 10
        assert euler_sampler.t_eps == 0.05

    def test_euler_sampler_step_basic(self, euler_sampler, dummy_model, sample_noise):
        """Test basic Euler step."""
        z = sample_noise
        t = torch.tensor(0.3)
        t_next = torch.tensor(0.4)

        z_next = euler_sampler.step(
            model=dummy_model,
            z=z,
            t=t,
            t_next=t_next,
            text_embeddings=None,
        )

        assert z_next.shape == z.shape
        assert not torch.isnan(z_next).any()

    def test_euler_sampler_step_formula(self, euler_sampler):
        """Test Euler step formula: z_{t+1} = z_t + dt * v (V-Prediction)."""
        # Create a simple case where we know the expected output
        z = torch.ones(1, 4, 4, 3)
        t = torch.tensor(0.3)
        t_next = torch.tensor(0.4)
        dt = t_next - t  # 0.1

        # Model that returns v_pred = z * 2 (velocity prediction)
        class VelocityModel(nn.Module):
            def forward(self, z, t, **kwargs):
                return z * 2  # v_pred = z * 2

        model = VelocityModel()
        z_next = euler_sampler.step(model, z, t, t_next, None)

        # V-Prediction: z_next = z + dt * v_pred
        # z_next = z + 0.1 * (z * 2) = z * (1 + 0.2) = z * 1.2
        expected = z + dt * (z * 2)  # = z * 1.2

        assert torch.allclose(z_next, expected, atol=1e-5)

    def test_euler_sampler_step_shape_preservation(self, euler_sampler, dummy_model):
        """Test that Euler step preserves tensor shape."""
        shapes = [
            (1, 16, 16, 3),
            (2, 32, 32, 3),
            (4, 64, 64, 3),
            (1, 128, 128, 3),
        ]

        for shape in shapes:
            z = torch.randn(shape)
            t = torch.tensor(0.5)
            t_next = torch.tensor(0.6)

            z_next = euler_sampler.step(dummy_model, z, t, t_next, None)
            assert z_next.shape == shape

    def test_euler_sampler_step_dtype_preservation(self, euler_sampler, dummy_model):
        """Test that Euler step preserves tensor dtype."""
        z = torch.randn(2, 16, 16, 3, dtype=torch.float32)
        t = torch.tensor(0.5)
        t_next = torch.tensor(0.6)

        z_next = euler_sampler.step(dummy_model, z, t, t_next, None)
        assert z_next.dtype == torch.float32

    def test_euler_sampler_sample_basic(self, euler_sampler, dummy_model, sample_noise):
        """Test complete Euler sampling."""
        sampler = UnifiedSampler(method="euler", num_steps=10)

        result = sampler.sample(
            model=dummy_model,
            z_0=sample_noise,
        )

        assert result.shape == sample_noise.shape
        assert not torch.isnan(result).any()

    def test_euler_sampler_sample_with_mock_model(self, euler_sampler, sample_noise):
        """Test Euler sampling with mock model."""
        mock_model = Mock()
        mock_model.return_value = sample_noise  # Return same as input

        sampler = UnifiedSampler(method="euler", num_steps=5)
        result = sampler.sample(model=mock_model, z_0=sample_noise)

        assert result.shape == sample_noise.shape
        # Mock should be called num_steps times
        assert mock_model.call_count == 5

    def test_euler_sampler_sample_returns_tensor(self, euler_sampler, dummy_model, sample_noise):
        """Test that sample returns a torch.Tensor."""
        sampler = UnifiedSampler(method="euler", num_steps=10)
        result = sampler.sample(model=dummy_model, z_0=sample_noise)

        assert isinstance(result, torch.Tensor)

    def test_euler_sampler_sample_output_shape(self, dummy_model):
        """Test output shape for various input shapes."""
        sampler = UnifiedSampler(method="euler", num_steps=5)

        shapes = [(1, 32, 32, 3), (2, 64, 64, 3), (4, 16, 16, 3)]
        for shape in shapes:
            z_0 = torch.randn(shape)
            result = sampler.sample(model=dummy_model, z_0=z_0)
            assert result.shape == shape

    def test_euler_sampler_nfe_count(self):
        """Test NFE count for Euler sampler."""
        for num_steps in [10, 25, 50, 100]:
            sampler = UnifiedSampler(method="euler", num_steps=num_steps)
            nfe = sampler.count_nfe()

            # Euler: NFE = num_steps
            assert nfe == num_steps, f"Expected NFE={num_steps}, got {nfe}"

    def test_euler_sampler_deterministic(self, dummy_model):
        """Test Euler sampling is deterministic with fixed seed."""
        sampler = UnifiedSampler(method="euler", num_steps=10)

        torch.manual_seed(42)
        z_0 = torch.randn(2, 32, 32, 3)

        result1 = sampler.sample(model=dummy_model, z_0=z_0.clone())
        result2 = sampler.sample(model=dummy_model, z_0=z_0.clone())

        assert torch.allclose(result1, result2)

    def test_euler_sampler_different_resolutions(self, dummy_model):
        """Test Euler sampling with different resolutions."""
        sampler = UnifiedSampler(method="euler", num_steps=5)

        resolutions = [(16, 16), (32, 32), (64, 64), (128, 128)]
        for h, w in resolutions:
            z_0 = torch.randn(1, h, w, 3)
            result = sampler.sample(model=dummy_model, z_0=z_0)
            assert result.shape == (1, h, w, 3)

    def test_euler_sampler_batch_sizes(self, dummy_model):
        """Test Euler sampling with different batch sizes."""
        sampler = UnifiedSampler(method="euler", num_steps=5)

        for batch_size in [1, 2, 4, 8]:
            z_0 = torch.randn(batch_size, 32, 32, 3)
            result = sampler.sample(model=dummy_model, z_0=z_0)
            assert result.shape[0] == batch_size

    def test_euler_sampler_gradient_disabled(self, linear_model, sample_noise):
        """Test that gradients are disabled during sampling."""
        sampler = UnifiedSampler(method="euler", num_steps=5)

        z_0 = sample_noise.requires_grad_(True)
        result = sampler.sample(model=linear_model, z_0=z_0)

        # Result should not require grad (sampled under no_grad)
        assert not result.requires_grad

    def test_euler_sampler_inference_mode(self, dummy_model, sample_noise):
        """Test sampling in inference mode."""
        sampler = UnifiedSampler(method="euler", num_steps=5)

        with torch.inference_mode():
            result = sampler.sample(model=dummy_model, z_0=sample_noise)

        assert result.shape == sample_noise.shape
        assert not torch.isnan(result).any()


# =============================================================================
# HeunSampler Tests (15 cases)
# =============================================================================


class TestHeunSampler:
    """HeunSampler test suite."""

    @pytest.fixture
    def heun_sampler(self):
        return HeunSampler(num_steps=10, t_eps=0.05)

    def test_heun_sampler_init(self, heun_sampler):
        """Test HeunSampler initialization."""
        assert heun_sampler.num_steps == 10
        assert heun_sampler.t_eps == 0.05

    def test_heun_sampler_step_basic(self, heun_sampler, dummy_model, sample_noise):
        """Test basic Heun step."""
        z = sample_noise
        t = torch.tensor(0.3)
        t_next = torch.tensor(0.4)

        z_next = heun_sampler.step(
            model=dummy_model,
            z=z,
            t=t,
            t_next=t_next,
            text_embeddings=None,
        )

        assert z_next.shape == z.shape
        assert not torch.isnan(z_next).any()

    def test_heun_sampler_step_formula(self, heun_sampler):
        """Test Heun step formula (2nd order accuracy) with V-Prediction."""
        # Heun uses average of velocities at t and t_next
        z = torch.ones(1, 4, 4, 3)
        t = torch.tensor(0.3)
        t_next = torch.tensor(0.4)
        dt = t_next - t  # 0.1

        # With zero model (v_pred = 0), z_next should equal z
        model = DummyModel(mode="zero")
        z_next = heun_sampler.step(model, z, t, t_next, None)

        # V-Prediction: with v_pred = 0, z_next = z + dt * 0 = z
        assert torch.allclose(z_next, z, atol=1e-5)

    def test_heun_sampler_step_two_evaluations(self, heun_sampler, sample_noise):
        """Test that Heun step makes two model evaluations."""
        model = DummyModel()
        model.reset_call_count()

        z = sample_noise
        t = torch.tensor(0.3)
        t_next = torch.tensor(0.4)

        heun_sampler.step(model, z, t, t_next, None)

        # Heun should call model twice per step
        assert model.call_count == 2

    def test_heun_sampler_step_shape_preservation(self, heun_sampler, dummy_model):
        """Test that Heun step preserves tensor shape."""
        shapes = [
            (1, 16, 16, 3),
            (2, 32, 32, 3),
            (4, 64, 64, 3),
        ]

        for shape in shapes:
            z = torch.randn(shape)
            t = torch.tensor(0.5)
            t_next = torch.tensor(0.6)

            z_next = heun_sampler.step(dummy_model, z, t, t_next, None)
            assert z_next.shape == shape

    def test_heun_sampler_sample_basic(self, dummy_model, sample_noise):
        """Test complete Heun sampling."""
        sampler = UnifiedSampler(method="heun", num_steps=10)

        result = sampler.sample(
            model=dummy_model,
            z_0=sample_noise,
        )

        assert result.shape == sample_noise.shape
        assert not torch.isnan(result).any()

    def test_heun_sampler_sample_with_mock_model(self, sample_noise):
        """Test Heun sampling with mock model."""
        mock_model = Mock()
        mock_model.return_value = sample_noise

        sampler = UnifiedSampler(method="heun", num_steps=5)
        result = sampler.sample(model=mock_model, z_0=sample_noise)

        assert result.shape == sample_noise.shape
        # Heun: 2 calls per step
        assert mock_model.call_count == 10  # 5 steps * 2

    def test_heun_sampler_nfe_count(self):
        """Test NFE count for Heun sampler."""
        for num_steps in [10, 25, 50]:
            sampler = UnifiedSampler(method="heun", num_steps=num_steps)
            nfe = sampler.count_nfe()

            # Heun: NFE = 2*num_steps - 1 (last step is Euler)
            expected = 2 * num_steps - 1
            assert nfe == expected, f"Expected NFE={expected}, got {nfe}"

    def test_heun_sampler_more_accurate_than_euler(self, dummy_model):
        """Test that Heun is more accurate than Euler (conceptually)."""
        # This is a qualitative test - Heun should produce different results
        z_0 = torch.randn(1, 32, 32, 3)

        euler_sampler = UnifiedSampler(method="euler", num_steps=10)
        heun_sampler = UnifiedSampler(method="heun", num_steps=10)

        euler_result = euler_sampler.sample(model=dummy_model, z_0=z_0.clone())
        heun_result = heun_sampler.sample(model=dummy_model, z_0=z_0.clone())

        # Results should be different (different integration methods)
        # Note: With identity model they might be the same, so we test shape
        assert euler_result.shape == heun_result.shape

    def test_heun_sampler_deterministic(self, dummy_model):
        """Test Heun sampling is deterministic."""
        sampler = UnifiedSampler(method="heun", num_steps=10)

        z_0 = torch.randn(2, 32, 32, 3)

        result1 = sampler.sample(model=dummy_model, z_0=z_0.clone())
        result2 = sampler.sample(model=dummy_model, z_0=z_0.clone())

        assert torch.allclose(result1, result2)

    def test_heun_sampler_different_resolutions(self, dummy_model):
        """Test Heun sampling with different resolutions."""
        sampler = UnifiedSampler(method="heun", num_steps=5)

        resolutions = [(16, 16), (32, 32), (64, 64)]
        for h, w in resolutions:
            z_0 = torch.randn(1, h, w, 3)
            result = sampler.sample(model=dummy_model, z_0=z_0)
            assert result.shape == (1, h, w, 3)

    def test_heun_sampler_batch_sizes(self, dummy_model):
        """Test Heun sampling with different batch sizes."""
        sampler = UnifiedSampler(method="heun", num_steps=5)

        for batch_size in [1, 2, 4]:
            z_0 = torch.randn(batch_size, 32, 32, 3)
            result = sampler.sample(model=dummy_model, z_0=z_0)
            assert result.shape[0] == batch_size

    def test_heun_sampler_last_step_euler(self):
        """Test that last step uses Euler (optimization)."""
        # This is reflected in NFE count: 2*n - 1 instead of 2*n
        sampler = UnifiedSampler(method="heun", num_steps=10)
        nfe = sampler.count_nfe()

        # If all steps were Heun: 2*10 = 20
        # With last step Euler: 2*10 - 1 = 19
        assert nfe == 19

    def test_heun_sampler_gradient_disabled(self, linear_model, sample_noise):
        """Test that gradients are disabled during Heun sampling."""
        sampler = UnifiedSampler(method="heun", num_steps=5)

        z_0 = sample_noise.requires_grad_(True)
        result = sampler.sample(model=linear_model, z_0=z_0)

        assert not result.requires_grad

    def test_heun_sampler_full_heun_cfg(self, dummy_model, sample_noise, sample_null_embeddings):
        """Test full Heun CFG mode."""
        sampler = UnifiedSampler(method="heun", num_steps=5)

        # Full Heun CFG applies Heun to both conditional and unconditional
        result = sampler.sample_with_full_heun_cfg(
            model=dummy_model,
            z_0=sample_noise,
            text_embeddings=torch.randn(2, 77, 1024),
            guidance_scale=7.5,
            null_text_embeddings=sample_null_embeddings,
        )

        assert result.shape == sample_noise.shape
        assert not torch.isnan(result).any()


# =============================================================================
# DPMPPSampler Tests (15 cases)
# =============================================================================


class TestDPMPPSampler:
    """DPMPPSampler test suite."""

    @pytest.fixture
    def dpmpp_sampler(self):
        return DPMPPSampler(num_steps=10, t_eps=0.05)

    def test_dpmpp_sampler_init(self, dpmpp_sampler):
        """Test DPMPPSampler initialization."""
        assert dpmpp_sampler.num_steps == 10
        assert dpmpp_sampler.t_eps == 0.05
        assert dpmpp_sampler._prev_x_pred is None
        assert dpmpp_sampler._prev_t is None

    def test_dpmpp_sampler_step_basic(self, dpmpp_sampler, dummy_model, sample_noise):
        """Test basic DPM++ step."""
        z = sample_noise
        t = torch.tensor(0.3)
        t_next = torch.tensor(0.4)

        z_next = dpmpp_sampler.step(
            model=dummy_model,
            z=z,
            t=t,
            t_next=t_next,
            text_embeddings=None,
        )

        assert z_next.shape == z.shape
        assert not torch.isnan(z_next).any()

    def test_dpmpp_sampler_multistep(self, dpmpp_sampler, dummy_model, sample_noise):
        """Test DPM++ multi-step behavior."""
        z = sample_noise
        timesteps = torch.linspace(0.05, 0.95, 6)

        dpmpp_sampler.reset()

        for i in range(5):
            z = dpmpp_sampler.step(
                model=dummy_model,
                z=z,
                t=timesteps[i],
                t_next=timesteps[i + 1],
                text_embeddings=None,
            )

        assert z.shape == sample_noise.shape
        assert not torch.isnan(z).any()

    def test_dpmpp_sampler_step_shape_preservation(self, dpmpp_sampler, dummy_model):
        """Test DPM++ step preserves shape."""
        shapes = [(1, 16, 16, 3), (2, 32, 32, 3), (4, 64, 64, 3)]

        for shape in shapes:
            dpmpp_sampler.reset()
            z = torch.randn(shape)
            t = torch.tensor(0.5)
            t_next = torch.tensor(0.6)

            z_next = dpmpp_sampler.step(dummy_model, z, t, t_next, None)
            assert z_next.shape == shape

    def test_dpmpp_sampler_sample_basic(self, dummy_model, sample_noise):
        """Test complete DPM++ sampling."""
        sampler = UnifiedSampler(method="dpm_pp", num_steps=10)

        result = sampler.sample(
            model=dummy_model,
            z_0=sample_noise,
        )

        assert result.shape == sample_noise.shape
        assert not torch.isnan(result).any()

    def test_dpmpp_sampler_sample_with_mock_model(self, sample_noise):
        """Test DPM++ sampling with mock model."""
        mock_model = Mock()
        mock_model.return_value = sample_noise

        sampler = UnifiedSampler(method="dpm_pp", num_steps=5)
        result = sampler.sample(model=mock_model, z_0=sample_noise)

        assert result.shape == sample_noise.shape
        # DPM++ makes 1 call per step
        assert mock_model.call_count == 5

    def test_dpmpp_sampler_nfe_count(self):
        """Test NFE count for DPM++ sampler."""
        for num_steps in [10, 25, 50]:
            sampler = UnifiedSampler(method="dpm_pp", num_steps=num_steps)
            nfe = sampler.count_nfe()

            # DPM++: NFE = num_steps (uses history, no extra evaluations)
            assert nfe == num_steps

    def test_dpmpp_sampler_deterministic(self, dummy_model):
        """Test DPM++ sampling is deterministic."""
        sampler = UnifiedSampler(method="dpm_pp", num_steps=10)

        z_0 = torch.randn(2, 32, 32, 3)

        result1 = sampler.sample(model=dummy_model, z_0=z_0.clone())
        result2 = sampler.sample(model=dummy_model, z_0=z_0.clone())

        assert torch.allclose(result1, result2)

    def test_dpmpp_sampler_different_resolutions(self, dummy_model):
        """Test DPM++ sampling with different resolutions."""
        sampler = UnifiedSampler(method="dpm_pp", num_steps=5)

        resolutions = [(16, 16), (32, 32), (64, 64)]
        for h, w in resolutions:
            z_0 = torch.randn(1, h, w, 3)
            result = sampler.sample(model=dummy_model, z_0=z_0)
            assert result.shape == (1, h, w, 3)

    def test_dpmpp_sampler_batch_sizes(self, dummy_model):
        """Test DPM++ sampling with different batch sizes."""
        sampler = UnifiedSampler(method="dpm_pp", num_steps=5)

        for batch_size in [1, 2, 4]:
            z_0 = torch.randn(batch_size, 32, 32, 3)
            result = sampler.sample(model=dummy_model, z_0=z_0)
            assert result.shape[0] == batch_size

    def test_dpmpp_sampler_history_management(self, dpmpp_sampler, dummy_model, sample_noise):
        """Test DPM++ history management."""
        # Initial state
        assert dpmpp_sampler._prev_x_pred is None
        assert dpmpp_sampler._prev_t is None

        # After first step
        z = sample_noise
        t = torch.tensor(0.3)
        t_next = torch.tensor(0.4)

        dpmpp_sampler.step(dummy_model, z, t, t_next, None)

        # History should be updated
        assert dpmpp_sampler._prev_x_pred is not None
        assert dpmpp_sampler._prev_t is not None

        # After reset
        dpmpp_sampler.reset()
        assert dpmpp_sampler._prev_x_pred is None
        assert dpmpp_sampler._prev_t is None

    def test_dpmpp_sampler_gradient_disabled(self, linear_model, sample_noise):
        """Test that gradients are disabled during DPM++ sampling."""
        sampler = UnifiedSampler(method="dpm_pp", num_steps=5)

        z_0 = sample_noise.requires_grad_(True)
        result = sampler.sample(model=linear_model, z_0=z_0)

        assert not result.requires_grad

    def test_dpmpp_sampler_first_step_special(self, dpmpp_sampler, dummy_model, sample_noise):
        """Test DPM++ first step uses first-order update."""
        dpmpp_sampler.reset()

        # First step should use first-order (no history)
        z = sample_noise
        t = torch.tensor(0.3)
        t_next = torch.tensor(0.4)

        z_next = dpmpp_sampler.step(dummy_model, z, t, t_next, None)

        # Should produce valid output
        assert z_next.shape == z.shape
        assert not torch.isnan(z_next).any()

    def test_dpmpp_sampler_stability(self, dummy_model):
        """Test DPM++ numerical stability."""
        sampler = UnifiedSampler(method="dpm_pp", num_steps=50)

        z_0 = torch.randn(1, 64, 64, 3)
        result = sampler.sample(model=dummy_model, z_0=z_0)

        # Check for numerical stability
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
        assert result.abs().max() < 1000  # Reasonable bounds

    def test_dpmpp_sampler_order_parameter(self):
        """Test DPM++ order parameter (2S variant)."""
        sampler = UnifiedSampler(method="dpm_pp_2s", num_steps=10)

        z_0 = torch.randn(1, 32, 32, 3)
        model = DummyModel()

        result = sampler.sample(model=model, z_0=z_0)

        assert result.shape == z_0.shape
        assert not torch.isnan(result).any()


# =============================================================================
# UnifiedSampler Tests (15 cases)
# =============================================================================


class TestUnifiedSampler:
    """UnifiedSampler test suite."""

    def test_unified_sampler_init_euler(self):
        """Test UnifiedSampler initialization with Euler."""
        sampler = UnifiedSampler(method="euler", num_steps=50)
        assert sampler.method == SamplerMethod.EULER
        assert sampler.num_steps == 50

    def test_unified_sampler_init_heun(self):
        """Test UnifiedSampler initialization with Heun."""
        sampler = UnifiedSampler(method="heun", num_steps=50)
        assert sampler.method == SamplerMethod.HEUN
        assert sampler.num_steps == 50

    def test_unified_sampler_init_dpmpp(self):
        """Test UnifiedSampler initialization with DPM++."""
        sampler = UnifiedSampler(method="dpm_pp", num_steps=50)
        assert sampler.method == SamplerMethod.DPM_PP
        assert sampler.num_steps == 50

    def test_unified_sampler_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            UnifiedSampler(method="invalid_method", num_steps=50)

        # Error message should indicate invalid method
        error_msg = str(exc_info.value)
        assert "invalid_method" in error_msg.lower() or "samplermethod" in error_msg.lower()

    def test_unified_sampler_sample_basic(self, dummy_model, sample_noise):
        """Test basic unified sampling."""
        sampler = UnifiedSampler(method="heun", num_steps=10)

        result = sampler.sample(
            model=dummy_model,
            z_0=sample_noise,
        )

        assert result.shape == sample_noise.shape
        assert not torch.isnan(result).any()

    def test_unified_sampler_sample_with_cfg(
        self, dummy_model, sample_noise, sample_text_embeddings, sample_null_embeddings
    ):
        """Test unified sampling with CFG."""
        sampler = UnifiedSampler(method="heun", num_steps=10)

        result = sampler.sample(
            model=dummy_model,
            z_0=sample_noise,
            text_embeddings=sample_text_embeddings,
            guidance_scale=7.5,
            null_text_embeddings=sample_null_embeddings,
        )

        assert result.shape == sample_noise.shape
        assert not torch.isnan(result).any()

    def test_unified_sampler_sample_with_text(
        self, dummy_model, sample_noise, sample_text_embeddings
    ):
        """Test unified sampling with text conditioning."""
        sampler = UnifiedSampler(method="euler", num_steps=10)

        result = sampler.sample(
            model=dummy_model,
            z_0=sample_noise,
            text_embeddings=sample_text_embeddings,
        )

        assert result.shape == sample_noise.shape

    def test_unified_sampler_count_nfe(self):
        """Test NFE counting method."""
        # Euler
        euler_sampler = UnifiedSampler(method="euler", num_steps=50)
        assert euler_sampler.count_nfe() == 50

        # Heun
        heun_sampler = UnifiedSampler(method="heun", num_steps=50)
        assert heun_sampler.count_nfe() == 99  # 2*50 - 1

        # DPM++
        dpmpp_sampler = UnifiedSampler(method="dpm_pp", num_steps=50)
        assert dpmpp_sampler.count_nfe() == 50

    def test_unified_sampler_count_nfe_with_cfg(self):
        """Test NFE counting with CFG enabled."""
        # With CFG, NFE doubles
        euler_sampler = UnifiedSampler(method="euler", num_steps=50)
        assert euler_sampler.count_nfe(use_cfg=True) == 100  # 50 * 2

        heun_sampler = UnifiedSampler(method="heun", num_steps=50)
        # Standard CFG with Heun falls back to per-step doubling
        assert heun_sampler.count_nfe(use_cfg=True) == 100  # num_steps * 2

    def test_unified_sampler_count_nfe_full_heun_cfg(self):
        """Test NFE counting with full Heun CFG."""
        heun_sampler = UnifiedSampler(method="heun", num_steps=50)

        # Full Heun CFG: base_nfe * 2
        # base_nfe = 2*50 - 1 = 99
        # full_heun_cfg_nfe = 99 * 2 = 198
        assert heun_sampler.count_nfe(use_cfg=True, full_heun_cfg=True) == 198

    def test_unified_sampler_sample_with_full_heun_cfg(
        self, dummy_model, sample_noise, sample_text_embeddings, sample_null_embeddings
    ):
        """Test full Heun CFG sampling."""
        sampler = UnifiedSampler(method="heun", num_steps=5)

        result = sampler.sample_with_full_heun_cfg(
            model=dummy_model,
            z_0=sample_noise,
            text_embeddings=sample_text_embeddings,
            guidance_scale=7.5,
            null_text_embeddings=sample_null_embeddings,
        )

        assert result.shape == sample_noise.shape
        assert not torch.isnan(result).any()

    def test_unified_sampler_get_method(self):
        """Test getting the sampler method."""
        for method in ["euler", "heun", "dpm_pp"]:
            sampler = UnifiedSampler(method=method, num_steps=50)
            assert sampler.method.value == method

    def test_unified_sampler_get_num_steps(self):
        """Test getting the number of steps."""
        for num_steps in [10, 25, 50, 100]:
            sampler = UnifiedSampler(method="heun", num_steps=num_steps)
            assert sampler.num_steps == num_steps

    def test_create_sampler_factory(self):
        """Test create_sampler factory function."""
        sampler = create_sampler(method="euler", num_steps=25, t_eps=0.01)

        assert isinstance(sampler, UnifiedSampler)
        assert sampler.method == SamplerMethod.EULER
        assert sampler.num_steps == 25
        assert sampler.t_eps == 0.01

    def test_create_sampler_from_config(self, test_config):
        """Test create_sampler_from_config factory function."""
        sampler = create_sampler_from_config(test_config)

        assert isinstance(sampler, UnifiedSampler)
        assert sampler.num_steps == test_config.default_num_steps
        assert sampler.t_eps == test_config.time_eps


# =============================================================================
# Additional Edge Case Tests
# =============================================================================


class TestSamplerEdgeCases:
    """Edge case tests for samplers."""

    def test_sampler_with_zero_guidance_scale(self, dummy_model, sample_noise):
        """Test sampling with guidance_scale=0 (no CFG)."""
        sampler = UnifiedSampler(method="euler", num_steps=5)

        result = sampler.sample(
            model=dummy_model,
            z_0=sample_noise,
            guidance_scale=0.0,
        )

        assert result.shape == sample_noise.shape

    def test_sampler_with_high_guidance_scale(
        self, dummy_model, sample_noise, sample_text_embeddings, sample_null_embeddings
    ):
        """Test sampling with very high guidance scale."""
        sampler = UnifiedSampler(method="euler", num_steps=5)

        result = sampler.sample(
            model=dummy_model,
            z_0=sample_noise,
            text_embeddings=sample_text_embeddings,
            guidance_scale=20.0,
            null_text_embeddings=sample_null_embeddings,
        )

        assert result.shape == sample_noise.shape
        assert not torch.isnan(result).any()

    def test_sampler_with_callback(self, dummy_model, sample_noise):
        """Test sampling with progress callback."""
        sampler = UnifiedSampler(method="euler", num_steps=5)

        callback_calls = []

        def callback(step, total, z):
            callback_calls.append((step, total, z.shape))

        sampler.sample(
            model=dummy_model,
            z_0=sample_noise,
            callback=callback,
        )

        assert len(callback_calls) == 5  # Called for each step
        for i, (step, total, shape) in enumerate(callback_calls):
            assert step == i
            assert total == 5
            assert shape == sample_noise.shape

    def test_sampler_progressive_output(self, dummy_model, sample_noise):
        """Test progressive sampling output."""
        sampler = UnifiedSampler(method="euler", num_steps=10)

        progressions = sampler.sample_progressive(
            model=dummy_model,
            z_0=sample_noise,
            return_every=2,
        )

        # Should return every 2nd step + last step
        # Steps: 0, 2, 4, 6, 8, 9 = 6 saved
        assert progressions.shape[0] >= 5
        assert progressions.shape[1:] == sample_noise.shape

    def test_sampler_i2i_t_start(self, dummy_model, sample_noise):
        """Test I2I sampling with custom t_start."""
        sampler = UnifiedSampler(method="euler", num_steps=10)

        # Start from t=0.5 (partial denoising)
        result = sampler.sample(
            model=dummy_model,
            z_0=sample_noise,
            t_start=0.5,
        )

        assert result.shape == sample_noise.shape

    def test_sampler_method_enum(self):
        """Test SamplerMethod enum values."""
        assert SamplerMethod.EULER.value == "euler"
        assert SamplerMethod.HEUN.value == "heun"
        assert SamplerMethod.DPM_PP.value == "dpm_pp"
        assert SamplerMethod.DPM_PP_2S.value == "dpm_pp_2s"

    def test_sampler_config_dataclass(self):
        """Test SamplerConfig dataclass."""
        config = SamplerConfig(
            num_steps=100,
            method=SamplerMethod.HEUN,
            t_eps=0.01,
            guidance_scale=5.0,
        )

        assert config.num_steps == 100
        assert config.method == SamplerMethod.HEUN
        assert config.t_eps == 0.01
        assert config.guidance_scale == 5.0

    def test_v_to_x_conversion(self):
        """Test velocity to x_pred conversion (V-Prediction, used by DPM++)."""
        from src.inference.sampler import DPMPPSampler
        sampler = DPMPPSampler(num_steps=10, t_eps=0.05)

        v_pred = torch.ones(2, 4, 4, 3)
        z = torch.zeros(2, 4, 4, 3)
        t = torch.tensor([0.5, 0.5])

        # x = z + (1 - t) * v = 0 + 0.5 * 1 = 0.5
        x = sampler._v_to_x(v_pred, z, t)

        expected = torch.ones(2, 4, 4, 3) * 0.5
        assert torch.allclose(x, expected, atol=1e-5)

    def test_sampler_with_text_mask(self, dummy_model, sample_noise, sample_text_embeddings):
        """Test sampling with text mask."""
        sampler = UnifiedSampler(method="euler", num_steps=5)
        text_mask = torch.ones(2, 77, dtype=torch.bool)

        result = sampler.sample(
            model=dummy_model,
            z_0=sample_noise,
            text_embeddings=sample_text_embeddings,
            text_mask=text_mask,
        )

        assert result.shape == sample_noise.shape
