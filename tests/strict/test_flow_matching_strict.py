"""
Flow Matching Strict Tests

Verifies the mathematical correctness of PixelHDMFlowMatching implementation.
Tests are designed based on STRICT_TEST_STRATEGY.md.

Key validations:
1. Default parameter behavior (JiT vs SD3/PixelHDM)
2. Config override mechanism
3. Timestep distribution statistics
4. Interpolation formula correctness
5. V-theta and V-target calculations
6. Loss computation

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

# Import the module under test
from src.training.flow_matching import (
    PixelHDMFlowMatching,
    PixelHDMSampler,
    create_flow_matching,
    create_flow_matching_from_config,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device() -> torch.device:
    """Return the device to use for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dtype() -> torch.dtype:
    """Return the default dtype for testing."""
    return torch.float32


@pytest.fixture
def sample_images(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create sample clean images for testing."""
    # Shape: (B, C, H, W) = (4, 3, 64, 64)
    torch.manual_seed(42)
    return torch.randn(4, 3, 64, 64, device=device, dtype=dtype)


@pytest.fixture
def sample_noise(sample_images: torch.Tensor) -> torch.Tensor:
    """Create sample noise matching image shape."""
    torch.manual_seed(123)
    return torch.randn_like(sample_images)


@dataclass
class MockPixelHDMConfig:
    """Mock config for testing config override behavior."""
    time_p_mean: float = 0.0
    time_p_std: float = 1.0
    time_eps: float = 0.05


@pytest.fixture
def mock_config() -> MockPixelHDMConfig:
    """Create a mock config with SD3/PixelHDM parameters."""
    return MockPixelHDMConfig()


@pytest.fixture
def mock_pixelhdm_config() -> MockPixelHDMConfig:
    """Create a mock config with JiT class-conditional parameters."""
    return MockPixelHDMConfig(time_p_mean=-0.8, time_p_std=0.8)


# =============================================================================
# Test 1: Default Parameters Without Config
# =============================================================================

class TestDefaultParamsWithoutConfig:
    """
    Verify behavior when no config is provided.

    The default parameters in PixelHDMFlowMatching.__init__ are SD3/PixelHDM
    values (p_mean=0.0, p_std=1.0), which are suitable for T2I tasks.
    """

    def test_default_p_mean_is_sd3_value(self):
        """Without config, p_mean defaults to 0.0 (SD3/PixelHDM T2I)."""
        fm = PixelHDMFlowMatching()

        # Document the behavior: default is SD3/PixelHDM T2I value
        assert fm.p_mean == 0.0, (
            f"Expected default p_mean=0.0 (SD3/PixelHDM), got {fm.p_mean}. "
            "If this changes, update test documentation."
        )

    def test_default_p_std_is_sd3_value(self):
        """Without config, p_std defaults to 1.0 (SD3/PixelHDM T2I)."""
        fm = PixelHDMFlowMatching()

        assert fm.p_std == 1.0, (
            f"Expected default p_std=1.0 (SD3/PixelHDM), got {fm.p_std}. "
            "If this changes, update test documentation."
        )

    def test_default_t_eps(self):
        """t_eps should default to 0.05."""
        fm = PixelHDMFlowMatching()

        assert fm.t_eps == 0.05, f"Expected t_eps=0.05, got {fm.t_eps}"

    def test_explicit_params_override_defaults(self):
        """Explicitly passed params should override defaults."""
        fm = PixelHDMFlowMatching(p_mean=0.0, p_std=1.0, t_eps=0.1)

        assert fm.p_mean == 0.0
        assert fm.p_std == 1.0
        assert fm.t_eps == 0.1

    def test_factory_function_default_params(self):
        """create_flow_matching factory uses SD3/PixelHDM defaults."""
        fm = create_flow_matching()

        # Factory uses SD3/PixelHDM defaults (p_mean=0.0, p_std=1.0)
        # This is the correct convention for T2I training
        assert fm.p_mean == 0.0
        assert fm.p_std == 1.0


# =============================================================================
# Test 2: Config Overrides Defaults
# =============================================================================

class TestConfigOverridesDefaults:
    """
    Verify that config parameters correctly override default values.

    This is CRITICAL for T2I training where we need SD3/PixelHDM parameters
    (p_mean=0.0, p_std=1.0) instead of JiT class-conditional defaults.
    """

    def test_config_p_mean_overrides_default(self, mock_config: MockPixelHDMConfig):
        """Config's time_p_mean should override default p_mean."""
        fm = PixelHDMFlowMatching(config=mock_config)

        assert fm.p_mean == 0.0, (
            f"Config p_mean=0.0 should override default -0.8, got {fm.p_mean}"
        )

    def test_config_p_std_overrides_default(self, mock_config: MockPixelHDMConfig):
        """Config's time_p_std should override default p_std."""
        fm = PixelHDMFlowMatching(config=mock_config)

        assert fm.p_std == 1.0, (
            f"Config p_std=1.0 should override default 0.8, got {fm.p_std}"
        )

    def test_config_t_eps_overrides_default(self):
        """Config's time_eps should override default t_eps."""
        custom_config = MockPixelHDMConfig(time_eps=0.02)
        fm = PixelHDMFlowMatching(config=custom_config)

        assert fm.t_eps == 0.02

    def test_config_takes_priority_over_explicit_params(self, mock_config: MockPixelHDMConfig):
        """Config should take priority over explicitly passed parameters."""
        # Even if we pass p_mean=-0.8 explicitly, config should override
        fm = PixelHDMFlowMatching(config=mock_config, p_mean=-0.8, p_std=0.8)

        assert fm.p_mean == 0.0, "Config should override explicit p_mean"
        assert fm.p_std == 1.0, "Config should override explicit p_std"

    def test_create_from_config_factory(self, mock_config: MockPixelHDMConfig):
        """create_flow_matching_from_config should correctly use config values."""
        fm = create_flow_matching_from_config(mock_config)

        assert fm.p_mean == 0.0
        assert fm.p_std == 1.0
        assert fm.t_eps == 0.05


# =============================================================================
# Test 3: Timestep Distribution Shape
# =============================================================================

class TestTimestepDistributionShape:
    """
    Verify the statistical properties of the timestep sampling distribution.

    Logit-Normal Distribution:
        t = sigmoid(p_mean + p_std * N(0,1))

    Expected behavior:
        - p_mean=0.0, p_std=1.0 (SD3/PixelHDM): ~5% samples with t>0.8
        - p_mean=-0.8, p_std=0.8 (PixelHDM): ~0.1% samples with t>0.8
    """

    @pytest.fixture
    def large_sample_size(self) -> int:
        """Number of samples for statistical tests."""
        return 100000

    def test_sd3_distribution_t_above_08(
        self,
        mock_config: MockPixelHDMConfig,
        large_sample_size: int,
        device: torch.device
    ):
        """SD3/PixelHDM params should have ~5% of t > 0.8."""
        fm = PixelHDMFlowMatching(config=mock_config)

        torch.manual_seed(42)
        t = fm.sample_timesteps(large_sample_size, device)

        # After rescaling: t in [t_eps, 1-t_eps] = [0.05, 0.95]
        # So t > 0.8 is in the upper portion
        ratio_above_08 = (t > 0.8).float().mean().item()

        # For p_mean=0.0, p_std=1.0, expect ~5% above 0.8
        # Allow tolerance: 3% to 8%
        assert 0.03 < ratio_above_08 < 0.08, (
            f"Expected ~5% of t > 0.8 for SD3 params, got {ratio_above_08*100:.1f}%"
        )

    def test_pixelhdm_distribution_t_above_08(
        self,
        mock_pixelhdm_config: MockPixelHDMConfig,
        large_sample_size: int,
        device: torch.device
    ):
        """JiT params should have very few (< 1%) of t > 0.8."""
        fm = PixelHDMFlowMatching(config=mock_pixelhdm_config)

        torch.manual_seed(42)
        t = fm.sample_timesteps(large_sample_size, device)

        ratio_above_08 = (t > 0.8).float().mean().item()

        # For p_mean=-0.8, p_std=0.8, expect < 1% above 0.8
        assert ratio_above_08 < 0.01, (
            f"Expected < 1% of t > 0.8 for JiT params, got {ratio_above_08*100:.2f}%"
        )

    def test_timesteps_are_in_valid_range(
        self,
        mock_config: MockPixelHDMConfig,
        device: torch.device
    ):
        """Timesteps should be in [t_eps, 1-t_eps]."""
        fm = PixelHDMFlowMatching(config=mock_config)

        torch.manual_seed(42)
        t = fm.sample_timesteps(10000, device)

        assert t.min() >= fm.t_eps, f"Min t={t.min():.4f} < t_eps={fm.t_eps}"
        assert t.max() <= 1 - fm.t_eps, f"Max t={t.max():.4f} > 1-t_eps={1-fm.t_eps}"

    def test_distribution_mean_shift(
        self,
        large_sample_size: int,
        device: torch.device
    ):
        """p_mean=-0.8 should shift distribution toward lower t values."""
        fm_sd3 = PixelHDMFlowMatching(p_mean=0.0, p_std=1.0)
        fm_jit = PixelHDMFlowMatching(p_mean=-0.8, p_std=0.8)

        torch.manual_seed(42)
        t_sd3 = fm_sd3.sample_timesteps(large_sample_size, device)

        torch.manual_seed(42)
        t_jit = fm_jit.sample_timesteps(large_sample_size, device)

        # JiT distribution should have lower mean (more noise-heavy)
        assert t_jit.mean() < t_sd3.mean(), (
            f"JiT mean ({t_jit.mean():.3f}) should be < SD3 mean ({t_sd3.mean():.3f})"
        )

    def test_distribution_reproducibility(self, device: torch.device):
        """Same seed should produce same timesteps."""
        fm = PixelHDMFlowMatching(p_mean=0.0, p_std=1.0)

        torch.manual_seed(42)
        t1 = fm.sample_timesteps(100, device)

        torch.manual_seed(42)
        t2 = fm.sample_timesteps(100, device)

        assert torch.allclose(t1, t2), "Same seed should produce identical timesteps"


# =============================================================================
# Test 4: Interpolation Formula
# =============================================================================

class TestInterpolationFormula:
    """
    Verify the JiT interpolation formula: z_t = t * x + (1 - t) * noise

    Boundary conditions:
        - t=0: z_t = noise (pure noise)
        - t=1: z_t = x (clean image)
    """

    def test_interpolation_at_t_zero(
        self,
        sample_images: torch.Tensor,
        sample_noise: torch.Tensor
    ):
        """At t=0, z_t should equal noise."""
        fm = PixelHDMFlowMatching()

        t = torch.zeros(sample_images.shape[0], device=sample_images.device)
        z_t = fm.interpolate(sample_images, sample_noise, t)

        assert torch.allclose(z_t, sample_noise, atol=1e-6), (
            "At t=0, z_t should equal noise"
        )

    def test_interpolation_at_t_one(
        self,
        sample_images: torch.Tensor,
        sample_noise: torch.Tensor
    ):
        """At t=1, z_t should equal clean image x."""
        fm = PixelHDMFlowMatching()

        t = torch.ones(sample_images.shape[0], device=sample_images.device)
        z_t = fm.interpolate(sample_images, sample_noise, t)

        assert torch.allclose(z_t, sample_images, atol=1e-6), (
            "At t=1, z_t should equal clean image x"
        )

    def test_interpolation_at_t_half(
        self,
        sample_images: torch.Tensor,
        sample_noise: torch.Tensor
    ):
        """At t=0.5, z_t should be midpoint between x and noise."""
        fm = PixelHDMFlowMatching()

        t = torch.full((sample_images.shape[0],), 0.5, device=sample_images.device)
        z_t = fm.interpolate(sample_images, sample_noise, t)

        expected = 0.5 * sample_images + 0.5 * sample_noise

        assert torch.allclose(z_t, expected, atol=1e-6), (
            "At t=0.5, z_t should be average of x and noise"
        )

    def test_interpolation_formula_manual(
        self,
        sample_images: torch.Tensor,
        sample_noise: torch.Tensor
    ):
        """Verify formula z_t = t * x + (1 - t) * noise for arbitrary t."""
        fm = PixelHDMFlowMatching()

        # Test with various t values
        t_values = [0.1, 0.3, 0.7, 0.9]

        for t_val in t_values:
            t = torch.full(
                (sample_images.shape[0],), t_val, device=sample_images.device
            )
            z_t = fm.interpolate(sample_images, sample_noise, t)

            # Manual calculation
            t_expanded = t.view(-1, 1, 1, 1)
            expected = t_expanded * sample_images + (1 - t_expanded) * sample_noise

            assert torch.allclose(z_t, expected, atol=1e-6), (
                f"Interpolation formula incorrect at t={t_val}"
            )

    def test_interpolation_batch_independence(
        self,
        sample_images: torch.Tensor,
        sample_noise: torch.Tensor
    ):
        """Each batch element should use its own t value."""
        fm = PixelHDMFlowMatching()

        # Different t for each batch element
        t = torch.tensor([0.1, 0.3, 0.5, 0.9], device=sample_images.device)
        z_t = fm.interpolate(sample_images, sample_noise, t)

        # Check each element separately
        for i, t_val in enumerate(t):
            expected_i = t_val * sample_images[i] + (1 - t_val) * sample_noise[i]
            assert torch.allclose(z_t[i], expected_i, atol=1e-6), (
                f"Batch element {i} interpolation incorrect"
            )


# =============================================================================
# Test 5: V-Prediction (compute_v_theta removed)
# =============================================================================

# NOTE: In V-Prediction, the model directly outputs velocity v = x - noise.
# There is no compute_v_theta method because no X->V conversion is needed.
# The old X-Prediction approach (v_theta = (x_pred - z_t) / (1-t)) has been
# removed to avoid 1/(1-t) numerical instability when t approaches 1.


# =============================================================================
# Test 6: V-Target Calculation
# =============================================================================

class TestVTargetCalculation:
    """
    Verify v_target calculation: v_target = x - noise

    This is the target velocity that the network should learn to predict.
    """

    def test_v_target_formula(
        self,
        sample_images: torch.Tensor,
        sample_noise: torch.Tensor
    ):
        """v_target should equal x - noise."""
        fm = PixelHDMFlowMatching()

        v_target = fm.compute_v_target(sample_images, sample_noise)

        expected = sample_images - sample_noise

        assert torch.allclose(v_target.float(), expected.float(), atol=1e-6), (
            "v_target should equal x - noise"
        )

    def test_v_target_batch_independence(
        self,
        sample_images: torch.Tensor,
        sample_noise: torch.Tensor
    ):
        """Each batch element should be computed independently."""
        fm = PixelHDMFlowMatching()

        v_target = fm.compute_v_target(sample_images, sample_noise)

        for i in range(sample_images.shape[0]):
            expected_i = sample_images[i] - sample_noise[i]
            assert torch.allclose(v_target[i].float(), expected_i.float(), atol=1e-6), (
                f"Batch element {i} v_target incorrect"
            )

    def test_v_target_zero_noise(
        self,
        sample_images: torch.Tensor
    ):
        """With zero noise, v_target should equal x."""
        fm = PixelHDMFlowMatching()

        zero_noise = torch.zeros_like(sample_images)
        v_target = fm.compute_v_target(sample_images, zero_noise)

        assert torch.allclose(v_target.float(), sample_images.float(), atol=1e-6)

    def test_v_target_returns_float32(
        self,
        sample_images: torch.Tensor,
        sample_noise: torch.Tensor
    ):
        """v_target should be computed in float32."""
        fm = PixelHDMFlowMatching()

        v_target = fm.compute_v_target(sample_images, sample_noise)

        assert v_target.dtype == torch.float32


# =============================================================================
# Test 7: Loss Computation
# =============================================================================

class TestLossComputation:
    """
    Verify the v-loss computation: L = MSE(v_pred, v_target)

    V-Prediction:
        - Model directly outputs velocity v_pred
        - v_target = x_clean - noise
        - Loss = MSE(v_pred, v_target)

    Properties to verify:
        - Loss is always non-negative
        - Loss is zero when prediction is perfect
        - Loss scales with prediction error
    """

    def test_loss_nonnegative(
        self,
        sample_images: torch.Tensor,
        sample_noise: torch.Tensor
    ):
        """Loss should always be non-negative."""
        fm = PixelHDMFlowMatching()

        # Random v_pred (not correct)
        v_pred = torch.randn_like(sample_images)

        loss = fm.compute_loss(v_pred, sample_images, sample_noise)

        assert loss >= 0, f"Loss should be non-negative, got {loss}"

    def test_loss_zero_when_perfect(
        self,
        sample_images: torch.Tensor,
        sample_noise: torch.Tensor,
        device: torch.device
    ):
        """Loss should be zero when v_pred equals v_target (perfect prediction)."""
        fm = PixelHDMFlowMatching()

        x = sample_images[:1]
        noise = sample_noise[:1]

        # Perfect prediction: v_pred = v_target = x - noise
        v_target = fm.compute_v_target(x, noise)
        v_pred = v_target.clone()

        loss = fm.compute_loss(v_pred, x, noise)

        assert torch.isclose(loss, torch.tensor(0.0, device=device), atol=1e-5), (
            f"Loss should be ~0 when prediction is perfect, got {loss}"
        )

    def test_loss_scales_with_error(
        self,
        sample_images: torch.Tensor,
        sample_noise: torch.Tensor,
        device: torch.device
    ):
        """Larger prediction errors should result in larger losses."""
        fm = PixelHDMFlowMatching()

        x = sample_images[:1]
        noise = sample_noise[:1]

        # Get correct v_target
        v_target = fm.compute_v_target(x, noise)

        # For deterministic test, use fixed perturbation
        v_pred_1 = v_target + 0.1  # Small error
        v_pred_2 = v_target + 0.5  # Large error

        loss_1 = fm.compute_loss(v_pred_1, x, noise)
        loss_2 = fm.compute_loss(v_pred_2, x, noise)

        assert loss_2 > loss_1, (
            f"Larger error should give larger loss: {loss_1} vs {loss_2}"
        )

    def test_loss_reduction_mean(
        self,
        sample_images: torch.Tensor,
        sample_noise: torch.Tensor
    ):
        """Test 'mean' reduction mode."""
        fm = PixelHDMFlowMatching()

        v_pred = torch.randn_like(sample_images)

        loss = fm.compute_loss(
            v_pred, sample_images, sample_noise,
            reduction="mean"
        )

        # Mean reduction should return scalar
        assert loss.dim() == 0, "Mean reduction should return scalar"

    def test_loss_reduction_sum(
        self,
        sample_images: torch.Tensor,
        sample_noise: torch.Tensor
    ):
        """Test 'sum' reduction mode."""
        fm = PixelHDMFlowMatching()

        v_pred = torch.randn_like(sample_images)

        loss_sum = fm.compute_loss(
            v_pred, sample_images, sample_noise,
            reduction="sum"
        )
        loss_mean = fm.compute_loss(
            v_pred, sample_images, sample_noise,
            reduction="mean"
        )

        # Sum should be larger than mean (for batch > 1)
        numel = sample_images.numel()
        assert torch.isclose(loss_sum, loss_mean * numel, rtol=1e-4), (
            f"Sum ({loss_sum}) should equal mean ({loss_mean}) * numel ({numel})"
        )

    def test_loss_reduction_none(
        self,
        sample_images: torch.Tensor,
        sample_noise: torch.Tensor
    ):
        """Test 'none' reduction mode (no reduction)."""
        fm = PixelHDMFlowMatching()

        v_pred = torch.randn_like(sample_images)

        loss = fm.compute_loss(
            v_pred, sample_images, sample_noise,
            reduction="none"
        )

        # No reduction should preserve shape
        assert loss.shape == sample_images.shape, (
            f"None reduction should preserve shape: {loss.shape} vs {sample_images.shape}"
        )


# =============================================================================
# Test: Prepare Training
# =============================================================================

class TestPrepareTraining:
    """Test the prepare_training method that combines multiple operations."""

    def test_prepare_training_returns_correct_shapes(
        self,
        sample_images: torch.Tensor
    ):
        """prepare_training should return correctly shaped outputs."""
        fm = PixelHDMFlowMatching()

        t, z_t, x, noise = fm.prepare_training(sample_images)

        assert t.shape == (sample_images.shape[0],), f"t shape: {t.shape}"
        assert z_t.shape == sample_images.shape, f"z_t shape: {z_t.shape}"
        assert x.shape == sample_images.shape, f"x shape: {x.shape}"
        assert noise.shape == sample_images.shape, f"noise shape: {noise.shape}"

    def test_prepare_training_with_custom_noise(
        self,
        sample_images: torch.Tensor,
        sample_noise: torch.Tensor
    ):
        """Custom noise should be used when provided."""
        fm = PixelHDMFlowMatching()

        t, z_t, x, noise = fm.prepare_training(sample_images, noise=sample_noise)

        assert torch.allclose(noise, sample_noise), "Custom noise should be used"

    def test_prepare_training_with_custom_t(
        self,
        sample_images: torch.Tensor
    ):
        """Custom timesteps should be used when provided."""
        fm = PixelHDMFlowMatching()

        custom_t = torch.tensor([0.2, 0.4, 0.6, 0.8], device=sample_images.device)
        t, z_t, x, noise = fm.prepare_training(sample_images, t=custom_t)

        assert torch.allclose(t, custom_t), "Custom t should be used"


# =============================================================================
# Test: Forward Method (Full Training Step)
# =============================================================================

class TestForwardMethod:
    """Test the complete forward pass with a model."""

    def test_forward_returns_loss_and_metrics(
        self,
        sample_images: torch.Tensor
    ):
        """forward() should return loss and metrics dict."""
        fm = PixelHDMFlowMatching()

        # Mock model that returns v_pred same shape as input (V-Prediction)
        def mock_model(z_t, t, **kwargs):
            return z_t.clone()  # Simple identity-like model

        loss, metrics = fm.forward(mock_model, sample_images)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "v_pred_mean" in metrics  # V-Prediction
        assert "v_pred_std" in metrics   # V-Prediction
        assert "t_mean" in metrics

    def test_forward_with_text_embeddings(
        self,
        sample_images: torch.Tensor
    ):
        """forward() should pass text_embeddings to model."""
        fm = PixelHDMFlowMatching()

        received_kwargs = {}

        def mock_model(z_t, t, **kwargs):
            received_kwargs.update(kwargs)
            return z_t.clone()

        text_emb = torch.randn(4, 10, 1024, device=sample_images.device)
        fm.forward(mock_model, sample_images, text_embeddings=text_emb)

        assert "text_embeddings" in received_kwargs
        assert torch.allclose(received_kwargs["text_embeddings"], text_emb)


# =============================================================================
# Test: PixelHDM Sampler
# =============================================================================

class TestPixelHDMSampler:
    """Test the PixelHDMSampler class for inference."""

    def test_sampler_timesteps_ascending(self):
        """JiT sampling timesteps should be ascending (0 to 1)."""
        sampler = PixelHDMSampler(num_steps=10)

        timesteps = sampler.get_timesteps()

        # All steps ascending
        for i in range(len(timesteps) - 1):
            assert timesteps[i] < timesteps[i + 1], (
                f"Timesteps should be ascending: {timesteps[i]} >= {timesteps[i+1]}"
            )

    def test_sampler_timesteps_range(self):
        """Timesteps should be in [t_eps, 1-t_eps]."""
        sampler = PixelHDMSampler(num_steps=50, t_eps=0.05)

        timesteps = sampler.get_timesteps()

        assert timesteps[0] == pytest.approx(0.05, abs=1e-6)
        assert timesteps[-1] == pytest.approx(0.95, abs=1e-6)

    def test_sampler_v_to_x_formula(
        self,
        sample_images: torch.Tensor,
        sample_noise: torch.Tensor
    ):
        """v_to_x should compute x = z + (1 - t) * v (V-Prediction)."""
        sampler = PixelHDMSampler()

        t = torch.tensor([0.5], device=sample_images.device)
        v_pred = sample_images[:1]  # Use images as v_pred for test
        z = sample_noise[:1]

        x = sampler.v_to_x(v_pred, z, t)

        # x = z + (1 - t) * v
        expected = z + (1 - 0.5) * v_pred
        assert torch.allclose(x, expected.float(), atol=1e-5)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
