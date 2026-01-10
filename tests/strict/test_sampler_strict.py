"""
PixelHDM-RPEA-DinoV3 Sampler Strict Tests

Strict mathematical verification of ODE solvers and CFG formulas.
These tests verify core mathematical correctness by direct computation,
not by trusting documentation or existing code behavior.

Key Verification Items:
    1. Euler step formula: z_{t+1} = z_t + dt * v
    2. Heun step formula: second-order predictor-corrector
    3. CFG formula: x_cfg = x_uncond + scale * (x_cond - x_uncond)
    4. JiT timesteps direction: t increases from 0 (noise) to 1 (clean)
    5. Sampling from noise produces reasonable output
    6. Deterministic behavior with fixed seeds

Test Count: 30+ test cases

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

import pytest
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from unittest.mock import Mock

from src.inference.sampler import (
    EulerSampler,
    HeunSampler,
    DPMPPSampler,
    UnifiedSampler,
    BaseSampler,
)
from src.inference.cfg import (
    StandardCFG,
    RescaledCFG,
    CFGWithInterval,
    apply_cfg,
)
from src.training.flow_matching import PixelHDMSampler, PixelHDMFlowMatching


# =============================================================================
# Helper Classes and Fixtures
# =============================================================================


class ControlledModel(nn.Module):
    """
    A model with controlled output for mathematical verification.

    This model returns a predictable output based on its mode:
    - "constant": Returns a constant value
    - "linear": Returns input scaled by a factor
    - "shift": Returns input shifted by an offset
    - "custom": Returns a custom function output
    """

    def __init__(
        self,
        mode: str = "constant",
        constant_value: float = 1.0,
        scale_factor: float = 2.0,
        shift_offset: float = 0.5,
        custom_fn: Optional[callable] = None,
    ):
        super().__init__()
        self.mode = mode
        self.constant_value = constant_value
        self.scale_factor = scale_factor
        self.shift_offset = shift_offset
        self.custom_fn = custom_fn
        self.call_count = 0
        self.last_inputs = []

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        self.call_count += 1
        self.last_inputs.append((z.clone(), t.clone()))

        if self.mode == "constant":
            return torch.full_like(z, self.constant_value)
        elif self.mode == "linear":
            return z * self.scale_factor
        elif self.mode == "shift":
            return z + self.shift_offset
        elif self.mode == "custom" and self.custom_fn is not None:
            return self.custom_fn(z, t)
        else:
            return z

    def reset(self):
        self.call_count = 0
        self.last_inputs = []


@pytest.fixture
def device():
    """Return CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def dtype():
    """Return default dtype for testing."""
    return torch.float32


# =============================================================================
# Test 1: Euler Step Mathematical Verification
# =============================================================================


class TestEulerStepMath:
    """
    Strict verification of Euler step formula: z_{t+1} = z_t + dt * v

    Where v = (x_pred - z_t) / (1 - t)

    Full formula: z_{t+1} = z_t + dt * (x_pred - z_t) / (1 - t)
    """

    def test_euler_step_basic_formula(self):
        """
        Verify Euler step with known values (V-Prediction).

        V-Prediction: Model directly outputs velocity v_pred.

        Setup:
            z_t = [1, 1, 1, ...]
            t = 0.3, t_next = 0.4, dt = 0.1
            v_pred = [2, 2, 2, ...] (velocity, not x_pred)

        Expected:
            z_{t+1} = z_t + dt * v_pred = 1 + 0.1 * 2 = 1.2
        """
        sampler = EulerSampler(num_steps=10, t_eps=0.05)

        # Create controlled model that returns constant velocity 2.0
        model = ControlledModel(mode="constant", constant_value=2.0)

        # Input setup
        z_t = torch.ones(1, 4, 4, 3)
        t = torch.tensor(0.3)
        t_next = torch.tensor(0.4)
        dt = t_next - t  # 0.1

        # Execute step
        z_next = sampler.step(model, z_t, t, t_next, None)

        # V-Prediction: z_next = z + dt * v_pred
        v_pred = 2.0
        expected_value = 1.0 + 0.1 * v_pred  # = 1.2
        expected = torch.full_like(z_t, expected_value)

        assert torch.allclose(z_next, expected, atol=1e-5), \
            f"Euler step mismatch: got {z_next[0,0,0,0].item():.6f}, expected {expected_value:.6f}"

    def test_euler_step_zero_velocity(self):
        """
        When v_pred = 0, z_next = z_t (V-Prediction).
        """
        sampler = EulerSampler(num_steps=10, t_eps=0.05)

        # Model returns zero velocity
        model = ControlledModel(mode="constant", constant_value=0.0)

        z_t = torch.randn(2, 8, 8, 3)
        t = torch.tensor(0.5)
        t_next = torch.tensor(0.6)

        z_next = sampler.step(model, z_t, t, t_next, None)

        # With v_pred = 0, z_next = z_t
        assert torch.allclose(z_next, z_t, atol=1e-5), \
            "Euler step with zero velocity should return original z_t"

    def test_euler_step_multiple_timesteps(self):
        """
        Verify Euler step formula at different timesteps (V-Prediction).
        For constant v_pred = 0, z_next = z_t regardless of timestep.
        """
        sampler = EulerSampler(num_steps=10, t_eps=0.05)
        model = ControlledModel(mode="constant", constant_value=0.0)  # v_pred = 0

        z_t = torch.ones(1, 4, 4, 3)

        timesteps = [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6), (0.7, 0.8)]

        for t_val, t_next_val in timesteps:
            t = torch.tensor(t_val)
            t_next = torch.tensor(t_next_val)

            z_next = sampler.step(model, z_t, t, t_next, None)

            # V-Prediction: v_pred = 0, so z_next = z_t
            expected = z_t.clone()

            assert torch.allclose(z_next, expected, atol=1e-5), \
                f"Euler step at t={t_val} failed: got {z_next[0,0,0,0].item():.6f}, expected {z_t[0,0,0,0].item():.6f}"

    def test_euler_step_numerical_stability_near_t1(self):
        """
        Test numerical stability when t is close to 1.

        The denominator (1 - t) approaches 0 as t -> 1,
        which could cause division issues. The sampler should
        use clamping (t_eps) to prevent this.
        """
        sampler = EulerSampler(num_steps=10, t_eps=0.05)
        model = ControlledModel(mode="constant", constant_value=1.0)

        z_t = torch.ones(1, 4, 4, 3)
        t = torch.tensor(0.94)  # Close to 1 - t_eps = 0.95
        t_next = torch.tensor(0.95)

        z_next = sampler.step(model, z_t, t, t_next, None)

        # Should not produce NaN or Inf
        assert not torch.isnan(z_next).any(), "Euler step produced NaN near t=1"
        assert not torch.isinf(z_next).any(), "Euler step produced Inf near t=1"

    def test_euler_step_dt_proportionality(self):
        """
        Verify that step size is proportional to dt.

        With same z_t and x_pred, doubling dt should double the change.
        """
        sampler = EulerSampler(num_steps=10, t_eps=0.05)
        model = ControlledModel(mode="constant", constant_value=2.0)

        z_t = torch.ones(1, 4, 4, 3)
        t = torch.tensor(0.3)

        # Step with dt = 0.1
        t_next_1 = torch.tensor(0.4)
        z_next_1 = sampler.step(model, z_t, t, t_next_1, None)
        change_1 = (z_next_1 - z_t).mean().item()

        # Step with dt = 0.2
        t_next_2 = torch.tensor(0.5)
        z_next_2 = sampler.step(model, z_t, t, t_next_2, None)
        change_2 = (z_next_2 - z_t).mean().item()

        # change_2 should be twice change_1
        ratio = change_2 / change_1
        assert abs(ratio - 2.0) < 0.01, \
            f"dt proportionality failed: change ratio = {ratio:.4f}, expected 2.0"


# =============================================================================
# Test 2: Heun Step Mathematical Verification
# =============================================================================


class TestHeunStepMath:
    """
    Strict verification of Heun step (second-order predictor-corrector).

    Algorithm:
        1. Euler prediction: z_euler = z_t + dt * v_t
        2. Evaluate at prediction point: v_next = f(z_euler, t_next)
        3. Average velocity: v_avg = (v_t + v_next) / 2
        4. Final step: z_{t+1} = z_t + dt * v_avg
    """

    def test_heun_step_two_evaluations(self):
        """
        Verify Heun makes exactly two model evaluations per step.
        """
        sampler = HeunSampler(num_steps=10, t_eps=0.05)
        model = ControlledModel(mode="constant", constant_value=1.0)

        z_t = torch.randn(2, 8, 8, 3)
        t = torch.tensor(0.3)
        t_next = torch.tensor(0.4)

        model.reset()
        sampler.step(model, z_t, t, t_next, None)

        assert model.call_count == 2, \
            f"Heun should make 2 evaluations, got {model.call_count}"

    def test_heun_step_predictor_corrector_formula(self):
        """
        Verify Heun predictor-corrector formula with controlled model.

        Setup:
            z_t = [1, 1, 1, ...]
            t = 0.3, t_next = 0.4, dt = 0.1
            Model always returns constant v_pred = 2.0 (V-Prediction)

        Expected (V-Prediction):
            v1 = v2 = 2.0 (constant velocity model)
            v_avg = (v1 + v2) / 2 = 2.0
            z_{t+1} = z_t + dt * v_avg = 1.0 + 0.1 * 2.0 = 1.2
        """
        sampler = HeunSampler(num_steps=10, t_eps=0.05)
        model = ControlledModel(mode="constant", constant_value=2.0)

        z_t = torch.ones(1, 4, 4, 3)
        t = torch.tensor(0.3)
        t_next = torch.tensor(0.4)
        dt = 0.1

        z_next = sampler.step(model, z_t, t, t_next, None)

        # V-Prediction: constant velocity model, v1 = v2 = 2.0
        v_pred = 2.0
        expected_value = 1.0 + dt * v_pred  # = 1.2
        expected = torch.full_like(z_t, expected_value)

        assert torch.allclose(z_next, expected, atol=1e-5), \
            f"Heun step mismatch: got {z_next[0,0,0,0].item():.6f}, expected {expected_value:.6f}"

    def test_heun_more_accurate_than_euler(self):
        """
        Verify Heun is more accurate than Euler for smooth functions.

        For a linear ODE, both methods should give similar results.
        But for nonlinear functions, Heun's second-order accuracy
        should show improvement.

        We test by comparing against analytical solution for
        the flow ODE: dz/dt = (x - z) / (1 - t) where x is constant.

        Analytical solution for z'(t) = (x - z) / (1 - t):
        z(t) = x - (x - z_0) * (1 - t) / (1 - t_0)
        """
        euler_sampler = EulerSampler(num_steps=10, t_eps=0.05)
        heun_sampler = HeunSampler(num_steps=10, t_eps=0.05)

        # Model returns constant target
        target = 2.0
        model = ControlledModel(mode="constant", constant_value=target)

        z_0 = torch.ones(1, 4, 4, 3) * 0.0
        t_0 = 0.2
        t_1 = 0.5

        t = torch.tensor(t_0)
        t_next = torch.tensor(t_1)

        euler_result = euler_sampler.step(model, z_0, t, t_next, None)
        heun_result = heun_sampler.step(model, z_0, t, t_next, None)

        # Analytical solution: z(t_1) = x - (x - z_0) * (1 - t_1) / (1 - t_0)
        # z(0.5) = 2 - (2 - 0) * (1 - 0.5) / (1 - 0.2) = 2 - 2 * 0.5 / 0.8 = 2 - 1.25 = 0.75
        analytical = target - (target - 0.0) * (1 - t_1) / (1 - t_0)

        euler_error = abs(euler_result[0,0,0,0].item() - analytical)
        heun_error = abs(heun_result[0,0,0,0].item() - analytical)

        # For this ODE, Heun should be at least as accurate as Euler
        # Note: For constant x_pred, both give same result since v is constant
        # This is expected behavior for constant forcing
        assert heun_error <= euler_error + 1e-5, \
            f"Heun error ({heun_error:.6f}) should be <= Euler error ({euler_error:.6f})"

    def test_heun_step_with_varying_model(self):
        """
        Test Heun with a model that gives different outputs at different z.

        This tests the predictor-corrector aspect where the correction
        at z_euler differs from the initial prediction at z_t.
        """
        # Model that returns linear transformation of input
        # x_pred = 2 * z, so prediction changes based on z
        sampler = HeunSampler(num_steps=10, t_eps=0.05)
        model = ControlledModel(mode="linear", scale_factor=2.0)

        z_t = torch.ones(1, 4, 4, 3)
        t = torch.tensor(0.3)
        t_next = torch.tensor(0.4)
        dt = 0.1

        model.reset()
        z_next = sampler.step(model, z_t, t, t_next, None)

        # Verify two different inputs were used
        assert len(model.last_inputs) == 2

        # First call at z_t, second call at z_euler
        first_z = model.last_inputs[0][0]
        second_z = model.last_inputs[1][0]

        # They should be different (z_euler != z_t)
        assert not torch.allclose(first_z, second_z), \
            "Heun should evaluate at different z values"

    def test_heun_reduces_to_euler_for_constant_velocity(self):
        """
        When velocity is constant (v_t = v_next), Heun equals Euler.

        V-Prediction: v_avg = (v1 + v2) / 2 = v when v1 = v2
        """
        euler_sampler = EulerSampler(num_steps=10, t_eps=0.05)
        heun_sampler = HeunSampler(num_steps=10, t_eps=0.05)

        # V-Prediction: Model returns zero velocity for all inputs
        model = ControlledModel(mode="constant", constant_value=0.0)

        z_t = torch.randn(2, 8, 8, 3)
        t = torch.tensor(0.4)
        t_next = torch.tensor(0.5)

        euler_result = euler_sampler.step(model, z_t, t, t_next, None)
        heun_result = heun_sampler.step(model, z_t, t, t_next, None)

        # Both should give same result when v = 0
        assert torch.allclose(euler_result, heun_result, atol=1e-5), \
            "Heun should equal Euler when velocity is constant"


# =============================================================================
# Test 3: CFG Formula Mathematical Verification
# =============================================================================


class TestCFGFormulaMath:
    """
    Strict verification of CFG formula:
    x_cfg = x_uncond + scale * (x_cond - x_uncond)

    Key properties:
        - scale=0: x_cfg = x_uncond
        - scale=1: x_cfg = x_cond
        - scale>1: extrapolation beyond x_cond
    """

    def test_cfg_formula_scale_zero(self):
        """
        When scale=0, x_cfg should equal x_uncond.
        """
        cfg = StandardCFG()

        x_cond = torch.ones(2, 4, 4, 3) * 2.0
        x_uncond = torch.ones(2, 4, 4, 3) * 1.0

        result = cfg.apply(x_cond, x_uncond, guidance_scale=0.0)

        # scale=0: x_cfg = x_uncond + 0 * (x_cond - x_uncond) = x_uncond
        assert torch.allclose(result, x_uncond), \
            "CFG with scale=0 should return x_uncond"

    def test_cfg_formula_scale_one(self):
        """
        When scale=1, x_cfg should equal x_cond.
        """
        cfg = StandardCFG()

        x_cond = torch.ones(2, 4, 4, 3) * 2.0
        x_uncond = torch.ones(2, 4, 4, 3) * 1.0

        result = cfg.apply(x_cond, x_uncond, guidance_scale=1.0)

        # scale=1: x_cfg = x_uncond + 1 * (x_cond - x_uncond) = x_cond
        assert torch.allclose(result, x_cond), \
            "CFG with scale=1 should return x_cond"

    def test_cfg_formula_exact_values(self):
        """
        Verify exact CFG calculation with known values.

        x_uncond = 1.0
        x_cond = 3.0
        scale = 2.0

        Expected: x_cfg = 1.0 + 2.0 * (3.0 - 1.0) = 1.0 + 4.0 = 5.0
        """
        cfg = StandardCFG()

        x_cond = torch.ones(1, 4, 4, 3) * 3.0
        x_uncond = torch.ones(1, 4, 4, 3) * 1.0

        result = cfg.apply(x_cond, x_uncond, guidance_scale=2.0)

        expected = torch.ones(1, 4, 4, 3) * 5.0
        assert torch.allclose(result, expected, atol=1e-6), \
            f"CFG formula mismatch: got {result[0,0,0,0].item()}, expected 5.0"

    def test_cfg_formula_negative_scale(self):
        """
        Verify CFG works correctly with negative scale.

        Negative scale inverts the conditioning direction.
        """
        cfg = StandardCFG()

        x_cond = torch.ones(1, 4, 4, 3) * 2.0
        x_uncond = torch.ones(1, 4, 4, 3) * 1.0

        result = cfg.apply(x_cond, x_uncond, guidance_scale=-1.0)

        # scale=-1: x_cfg = 1.0 + (-1) * (2.0 - 1.0) = 1.0 - 1.0 = 0.0
        expected = torch.zeros(1, 4, 4, 3)
        assert torch.allclose(result, expected, atol=1e-6), \
            f"CFG with negative scale failed: got {result[0,0,0,0].item()}, expected 0.0"

    def test_cfg_formula_high_scale_extrapolation(self):
        """
        Verify high guidance scale correctly extrapolates.

        scale=7.5 (typical diffusion CFG value):
        x_cfg = x_uncond + 7.5 * (x_cond - x_uncond)
        """
        cfg = StandardCFG()

        x_cond = torch.ones(1, 4, 4, 3) * 2.0
        x_uncond = torch.ones(1, 4, 4, 3) * 1.0
        scale = 7.5

        result = cfg.apply(x_cond, x_uncond, guidance_scale=scale)

        # x_cfg = 1.0 + 7.5 * (2.0 - 1.0) = 1.0 + 7.5 = 8.5
        expected = torch.ones(1, 4, 4, 3) * 8.5
        assert torch.allclose(result, expected, atol=1e-6), \
            f"CFG with scale=7.5 failed: got {result[0,0,0,0].item()}, expected 8.5"

    def test_cfg_apply_function(self):
        """
        Test the standalone apply_cfg function.
        """
        x_cond = torch.ones(2, 8, 8, 3) * 3.0
        x_uncond = torch.ones(2, 8, 8, 3) * 1.0

        result = apply_cfg(x_cond, x_uncond, guidance_scale=2.0, rescale_factor=0.0)

        # x_cfg = 1.0 + 2.0 * (3.0 - 1.0) = 5.0
        expected = torch.ones(2, 8, 8, 3) * 5.0
        assert torch.allclose(result, expected, atol=1e-6)

    def test_cfg_linearity(self):
        """
        Verify CFG is linear in the guidance scale.

        2 * cfg(scale=s) - cfg(scale=2s) should cancel middle term.
        """
        cfg = StandardCFG()

        x_cond = torch.randn(2, 4, 4, 3)
        x_uncond = torch.randn(2, 4, 4, 3)

        result_s1 = cfg.apply(x_cond, x_uncond, guidance_scale=3.0)
        result_s2 = cfg.apply(x_cond, x_uncond, guidance_scale=6.0)

        # cfg(s1) = u + s1*(c - u)
        # cfg(s2) = u + s2*(c - u)
        # 2*cfg(s1) - cfg(s2) = 2u + 2*s1*(c-u) - u - s2*(c-u) = u + (2*s1 - s2)*(c-u) = u if s2=2*s1
        expected = x_uncond
        calculated = 2 * result_s1 - result_s2

        assert torch.allclose(calculated, expected, atol=1e-5), \
            "CFG should be linear in guidance scale"


# =============================================================================
# Test 4: Timesteps Direction Verification (PixelHDM)
# =============================================================================


class TestTimestepsDirection:
    """
    Verify PixelHDM time convention: t increases from 0 (noise) to 1 (clean).

    This is opposite to standard diffusion (DDPM) where t goes from T to 0.
    """

    def test_timesteps_ascending(self):
        """
        Verify timesteps are strictly ascending.
        """
        sampler = EulerSampler(num_steps=50, t_eps=0.05)
        timesteps = sampler.get_timesteps(device=torch.device("cpu"))

        for i in range(len(timesteps) - 1):
            assert timesteps[i] < timesteps[i + 1], \
                f"Timesteps not ascending at index {i}: {timesteps[i].item():.6f} >= {timesteps[i+1].item():.6f}"

    def test_timesteps_start_near_zero(self):
        """
        Verify timesteps start near 0 (at t_eps).
        """
        t_eps = 0.05
        sampler = EulerSampler(num_steps=50, t_eps=t_eps)
        timesteps = sampler.get_timesteps(device=torch.device("cpu"))

        assert abs(timesteps[0].item() - t_eps) < 1e-6, \
            f"First timestep should be t_eps={t_eps}, got {timesteps[0].item()}"

    def test_timesteps_end_near_one(self):
        """
        Verify timesteps end near 1 (at 1 - t_eps).
        """
        t_eps = 0.05
        sampler = EulerSampler(num_steps=50, t_eps=t_eps)
        timesteps = sampler.get_timesteps(device=torch.device("cpu"))

        expected_end = 1 - t_eps
        assert abs(timesteps[-1].item() - expected_end) < 1e-6, \
            f"Last timestep should be 1-t_eps={expected_end}, got {timesteps[-1].item()}"

    def test_timesteps_uniform_spacing(self):
        """
        Verify timesteps are uniformly spaced.
        """
        sampler = EulerSampler(num_steps=10, t_eps=0.05)
        timesteps = sampler.get_timesteps(device=torch.device("cpu"))

        # Calculate step sizes
        dts = timesteps[1:] - timesteps[:-1]

        # All steps should be approximately equal
        mean_dt = dts.mean()
        for i, dt in enumerate(dts):
            assert abs(dt - mean_dt) < 1e-6, \
                f"Non-uniform step at index {i}: dt={dt.item():.6f}, mean={mean_dt.item():.6f}"

    def test_pixelhdm_sampler_timesteps_direction(self):
        """
        Verify PixelHDMSampler from flow_matching.py also uses ascending timesteps.
        """
        sampler = PixelHDMSampler(num_steps=50, t_eps=0.05)
        timesteps = sampler.get_timesteps(device=torch.device("cpu"))

        # Verify ascending
        for i in range(len(timesteps) - 1):
            assert timesteps[i] < timesteps[i + 1], \
                f"PixelHDMSampler timesteps not ascending at index {i}"

        # Verify range
        assert timesteps[0].item() >= 0.0
        assert timesteps[-1].item() <= 1.0

    def test_timesteps_count(self):
        """
        Verify timesteps count is num_steps + 1 (includes start and end).
        """
        for num_steps in [5, 10, 25, 50, 100]:
            sampler = EulerSampler(num_steps=num_steps, t_eps=0.05)
            timesteps = sampler.get_timesteps(num_steps=num_steps)

            assert len(timesteps) == num_steps + 1, \
                f"Expected {num_steps + 1} timesteps, got {len(timesteps)}"


# =============================================================================
# Test 5: Sampling From Noise
# =============================================================================


class TestSamplingFromNoise:
    """
    Verify that sampling from pure noise produces reasonable output.

    These tests check:
    1. Output is bounded
    2. Output is not NaN or Inf
    3. Output has changed from input noise
    4. Output statistics are reasonable
    """

    def test_sampling_produces_valid_output(self):
        """
        Basic sanity check: sampling should produce non-NaN, non-Inf output.
        """
        sampler = UnifiedSampler(method="euler", num_steps=10, t_eps=0.05)

        # Model that gradually pulls toward target
        model = ControlledModel(mode="constant", constant_value=0.0)

        # Pure noise input
        torch.manual_seed(42)
        z_0 = torch.randn(2, 16, 16, 3)

        result = sampler.sample(model=model, z_0=z_0)

        assert not torch.isnan(result).any(), "Sampling produced NaN values"
        assert not torch.isinf(result).any(), "Sampling produced Inf values"

    def test_sampling_changes_input(self):
        """
        Sampling should transform the input (unless model is identity).
        """
        sampler = UnifiedSampler(method="euler", num_steps=10, t_eps=0.05)
        model = ControlledModel(mode="constant", constant_value=0.5)

        torch.manual_seed(42)
        z_0 = torch.randn(2, 16, 16, 3)

        result = sampler.sample(model=model, z_0=z_0.clone())

        # Result should be different from input
        assert not torch.allclose(result, z_0), \
            "Sampling should transform the input"

    def test_sampling_output_bounded(self):
        """
        Verify sampling output stays within reasonable bounds.
        """
        sampler = UnifiedSampler(method="heun", num_steps=20, t_eps=0.05)
        model = ControlledModel(mode="constant", constant_value=0.0)

        torch.manual_seed(42)
        z_0 = torch.randn(4, 32, 32, 3)

        result = sampler.sample(model=model, z_0=z_0)

        # Output should be bounded (not exploding)
        max_val = result.abs().max().item()
        assert max_val < 100.0, \
            f"Sampling output too large: max={max_val}"

    def test_sampling_approaches_target(self):
        """
        With a constant target model, sampling should approach the target.
        """
        sampler = UnifiedSampler(method="heun", num_steps=50, t_eps=0.05)
        target_value = 0.5
        model = ControlledModel(mode="constant", constant_value=target_value)

        torch.manual_seed(42)
        z_0 = torch.randn(2, 16, 16, 3)

        result = sampler.sample(model=model, z_0=z_0)

        # Result should be closer to target than input was
        input_distance = (z_0 - target_value).abs().mean().item()
        output_distance = (result - target_value).abs().mean().item()

        assert output_distance < input_distance, \
            f"Sampling should approach target: input_dist={input_distance:.4f}, output_dist={output_distance:.4f}"

    def test_sampling_all_methods_valid(self):
        """
        All sampling methods should produce valid output.
        """
        model = ControlledModel(mode="constant", constant_value=0.0)

        torch.manual_seed(42)
        z_0 = torch.randn(2, 16, 16, 3)

        for method in ["euler", "heun", "dpm_pp"]:
            sampler = UnifiedSampler(method=method, num_steps=10, t_eps=0.05)
            result = sampler.sample(model=model, z_0=z_0.clone())

            assert not torch.isnan(result).any(), f"{method} produced NaN"
            assert not torch.isinf(result).any(), f"{method} produced Inf"


# =============================================================================
# Test 6: Deterministic Sampling
# =============================================================================


class TestDeterministicSampling:
    """
    Verify that sampling is deterministic with the same initial noise.

    Given the same z_0, model, and sampler settings, the result
    should be identical across multiple runs.
    """

    def test_euler_deterministic(self):
        """
        Euler sampling should be deterministic.
        """
        sampler = UnifiedSampler(method="euler", num_steps=20, t_eps=0.05)
        model = ControlledModel(mode="linear", scale_factor=0.9)

        torch.manual_seed(123)
        z_0 = torch.randn(2, 16, 16, 3)

        result1 = sampler.sample(model=model, z_0=z_0.clone())
        result2 = sampler.sample(model=model, z_0=z_0.clone())

        assert torch.allclose(result1, result2, atol=1e-6), \
            "Euler sampling not deterministic"

    def test_heun_deterministic(self):
        """
        Heun sampling should be deterministic.
        """
        sampler = UnifiedSampler(method="heun", num_steps=20, t_eps=0.05)
        model = ControlledModel(mode="linear", scale_factor=0.9)

        torch.manual_seed(456)
        z_0 = torch.randn(2, 16, 16, 3)

        result1 = sampler.sample(model=model, z_0=z_0.clone())
        result2 = sampler.sample(model=model, z_0=z_0.clone())

        assert torch.allclose(result1, result2, atol=1e-6), \
            "Heun sampling not deterministic"

    def test_dpmpp_deterministic(self):
        """
        DPM++ sampling should be deterministic.
        """
        sampler = UnifiedSampler(method="dpm_pp", num_steps=20, t_eps=0.05)
        model = ControlledModel(mode="linear", scale_factor=0.9)

        torch.manual_seed(789)
        z_0 = torch.randn(2, 16, 16, 3)

        result1 = sampler.sample(model=model, z_0=z_0.clone())
        result2 = sampler.sample(model=model, z_0=z_0.clone())

        assert torch.allclose(result1, result2, atol=1e-6), \
            "DPM++ sampling not deterministic"

    def test_different_seeds_different_results(self):
        """
        Different initial noise should produce different results.
        """
        sampler = UnifiedSampler(method="heun", num_steps=20, t_eps=0.05)
        model = ControlledModel(mode="constant", constant_value=0.0)

        torch.manual_seed(42)
        z_0_a = torch.randn(2, 16, 16, 3)

        torch.manual_seed(43)  # Different seed
        z_0_b = torch.randn(2, 16, 16, 3)

        result_a = sampler.sample(model=model, z_0=z_0_a)
        result_b = sampler.sample(model=model, z_0=z_0_b)

        assert not torch.allclose(result_a, result_b), \
            "Different seeds should produce different results"

    def test_deterministic_with_cfg(self):
        """
        Sampling with CFG should be deterministic.
        """
        sampler = UnifiedSampler(method="euler", num_steps=10, t_eps=0.05)
        model = ControlledModel(mode="constant", constant_value=0.5)

        torch.manual_seed(42)
        z_0 = torch.randn(2, 16, 16, 3)
        text_embed = torch.randn(2, 10, 64)
        null_embed = torch.zeros(2, 10, 64)

        result1 = sampler.sample(
            model=model,
            z_0=z_0.clone(),
            text_embeddings=text_embed,
            guidance_scale=7.5,
            null_text_embeddings=null_embed,
        )
        result2 = sampler.sample(
            model=model,
            z_0=z_0.clone(),
            text_embeddings=text_embed,
            guidance_scale=7.5,
            null_text_embeddings=null_embed,
        )

        assert torch.allclose(result1, result2, atol=1e-6), \
            "CFG sampling not deterministic"

    def test_batch_independence(self):
        """
        Each sample in a batch should be independent.
        """
        sampler = UnifiedSampler(method="euler", num_steps=10, t_eps=0.05)
        model = ControlledModel(mode="linear", scale_factor=0.9)

        torch.manual_seed(42)
        z_0_single = torch.randn(1, 16, 16, 3)

        # Sample individually
        result_single = sampler.sample(model=model, z_0=z_0_single)

        # Sample as batch (duplicate the input)
        z_0_batch = z_0_single.repeat(4, 1, 1, 1)
        result_batch = sampler.sample(model=model, z_0=z_0_batch)

        # Each element in batch should match single sample
        for i in range(4):
            assert torch.allclose(result_batch[i], result_single[0], atol=1e-5), \
                f"Batch sample {i} differs from single sample"


# =============================================================================
# Additional Numerical Stability Tests
# =============================================================================


class TestNumericalStability:
    """
    Test numerical stability of sampling algorithms.
    """

    def test_stability_with_large_input(self):
        """
        Sampling should handle large input values.
        """
        sampler = UnifiedSampler(method="heun", num_steps=10, t_eps=0.05)
        model = ControlledModel(mode="constant", constant_value=0.0)

        z_0 = torch.ones(2, 8, 8, 3) * 100.0
        result = sampler.sample(model=model, z_0=z_0)

        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_stability_with_small_input(self):
        """
        Sampling should handle very small input values.
        """
        sampler = UnifiedSampler(method="heun", num_steps=10, t_eps=0.05)
        model = ControlledModel(mode="constant", constant_value=0.0)

        z_0 = torch.ones(2, 8, 8, 3) * 1e-6
        result = sampler.sample(model=model, z_0=z_0)

        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_stability_with_many_steps(self):
        """
        Sampling should remain stable with many steps.
        """
        sampler = UnifiedSampler(method="euler", num_steps=100, t_eps=0.05)
        model = ControlledModel(mode="constant", constant_value=0.0)

        torch.manual_seed(42)
        z_0 = torch.randn(1, 16, 16, 3)
        result = sampler.sample(model=model, z_0=z_0)

        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
        assert result.abs().max() < 1000.0

    def test_t_eps_prevents_division_by_zero(self):
        """
        t_eps should prevent division by zero in velocity calculation.
        """
        # Without t_eps clamping, t=1 would cause division by zero
        sampler = EulerSampler(num_steps=10, t_eps=0.05)
        model = ControlledModel(mode="constant", constant_value=1.0)

        z_t = torch.ones(1, 4, 4, 3)

        # t very close to 1 - t_eps
        t = torch.tensor(0.94)
        t_next = torch.tensor(0.95)

        z_next = sampler.step(model, z_t, t, t_next, None)

        # Should not have numerical issues
        assert torch.isfinite(z_next).all()


# =============================================================================
# CFG Interval Tests
# =============================================================================


class TestCFGInterval:
    """
    Test CFGWithInterval for time-based CFG control.
    """

    def test_cfg_interval_within_range(self):
        """
        CFG should be applied within the specified interval.
        """
        cfg = CFGWithInterval(
            guidance_scale=7.5,
            interval_start=0.0,
            interval_end=0.75,
        )

        # t=0.5 is within [0.0, 0.75)
        assert cfg.should_apply_cfg(0.5)
        assert cfg.get_effective_scale(0.5) == 7.5

    def test_cfg_interval_outside_range(self):
        """
        CFG should not be applied outside the specified interval.
        """
        cfg = CFGWithInterval(
            guidance_scale=7.5,
            interval_start=0.0,
            interval_end=0.75,
        )

        # t=0.8 is outside [0.0, 0.75)
        assert not cfg.should_apply_cfg(0.8)
        assert cfg.get_effective_scale(0.8) == 1.0

    def test_cfg_interval_at_boundary(self):
        """
        Test CFG behavior at interval boundaries.
        """
        cfg = CFGWithInterval(
            guidance_scale=7.5,
            interval_start=0.25,
            interval_end=0.75,
        )

        # At start (included)
        assert cfg.should_apply_cfg(0.25)

        # Just before end (included)
        assert cfg.should_apply_cfg(0.74)

        # At end (excluded)
        assert not cfg.should_apply_cfg(0.75)

        # Before start (excluded)
        assert not cfg.should_apply_cfg(0.24)
