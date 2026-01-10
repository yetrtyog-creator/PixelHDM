"""
PixelHDM-RPEA-DinoV3 VLoss Tests (V-Prediction)

Tests for V-Loss implementation based on PixelHDM paper.

V-Prediction Design:
    - v_pred = model(z_t, t)  # Network directly outputs velocity
    - v_target = x - noise
    - L = E[||v_pred - v_target||^2]

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.config.model_config import PixelHDMConfig
from src.training.losses.vloss import (
    VLoss,
    VLossWithVelocity,
    create_vloss,
    create_vloss_from_config,
    create_vloss_with_velocity,
)


class TestVLoss:
    """VLoss test suite (V-Prediction)."""

    # =========================================================================
    # Fixtures
    # =========================================================================

    @pytest.fixture
    def config(self) -> PixelHDMConfig:
        """Create test configuration."""
        return PixelHDMConfig.for_testing()

    @pytest.fixture
    def vloss(self, config: PixelHDMConfig) -> VLoss:
        """Create VLoss instance from config."""
        return VLoss(config=config)

    @pytest.fixture
    def vloss_default(self) -> VLoss:
        """Create VLoss instance with default parameters."""
        return VLoss(config=None, t_eps=0.05)

    @pytest.fixture
    def sample_tensors(self) -> dict:
        """Create sample tensors for testing."""
        torch.manual_seed(42)
        B, C, H, W = 2, 3, 64, 64
        return {
            "v_pred": torch.randn(B, C, H, W),
            "x_clean": torch.randn(B, C, H, W),
            "noise": torch.randn(B, C, H, W),
            "B": B,
            "C": C,
            "H": H,
            "W": W,
        }

    # =========================================================================
    # Core Formula Tests
    # =========================================================================

    def test_compute_v_target_formula(
        self, vloss: VLoss, sample_tensors: dict
    ) -> None:
        """
        Test v_target computation formula.

        v_target = x - noise

        Verifies that v_target equals the clean image minus noise.
        """
        x_clean = sample_tensors["x_clean"]
        noise = sample_tensors["noise"]

        v_target = vloss.compute_v_target(x_clean, noise)

        expected = x_clean - noise

        assert torch.allclose(v_target, expected, atol=1e-6), \
            f"v_target formula mismatch: got {v_target[0, 0, 0, :5]}, expected {expected[0, 0, 0, :5]}"

    # =========================================================================
    # Reduction Tests
    # =========================================================================

    def test_forward_returns_scalar(
        self, vloss: VLoss, sample_tensors: dict
    ) -> None:
        """
        Test that forward with reduction='mean' returns a scalar tensor.
        """
        loss = vloss(
            v_pred=sample_tensors["v_pred"],
            x_clean=sample_tensors["x_clean"],
            noise=sample_tensors["noise"],
            reduction="mean",
        )

        assert loss.dim() == 0, f"Expected scalar (0D), got {loss.dim()}D tensor"
        assert loss.numel() == 1, f"Expected 1 element, got {loss.numel()}"

    def test_forward_reduction_sum(
        self, vloss: VLoss, sample_tensors: dict
    ) -> None:
        """
        Test that forward with reduction='sum' returns a scalar tensor.
        """
        loss = vloss(
            v_pred=sample_tensors["v_pred"],
            x_clean=sample_tensors["x_clean"],
            noise=sample_tensors["noise"],
            reduction="sum",
        )

        assert loss.dim() == 0, f"Expected scalar (0D), got {loss.dim()}D tensor"

    def test_forward_reduction_none(
        self, vloss: VLoss, sample_tensors: dict
    ) -> None:
        """
        Test that forward with reduction='none' preserves input shape.
        """
        B, C, H, W = (
            sample_tensors["B"],
            sample_tensors["C"],
            sample_tensors["H"],
            sample_tensors["W"],
        )

        loss = vloss(
            v_pred=sample_tensors["v_pred"],
            x_clean=sample_tensors["x_clean"],
            noise=sample_tensors["noise"],
            reduction="none",
        )

        assert loss.shape == (B, C, H, W), \
            f"Expected shape {(B, C, H, W)}, got {loss.shape}"

    # =========================================================================
    # Numerical Stability Tests
    # =========================================================================

    def test_numerical_stability_large_values(
        self, vloss: VLoss
    ) -> None:
        """
        Test numerical stability with large input values.
        """
        B, C, H, W = 2, 3, 32, 32
        v_pred = torch.randn(B, C, H, W) * 100
        x_clean = torch.randn(B, C, H, W) * 100
        noise = torch.randn(B, C, H, W) * 100

        loss = vloss(
            v_pred=v_pred,
            x_clean=x_clean,
            noise=noise,
            reduction="mean",
        )

        assert not torch.isnan(loss), "Loss is NaN for large values"
        assert not torch.isinf(loss), "Loss is Inf for large values"
        assert loss >= 0, "Loss should be non-negative"

    def test_numerical_stability_small_values(
        self, vloss: VLoss
    ) -> None:
        """
        Test numerical stability with small input values.
        """
        B, C, H, W = 2, 3, 32, 32
        v_pred = torch.randn(B, C, H, W) * 1e-6
        x_clean = torch.randn(B, C, H, W) * 1e-6
        noise = torch.randn(B, C, H, W) * 1e-6

        loss = vloss(
            v_pred=v_pred,
            x_clean=x_clean,
            noise=noise,
            reduction="mean",
        )

        assert not torch.isnan(loss), "Loss is NaN for small values"
        assert not torch.isinf(loss), "Loss is Inf for small values"
        assert loss >= 0, "Loss should be non-negative"

    # =========================================================================
    # Precision Tests
    # =========================================================================

    def test_float32_computation(self, vloss: VLoss) -> None:
        """
        Test that float16 inputs use float32 for internal computation.

        This ensures numerical stability when using mixed precision training.
        """
        B, C, H, W = 2, 3, 32, 32
        v_pred = torch.randn(B, C, H, W, dtype=torch.float16)
        x_clean = torch.randn(B, C, H, W, dtype=torch.float16)
        noise = torch.randn(B, C, H, W, dtype=torch.float16)

        loss = vloss(
            v_pred=v_pred,
            x_clean=x_clean,
            noise=noise,
            reduction="mean",
        )

        # Loss should be returned in original dtype
        assert loss.dtype == torch.float16, \
            f"Expected float16 output, got {loss.dtype}"
        assert not torch.isnan(loss), "Loss is NaN with float16 input"

    # =========================================================================
    # Input Shape Tests
    # =========================================================================

    def test_input_4d_tensor(
        self, vloss: VLoss, sample_tensors: dict
    ) -> None:
        """
        Test that 4D tensors (B, C, H, W) are processed correctly.
        """
        loss = vloss(
            v_pred=sample_tensors["v_pred"],
            x_clean=sample_tensors["x_clean"],
            noise=sample_tensors["noise"],
            reduction="mean",
        )

        assert loss.dim() == 0, "Loss should be scalar"
        assert not torch.isnan(loss), "Loss is NaN"

    def test_batch_size_mismatch_raises(self, vloss: VLoss) -> None:
        """
        Test that mismatched batch sizes raise RuntimeError.
        """
        B1, B2, C, H, W = 2, 4, 3, 32, 32

        v_pred = torch.randn(B1, C, H, W)
        x_clean = torch.randn(B2, C, H, W)  # Different batch size
        noise = torch.randn(B1, C, H, W)

        with pytest.raises(RuntimeError):
            vloss(
                v_pred=v_pred,
                x_clean=x_clean,
                noise=noise,
                reduction="mean",
            )

    # =========================================================================
    # VLossWithVelocity Tests
    # =========================================================================

    def test_with_velocity_returns_tuple(
        self, config: PixelHDMConfig, sample_tensors: dict
    ) -> None:
        """
        Test that VLossWithVelocity.forward_with_velocity returns (loss, v_pred, v_target).
        """
        vloss_with_v = VLossWithVelocity(config=config)

        result = vloss_with_v.forward_with_velocity(
            v_pred=sample_tensors["v_pred"],
            x_clean=sample_tensors["x_clean"],
            noise=sample_tensors["noise"],
            reduction="mean",
        )

        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 3, f"Expected 3 elements, got {len(result)}"

        loss, v_pred_out, v_target = result

        assert loss.dim() == 0, "Loss should be scalar"
        assert v_pred_out.shape == sample_tensors["v_pred"].shape, \
            f"v_pred shape mismatch: {v_pred_out.shape} vs {sample_tensors['v_pred'].shape}"
        assert v_target.shape == sample_tensors["x_clean"].shape, \
            f"v_target shape mismatch: {v_target.shape} vs {sample_tensors['x_clean'].shape}"

    # =========================================================================
    # Special Case Tests
    # =========================================================================

    def test_perfect_prediction_loss_zero(self, vloss: VLoss) -> None:
        """
        Test that perfect prediction results in near-zero loss.

        When v_pred exactly equals v_target = x - noise, loss should be zero.
        """
        B, C, H, W = 2, 3, 32, 32

        # Create consistent data
        x_clean = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)

        # Perfect prediction: v_pred = v_target = x - noise
        v_pred = x_clean - noise

        loss = vloss(
            v_pred=v_pred,
            x_clean=x_clean,
            noise=noise,
            reduction="mean",
        )

        assert loss < 1e-6, f"Expected near-zero loss for perfect prediction, got {loss}"

    def test_random_prediction_loss_positive(self, vloss: VLoss) -> None:
        """
        Test that random predictions result in positive loss.
        """
        B, C, H, W = 2, 3, 32, 32
        torch.manual_seed(123)

        v_pred = torch.randn(B, C, H, W)
        x_clean = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)

        loss = vloss(
            v_pred=v_pred,
            x_clean=x_clean,
            noise=noise,
            reduction="mean",
        )

        assert loss > 0, f"Expected positive loss for random prediction, got {loss}"

    # =========================================================================
    # Gradient Tests
    # =========================================================================

    def test_gradient_flow(self, vloss: VLoss) -> None:
        """
        Test that gradients flow correctly through the loss computation.
        """
        B, C, H, W = 2, 3, 32, 32

        v_pred = torch.randn(B, C, H, W, requires_grad=True)
        x_clean = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)

        loss = vloss(
            v_pred=v_pred,
            x_clean=x_clean,
            noise=noise,
            reduction="mean",
        )

        loss.backward()

        assert v_pred.grad is not None, "Gradient not computed for v_pred"
        assert not torch.isnan(v_pred.grad).any(), "Gradient contains NaN"
        assert v_pred.grad.shape == v_pred.shape, \
            f"Gradient shape mismatch: {v_pred.grad.shape} vs {v_pred.shape}"


class TestVLossFactoryFunctions:
    """Test factory functions for VLoss."""

    def test_create_vloss(self) -> None:
        """Test create_vloss factory function."""
        vloss = create_vloss(t_eps=0.1)

        assert isinstance(vloss, VLoss)
        assert vloss.t_eps == 0.1

    def test_create_vloss_from_config(self) -> None:
        """Test create_vloss_from_config factory function."""
        config = PixelHDMConfig.for_testing()
        vloss = create_vloss_from_config(config)

        assert isinstance(vloss, VLoss)
        assert vloss.t_eps == config.time_eps

    def test_create_vloss_with_velocity(self) -> None:
        """Test create_vloss_with_velocity factory function."""
        vloss = create_vloss_with_velocity(t_eps=0.1)

        assert isinstance(vloss, VLossWithVelocity)
        assert vloss.t_eps == 0.1
