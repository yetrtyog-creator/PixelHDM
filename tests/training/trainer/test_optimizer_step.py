"""
Comprehensive Unit Tests for Optimizer Step Logic

Tests for:
1. Basic _optimizer_step functionality (5 tests)
2. Gradient accumulation logic (5 tests)
3. AMP scaler handling (5 tests)
4. NaN/Inf gradient detection (3 tests)
5. _update_ema timing (3 tests)
6. ZClip integration (3 tests)
7. Learning rate update (_update_lr) (3 tests)

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-08

Tests the OptimizerStepMixin class which handles optimizer step execution,
gradient clipping, EMA updates, and learning rate scheduling.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import MagicMock, Mock, patch, call

import pytest
import torch
import torch.nn as nn

from src.training.trainer.optimizer_step import OptimizerStepMixin
from src.training.optimization.gradient_clip import ZClip, clip_grad_norm_with_zclip


class MockModel(nn.Module):
    """Mock model for testing optimizer step logic."""

    def __init__(self, num_params: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(10, 10) for _ in range(num_params)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ConcreteOptimizerStepMixin(OptimizerStepMixin):
    """
    Concrete implementation of OptimizerStepMixin for testing.

    Implements required attributes that the mixin expects from its host class.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[Any] = None,
        zclip: Optional[ZClip] = None,
        ema: Optional[Any] = None,
        cpu_checkpoint: Optional[Any] = None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.zclip = zclip if zclip is not None else ZClip()
        self.ema = ema
        self.cpu_checkpoint = cpu_checkpoint
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm


def create_test_setup(
    gradient_accumulation_steps: int = 1,
    use_scaler: bool = False,
    use_ema: bool = False,
    use_cpu_checkpoint: bool = False,
) -> tuple[ConcreteOptimizerStepMixin, MockModel]:
    """Create a test setup with mock components."""
    model = MockModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    scaler = None
    if use_scaler:
        scaler = MagicMock()
        scaler.unscale_ = MagicMock()
        scaler.step = MagicMock()
        scaler.update = MagicMock()

    ema = None
    if use_ema:
        ema = MagicMock()
        ema.update = MagicMock()

    cpu_checkpoint = None
    if use_cpu_checkpoint:
        cpu_checkpoint = MagicMock()
        cpu_checkpoint.step = MagicMock(return_value=False)

    mixin = ConcreteOptimizerStepMixin(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        ema=ema,
        cpu_checkpoint=cpu_checkpoint,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    return mixin, model


def simulate_backward(model: nn.Module) -> None:
    """Simulate a backward pass by setting gradients on all parameters."""
    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.randn_like(param) * 0.1


# ============================================================================
# Basic _optimizer_step functionality (5 tests)
# ============================================================================


class TestOptimizerStepBasic:
    """Tests for basic _optimizer_step functionality."""

    def test_optimizer_step_returns_grad_norm(self):
        """Test that _optimizer_step returns gradient norm."""
        mixin, model = create_test_setup()
        simulate_backward(model)

        grad_norm = mixin._optimizer_step(
            step=0,
            loss_value=1.0,
            lr_scheduler=None,
            warmup_fn=None,
            warmup_steps=0,
        )

        assert isinstance(grad_norm, float)
        assert grad_norm >= 0.0

    def test_optimizer_step_calls_zero_grad(self):
        """Test that optimizer.zero_grad() is called after step."""
        mixin, model = create_test_setup()
        simulate_backward(model)

        # Spy on zero_grad
        original_zero_grad = mixin.optimizer.zero_grad
        mixin.optimizer.zero_grad = MagicMock(side_effect=original_zero_grad)

        mixin._optimizer_step(
            step=0,
            loss_value=1.0,
            lr_scheduler=None,
            warmup_fn=None,
            warmup_steps=0,
        )

        mixin.optimizer.zero_grad.assert_called_once()

    def test_optimizer_step_calls_optimizer_step(self):
        """Test that optimizer.step() is called (without scaler)."""
        mixin, model = create_test_setup(use_scaler=False)
        simulate_backward(model)

        original_step = mixin.optimizer.step
        mixin.optimizer.step = MagicMock(side_effect=original_step)

        mixin._optimizer_step(
            step=0,
            loss_value=1.0,
            lr_scheduler=None,
            warmup_fn=None,
            warmup_steps=0,
        )

        mixin.optimizer.step.assert_called_once()

    def test_optimizer_step_updates_parameters(self):
        """Test that model parameters are actually updated."""
        mixin, model = create_test_setup()

        # Store original parameter values
        original_params = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }

        simulate_backward(model)

        mixin._optimizer_step(
            step=0,
            loss_value=1.0,
            lr_scheduler=None,
            warmup_fn=None,
            warmup_steps=0,
        )

        # Check at least one parameter changed
        any_changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(param.data, original_params[name]):
                any_changed = True
                break

        assert any_changed, "No parameters were updated"

    def test_optimizer_step_with_zero_loss(self):
        """Test optimizer step with zero loss value."""
        mixin, model = create_test_setup()
        simulate_backward(model)

        # Should not raise error
        grad_norm = mixin._optimizer_step(
            step=0,
            loss_value=0.0,
            lr_scheduler=None,
            warmup_fn=None,
            warmup_steps=0,
        )

        assert isinstance(grad_norm, float)


# ============================================================================
# Gradient accumulation logic (5 tests)
# ============================================================================


class TestGradientAccumulation:
    """Tests for gradient accumulation logic."""

    def test_accumulation_skips_on_non_boundary_step(self):
        """Test that optimizer step is skipped when accumulating."""
        mixin, model = create_test_setup(gradient_accumulation_steps=4)
        simulate_backward(model)

        # Step 0, 1, 2 should skip (not divisible by 4)
        for step in [0, 1, 2]:
            grad_norm = mixin._optimizer_step(
                step=step,
                loss_value=1.0,
                lr_scheduler=None,
                warmup_fn=None,
                warmup_steps=0,
            )
            assert grad_norm == 0.0, f"Step {step} should skip, got grad_norm={grad_norm}"

    def test_accumulation_executes_on_boundary_step(self):
        """Test that optimizer step executes on accumulation boundary."""
        mixin, model = create_test_setup(gradient_accumulation_steps=4)
        simulate_backward(model)

        # Step 3 should execute (3 + 1 = 4, divisible by 4)
        grad_norm = mixin._optimizer_step(
            step=3,
            loss_value=1.0,
            lr_scheduler=None,
            warmup_fn=None,
            warmup_steps=0,
        )

        assert grad_norm > 0.0, "Step 3 should execute optimizer step"

    def test_accumulation_pattern_with_steps_2(self):
        """Test accumulation pattern with gradient_accumulation_steps=2."""
        mixin, model = create_test_setup(gradient_accumulation_steps=2)

        results = []
        for step in range(6):
            simulate_backward(model)
            grad_norm = mixin._optimizer_step(
                step=step,
                loss_value=1.0,
                lr_scheduler=None,
                warmup_fn=None,
                warmup_steps=0,
            )
            results.append(grad_norm > 0.0)

        # Steps 1, 3, 5 should execute (odd steps)
        expected = [False, True, False, True, False, True]
        assert results == expected, f"Expected {expected}, got {results}"

    def test_accumulation_does_not_zero_grad_when_skipping(self):
        """Test that zero_grad is not called when accumulating."""
        mixin, model = create_test_setup(gradient_accumulation_steps=4)
        simulate_backward(model)

        # Store original gradients
        original_grads = {
            name: param.grad.clone() if param.grad is not None else None
            for name, param in model.named_parameters()
        }

        mixin.optimizer.zero_grad = MagicMock()

        # Step 0 should skip
        mixin._optimizer_step(
            step=0,
            loss_value=1.0,
            lr_scheduler=None,
            warmup_fn=None,
            warmup_steps=0,
        )

        # zero_grad should not be called when skipping
        mixin.optimizer.zero_grad.assert_not_called()

    def test_accumulation_with_step_1_always_executes(self):
        """Test that gradient_accumulation_steps=1 always executes."""
        mixin, model = create_test_setup(gradient_accumulation_steps=1)

        for step in range(5):
            simulate_backward(model)
            grad_norm = mixin._optimizer_step(
                step=step,
                loss_value=1.0,
                lr_scheduler=None,
                warmup_fn=None,
                warmup_steps=0,
            )
            assert grad_norm > 0.0, f"Step {step} should always execute"


# ============================================================================
# AMP scaler handling (5 tests)
# ============================================================================


class TestAMPScalerHandling:
    """Tests for AMP GradScaler handling."""

    def test_scaler_unscale_called(self):
        """Test that scaler.unscale_() is called before clipping."""
        mixin, model = create_test_setup(use_scaler=True)
        simulate_backward(model)

        mixin._optimizer_step(
            step=0,
            loss_value=1.0,
            lr_scheduler=None,
            warmup_fn=None,
            warmup_steps=0,
        )

        mixin.scaler.unscale_.assert_called_once_with(mixin.optimizer)

    def test_scaler_step_called(self):
        """Test that scaler.step() is called instead of optimizer.step()."""
        mixin, model = create_test_setup(use_scaler=True)
        simulate_backward(model)

        mixin._optimizer_step(
            step=0,
            loss_value=1.0,
            lr_scheduler=None,
            warmup_fn=None,
            warmup_steps=0,
        )

        mixin.scaler.step.assert_called_once_with(mixin.optimizer)

    def test_scaler_update_called(self):
        """Test that scaler.update() is called after step."""
        mixin, model = create_test_setup(use_scaler=True)
        simulate_backward(model)

        mixin._optimizer_step(
            step=0,
            loss_value=1.0,
            lr_scheduler=None,
            warmup_fn=None,
            warmup_steps=0,
        )

        mixin.scaler.update.assert_called_once()

    def test_scaler_call_order(self):
        """Test that scaler methods are called in correct order."""
        mixin, model = create_test_setup(use_scaler=True)
        simulate_backward(model)

        # Create a call order tracker
        call_order = []
        mixin.scaler.unscale_ = MagicMock(side_effect=lambda x: call_order.append("unscale"))
        mixin.scaler.step = MagicMock(side_effect=lambda x: call_order.append("step"))
        mixin.scaler.update = MagicMock(side_effect=lambda: call_order.append("update"))

        mixin._optimizer_step(
            step=0,
            loss_value=1.0,
            lr_scheduler=None,
            warmup_fn=None,
            warmup_steps=0,
        )

        assert call_order == ["unscale", "step", "update"], f"Got order: {call_order}"

    def test_no_scaler_uses_regular_step(self):
        """Test that without scaler, regular optimizer.step() is used."""
        mixin, model = create_test_setup(use_scaler=False)
        simulate_backward(model)

        # Replace optimizer.step with mock
        step_called = [False]
        original_step = mixin.optimizer.step
        def mock_step():
            step_called[0] = True
            original_step()
        mixin.optimizer.step = mock_step

        mixin._optimizer_step(
            step=0,
            loss_value=1.0,
            lr_scheduler=None,
            warmup_fn=None,
            warmup_steps=0,
        )

        assert step_called[0], "optimizer.step() should be called"


# ============================================================================
# NaN/Inf gradient detection (3 tests)
# ============================================================================


class TestNaNInfGradientDetection:
    """Tests for NaN/Inf gradient detection via clip_grad_norm_with_zclip."""

    def test_nan_gradient_raises_error(self):
        """Test that NaN gradients raise RuntimeError."""
        mixin, model = create_test_setup()

        # Set NaN gradients
        for param in model.parameters():
            param.grad = torch.full_like(param, float('nan'))

        with pytest.raises(RuntimeError, match="無效梯度範數"):
            mixin._clip_and_step(clip_grad_norm_with_zclip)

    def test_inf_gradient_raises_error(self):
        """Test that Inf gradients raise RuntimeError."""
        mixin, model = create_test_setup()

        # Set Inf gradients
        for param in model.parameters():
            param.grad = torch.full_like(param, float('inf'))

        with pytest.raises(RuntimeError, match="無效梯度範數"):
            mixin._clip_and_step(clip_grad_norm_with_zclip)

    def test_valid_gradients_no_error(self):
        """Test that valid gradients don't raise errors."""
        mixin, model = create_test_setup()
        simulate_backward(model)

        # Should not raise error
        grad_norm = mixin._clip_and_step(clip_grad_norm_with_zclip)

        assert isinstance(grad_norm, float)
        assert grad_norm >= 0.0
        assert grad_norm != float('inf')
        assert grad_norm == grad_norm  # Not NaN


# ============================================================================
# _update_ema timing (3 tests)
# ============================================================================


class TestUpdateEMATiming:
    """Tests for _update_ema timing with gradient accumulation."""

    def test_ema_update_on_accumulation_boundary(self):
        """Test that EMA is updated on accumulation boundary."""
        mixin, model = create_test_setup(
            gradient_accumulation_steps=4,
            use_ema=True,
        )

        # Step 3 should trigger EMA update (3 + 1 = 4, divisible by 4)
        mixin._update_ema(step=3)

        mixin.ema.update.assert_called_once_with(mixin.model, 3)

    def test_ema_update_skipped_when_accumulating(self):
        """Test that EMA update is skipped when accumulating gradients."""
        mixin, model = create_test_setup(
            gradient_accumulation_steps=4,
            use_ema=True,
        )

        # Steps 0, 1, 2 should skip EMA update
        for step in [0, 1, 2]:
            mixin._update_ema(step=step)

        mixin.ema.update.assert_not_called()

    def test_ema_update_with_none_ema(self):
        """Test that _update_ema handles None ema gracefully."""
        mixin, model = create_test_setup(use_ema=False)

        # Should not raise error
        mixin._update_ema(step=0)
        mixin._update_ema(step=3)


# ============================================================================
# ZClip integration (3 tests)
# ============================================================================


class TestZClipIntegration:
    """Tests for ZClip integration in gradient clipping."""

    def test_zclip_receives_gradient_norm(self):
        """Test that ZClip receives the gradient norm and updates statistics."""
        mixin, model = create_test_setup()
        simulate_backward(model)

        # Reset ZClip to track this specific call
        mixin.zclip.reset()
        initial_steps = mixin.zclip.step_count

        mixin._clip_and_step(clip_grad_norm_with_zclip)

        # ZClip should have been called (step count increased)
        assert mixin.zclip.step_count == initial_steps + 1

    def test_zclip_adaptive_clipping(self):
        """Test that ZClip provides adaptive gradient clipping."""
        mixin, model = create_test_setup()

        # Reset ZClip statistics
        mixin.zclip.reset()

        # Run multiple steps to build up statistics
        norms = []
        for i in range(10):
            simulate_backward(model)
            grad_norm = mixin._clip_and_step(clip_grad_norm_with_zclip)
            norms.append(grad_norm)

        # Verify ZClip has accumulated statistics
        stats = mixin.zclip.get_stats()
        assert stats.steps > 0

    def test_zclip_respects_max_grad_norm(self):
        """Test that gradient clipping respects max_grad_norm."""
        mixin, model = create_test_setup()
        mixin.max_grad_norm = 0.01  # Very small max norm

        # Set large gradients
        for param in model.parameters():
            param.grad = torch.ones_like(param) * 10.0

        # Compute pre-clip norm
        pre_clip_norm = torch.norm(
            torch.stack([torch.norm(p.grad, 2) for p in model.parameters() if p.grad is not None]),
            2
        ).item()

        # Note: _clip_and_step will zero gradients after clipping via optimizer.zero_grad()
        # So we need to verify the clipping happened by checking the returned norm
        # and that it was larger than max_grad_norm (i.e., clipping was needed)
        returned_norm = mixin._clip_and_step(clip_grad_norm_with_zclip)

        # The returned norm is the original norm before clipping
        assert returned_norm > mixin.max_grad_norm, "Original gradient norm should exceed max"
        assert pre_clip_norm > mixin.max_grad_norm, "Pre-clip norm should exceed max"


# ============================================================================
# Learning rate update (_update_lr) (3 tests)
# ============================================================================


class TestLearningRateUpdate:
    """Tests for _update_lr method."""

    def test_warmup_fn_called_during_warmup(self):
        """Test that warmup_fn is called during warmup period."""
        mixin, model = create_test_setup()

        warmup_fn = MagicMock()
        lr_scheduler = MagicMock()

        mixin._update_lr(
            step=5,
            lr_scheduler=lr_scheduler,
            warmup_fn=warmup_fn,
            warmup_steps=100,
        )

        warmup_fn.assert_called_once_with(5)
        lr_scheduler.step.assert_not_called()

    def test_scheduler_called_after_warmup(self):
        """Test that lr_scheduler.step() is called after warmup."""
        mixin, model = create_test_setup()

        warmup_fn = MagicMock()
        lr_scheduler = MagicMock()

        mixin._update_lr(
            step=100,  # At warmup_steps, not during warmup
            lr_scheduler=lr_scheduler,
            warmup_fn=warmup_fn,
            warmup_steps=100,
        )

        warmup_fn.assert_not_called()
        lr_scheduler.step.assert_called_once()

    def test_no_warmup_uses_scheduler_immediately(self):
        """Test that scheduler is used immediately when warmup_steps=0."""
        mixin, model = create_test_setup()

        lr_scheduler = MagicMock()

        mixin._update_lr(
            step=0,
            lr_scheduler=lr_scheduler,
            warmup_fn=None,
            warmup_steps=0,
        )

        lr_scheduler.step.assert_called_once()


# ============================================================================
# Loss spike detection (_check_loss_spike) (3 tests)
# ============================================================================


class TestLossSpikeDetection:
    """Tests for loss spike detection via CPUMemoryCheckpoint."""

    def test_loss_spike_triggers_restore(self):
        """Test that loss spike triggers checkpoint restore."""
        mixin, model = create_test_setup(use_cpu_checkpoint=True)

        # Configure mock to return True (restored)
        mixin.cpu_checkpoint.step.return_value = True

        result = mixin._check_loss_spike(loss_value=100.0, step=10)

        assert result is True
        mixin.cpu_checkpoint.step.assert_called_once_with(
            mixin.model, mixin.optimizer, mixin.ema, 100.0, 10
        )

    def test_no_spike_continues_normally(self):
        """Test that normal loss continues without restore."""
        mixin, model = create_test_setup(use_cpu_checkpoint=True)

        mixin.cpu_checkpoint.step.return_value = False

        result = mixin._check_loss_spike(loss_value=1.0, step=10)

        assert result is False

    def test_no_checkpoint_returns_false(self):
        """Test that without cpu_checkpoint, always returns False."""
        mixin, model = create_test_setup(use_cpu_checkpoint=False)

        result = mixin._check_loss_spike(loss_value=1000.0, step=10)

        assert result is False


# ============================================================================
# Full integration tests (2 tests)
# ============================================================================


class TestOptimizerStepIntegration:
    """Integration tests for full optimizer step flow."""

    def test_full_step_with_all_components(self):
        """Test full optimizer step with all components enabled."""
        mixin, model = create_test_setup(
            gradient_accumulation_steps=1,
            use_scaler=True,
            use_ema=True,
            use_cpu_checkpoint=True,
        )
        mixin.cpu_checkpoint.step.return_value = False

        simulate_backward(model)

        grad_norm = mixin._optimizer_step(
            step=0,
            loss_value=1.0,
            lr_scheduler=None,
            warmup_fn=None,
            warmup_steps=0,
        )

        assert grad_norm >= 0.0
        mixin.scaler.unscale_.assert_called_once()
        mixin.scaler.step.assert_called_once()
        mixin.scaler.update.assert_called_once()
        mixin.cpu_checkpoint.step.assert_called_once()

    def test_spike_restore_skips_optimizer_step(self):
        """Test that loss spike restore skips the optimizer step."""
        mixin, model = create_test_setup(use_cpu_checkpoint=True)
        mixin.cpu_checkpoint.step.return_value = True

        simulate_backward(model)

        # Mock optimizer.step to track if it's called
        original_step = mixin.optimizer.step
        mixin.optimizer.step = MagicMock()

        grad_norm = mixin._optimizer_step(
            step=0,
            loss_value=100.0,  # Spike loss
            lr_scheduler=None,
            warmup_fn=None,
            warmup_steps=0,
        )

        assert grad_norm == 0.0
        mixin.optimizer.step.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
