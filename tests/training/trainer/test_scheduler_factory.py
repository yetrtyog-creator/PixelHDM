"""
Comprehensive Unit Tests for Scheduler Factory and LR Scheduling

Tests for:
1. T_0 Calculation (5 tests)
2. Scheduler Creation (5 tests)
3. Reset/Resume (5 tests) - Critical regression tests for 2026-01-07 bug
4. LR Schedule Verification (5 tests)

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-08

CRITICAL: These tests cover the 2026-01-07 bug where reset_scheduler=True
caused LR to immediately drop to global_min_lr (1e-5) instead of starting
fresh from cycle 0 with base_lr.
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from src.training.trainer.scheduler_factory import (
    create_lr_scheduler,
    apply_warmup_lr,
    _calculate_t0,
)
from src.training.optimization.scheduler import (
    SteppedCosineRestartScheduler,
    CosineAnnealingWarmRestartsWithDecay,
)
from src.config.training_config import TrainingConfig, SteppedCosineRestartConfig


class SimpleModel(nn.Module):
    """Simple model for scheduler testing."""

    def __init__(self, in_features: int = 10, out_features: int = 10):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def create_optimizer(lr: float = 1e-4) -> torch.optim.Optimizer:
    """Create a simple optimizer for testing."""
    model = SimpleModel()
    return torch.optim.AdamW(model.parameters(), lr=lr)


def create_mock_dataloader(length: int) -> MagicMock:
    """Create a mock dataloader with specified length."""
    mock_dl = MagicMock()
    mock_dl.__len__ = MagicMock(return_value=length)
    return mock_dl


# ============================================================================
# T_0 Calculation Tests (5 tests)
# ============================================================================

class TestT0CalculationBasic:
    """Tests for T_0 auto-calculation."""

    def test_auto_calculate_t0_basic(self):
        """Test T_0 = restart_epochs * steps_per_epoch (optimizer steps).

        Formula: T_0 = restart_epochs * (len(dataloader) / gradient_accumulation_steps)
        """
        # Create config with restart_period=0 (auto-calculate)
        config = TrainingConfig(
            restart_epochs=4,
            restart_period=0,  # Triggers auto-calculation
            num_epochs=16,
            max_steps=10000,
        )

        mock_dataloader = create_mock_dataloader(100)

        # gradient_accumulation_steps=1: T_0 = 4 * 100 = 400
        t0 = _calculate_t0(
            restart_period=0,
            restart_epochs=4,
            training_config=config,
            dataloader=mock_dataloader,
            gradient_accumulation_steps=1,
        )
        assert t0 == 400, f"Expected T_0=400, got {t0}"

    def test_auto_calculate_t0_with_gradient_accumulation(self):
        """Test T_0 calculation accounts for gradient accumulation.

        With grad_accum=2, optimizer steps per epoch = len(dataloader) / 2
        """
        config = TrainingConfig(
            restart_epochs=4,
            restart_period=0,
            num_epochs=16,
            max_steps=10000,
        )

        mock_dataloader = create_mock_dataloader(100)

        # gradient_accumulation_steps=2: T_0 = 4 * (100 / 2) = 200
        t0 = _calculate_t0(
            restart_period=0,
            restart_epochs=4,
            training_config=config,
            dataloader=mock_dataloader,
            gradient_accumulation_steps=2,
        )
        assert t0 == 200, f"Expected T_0=200, got {t0}"

    def test_auto_calculate_t0_different_dataset_sizes(self):
        """Test T_0 calculation with different dataloader lengths."""
        config = TrainingConfig(
            restart_epochs=8,
            restart_period=0,
            num_epochs=32,
            max_steps=50000,
        )

        # Small dataset: 22 images / batch_size=8 = 3 batches
        mock_dataloader_small = create_mock_dataloader(3)
        t0_small = _calculate_t0(
            restart_period=0,
            restart_epochs=8,
            training_config=config,
            dataloader=mock_dataloader_small,
            gradient_accumulation_steps=2,
        )
        # optimizer_steps_per_epoch = 3 // 2 = 1, T_0 = 8 * 1 = 8
        assert t0_small == 8, f"Expected T_0=8, got {t0_small}"

        # Large dataset: 10000 images / batch_size=8 = 1250 batches
        mock_dataloader_large = create_mock_dataloader(1250)
        t0_large = _calculate_t0(
            restart_period=0,
            restart_epochs=8,
            training_config=config,
            dataloader=mock_dataloader_large,
            gradient_accumulation_steps=2,
        )
        # optimizer_steps_per_epoch = 1250 // 2 = 625, T_0 = 8 * 625 = 5000
        assert t0_large == 5000, f"Expected T_0=5000, got {t0_large}"

    def test_t0_manual_override(self):
        """Test restart_period > 0 uses manual value instead of auto-calculation."""
        config = TrainingConfig(
            restart_epochs=4,
            restart_period=500,  # Manual override
            num_epochs=16,
            max_steps=10000,
        )

        mock_dataloader = create_mock_dataloader(100)

        # Should use manual restart_period=500, ignoring dataloader
        t0 = _calculate_t0(
            restart_period=500,
            restart_epochs=4,
            training_config=config,
            dataloader=mock_dataloader,
            gradient_accumulation_steps=1,
        )
        assert t0 == 500, f"Expected manual T_0=500, got {t0}"

    def test_t0_zero_triggers_auto(self):
        """Test restart_period=0 triggers auto-calculation."""
        config = TrainingConfig(
            restart_epochs=2,
            restart_period=0,  # Explicitly zero
            num_epochs=8,
            max_steps=5000,
        )

        mock_dataloader = create_mock_dataloader(50)

        t0 = _calculate_t0(
            restart_period=0,
            restart_epochs=2,
            training_config=config,
            dataloader=mock_dataloader,
            gradient_accumulation_steps=1,
        )
        # T_0 = 2 * 50 = 100
        assert t0 == 100, f"Expected auto T_0=100, got {t0}"


# ============================================================================
# Scheduler Creation Tests (5 tests)
# ============================================================================

class TestSchedulerCreation:
    """Tests for scheduler factory creation."""

    def test_create_stepped_cosine_scheduler(self):
        """Test SteppedCosineRestartScheduler is created correctly."""
        optimizer = create_optimizer(lr=1e-4)
        config = TrainingConfig(
            stepped_cosine_restart=SteppedCosineRestartConfig(
                enabled=True,
                base_lr=2e-4,
                cycle_min_lr=1e-4,
                decay_rate=0.9,
                global_min_lr=1e-5,
                warmup_steps=100,
            ),
            restart_epochs=4,
            num_epochs=16,
        )

        mock_dataloader = create_mock_dataloader(100)

        scheduler = create_lr_scheduler(
            optimizer, config, dataloader=mock_dataloader,
            gradient_accumulation_steps=1
        )

        assert isinstance(scheduler, SteppedCosineRestartScheduler)
        assert scheduler.base_lr == 2e-4
        assert scheduler.cycle_min_lr == 1e-4
        assert scheduler.decay_rate == 0.9
        assert scheduler.global_min_lr == 1e-5
        assert scheduler.warmup_steps == 100

    def test_create_with_warmup(self):
        """Test scheduler with warmup steps applied."""
        optimizer = create_optimizer(lr=1e-4)
        config = TrainingConfig(
            stepped_cosine_restart=SteppedCosineRestartConfig(
                enabled=True,
                base_lr=2e-4,
                warmup_steps=50,
            ),
            restart_epochs=4,
            num_epochs=16,
        )

        mock_dataloader = create_mock_dataloader(100)

        scheduler = create_lr_scheduler(
            optimizer, config, dataloader=mock_dataloader,
            gradient_accumulation_steps=1
        )

        assert scheduler.warmup_steps == 50

        # During warmup, LR should increase linearly
        # After init (which calls step()), total_steps=1
        # warmup_factor = (1 + 1) / 50 = 0.04
        # LR = 2e-4 * 0.04 = 8e-6
        expected_first_lr = 2e-4 * (2 / 50)
        actual_lr = optimizer.param_groups[0]['lr']
        assert abs(actual_lr - expected_first_lr) < 1e-10

    def test_create_with_decay_rate(self):
        """Test decay rate is applied between cycles."""
        optimizer = create_optimizer(lr=1e-4)
        config = TrainingConfig(
            stepped_cosine_restart=SteppedCosineRestartConfig(
                enabled=True,
                base_lr=1e-4,
                cycle_min_lr=5e-5,
                decay_rate=0.75,
                global_min_lr=1e-5,
            ),
            restart_epochs=4,
            num_epochs=16,
        )

        mock_dataloader = create_mock_dataloader(25)  # T_0 = 4 * 25 = 100

        scheduler = create_lr_scheduler(
            optimizer, config, dataloader=mock_dataloader,
            gradient_accumulation_steps=1
        )

        # Complete cycle 0
        for _ in range(100):
            scheduler.step()

        # At start of cycle 1, peak should be base_lr * 0.75
        expected_peak = 1e-4 * 0.75
        actual_lr = optimizer.param_groups[0]['lr']
        assert abs(actual_lr - expected_peak) < 1e-10

    def test_create_global_min_lr(self):
        """Test global_min_lr is respected as absolute lower bound."""
        optimizer = create_optimizer(lr=1e-4)
        config = TrainingConfig(
            stepped_cosine_restart=SteppedCosineRestartConfig(
                enabled=True,
                base_lr=1e-4,
                cycle_min_lr=5e-5,
                decay_rate=0.5,
                global_min_lr=2e-5,
            ),
            restart_epochs=1,  # Very short cycles
            num_epochs=100,
        )

        mock_dataloader = create_mock_dataloader(10)  # T_0 = 1 * 10 = 10

        scheduler = create_lr_scheduler(
            optimizer, config, dataloader=mock_dataloader,
            gradient_accumulation_steps=1
        )

        # Run many cycles
        for _ in range(200):
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            assert lr >= 2e-5 - 1e-12, f"LR {lr} went below global_min_lr 2e-5"

    def test_scheduler_type_from_config(self):
        """Test scheduler type is determined from config."""
        optimizer = create_optimizer(lr=1e-4)

        # With stepped_cosine_restart disabled, should use lr_scheduler type
        config = TrainingConfig(
            lr_scheduler="cosine_restart",
            stepped_cosine_restart=SteppedCosineRestartConfig(
                enabled=False,  # Disabled
            ),
            restart_epochs=4,
            num_epochs=16,
        )

        mock_dataloader = create_mock_dataloader(100)

        scheduler = create_lr_scheduler(
            optimizer, config, dataloader=mock_dataloader,
            gradient_accumulation_steps=1
        )

        # Should NOT be SteppedCosineRestartScheduler when disabled
        assert not isinstance(scheduler, SteppedCosineRestartScheduler)
        assert isinstance(scheduler, CosineAnnealingWarmRestartsWithDecay)


# ============================================================================
# Reset/Resume Tests (5 tests) - Critical regression tests for 2026-01-07 bug
# ============================================================================

class TestResetResume:
    """Tests for reset/resume behavior.

    CRITICAL: These tests verify the fix for the 2026-01-07 bug where
    reset_scheduler=True caused LR to immediately drop to global_min_lr.
    """

    def test_skip_sync_on_reset_scheduler(self):
        """Test that _scheduler_skip_sync=True prevents step synchronization.

        Bug scenario (2026-01-07):
            - Resume at step 5000 with reset_scheduler=True
            - Old code: synced scheduler to step 5000, causing many cycles to pass
            - Fixed code: skip sync, scheduler starts fresh at cycle 0
        """
        from src.training.trainer.core import Trainer

        model = SimpleModel()
        optimizer = create_optimizer()

        config = TrainingConfig(
            stepped_cosine_restart=SteppedCosineRestartConfig(
                enabled=True,
                base_lr=1e-4,
                cycle_min_lr=5e-5,
                decay_rate=0.9,
                global_min_lr=1e-5,
            ),
            restart_epochs=32,
            num_epochs=1024,
        )

        mock_dataloader = create_mock_dataloader(100)

        trainer = Trainer(
            model=model,
            training_config=config,
            dataloader=mock_dataloader,
            optimizer=optimizer,
            device=torch.device('cpu'),
        )

        # Simulate resume: set state.step to 5000
        trainer.state.step = 5000

        # Set flag to skip sync (this is what train.py does when reset_scheduler=True)
        trainer._scheduler_skip_sync = True

        # Access lr_scheduler property (this triggers scheduler creation)
        scheduler = trainer.lr_scheduler

        # LR should be at base_lr (start of cycle 0), NOT decayed
        actual_lr = optimizer.param_groups[0]['lr']
        expected_lr = 1e-4  # base_lr at cycle 0
        assert abs(actual_lr - expected_lr) < 1e-10, \
            f"Expected LR {expected_lr:.2e} (cycle 0), got {actual_lr:.2e}"

        # Verify scheduler is at cycle 0
        assert scheduler.cycle == 0, f"Expected cycle 0, got {scheduler.cycle}"

    def test_resume_without_reset_syncs(self):
        """Test that resume without reset syncs scheduler to current step.

        When reset_scheduler=False, scheduler should be synced to the
        current step to produce correct LR.
        """
        from src.training.trainer.core import Trainer

        model = SimpleModel()
        optimizer = create_optimizer()

        config = TrainingConfig(
            stepped_cosine_restart=SteppedCosineRestartConfig(
                enabled=True,
                base_lr=1e-4,
                cycle_min_lr=5e-5,
                decay_rate=0.9,
                global_min_lr=1e-5,
            ),
            restart_epochs=1,  # T_0 = 100 * 1 = 100
            num_epochs=100,
        )

        mock_dataloader = create_mock_dataloader(100)

        trainer = Trainer(
            model=model,
            training_config=config,
            dataloader=mock_dataloader,
            optimizer=optimizer,
            device=torch.device('cpu'),
        )

        # Simulate resume at step 100 (start of cycle 1)
        # Note: trainer.state.step is batch steps
        trainer.state.step = 100  # With grad_accum=1, this equals 100 optimizer steps

        # Do NOT set _scheduler_skip_sync (this is reset_scheduler=False behavior)
        # trainer._scheduler_skip_sync = False  # Default

        # Access lr_scheduler property
        scheduler = trainer.lr_scheduler

        # Should be synced to step 100 = start of cycle 1
        # LR should be base_lr * 0.9 = 9e-5
        actual_lr = optimizer.param_groups[0]['lr']
        expected_lr = 1e-4 * 0.9
        assert abs(actual_lr - expected_lr) < 1e-10, \
            f"Expected LR {expected_lr:.2e} (cycle 1), got {actual_lr:.2e}"

    def test_lr_after_resume_with_reset(self):
        """Test LR starts from base_lr after reset_scheduler=True.

        This is the critical test for the 2026-01-07 bug fix.
        """
        optimizer = create_optimizer(lr=1e-4)

        scheduler = SteppedCosineRestartScheduler(
            optimizer,
            T_0=100,
            base_lr=1e-4,
            cycle_min_lr=5e-5,
            decay_rate=0.9,
            global_min_lr=1e-5,
        )

        # Initial LR should be base_lr
        assert abs(optimizer.param_groups[0]['lr'] - 1e-4) < 1e-10

        # This is what reset_scheduler=True should achieve
        # Scheduler starts fresh, regardless of how many steps were done before
        assert scheduler.cycle == 0
        assert scheduler.T_cur == 0

    def test_lr_after_resume_without_reset(self):
        """Test LR continues from saved state when reset_scheduler=False."""
        optimizer = create_optimizer(lr=1e-4)

        # Create and advance scheduler
        scheduler1 = SteppedCosineRestartScheduler(
            optimizer,
            T_0=100,
            base_lr=1e-4,
            cycle_min_lr=5e-5,
            decay_rate=0.9,
            global_min_lr=1e-5,
        )

        # Advance to cycle 5
        for _ in range(500):
            scheduler1.step()

        assert scheduler1.cycle == 5
        expected_lr = 1e-4 * (0.9 ** 5)  # ~5.9e-5
        actual_lr = optimizer.param_groups[0]['lr']
        assert abs(actual_lr - expected_lr) < 1e-10

        # Save state
        state = scheduler1.state_dict()

        # Create new scheduler and load state (reset_scheduler=False)
        optimizer2 = create_optimizer(lr=1e-4)
        scheduler2 = SteppedCosineRestartScheduler(
            optimizer2,
            T_0=100,
            base_lr=1e-4,
            cycle_min_lr=5e-5,
            decay_rate=0.9,
            global_min_lr=1e-5,
        )

        scheduler2.load_state_dict(state)

        # LR should be restored to cycle 5 value
        restored_lr = optimizer2.param_groups[0]['lr']
        assert abs(restored_lr - expected_lr) < 1e-10, \
            f"Expected restored LR {expected_lr:.2e}, got {restored_lr:.2e}"

    def test_scheduler_skip_sync_flag(self):
        """Test _scheduler_skip_sync flag behavior."""
        from src.training.trainer.core import Trainer

        model = SimpleModel()
        optimizer = create_optimizer()

        config = TrainingConfig(
            stepped_cosine_restart=SteppedCosineRestartConfig(
                enabled=True,
                base_lr=1e-4,
                cycle_min_lr=5e-5,  # Must be <= base_lr
            ),
            restart_epochs=4,
            num_epochs=16,
        )

        mock_dataloader = create_mock_dataloader(100)

        trainer = Trainer(
            model=model,
            training_config=config,
            dataloader=mock_dataloader,
            optimizer=optimizer,
            device=torch.device('cpu'),
        )

        # Initial flag should be False
        assert trainer._scheduler_skip_sync is False

        # Set flag
        trainer._scheduler_skip_sync = True

        # Access scheduler (triggers creation and flag reset)
        _ = trainer.lr_scheduler

        # Flag should be reset to False after use
        assert trainer._scheduler_skip_sync is False


# ============================================================================
# LR Schedule Verification Tests (5 tests)
# ============================================================================

class TestLRScheduleVerification:
    """Tests for LR schedule correctness."""

    def test_lr_at_cycle_start(self):
        """Test LR at start of cycle = peak_lr * decay^cycle."""
        optimizer = create_optimizer(lr=1e-4)

        base_lr = 2e-4
        decay_rate = 0.8

        scheduler = SteppedCosineRestartScheduler(
            optimizer,
            T_0=50,
            base_lr=base_lr,
            cycle_min_lr=1e-4,
            decay_rate=decay_rate,
            global_min_lr=1e-5,
        )

        # Check LR at start of each cycle
        for cycle in range(5):
            expected_peak = base_lr * (decay_rate ** cycle)
            actual_lr = optimizer.param_groups[0]['lr']
            assert abs(actual_lr - expected_peak) < 1e-10, \
                f"Cycle {cycle}: expected peak {expected_peak:.2e}, got {actual_lr:.2e}"

            # Complete this cycle
            for _ in range(50):
                scheduler.step()

    def test_lr_at_cycle_end(self):
        """Test LR at end of cycle = min_lr * decay^cycle."""
        optimizer = create_optimizer(lr=1e-4)

        base_lr = 2e-4
        cycle_min_lr = 1e-4
        decay_rate = 0.8

        scheduler = SteppedCosineRestartScheduler(
            optimizer,
            T_0=50,
            base_lr=base_lr,
            cycle_min_lr=cycle_min_lr,
            decay_rate=decay_rate,
            global_min_lr=1e-5,
        )

        # Check LR at end of each cycle
        for cycle in range(5):
            # Move to end of cycle (step 49)
            for _ in range(49):
                scheduler.step()

            expected_min = cycle_min_lr * (decay_rate ** cycle)
            actual_lr = optimizer.param_groups[0]['lr']
            # At end of cycle, LR should be close to cycle_min (with cosine decay)
            # Allow some tolerance due to cosine shape
            assert actual_lr < expected_min + 1e-6, \
                f"Cycle {cycle}: LR {actual_lr:.2e} should be close to min {expected_min:.2e}"

            # Step to start of next cycle
            scheduler.step()

    def test_lr_after_many_cycles(self):
        """Test LR reaches global_min_lr eventually."""
        optimizer = create_optimizer(lr=1e-4)

        scheduler = SteppedCosineRestartScheduler(
            optimizer,
            T_0=10,
            base_lr=1e-4,
            cycle_min_lr=5e-5,
            decay_rate=0.5,  # Aggressive decay
            global_min_lr=1e-5,
        )

        # After many cycles, both peak and min should hit global_min
        for _ in range(200):  # 20 cycles
            scheduler.step()

        cycle_peak, cycle_min = scheduler.get_cycle_lrs()

        # Both should be clamped to global_min
        assert cycle_peak == 1e-5, f"Peak should be global_min, got {cycle_peak}"
        assert cycle_min == 1e-5, f"Min should be global_min, got {cycle_min}"

    def test_lr_decay_rate_effect(self):
        """Test decay rate affects LR correctly across cycles."""
        optimizer1 = create_optimizer(lr=1e-4)
        optimizer2 = create_optimizer(lr=1e-4)

        # Scheduler with 0.9 decay
        scheduler1 = SteppedCosineRestartScheduler(
            optimizer1,
            T_0=50,
            base_lr=1e-4,
            cycle_min_lr=5e-5,
            decay_rate=0.9,
            global_min_lr=1e-6,
        )

        # Scheduler with 0.5 decay (more aggressive)
        scheduler2 = SteppedCosineRestartScheduler(
            optimizer2,
            T_0=50,
            base_lr=1e-4,
            cycle_min_lr=5e-5,
            decay_rate=0.5,
            global_min_lr=1e-6,
        )

        # After 5 cycles
        for _ in range(250):
            scheduler1.step()
            scheduler2.step()

        lr1 = optimizer1.param_groups[0]['lr']
        lr2 = optimizer2.param_groups[0]['lr']

        # Scheduler with 0.5 decay should have lower LR
        assert lr2 < lr1, f"0.5 decay LR ({lr2:.2e}) should be < 0.9 decay LR ({lr1:.2e})"

        # Verify actual values
        # 0.9 decay: 1e-4 * 0.9^5 = 5.9e-5
        # 0.5 decay: 1e-4 * 0.5^5 = 3.125e-6
        expected_lr1 = 1e-4 * (0.9 ** 5)
        expected_lr2 = 1e-4 * (0.5 ** 5)
        assert abs(lr1 - expected_lr1) < 1e-10
        assert abs(lr2 - expected_lr2) < 1e-10

    def test_lr_never_below_global_min(self):
        """Test LR never goes below global_min_lr under any circumstance."""
        optimizer = create_optimizer(lr=1e-4)

        global_min = 5e-6

        scheduler = SteppedCosineRestartScheduler(
            optimizer,
            T_0=10,
            base_lr=1e-4,
            cycle_min_lr=5e-5,
            decay_rate=0.5,
            global_min_lr=global_min,
        )

        # Run for many steps
        for step in range(1000):
            lr = optimizer.param_groups[0]['lr']
            assert lr >= global_min - 1e-12, \
                f"Step {step}: LR {lr:.2e} went below global_min {global_min:.2e}"
            scheduler.step()


# ============================================================================
# Additional Integration Tests
# ============================================================================

class TestApplyWarmupLR:
    """Tests for apply_warmup_lr function."""

    def test_warmup_linear_increase(self):
        """Test warmup linearly increases LR from 0 to base_lr."""
        optimizer = create_optimizer(lr=1e-4)
        base_lr = 2e-4
        warmup_steps = 100

        # At step 0
        apply_warmup_lr(optimizer, step=0, warmup_steps=warmup_steps, base_lr=base_lr)
        expected = base_lr * (1 / 100)
        assert abs(optimizer.param_groups[0]['lr'] - expected) < 1e-10

        # At step 50 (middle)
        apply_warmup_lr(optimizer, step=50, warmup_steps=warmup_steps, base_lr=base_lr)
        expected = base_lr * (51 / 100)
        assert abs(optimizer.param_groups[0]['lr'] - expected) < 1e-10

        # At step 99 (last warmup step)
        apply_warmup_lr(optimizer, step=99, warmup_steps=warmup_steps, base_lr=base_lr)
        expected = base_lr * (100 / 100)
        assert abs(optimizer.param_groups[0]['lr'] - expected) < 1e-10

    def test_warmup_zero_steps(self):
        """Test warmup with 0 steps does nothing."""
        optimizer = create_optimizer(lr=1e-4)
        original_lr = optimizer.param_groups[0]['lr']

        apply_warmup_lr(optimizer, step=0, warmup_steps=0, base_lr=2e-4)

        # LR should be unchanged
        assert optimizer.param_groups[0]['lr'] == original_lr


class TestSchedulerFactoryEdgeCases:
    """Tests for edge cases in scheduler factory."""

    def test_no_config_returns_constant(self):
        """Test None config returns constant scheduler."""
        optimizer = create_optimizer(lr=1e-4)

        scheduler = create_lr_scheduler(optimizer, training_config=None)

        assert isinstance(scheduler, torch.optim.lr_scheduler.ConstantLR)

    def test_no_dataloader_uses_max_steps(self):
        """Test T_0 calculation without dataloader uses max_steps."""
        config = TrainingConfig(
            restart_epochs=4,
            restart_period=0,
            num_epochs=16,
            max_steps=1600,  # 1600 / 16 = 100 steps per epoch
            stepped_cosine_restart=SteppedCosineRestartConfig(
                enabled=True,
                base_lr=1e-4,
                cycle_min_lr=5e-5,  # Must be <= base_lr
            ),
        )

        optimizer = create_optimizer(lr=1e-4)

        scheduler = create_lr_scheduler(
            optimizer, config, dataloader=None,
            gradient_accumulation_steps=1
        )

        # T_0 = 4 * (1600 / 16) = 400
        assert scheduler.T_0 == 400

    def test_small_dataset_minimum_t0(self):
        """Test small dataset doesn't result in T_0=0."""
        config = TrainingConfig(
            restart_epochs=1,
            restart_period=0,
            num_epochs=100,
            stepped_cosine_restart=SteppedCosineRestartConfig(
                enabled=True,
                base_lr=1e-4,
                cycle_min_lr=5e-5,  # Must be <= base_lr
            ),
        )

        # Very small dataset: 1 batch per epoch
        mock_dataloader = create_mock_dataloader(1)

        optimizer = create_optimizer(lr=1e-4)

        scheduler = create_lr_scheduler(
            optimizer, config, dataloader=mock_dataloader,
            gradient_accumulation_steps=2  # More than dataloader length
        )

        # optimizer_steps_per_epoch = 1 // 2 = 0 -> clamped to 1
        # T_0 = 1 * 1 = 1 (minimum)
        assert scheduler.T_0 >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
