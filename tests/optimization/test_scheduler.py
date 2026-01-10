"""
Learning Rate Scheduler tests.

Tests for:
- CosineWarmupScheduler (5 tests)
- LinearWarmupScheduler (3 tests)
- Factory functions (2 tests)

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn

from src.training.optimization.scheduler import (
    CosineWarmupScheduler,
    LinearWarmupScheduler,
    ConstantWarmupScheduler,
    CosineAnnealingWarmRestartsWithDecay,
    create_scheduler,
    create_scheduler_from_config,
)


class SimpleModel(nn.Module):
    """Simple model for scheduler testing."""

    def __init__(self, in_features: int = 10, out_features: int = 10):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def step_scheduler(optimizer: torch.optim.Optimizer, scheduler, n: int = 1) -> None:
    """
    Step scheduler n times with proper optimizer.step() calls.

    PyTorch 1.1.0+ requires optimizer.step() before scheduler.step().
    This helper ensures correct ordering to avoid warnings.
    """
    for _ in range(n):
        optimizer.step()
        scheduler.step()


# ============================================================================
# CosineWarmupScheduler Tests (5 tests)
# ============================================================================

class TestCosineWarmupSchedulerInit:
    """Tests for CosineWarmupScheduler initialization."""

    def test_cosine_warmup_init(self):
        """Test CosineWarmupScheduler initializes correctly."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            min_lr=1e-6,
        )

        assert scheduler.warmup_steps == 100
        assert scheduler.total_steps == 1000
        assert scheduler.min_lr == 1e-6
        assert scheduler.base_lrs == [1e-4]

    def test_cosine_warmup_init_invalid_warmup(self):
        """Test CosineWarmupScheduler rejects invalid warmup_steps."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        with pytest.raises(ValueError, match="warmup_steps"):
            CosineWarmupScheduler(optimizer, warmup_steps=-1, total_steps=1000)

    def test_cosine_warmup_init_warmup_exceeds_total(self):
        """Test CosineWarmupScheduler rejects warmup >= total."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        with pytest.raises(ValueError, match="warmup_steps.*total_steps"):
            CosineWarmupScheduler(optimizer, warmup_steps=1000, total_steps=100)

    def test_cosine_warmup_init_invalid_total(self):
        """Test CosineWarmupScheduler rejects invalid total_steps."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        with pytest.raises(ValueError, match="total_steps"):
            CosineWarmupScheduler(optimizer, warmup_steps=100, total_steps=0)


class TestCosineWarmupWarmupPhase:
    """Tests for CosineWarmupScheduler warmup phase."""

    def test_cosine_warmup_warmup_phase(self):
        """Test learning rate increases linearly during warmup."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            min_lr=0.0,
        )

        # At step 0, LR should be 0
        assert optimizer.param_groups[0]["lr"] == 0.0

        # At step 50 (middle of warmup), LR should be base_lr * 0.5
        step_scheduler(optimizer, scheduler, 50)

        expected_lr = 1e-4 * (50 / 100)  # 50/100 = 0.5
        actual_lr = optimizer.param_groups[0]["lr"]
        assert abs(actual_lr - expected_lr) < 1e-10

        # At step 100 (end of warmup), LR should be base_lr
        step_scheduler(optimizer, scheduler, 50)

        expected_lr = 1e-4  # Full warmup
        actual_lr = optimizer.param_groups[0]["lr"]
        assert abs(actual_lr - expected_lr) < 1e-10

    def test_cosine_warmup_warmup_linear_increase(self):
        """Test warmup phase is truly linear."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            min_lr=0.0,
        )

        lrs = []
        for _ in range(100):
            optimizer.step()
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])

        # Check linearity: difference between consecutive LRs should be constant
        diffs = [lrs[i+1] - lrs[i] for i in range(len(lrs) - 1)]
        avg_diff = sum(diffs) / len(diffs)

        for diff in diffs:
            assert abs(diff - avg_diff) < 1e-12  # All diffs should be same


class TestCosineWarmupAnnealingPhase:
    """Tests for CosineWarmupScheduler annealing phase."""

    def test_cosine_warmup_annealing_phase(self):
        """Test learning rate follows cosine decay after warmup."""
        model = SimpleModel()
        base_lr = 1e-4
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)

        warmup_steps = 100
        total_steps = 1000
        min_lr = 0.0

        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=min_lr,
        )

        # Skip warmup
        step_scheduler(optimizer, scheduler, warmup_steps)

        # At end of warmup (step 100), LR = base_lr
        assert abs(optimizer.param_groups[0]["lr"] - base_lr) < 1e-10

        # At halfway through annealing (step 550 = 100 + 450)
        step_scheduler(optimizer, scheduler, 450)

        # progress = 450 / 900 = 0.5
        progress = 450 / (total_steps - warmup_steps)
        expected_lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
        actual_lr = optimizer.param_groups[0]["lr"]
        assert abs(actual_lr - expected_lr) < 1e-10

    def test_cosine_warmup_decreases_monotonically(self):
        """Test LR decreases monotonically after warmup."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            min_lr=0.0,
        )

        # Skip warmup
        step_scheduler(optimizer, scheduler, 100)

        prev_lr = optimizer.param_groups[0]["lr"]

        # Check all remaining steps decrease
        for _ in range(899):  # 1000 - 100 - 1 steps
            optimizer.step()
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            assert current_lr <= prev_lr + 1e-12  # Allow tiny numerical error
            prev_lr = current_lr


class TestCosineWarmupMinLR:
    """Tests for minimum learning rate."""

    def test_cosine_warmup_min_lr(self):
        """Test LR reaches min_lr at end of training."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        min_lr = 1e-6
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            min_lr=min_lr,
        )

        # Run to completion
        step_scheduler(optimizer, scheduler, 1000)

        final_lr = optimizer.param_groups[0]["lr"]
        assert abs(final_lr - min_lr) < 1e-10

    def test_cosine_warmup_never_below_min_lr(self):
        """Test LR never goes below min_lr."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        min_lr = 1e-6
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            min_lr=min_lr,
        )

        # Run past total_steps
        for _ in range(1500):  # 500 steps beyond total
            optimizer.step()
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            assert current_lr >= min_lr - 1e-12


class TestCosineWarmupStep:
    """Tests for step() method."""

    def test_cosine_warmup_step_updates_lr(self):
        """Test step() correctly updates learning rate."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            min_lr=0.0,
        )

        initial_lr = optimizer.param_groups[0]["lr"]
        assert initial_lr == 0.0  # At step 0

        optimizer.step()
        scheduler.step()
        lr_after_one_step = optimizer.param_groups[0]["lr"]
        assert lr_after_one_step > 0.0  # Should have increased

        optimizer.step()
        scheduler.step()
        lr_after_two_steps = optimizer.param_groups[0]["lr"]
        assert lr_after_two_steps > lr_after_one_step  # Still increasing in warmup


# ============================================================================
# LinearWarmupScheduler Tests (3 tests)
# ============================================================================

class TestLinearWarmupSchedulerInit:
    """Tests for LinearWarmupScheduler initialization."""

    def test_linear_warmup_init(self):
        """Test LinearWarmupScheduler initializes correctly."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        scheduler = LinearWarmupScheduler(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            min_lr=1e-6,
        )

        assert scheduler.warmup_steps == 100
        assert scheduler.total_steps == 1000
        assert scheduler.min_lr == 1e-6


class TestLinearWarmupWarmupPhase:
    """Tests for LinearWarmupScheduler warmup phase."""

    def test_linear_warmup_warmup_phase(self):
        """Test warmup phase is linear (same as cosine warmup)."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        scheduler = LinearWarmupScheduler(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            min_lr=0.0,
        )

        # At step 50, should be 50% of base_lr
        step_scheduler(optimizer, scheduler, 50)

        expected_lr = 1e-4 * 0.5
        actual_lr = optimizer.param_groups[0]["lr"]
        assert abs(actual_lr - expected_lr) < 1e-10


class TestLinearWarmupDecayPhase:
    """Tests for LinearWarmupScheduler decay phase."""

    def test_linear_warmup_decay_phase(self):
        """Test decay phase is linear (not cosine)."""
        model = SimpleModel()
        base_lr = 1e-4
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)

        warmup_steps = 100
        total_steps = 1000
        min_lr = 0.0

        scheduler = LinearWarmupScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=min_lr,
        )

        # Skip warmup
        step_scheduler(optimizer, scheduler, warmup_steps)

        # Record LRs during decay
        lrs = []
        for _ in range(total_steps - warmup_steps):
            optimizer.step()
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])

        # Check linearity of decay
        diffs = [lrs[i] - lrs[i+1] for i in range(len(lrs) - 1)]
        avg_diff = sum(diffs) / len(diffs)

        for diff in diffs:
            assert abs(diff - avg_diff) < 1e-12  # All diffs should be same

    def test_linear_warmup_reaches_min_lr(self):
        """Test LinearWarmupScheduler reaches min_lr at end."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        min_lr = 1e-6
        scheduler = LinearWarmupScheduler(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            min_lr=min_lr,
        )

        step_scheduler(optimizer, scheduler, 1000)

        final_lr = optimizer.param_groups[0]["lr"]
        assert abs(final_lr - min_lr) < 1e-10


# ============================================================================
# ConstantWarmupScheduler Tests
# ============================================================================

class TestConstantWarmupScheduler:
    """Tests for ConstantWarmupScheduler."""

    def test_constant_warmup_init(self):
        """Test ConstantWarmupScheduler initializes correctly."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        scheduler = ConstantWarmupScheduler(
            optimizer,
            warmup_steps=100,
        )

        assert scheduler.warmup_steps == 100

    def test_constant_warmup_stays_constant(self):
        """Test LR stays constant after warmup."""
        model = SimpleModel()
        base_lr = 1e-4
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)

        scheduler = ConstantWarmupScheduler(
            optimizer,
            warmup_steps=100,
        )

        # Skip warmup
        step_scheduler(optimizer, scheduler, 100)

        # LR should be constant at base_lr
        for _ in range(500):
            optimizer.step()
            scheduler.step()
            assert optimizer.param_groups[0]["lr"] == base_lr


# ============================================================================
# Factory Function Tests (2 tests)
# ============================================================================

class TestCreateScheduler:
    """Tests for create_scheduler factory function."""

    def test_create_scheduler_cosine(self):
        """Test creating cosine scheduler."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        scheduler = create_scheduler(
            optimizer,
            scheduler_type="cosine",
            warmup_steps=100,
            total_steps=1000,
            min_lr=1e-6,
        )

        assert isinstance(scheduler, CosineWarmupScheduler)
        assert scheduler.warmup_steps == 100
        assert scheduler.total_steps == 1000

    def test_create_scheduler_linear(self):
        """Test creating linear scheduler."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        scheduler = create_scheduler(
            optimizer,
            scheduler_type="linear",
            warmup_steps=100,
            total_steps=1000,
        )

        assert isinstance(scheduler, LinearWarmupScheduler)

    def test_create_scheduler_constant(self):
        """Test creating constant scheduler."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        scheduler = create_scheduler(
            optimizer,
            scheduler_type="constant",
            warmup_steps=100,
            total_steps=1000,
        )

        assert isinstance(scheduler, ConstantWarmupScheduler)

    def test_create_scheduler_invalid_type(self):
        """Test invalid scheduler type raises error."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        with pytest.raises(ValueError, match="Unknown scheduler type"):
            create_scheduler(
                optimizer,
                scheduler_type="invalid",
                warmup_steps=100,
                total_steps=1000,
            )


class TestCreateSchedulerFromConfig:
    """Tests for create_scheduler_from_config function."""

    def test_create_scheduler_from_config(self):
        """Test creating scheduler from real TrainingConfig."""
        from src.config.model_config import TrainingConfig

        # 使用真實的 TrainingConfig，而非 Mock
        config = TrainingConfig(
            lr_scheduler="cosine",
            warmup_steps=500,
            max_steps=50000,
            min_lr=1e-6,
        )

        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        scheduler = create_scheduler_from_config(optimizer, config)

        assert isinstance(scheduler, CosineWarmupScheduler)
        assert scheduler.warmup_steps == 500
        assert scheduler.total_steps == 50000
        assert scheduler.min_lr == 1e-6

    def test_create_scheduler_from_config_linear(self):
        """Test creating linear scheduler from real TrainingConfig."""
        from src.config.model_config import TrainingConfig

        # 使用真實的 TrainingConfig，而非 Mock
        config = TrainingConfig(
            lr_scheduler="linear",
            warmup_steps=1000,
            max_steps=100000,
            min_lr=0.0,
        )

        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        scheduler = create_scheduler_from_config(optimizer, config)

        assert isinstance(scheduler, LinearWarmupScheduler)


# ============================================================================
# CosineAnnealingWarmRestartsWithDecay Tests
# ============================================================================

class TestCosineAnnealingWarmRestartsWithDecayInit:
    """Tests for CosineAnnealingWarmRestartsWithDecay initialization."""

    def test_init_default(self):
        """Test default initialization."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        scheduler = CosineAnnealingWarmRestartsWithDecay(
            optimizer,
            T_0=100,
        )

        assert scheduler.T_0 == 100
        assert scheduler.T_mult == 1
        assert scheduler.eta_min == 0.0
        assert scheduler.lr_decay_per_cycle == 1.0
        assert scheduler.cycle == 0
        assert scheduler.T_cur == 0

    def test_init_with_decay(self):
        """Test initialization with decay parameter."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        scheduler = CosineAnnealingWarmRestartsWithDecay(
            optimizer,
            T_0=100,
            eta_min=1e-6,
            lr_decay_per_cycle=0.9,
        )

        assert scheduler.lr_decay_per_cycle == 0.9
        assert scheduler.eta_min == 1e-6

    def test_init_invalid_T_0(self):
        """Test invalid T_0 raises error."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        with pytest.raises(ValueError, match="T_0 must be positive"):
            CosineAnnealingWarmRestartsWithDecay(optimizer, T_0=0)

    def test_init_invalid_decay(self):
        """Test invalid lr_decay_per_cycle raises error."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        with pytest.raises(ValueError, match="lr_decay_per_cycle must be in"):
            CosineAnnealingWarmRestartsWithDecay(optimizer, T_0=100, lr_decay_per_cycle=0)

        with pytest.raises(ValueError, match="lr_decay_per_cycle must be in"):
            CosineAnnealingWarmRestartsWithDecay(optimizer, T_0=100, lr_decay_per_cycle=1.5)


class TestCosineAnnealingWarmRestartsWithDecayCycleRestart:
    """Tests for cycle restart behavior."""

    def test_single_cycle_lr_curve(self):
        """Test LR curve within a single cycle (no decay effect)."""
        model = SimpleModel()
        base_lr = 1e-4
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)

        T_0 = 100
        eta_min = 1e-6

        scheduler = CosineAnnealingWarmRestartsWithDecay(
            optimizer,
            T_0=T_0,
            eta_min=eta_min,
            lr_decay_per_cycle=0.9,
        )

        # At step 0, LR should be peak (base_lr)
        assert abs(optimizer.param_groups[0]["lr"] - base_lr) < 1e-10

        # At step T_0/2 (middle of cycle), LR should be between peak and min
        for _ in range(T_0 // 2):
            optimizer.step()
            scheduler.step()

        mid_lr = optimizer.param_groups[0]["lr"]
        # 餘弦中點: 0.5 * (peak + min)
        expected_mid_lr = 0.5 * (base_lr + eta_min)
        assert abs(mid_lr - expected_mid_lr) < 1e-10

        # At step T_0 - 1 (end of cycle), LR should be close to eta_min
        for _ in range(T_0 // 2 - 1):
            optimizer.step()
            scheduler.step()

        # Step 99: almost at eta_min
        end_lr = optimizer.param_groups[0]["lr"]
        assert end_lr < base_lr / 2  # Should be much lower than peak

    def test_cycle_restart_with_decay(self):
        """Test LR restarts with decay after each cycle."""
        model = SimpleModel()
        base_lr = 1e-4
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)

        T_0 = 100
        eta_min = 0.0
        lr_decay = 0.9

        scheduler = CosineAnnealingWarmRestartsWithDecay(
            optimizer,
            T_0=T_0,
            eta_min=eta_min,
            lr_decay_per_cycle=lr_decay,
        )

        # Complete first cycle
        for _ in range(T_0):
            optimizer.step()
            scheduler.step()

        # After cycle 0 ends, we're now at cycle 1, step 0
        # Peak LR should be base_lr * lr_decay
        assert scheduler.cycle == 1
        expected_peak = base_lr * lr_decay  # 1e-4 * 0.9 = 9e-5
        actual_lr = optimizer.param_groups[0]["lr"]
        assert abs(actual_lr - expected_peak) < 1e-10

        # Complete second cycle
        for _ in range(T_0):
            optimizer.step()
            scheduler.step()

        # After cycle 1 ends, we're at cycle 2, step 0
        # Peak LR should be base_lr * lr_decay^2
        assert scheduler.cycle == 2
        expected_peak = base_lr * (lr_decay ** 2)  # 1e-4 * 0.81 = 8.1e-5
        actual_lr = optimizer.param_groups[0]["lr"]
        assert abs(actual_lr - expected_peak) < 1e-10

    def test_decay_respects_min_lr(self):
        """Test decayed peak LR does not go below min_lr."""
        model = SimpleModel()
        base_lr = 1e-4
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)

        T_0 = 10
        eta_min = 5e-5  # Relatively high min_lr
        lr_decay = 0.5   # Aggressive decay

        scheduler = CosineAnnealingWarmRestartsWithDecay(
            optimizer,
            T_0=T_0,
            eta_min=eta_min,
            lr_decay_per_cycle=lr_decay,
        )

        # Cycle 0: peak = 1e-4
        # Cycle 1: peak = 1e-4 * 0.5 = 5e-5 (equals min_lr)
        # Cycle 2: peak = 1e-4 * 0.25 = 2.5e-5, but clamped to min_lr = 5e-5

        # Complete 3 cycles
        for cycle in range(3):
            for _ in range(T_0):
                optimizer.step()
                scheduler.step()

        # After 3 cycles, peak should be clamped to min_lr
        assert scheduler.cycle == 3
        expected_peak = eta_min  # Clamped
        actual_lr = optimizer.param_groups[0]["lr"]
        assert actual_lr >= eta_min - 1e-12


class TestCosineAnnealingWarmRestartsWithDecayStateDictSaveLoad:
    """Tests for state_dict save/load."""

    def test_state_dict_save_load(self):
        """Test scheduler state can be saved and loaded."""
        model = SimpleModel()
        base_lr = 1e-4
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)

        scheduler = CosineAnnealingWarmRestartsWithDecay(
            optimizer,
            T_0=100,
            eta_min=1e-6,
            lr_decay_per_cycle=0.9,
        )

        # Run for 150 steps (1.5 cycles)
        for _ in range(150):
            optimizer.step()
            scheduler.step()

        # Save state
        state = scheduler.state_dict()
        lr_before = optimizer.param_groups[0]["lr"]
        cycle_before = scheduler.cycle
        T_cur_before = scheduler.T_cur

        # Create new scheduler and load state
        new_optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
        new_scheduler = CosineAnnealingWarmRestartsWithDecay(
            new_optimizer,
            T_0=100,
            eta_min=1e-6,
            lr_decay_per_cycle=0.9,
        )

        new_scheduler.load_state_dict(state)

        # Verify state restored
        assert new_scheduler.cycle == cycle_before
        assert new_scheduler.T_cur == T_cur_before

        # Continue training and verify LR is consistent
        optimizer.step()
        scheduler.step()
        new_optimizer.step()
        new_scheduler.step()

        assert abs(optimizer.param_groups[0]["lr"] - new_optimizer.param_groups[0]["lr"]) < 1e-10


class TestCosineAnnealingWarmRestartsWithDecayNoDecay:
    """Tests for lr_decay_per_cycle=1.0 (no decay)."""

    def test_no_decay_matches_pytorch(self):
        """Test with decay=1.0 behaves like standard CosineAnnealingWarmRestarts."""
        model = SimpleModel()
        base_lr = 1e-4
        eta_min = 1e-6
        T_0 = 50

        optimizer1 = torch.optim.AdamW(model.parameters(), lr=base_lr)
        scheduler1 = CosineAnnealingWarmRestartsWithDecay(
            optimizer1,
            T_0=T_0,
            eta_min=eta_min,
            lr_decay_per_cycle=1.0,  # No decay
        )

        optimizer2 = torch.optim.AdamW(model.parameters(), lr=base_lr)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer2,
            T_0=T_0,
            eta_min=eta_min,
        )

        # Run for 200 steps (4 cycles) and compare LRs
        for _ in range(200):
            optimizer1.step()
            scheduler1.step()
            optimizer2.step()
            scheduler2.step()

            lr1 = optimizer1.param_groups[0]["lr"]
            lr2 = optimizer2.param_groups[0]["lr"]
            assert abs(lr1 - lr2) < 1e-10, f"LR mismatch: {lr1} vs {lr2}"


class TestCosineAnnealingWarmRestartsWithDecayTMult:
    """Tests for T_mult > 1 (increasing cycle length)."""

    def test_t_mult_increases_cycle_length(self):
        """Test T_mult correctly increases cycle length."""
        model = SimpleModel()
        base_lr = 1e-4
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)

        T_0 = 10
        T_mult = 2

        scheduler = CosineAnnealingWarmRestartsWithDecay(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=0.0,
            lr_decay_per_cycle=0.9,
        )

        # Cycle 0: T_i = 10
        for _ in range(T_0):
            optimizer.step()
            scheduler.step()

        assert scheduler.cycle == 1
        assert scheduler.T_i == T_0 * T_mult  # 20

        # Cycle 1: T_i = 20
        for _ in range(T_0 * T_mult):
            optimizer.step()
            scheduler.step()

        assert scheduler.cycle == 2
        assert scheduler.T_i == T_0 * T_mult * T_mult  # 40
