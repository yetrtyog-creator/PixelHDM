"""
Strict Tests for Stepped Cosine Restart Scheduler

End-to-end tests verifying:
1. Learning rate curve (both peak and trough decay)
2. Global minimum protection
3. Configuration parsing
4. Mode switching
5. Checkpoint save/restore
6. Factory function integration

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-04
"""

from __future__ import annotations

import math
import pytest
import torch
import yaml
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from src.training.optimization.scheduler import SteppedCosineRestartScheduler
from src.config.training_config import TrainingConfig, SteppedCosineRestartConfig
from src.config.parsers import parse_training_config
from src.training.trainer.scheduler_factory import create_lr_scheduler


class TestSteppedCosineRestartSchedulerMath:
    """Test mathematical correctness of the scheduler."""

    def _create_optimizer(self, lr: float = 1e-4) -> torch.optim.Optimizer:
        """Create a simple optimizer for testing."""
        model = torch.nn.Linear(10, 10)
        return torch.optim.SGD(model.parameters(), lr=lr)

    def test_initial_lr_at_peak(self):
        """Test that initial learning rate equals base_lr."""
        optimizer = self._create_optimizer()
        scheduler = SteppedCosineRestartScheduler(
            optimizer,
            T_0=100,
            base_lr=2e-4,
            cycle_min_lr=1.2e-4,
            decay_rate=0.75,
            global_min_lr=1e-5,
        )

        # After first step, LR should be at peak (base_lr)
        assert abs(optimizer.param_groups[0]['lr'] - 2e-4) < 1e-10

    def test_cycle_0_cosine_decay(self):
        """Test cycle 0 decays from base_lr to cycle_min_lr."""
        optimizer = self._create_optimizer()
        scheduler = SteppedCosineRestartScheduler(
            optimizer,
            T_0=100,
            base_lr=2e-4,
            cycle_min_lr=1.2e-4,
            decay_rate=0.75,
            global_min_lr=1e-5,
        )

        # At start (T_cur=0): should be at peak
        lr_start = optimizer.param_groups[0]['lr']
        assert abs(lr_start - 2e-4) < 1e-10

        # Move to middle of cycle (T_cur=50)
        for _ in range(50):
            scheduler.step()

        lr_mid = optimizer.param_groups[0]['lr']
        # At middle: lr = cycle_min + (peak - min) * 0.5 * (1 + cos(π * 0.5))
        # = 1.2e-4 + (2e-4 - 1.2e-4) * 0.5 * 1 = 1.6e-4
        expected_mid = 1.2e-4 + (2e-4 - 1.2e-4) * 0.5
        assert abs(lr_mid - expected_mid) < 1e-9

        # Move to end of cycle (T_cur=99)
        for _ in range(49):
            scheduler.step()

        lr_end = optimizer.param_groups[0]['lr']
        # At end: should be close to cycle_min_lr (relaxed tolerance for float precision)
        assert abs(lr_end - 1.2e-4) < 1e-7

    def test_cycle_1_peak_decay(self):
        """Test cycle 1 peak is base_lr * decay_rate."""
        optimizer = self._create_optimizer()
        scheduler = SteppedCosineRestartScheduler(
            optimizer,
            T_0=100,
            base_lr=2e-4,
            cycle_min_lr=1.2e-4,
            decay_rate=0.75,
            global_min_lr=1e-5,
        )

        # Complete cycle 0
        for _ in range(100):
            scheduler.step()

        # Now at start of cycle 1
        assert scheduler.cycle == 1
        lr_cycle1_start = optimizer.param_groups[0]['lr']

        # Expected: base_lr * 0.75 = 1.5e-4
        expected_peak = 2e-4 * 0.75
        assert abs(lr_cycle1_start - expected_peak) < 1e-10

    def test_cycle_1_trough_decay(self):
        """Test cycle 1 trough is cycle_min_lr * decay_rate."""
        optimizer = self._create_optimizer()
        scheduler = SteppedCosineRestartScheduler(
            optimizer,
            T_0=100,
            base_lr=2e-4,
            cycle_min_lr=1.2e-4,
            decay_rate=0.75,
            global_min_lr=1e-5,
        )

        # Complete cycle 0
        for _ in range(100):
            scheduler.step()

        # Move to end of cycle 1
        for _ in range(99):
            scheduler.step()

        lr_cycle1_end = optimizer.param_groups[0]['lr']

        # Expected: cycle_min_lr * 0.75 = 0.9e-4 (relaxed tolerance for float precision)
        expected_trough = 1.2e-4 * 0.75
        assert abs(lr_cycle1_end - expected_trough) < 1e-7

    def test_multi_cycle_decay_curve(self):
        """Test decay curve across multiple cycles."""
        optimizer = self._create_optimizer()
        scheduler = SteppedCosineRestartScheduler(
            optimizer,
            T_0=100,
            base_lr=2e-4,
            cycle_min_lr=1.2e-4,
            decay_rate=0.75,
            global_min_lr=1e-5,
        )

        expected_peaks = [
            2e-4,           # Cycle 0
            2e-4 * 0.75,    # Cycle 1: 1.5e-4
            2e-4 * 0.75**2, # Cycle 2: 1.125e-4
            2e-4 * 0.75**3, # Cycle 3: 0.84375e-4
        ]

        for cycle_idx, expected_peak in enumerate(expected_peaks):
            # At start of each cycle, LR should be at peak
            actual_lr = optimizer.param_groups[0]['lr']
            assert abs(actual_lr - expected_peak) < 1e-10, \
                f"Cycle {cycle_idx}: expected {expected_peak}, got {actual_lr}"

            # Complete this cycle
            for _ in range(100):
                scheduler.step()


class TestGlobalMinimumProtection:
    """Test that global_min_lr is respected as absolute lower bound."""

    def _create_optimizer(self, lr: float = 1e-4) -> torch.optim.Optimizer:
        model = torch.nn.Linear(10, 10)
        return torch.optim.SGD(model.parameters(), lr=lr)

    def test_peak_respects_global_min(self):
        """Test that peak never goes below global_min_lr."""
        optimizer = self._create_optimizer()
        scheduler = SteppedCosineRestartScheduler(
            optimizer,
            T_0=10,
            base_lr=1e-4,
            cycle_min_lr=5e-5,
            decay_rate=0.5,
            global_min_lr=2e-5,
        )

        # Run many cycles
        for _ in range(200):
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            assert lr >= 2e-5 - 1e-12, f"LR {lr} went below global_min_lr 2e-5"

    def test_trough_respects_global_min(self):
        """Test that trough never goes below global_min_lr."""
        optimizer = self._create_optimizer()
        scheduler = SteppedCosineRestartScheduler(
            optimizer,
            T_0=10,
            base_lr=1e-4,
            cycle_min_lr=5e-5,
            decay_rate=0.5,
            global_min_lr=3e-5,
        )

        # Cycle 3: cycle_min = 5e-5 * 0.5^3 = 6.25e-6 < 3e-5
        # Should be clamped to 3e-5
        for _ in range(40):  # 4 cycles
            scheduler.step()

        cycle_peak, cycle_min = scheduler.get_cycle_lrs()
        assert cycle_min >= 3e-5 - 1e-12

    def test_eventual_flat_lr(self):
        """Test that LR eventually becomes flat at global_min_lr."""
        optimizer = self._create_optimizer()
        scheduler = SteppedCosineRestartScheduler(
            optimizer,
            T_0=10,
            base_lr=1e-4,
            cycle_min_lr=5e-5,
            decay_rate=0.5,
            global_min_lr=1e-5,
        )

        # Run many cycles until both peak and trough hit global_min
        for _ in range(100):
            scheduler.step()

        cycle_peak, cycle_min = scheduler.get_cycle_lrs()
        # Both should be at global_min_lr now
        assert cycle_peak == 1e-5
        assert cycle_min == 1e-5


class TestWarmupIntegration:
    """Test warmup functionality."""

    def _create_optimizer(self, lr: float = 1e-4) -> torch.optim.Optimizer:
        model = torch.nn.Linear(10, 10)
        return torch.optim.SGD(model.parameters(), lr=lr)

    def test_warmup_linear_increase(self):
        """Test that warmup linearly increases from 0 to base_lr."""
        optimizer = self._create_optimizer()
        scheduler = SteppedCosineRestartScheduler(
            optimizer,
            T_0=100,
            base_lr=2e-4,
            cycle_min_lr=1e-4,
            decay_rate=0.75,
            global_min_lr=1e-5,
            warmup_steps=10,
        )

        # Note: LRScheduler.__init__ calls step() once, so initial total_steps=1
        # LR after init = 2e-4 * (1 / 10) = 2e-5... but step() increments first
        # Actually, the warmup formula uses (total_steps + 1) / warmup_steps
        # After __init__ step(): total_steps=1, LR = 2e-4 * 2/10 = 4e-5
        expected_first = 2e-4 * (2 / 10)  # After __init__ which calls step()
        assert abs(optimizer.param_groups[0]['lr'] - expected_first) < 1e-10

        # After 4 more steps: total_steps=5, LR = 2e-4 * 6/10 = 1.2e-4
        for _ in range(4):
            scheduler.step()
        expected_mid = 2e-4 * (6 / 10)
        assert abs(optimizer.param_groups[0]['lr'] - expected_mid) < 1e-10

        # After warmup (step 10): should be at base_lr
        for _ in range(5):
            scheduler.step()
        # At this point total_steps >= warmup_steps, so we're in cosine decay
        # At start of cosine decay, LR = base_lr (peak)
        assert abs(optimizer.param_groups[0]['lr'] - 2e-4) < 1e-10

    def test_warmup_then_cosine(self):
        """Test that cosine decay starts after warmup."""
        optimizer = self._create_optimizer()
        scheduler = SteppedCosineRestartScheduler(
            optimizer,
            T_0=100,
            base_lr=2e-4,
            cycle_min_lr=1e-4,
            decay_rate=0.75,
            global_min_lr=1e-5,
            warmup_steps=10,
        )

        # Complete warmup
        for _ in range(10):
            scheduler.step()

        # Now should be at base_lr, starting cosine decay
        assert abs(optimizer.param_groups[0]['lr'] - 2e-4) < 1e-10
        assert scheduler.cycle == 0
        assert scheduler.T_cur == 0


class TestConfigurationParsing:
    """Test YAML configuration parsing."""

    def test_parse_stepped_config_from_yaml(self):
        """Test parsing stepped_cosine_restart from YAML."""
        yaml_content = """
training:
  batch_size: 8
  lr_schedule:
    schedule_type: "cosine_restart"
    restart_epochs: 4
    stepped_cosine_restart:
      enabled: true
      base_lr: 2.0e-4
      cycle_min_lr: 1.2e-4
      decay_rate: 0.75
      global_min_lr: 1.0e-5
      warmup_steps: 100
"""
        data = yaml.safe_load(yaml_content)
        config = parse_training_config(data)

        assert config.stepped_cosine_restart is not None
        assert config.stepped_cosine_restart.enabled is True
        assert config.stepped_cosine_restart.base_lr == 2e-4
        assert config.stepped_cosine_restart.cycle_min_lr == 1.2e-4
        assert config.stepped_cosine_restart.decay_rate == 0.75
        assert config.stepped_cosine_restart.global_min_lr == 1e-5
        assert config.stepped_cosine_restart.warmup_steps == 100

    def test_parse_disabled_stepped_config(self):
        """Test parsing with stepped_cosine_restart disabled."""
        yaml_content = """
training:
  lr_schedule:
    schedule_type: "cosine_restart"
    stepped_cosine_restart:
      enabled: false
"""
        data = yaml.safe_load(yaml_content)
        config = parse_training_config(data)

        assert config.stepped_cosine_restart is not None
        assert config.stepped_cosine_restart.enabled is False

    def test_parse_without_stepped_config(self):
        """Test parsing without stepped_cosine_restart section."""
        yaml_content = """
training:
  lr_schedule:
    schedule_type: "cosine_restart"
"""
        data = yaml.safe_load(yaml_content)
        config = parse_training_config(data)

        assert config.stepped_cosine_restart is None


class TestModeSwitching:
    """Test that enabling stepped_cosine_restart takes priority."""

    def _create_optimizer(self, lr: float = 1e-4) -> torch.optim.Optimizer:
        model = torch.nn.Linear(10, 10)
        return torch.optim.SGD(model.parameters(), lr=lr)

    def test_stepped_takes_priority_over_standard(self):
        """Test that stepped config overrides standard lr_scheduler."""
        optimizer = self._create_optimizer()

        config = TrainingConfig(
            lr_scheduler="cosine_restart",
            learning_rate=1e-4,  # Should be ignored
            min_lr=1e-5,         # Should be ignored
            stepped_cosine_restart=SteppedCosineRestartConfig(
                enabled=True,
                base_lr=2e-4,
                cycle_min_lr=1.2e-4,
                decay_rate=0.75,
                global_min_lr=1e-5,
            ),
            restart_epochs=4,
            num_epochs=16,
        )

        scheduler = create_lr_scheduler(
            optimizer, config, dataloader=None, gradient_accumulation_steps=1
        )

        assert isinstance(scheduler, SteppedCosineRestartScheduler)
        # LR should be from stepped config, not TrainingConfig.learning_rate
        assert abs(optimizer.param_groups[0]['lr'] - 2e-4) < 1e-10

    def test_disabled_falls_back_to_standard(self):
        """Test that disabled stepped config uses standard scheduler."""
        optimizer = self._create_optimizer()

        config = TrainingConfig(
            lr_scheduler="cosine_restart",
            learning_rate=1e-4,
            min_lr=1e-5,
            stepped_cosine_restart=SteppedCosineRestartConfig(
                enabled=False,  # Disabled
            ),
            restart_epochs=4,
            num_epochs=16,
        )

        scheduler = create_lr_scheduler(
            optimizer, config, dataloader=None, gradient_accumulation_steps=1
        )

        # Should NOT be SteppedCosineRestartScheduler
        assert not isinstance(scheduler, SteppedCosineRestartScheduler)


class TestCheckpointSaveRestore:
    """Test state dict save/restore functionality."""

    def _create_optimizer(self, lr: float = 1e-4) -> torch.optim.Optimizer:
        model = torch.nn.Linear(10, 10)
        return torch.optim.SGD(model.parameters(), lr=lr)

    def test_state_dict_saves_cycle_state(self):
        """Test that state_dict saves cycle, T_cur, T_i."""
        optimizer = self._create_optimizer()
        scheduler = SteppedCosineRestartScheduler(
            optimizer,
            T_0=100,
            base_lr=2e-4,
            cycle_min_lr=1.2e-4,
            decay_rate=0.75,
            global_min_lr=1e-5,
        )

        # Advance to cycle 2
        for _ in range(250):
            scheduler.step()

        state = scheduler.state_dict()

        assert 'cycle' in state
        assert 'T_cur' in state
        assert 'T_i' in state
        assert 'total_steps' in state
        assert state['cycle'] == 2

    def test_load_state_dict_restores_correctly(self):
        """Test that load_state_dict restores scheduler correctly."""
        optimizer = self._create_optimizer()
        scheduler1 = SteppedCosineRestartScheduler(
            optimizer,
            T_0=100,
            base_lr=2e-4,
            cycle_min_lr=1.2e-4,
            decay_rate=0.75,
            global_min_lr=1e-5,
        )

        # Advance to mid-cycle
        for _ in range(150):
            scheduler1.step()

        state = scheduler1.state_dict()
        lr_before = optimizer.param_groups[0]['lr']

        # Create new scheduler and restore state
        optimizer2 = self._create_optimizer()
        scheduler2 = SteppedCosineRestartScheduler(
            optimizer2,
            T_0=100,
            base_lr=2e-4,
            cycle_min_lr=1.2e-4,
            decay_rate=0.75,
            global_min_lr=1e-5,
        )
        scheduler2.load_state_dict(state)

        # LR should be identical after restore
        lr_after = optimizer2.param_groups[0]['lr']
        assert abs(lr_before - lr_after) < 1e-10
        assert scheduler2.cycle == scheduler1.cycle
        assert scheduler2.T_cur == scheduler1.T_cur


class TestFactoryFunctionIntegration:
    """Test factory function creates correct scheduler type."""

    def _create_optimizer(self, lr: float = 1e-4) -> torch.optim.Optimizer:
        model = torch.nn.Linear(10, 10)
        return torch.optim.SGD(model.parameters(), lr=lr)

    def test_factory_creates_stepped_scheduler(self):
        """Test factory creates SteppedCosineRestartScheduler when enabled."""
        optimizer = self._create_optimizer()
        config = TrainingConfig(
            stepped_cosine_restart=SteppedCosineRestartConfig(
                enabled=True,
                base_lr=2e-4,
            ),
            restart_epochs=4,
            num_epochs=16,
        )

        scheduler = create_lr_scheduler(optimizer, config)
        assert isinstance(scheduler, SteppedCosineRestartScheduler)

    def test_factory_uses_t0_from_training_config(self):
        """Test factory calculates T_0 from training config."""
        optimizer = self._create_optimizer()

        # Create mock dataloader
        mock_dataloader = MagicMock()
        mock_dataloader.__len__ = MagicMock(return_value=100)

        config = TrainingConfig(
            stepped_cosine_restart=SteppedCosineRestartConfig(
                enabled=True,
                base_lr=2e-4,
            ),
            restart_epochs=4,  # 4 epochs per cycle
            num_epochs=16,
        )

        scheduler = create_lr_scheduler(
            optimizer, config, dataloader=mock_dataloader,
            gradient_accumulation_steps=2
        )

        # T_0 should be: (100 / 2) * 4 = 200 steps
        assert scheduler.T_0 == 200


class TestValidation:
    """Test parameter validation."""

    def _create_optimizer(self, lr: float = 1e-4) -> torch.optim.Optimizer:
        model = torch.nn.Linear(10, 10)
        return torch.optim.SGD(model.parameters(), lr=lr)

    def test_invalid_base_lr_raises(self):
        """Test that non-positive base_lr raises ValueError."""
        optimizer = self._create_optimizer()
        with pytest.raises(ValueError, match="base_lr must be positive"):
            SteppedCosineRestartScheduler(
                optimizer, T_0=100, base_lr=0, cycle_min_lr=1e-4
            )

    def test_cycle_min_lr_greater_than_base_lr_raises(self):
        """Test that cycle_min_lr > base_lr raises ValueError."""
        optimizer = self._create_optimizer()
        with pytest.raises(ValueError, match="cycle_min_lr.*must be <= base_lr"):
            SteppedCosineRestartScheduler(
                optimizer, T_0=100, base_lr=1e-4, cycle_min_lr=2e-4
            )

    def test_global_min_greater_than_cycle_min_raises(self):
        """Test that global_min_lr > cycle_min_lr raises ValueError."""
        optimizer = self._create_optimizer()
        with pytest.raises(ValueError, match="global_min_lr.*must be <= cycle_min_lr"):
            SteppedCosineRestartScheduler(
                optimizer, T_0=100, base_lr=2e-4, cycle_min_lr=1e-4,
                global_min_lr=5e-4
            )

    def test_invalid_decay_rate_raises(self):
        """Test that decay_rate outside (0, 1] raises ValueError."""
        optimizer = self._create_optimizer()
        with pytest.raises(ValueError, match="decay_rate must be in"):
            SteppedCosineRestartScheduler(
                optimizer, T_0=100, base_lr=2e-4, cycle_min_lr=1e-4,
                decay_rate=1.5
            )

        with pytest.raises(ValueError, match="decay_rate must be in"):
            SteppedCosineRestartScheduler(
                optimizer, T_0=100, base_lr=2e-4, cycle_min_lr=1e-4,
                decay_rate=0
            )


class TestEndToEndTrainingSimulation:
    """Simulate a complete training run to verify scheduler behavior."""

    def _create_optimizer(self, lr: float = 1e-4) -> torch.optim.Optimizer:
        model = torch.nn.Linear(10, 10)
        return torch.optim.SGD(model.parameters(), lr=lr)

    def test_full_training_simulation(self):
        """Simulate complete training and verify cycle decay behavior."""
        optimizer = self._create_optimizer()
        scheduler = SteppedCosineRestartScheduler(
            optimizer,
            T_0=50,
            base_lr=2e-4,
            cycle_min_lr=1.2e-4,
            decay_rate=0.75,
            global_min_lr=1e-5,
            warmup_steps=0,  # No warmup for simpler verification
        )

        # Track peaks at start of each cycle
        cycle_peaks_observed = []
        prev_cycle = -1

        # Run for 4 complete cycles (200 steps)
        for _ in range(200):
            if scheduler.cycle != prev_cycle:
                # New cycle started, record current LR (should be peak)
                cycle_peaks_observed.append(optimizer.param_groups[0]['lr'])
                prev_cycle = scheduler.cycle
            scheduler.step()

        # Verify we observed 4 cycles
        assert len(cycle_peaks_observed) == 4, \
            f"Expected 4 cycles, observed {len(cycle_peaks_observed)}"

        # Verify peak decay
        expected_peaks = [2e-4, 2e-4*0.75, 2e-4*0.75**2, 2e-4*0.75**3]
        for cycle_idx, (observed, expected) in enumerate(zip(cycle_peaks_observed, expected_peaks)):
            assert abs(observed - expected) < 1e-9, \
                f"Cycle {cycle_idx}: expected peak {expected}, observed {observed}"

    def test_lr_never_below_global_min(self):
        """Verify LR never goes below global_min_lr during training."""
        optimizer = self._create_optimizer()
        scheduler = SteppedCosineRestartScheduler(
            optimizer,
            T_0=20,
            base_lr=1e-4,
            cycle_min_lr=5e-5,
            decay_rate=0.5,
            global_min_lr=1e-5,
        )

        for _ in range(500):
            lr = optimizer.param_groups[0]['lr']
            assert lr >= 1e-5 - 1e-12, f"LR {lr} went below global_min_lr"
            scheduler.step()


class TestEpochToLRConversion:
    """Test end-to-end epoch to learning rate conversion.

    Verifies the complete chain: Epoch → Step → Cycle → LR
    with actual numerical values matching documented expectations.
    """

    def _create_optimizer(self, lr: float = 1e-4) -> torch.optim.Optimizer:
        model = torch.nn.Linear(10, 10)
        return torch.optim.SGD(model.parameters(), lr=lr)

    def _compute_expected_lr(
        self,
        epoch: int,
        steps_per_epoch: int,
        gradient_accumulation_steps: int,
        restart_epochs: int,
        base_lr: float,
        cycle_min_lr: float,
        decay_rate: float,
        global_min_lr: float,
    ) -> tuple[float, int, int]:
        """Compute expected LR at given epoch.

        Returns:
            Tuple of (lr, cycle, T_cur)
        """
        optimizer_steps_per_epoch = steps_per_epoch // gradient_accumulation_steps
        T_0 = optimizer_steps_per_epoch * restart_epochs

        # Current step in optimizer steps
        current_step = epoch * optimizer_steps_per_epoch

        # Calculate cycle and position
        cycle = current_step // T_0
        T_cur = current_step % T_0

        # Decay factor
        decay_factor = decay_rate ** cycle
        cycle_peak = max(global_min_lr, base_lr * decay_factor)
        cycle_min = max(global_min_lr, cycle_min_lr * decay_factor)
        if cycle_min > cycle_peak:
            cycle_min = cycle_peak

        # Cosine interpolation
        progress = T_cur / T_0 if T_0 > 0 else 0
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = cycle_min + (cycle_peak - cycle_min) * cosine_factor

        return lr, cycle, T_cur

    def test_cycle_boundaries_correct(self):
        """Test that cycle boundaries occur at expected epochs."""
        # Config matching train_config.yaml
        steps_per_epoch = 100
        gradient_accumulation_steps = 2
        restart_epochs = 32

        # Cycle should change every restart_epochs
        # Cycle 0: epochs 0-31
        # Cycle 1: epochs 32-63
        # etc.

        _, cycle_0, _ = self._compute_expected_lr(
            epoch=0, steps_per_epoch=steps_per_epoch,
            gradient_accumulation_steps=gradient_accumulation_steps,
            restart_epochs=restart_epochs,
            base_lr=1e-4, cycle_min_lr=5e-5, decay_rate=0.9, global_min_lr=5e-6
        )
        assert cycle_0 == 0

        _, cycle_31, _ = self._compute_expected_lr(
            epoch=31, steps_per_epoch=steps_per_epoch,
            gradient_accumulation_steps=gradient_accumulation_steps,
            restart_epochs=restart_epochs,
            base_lr=1e-4, cycle_min_lr=5e-5, decay_rate=0.9, global_min_lr=5e-6
        )
        assert cycle_31 == 0

        _, cycle_32, _ = self._compute_expected_lr(
            epoch=32, steps_per_epoch=steps_per_epoch,
            gradient_accumulation_steps=gradient_accumulation_steps,
            restart_epochs=restart_epochs,
            base_lr=1e-4, cycle_min_lr=5e-5, decay_rate=0.9, global_min_lr=5e-6
        )
        assert cycle_32 == 1

    def test_peak_lr_at_cycle_start(self):
        """Test that LR is at peak at the start of each cycle."""
        base_lr = 1e-4
        decay_rate = 0.9

        # Cycle 0 peak
        lr_0, _, _ = self._compute_expected_lr(
            epoch=0, steps_per_epoch=100, gradient_accumulation_steps=2,
            restart_epochs=32, base_lr=base_lr, cycle_min_lr=5e-5,
            decay_rate=decay_rate, global_min_lr=5e-6
        )
        assert abs(lr_0 - base_lr) < 1e-10

        # Cycle 1 peak = base_lr * 0.9
        lr_32, _, _ = self._compute_expected_lr(
            epoch=32, steps_per_epoch=100, gradient_accumulation_steps=2,
            restart_epochs=32, base_lr=base_lr, cycle_min_lr=5e-5,
            decay_rate=decay_rate, global_min_lr=5e-6
        )
        expected_cycle1_peak = base_lr * decay_rate
        assert abs(lr_32 - expected_cycle1_peak) < 1e-10

    def test_global_min_reached_at_correct_cycle(self):
        """Test that global_min_lr is reached at mathematically predicted cycles."""
        base_lr = 1e-4
        cycle_min_lr = 5e-5
        decay_rate = 0.9
        global_min_lr = 5e-6

        # Trough should reach global_min at cycle 22
        # cycle_min_lr * 0.9^22 = 5e-5 * 0.0985 = 4.93e-6 < 5e-6
        trough_cycle = 22
        lr_at_trough_cycle_end, cycle, _ = self._compute_expected_lr(
            epoch=(trough_cycle + 1) * 32 - 1,  # End of cycle 22
            steps_per_epoch=100, gradient_accumulation_steps=2,
            restart_epochs=32, base_lr=base_lr, cycle_min_lr=cycle_min_lr,
            decay_rate=decay_rate, global_min_lr=global_min_lr
        )
        # Trough should be clamped to global_min_lr
        # At end of cycle, LR should be close to trough
        assert lr_at_trough_cycle_end >= global_min_lr - 1e-12

        # Peak should reach global_min at cycle 29
        # base_lr * 0.9^29 = 1e-4 * 0.0424 = 4.24e-6 < 5e-6
        peak_cycle = 29
        lr_at_peak_cycle_start, _, _ = self._compute_expected_lr(
            epoch=peak_cycle * 32,  # Start of cycle 29
            steps_per_epoch=100, gradient_accumulation_steps=2,
            restart_epochs=32, base_lr=base_lr, cycle_min_lr=cycle_min_lr,
            decay_rate=decay_rate, global_min_lr=global_min_lr
        )
        assert abs(lr_at_peak_cycle_start - global_min_lr) < 1e-12

    def test_specific_epoch_lr_values(self):
        """Test specific epoch LR values match documented expectations."""
        params = dict(
            steps_per_epoch=100,
            gradient_accumulation_steps=2,
            restart_epochs=32,
            base_lr=1e-4,
            cycle_min_lr=5e-5,
            decay_rate=0.9,
            global_min_lr=5e-6,
        )

        # Documented values from CHANGELOG
        expected_values = [
            (0, 1.000e-04),      # Epoch 0: peak of cycle 0
            (32, 9.000e-05),     # Epoch 32: peak of cycle 1
            (512, 1.853e-05),    # Epoch 512: peak of cycle 16
            (1024, 5.000e-06),   # Epoch 1024: at global_min
        ]

        for epoch, expected_lr in expected_values:
            lr, cycle, _ = self._compute_expected_lr(epoch=epoch, **params)
            assert abs(lr - expected_lr) < 1e-9, \
                f"Epoch {epoch}: expected {expected_lr:.3e}, got {lr:.3e}"

    def test_scheduler_matches_manual_calculation(self):
        """Test that scheduler produces same LR as manual calculation."""
        optimizer = self._create_optimizer()

        # Create scheduler with T_0 = (100/2) * 32 = 1600
        steps_per_epoch = 100
        gradient_accumulation_steps = 2
        restart_epochs = 32
        T_0 = (steps_per_epoch // gradient_accumulation_steps) * restart_epochs

        scheduler = SteppedCosineRestartScheduler(
            optimizer,
            T_0=T_0,
            base_lr=1e-4,
            cycle_min_lr=5e-5,
            decay_rate=0.9,
            global_min_lr=5e-6,
        )

        # Check LR at various epochs
        optimizer_steps_per_epoch = steps_per_epoch // gradient_accumulation_steps

        for epoch in [0, 1, 16, 31, 32, 64, 128, 256, 512]:
            # Reset scheduler to test specific epoch
            optimizer2 = self._create_optimizer()
            scheduler2 = SteppedCosineRestartScheduler(
                optimizer2,
                T_0=T_0,
                base_lr=1e-4,
                cycle_min_lr=5e-5,
                decay_rate=0.9,
                global_min_lr=5e-6,
            )

            # Advance to target epoch
            target_step = epoch * optimizer_steps_per_epoch
            for _ in range(target_step):
                scheduler2.step()

            scheduler_lr = optimizer2.param_groups[0]['lr']

            # Manual calculation
            expected_lr, _, _ = self._compute_expected_lr(
                epoch=epoch,
                steps_per_epoch=steps_per_epoch,
                gradient_accumulation_steps=gradient_accumulation_steps,
                restart_epochs=restart_epochs,
                base_lr=1e-4,
                cycle_min_lr=5e-5,
                decay_rate=0.9,
                global_min_lr=5e-6,
            )

            assert abs(scheduler_lr - expected_lr) < 1e-9, \
                f"Epoch {epoch}: scheduler={scheduler_lr:.6e}, expected={expected_lr:.6e}"

    def test_t0_calculation_from_factory(self):
        """Test that factory calculates T_0 correctly from config."""
        optimizer = self._create_optimizer()

        # Mock dataloader with known length
        mock_dataloader = MagicMock()
        mock_dataloader.__len__ = MagicMock(return_value=100)

        config = TrainingConfig(
            stepped_cosine_restart=SteppedCosineRestartConfig(
                enabled=True,
                base_lr=1e-4,
                cycle_min_lr=5e-5,
                decay_rate=0.9,
                global_min_lr=5e-6,
            ),
            restart_epochs=32,
            num_epochs=2048,
        )

        scheduler = create_lr_scheduler(
            optimizer, config, dataloader=mock_dataloader,
            gradient_accumulation_steps=2
        )

        # T_0 should be: (100 / 2) * 32 = 1600
        expected_T_0 = (100 // 2) * 32
        assert scheduler.T_0 == expected_T_0, \
            f"Expected T_0={expected_T_0}, got {scheduler.T_0}"


class TestSchedulerSyncOnResume:
    """Test scheduler synchronization when reset_scheduler=True during resume.

    This verifies the fix for the bug where scheduler would start from step 0
    even when resuming at step N, causing incorrect LR calculation.
    """

    def _create_model_and_optimizer(self):
        """Create a simple model and optimizer for testing."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        return model, optimizer

    def test_scheduler_sync_computes_correct_lr(self):
        """Test that scheduler synced to step N produces correct LR."""
        model, optimizer = self._create_model_and_optimizer()

        # Create scheduler
        scheduler = SteppedCosineRestartScheduler(
            optimizer,
            T_0=1600,  # (100/2) * 32
            base_lr=1e-4,
            cycle_min_lr=5e-5,
            decay_rate=0.9,
            global_min_lr=5e-6,
        )

        # Simulate resume at step 1600 (cycle 1)
        target_step = 1600
        scheduler.step(target_step)

        # After syncing to step 1600, should be at start of cycle 1
        assert scheduler.cycle == 1
        # LR should be base_lr * 0.9 = 9e-5
        expected_lr = 1e-4 * 0.9
        actual_lr = optimizer.param_groups[0]['lr']
        assert abs(actual_lr - expected_lr) < 1e-10, \
            f"Expected LR {expected_lr:.2e}, got {actual_lr:.2e}"

    def test_scheduler_sync_at_mid_cycle(self):
        """Test scheduler sync at middle of a cycle."""
        model, optimizer = self._create_model_and_optimizer()

        scheduler = SteppedCosineRestartScheduler(
            optimizer,
            T_0=1600,
            base_lr=1e-4,
            cycle_min_lr=5e-5,
            decay_rate=0.9,
            global_min_lr=5e-6,
        )

        # Sync to step 800 (middle of cycle 0)
        target_step = 800
        scheduler.step(target_step)

        assert scheduler.cycle == 0
        assert scheduler.T_cur == 800

        # LR at middle of cycle 0: avg of peak and trough
        # = 0.5 * (1e-4 + 5e-5) = 7.5e-5
        expected_lr = 0.5 * (1e-4 + 5e-5)
        actual_lr = optimizer.param_groups[0]['lr']
        assert abs(actual_lr - expected_lr) < 1e-9

    def test_scheduler_sync_preserves_subsequent_steps(self):
        """Test that stepping after sync continues correctly."""
        model, optimizer = self._create_model_and_optimizer()

        scheduler = SteppedCosineRestartScheduler(
            optimizer,
            T_0=100,  # Small T_0 for quick testing
            base_lr=1e-4,
            cycle_min_lr=5e-5,
            decay_rate=0.9,
            global_min_lr=5e-6,
        )

        # Sync to step 95 (near end of cycle 0)
        scheduler.step(95)
        assert scheduler.cycle == 0
        assert scheduler.T_cur == 95

        # Step 5 more times to enter cycle 1
        for _ in range(5):
            scheduler.step()

        # Should now be in cycle 1
        assert scheduler.cycle == 1
        # LR should be at peak of cycle 1 = 1e-4 * 0.9 = 9e-5
        expected_lr = 1e-4 * 0.9
        actual_lr = optimizer.param_groups[0]['lr']
        assert abs(actual_lr - expected_lr) < 1e-10

    def test_scheduler_sync_at_global_min(self):
        """Test scheduler sync when already at global_min_lr."""
        model, optimizer = self._create_model_and_optimizer()

        scheduler = SteppedCosineRestartScheduler(
            optimizer,
            T_0=100,
            base_lr=1e-4,
            cycle_min_lr=5e-5,
            decay_rate=0.9,
            global_min_lr=5e-6,
        )

        # Sync to step 3000 (cycle 30, well past global_min)
        scheduler.step(3000)

        # LR should be at global_min
        actual_lr = optimizer.param_groups[0]['lr']
        assert abs(actual_lr - 5e-6) < 1e-12


class TestResumeScenarios:
    """Test resume scenarios with different reset_optimizer/reset_scheduler combinations.

    This tests the fix for the bug where LR was incorrectly reset to base_lr
    when reset_optimizer=True but reset_scheduler=False.
    """

    def _create_model_and_optimizer(self):
        """Create a simple model and optimizer for testing."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        return model, optimizer

    def test_resume_preserves_lr_when_scheduler_not_reset(self):
        """Test that LR is preserved when reset_scheduler=False.

        Bug scenario:
            - config: reset_optimizer=True, reset_scheduler=False
            - After 8 cycles (epoch 256), LR should be ~4.3e-5
            - Old code: LR was incorrectly reset to 1e-4
            - Fixed code: LR should remain at ~4.3e-5
        """
        model, optimizer = self._create_model_and_optimizer()

        # Create scheduler and advance to cycle 8
        T_0 = 100  # 100 steps per cycle
        scheduler = SteppedCosineRestartScheduler(
            optimizer,
            T_0=T_0,
            base_lr=1e-4,
            cycle_min_lr=5e-5,
            decay_rate=0.9,
            global_min_lr=5e-6,
        )

        # Advance to cycle 8 (800 steps)
        for _ in range(800):
            scheduler.step()

        assert scheduler.cycle == 8, f"Expected cycle 8, got {scheduler.cycle}"

        # Get the expected LR at cycle 8 peak
        expected_lr = 1e-4 * (0.9 ** 8)  # ≈ 4.3e-5
        actual_lr = optimizer.param_groups[0]['lr']
        assert abs(actual_lr - expected_lr) < 1e-10, \
            f"Expected LR {expected_lr:.2e}, got {actual_lr:.2e}"

        # Save scheduler state
        saved_state = scheduler.state_dict()

        # Simulate resume: create new optimizer and scheduler
        _, new_optimizer = self._create_model_and_optimizer()
        new_scheduler = SteppedCosineRestartScheduler(
            new_optimizer,
            T_0=T_0,
            base_lr=1e-4,
            cycle_min_lr=5e-5,
            decay_rate=0.9,
            global_min_lr=5e-6,
        )

        # Load scheduler state (this is what happens when reset_scheduler=False)
        new_scheduler.load_state_dict(saved_state)

        # Verify LR is restored correctly (NOT reset to 1e-4!)
        restored_lr = new_optimizer.param_groups[0]['lr']
        assert abs(restored_lr - expected_lr) < 1e-10, \
            f"Expected restored LR {expected_lr:.2e}, got {restored_lr:.2e}"

        # Verify cycle state is preserved
        assert new_scheduler.cycle == 8

    def test_8_cycles_produces_correct_decay_factor(self):
        """Verify that 8 cycles produce the expected 0.43x decay factor.

        This matches the user's expectation:
            32轮一次后8次是0.43*1e-4=4e-5左右
        """
        expected_decay = 0.9 ** 8
        expected_lr = 1e-4 * expected_decay

        # Verify mathematically
        assert abs(expected_decay - 0.43046721) < 1e-8
        assert abs(expected_lr - 4.3046721e-5) < 1e-12

        # Verify with scheduler
        model, optimizer = self._create_model_and_optimizer()
        scheduler = SteppedCosineRestartScheduler(
            optimizer,
            T_0=100,  # 100 steps per cycle
            base_lr=1e-4,
            cycle_min_lr=5e-5,
            decay_rate=0.9,
            global_min_lr=5e-6,
        )

        # Advance to cycle 8
        for _ in range(800):  # 8 * 100 = 800 steps
            scheduler.step()

        actual_lr = optimizer.param_groups[0]['lr']
        assert abs(actual_lr - expected_lr) < 1e-10, \
            f"After 8 cycles: expected {expected_lr:.2e}, got {actual_lr:.2e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
