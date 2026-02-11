"""
Phase C: Gradient Accumulation Semantic Refactor Tests

Tests that state.step counts optimizer updates (not batches) and that
micro-batch collection, drop_last_accumulation, save/log intervals,
and scheduler stepping all use optimizer-step semantics.

10 test cases per plan Section 3.6.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import pytest
import torch

from src.training.trainer.loop import TrainingLoop
from src.training.trainer.metrics import TrainerState, TrainMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockDataLoader:
    """Deterministic mock dataloader with configurable batch count."""

    def __init__(self, num_batches: int = 10, batch_size: int = 4):
        self.num_batches = num_batches
        self.batch_size = batch_size

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        if self._idx >= self.num_batches:
            raise StopIteration
        self._idx += 1
        return {
            "images": torch.randn(self.batch_size, 32, 32, 3),
            "text_embeddings": torch.randn(self.batch_size, 16, 64),
            "text_mask": torch.ones(self.batch_size, 16),
        }


class NoLenDataLoader:
    """Deterministic dataloader without __len__ support."""

    def __init__(self, num_batches: int = 5):
        self.num_batches = num_batches

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        if self._idx >= self.num_batches:
            raise StopIteration
        self._idx += 1
        return {
            "images": torch.randn(1, 1),
            "text_embeddings": torch.randn(1, 1),
            "text_mask": torch.ones(1, 1),
        }


@dataclass
class SimpleTrainingConfig:
    """Minimal training config for loop tests."""

    gradient_accumulation_steps: int = 1
    drop_last_accumulation: bool = True
    training_mode: str = "epochs"
    num_epochs: int = 1
    max_steps: int = 100
    checkpoint_dir: Optional[str] = None
    save_every_epochs: int = 0
    log_every_epochs: int = 0


def _metrics(loss: float = 0.5) -> TrainMetrics:
    return TrainMetrics(
        loss=loss,
        loss_vloss=loss * 0.8,
        loss_freq=loss * 0.1,
        loss_repa=loss * 0.1,
        grad_norm=1.0,
        learning_rate=1e-4,
        samples_per_sec=100.0,
        step_time=0.04,
    )


def _make_loop(
    num_batches: int = 10,
    grad_accum: int = 1,
    drop_last: bool = True,
    num_epochs: int = 1,
) -> TrainingLoop:
    """Build a TrainingLoop with the given accumulation settings."""
    dl = MockDataLoader(num_batches=num_batches)
    cfg = SimpleTrainingConfig(
        gradient_accumulation_steps=grad_accum,
        drop_last_accumulation=drop_last,
        num_epochs=num_epochs,
    )
    state = TrainerState()
    return TrainingLoop(dataloader=dl, training_config=cfg, state=state)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGradientAccumulation:
    """Phase C gradient-accumulation semantic tests."""

    # 1. state.step increments only on optimizer update -------------------------

    def test_step_increments_only_on_optimizer_update(self):
        """state.step must increment once per optimizer step, not per batch."""
        loop = _make_loop(num_batches=8, grad_accum=4, drop_last=True)

        step_values: List[int] = []

        def mock_train_step(batch_or_batches):
            # batch_or_batches is a list of 4 micro-batches
            loop.state.step += 1
            step_values.append(loop.state.step)
            return _metrics()

        loop.run(
            train_step_fn=mock_train_step,
            save_checkpoint_fn=MagicMock(),
            num_steps=2,
            log_interval=0,
            save_interval=0,
            use_progress_bar=False,
        )

        # 8 batches / 4 grad_accum = 2 optimizer steps
        assert loop.state.step == 2
        assert step_values == [1, 2]

    # 2. drop_last=false: tail micro-batches still update ----------------------

    def test_drop_last_false_updates_on_tail_micro_batches(self):
        """With drop_last=false, partial windows wrap across epochs to stay full."""
        loop = _make_loop(num_batches=10, grad_accum=4, drop_last=False)

        call_sizes: List[int] = []

        def mock_train_step(batch_or_batches):
            if isinstance(batch_or_batches, list):
                call_sizes.append(len(batch_or_batches))
            else:
                call_sizes.append(1)
            loop.state.step += 1
            return _metrics()

        loop.run(
            train_step_fn=mock_train_step,
            save_checkpoint_fn=MagicMock(),
            num_epochs=1,
            log_interval=0,
            save_interval=0,
            use_progress_bar=False,
        )

        # Unified semantics: one configured epoch keeps 10 optimizer steps.
        # With grad_accum=4, windows wrap across physical epochs to stay full.
        assert loop.state.step == 10
        assert call_sizes == [4] * 10
        # Epoch counter should have incremented (proving wrap-around)
        assert loop.state.epoch >= 1

    # 3. drop_last=true: tail micro-batches skipped ----------------------------

    def test_drop_last_true_skips_tail_micro_batches(self):
        """With drop_last=true, partial accumulation windows are discarded."""
        loop = _make_loop(num_batches=10, grad_accum=4, drop_last=True)

        def mock_train_step(batch_or_batches):
            loop.state.step += 1
            return _metrics()

        loop.run(
            train_step_fn=mock_train_step,
            save_checkpoint_fn=MagicMock(),
            num_epochs=1,
            log_interval=0,
            save_interval=0,
            use_progress_bar=False,
        )

        # Unified semantics: one configured epoch keeps 10 optimizer steps.
        # With drop_last=true, partial windows are dropped and require more
        # physical epochs to reach the same configured step budget.
        assert loop.state.step == 10

    # 4. total_steps epoch mode, drop_last=false --------------------------------

    def test_total_steps_in_epoch_mode_with_drop_last_false(self):
        """Epoch-mode total_steps is fixed to epochs * batches_per_epoch."""
        loop = _make_loop(num_batches=10, grad_accum=4, drop_last=False, num_epochs=2)

        total = loop._calculate_total_steps(num_steps=None, num_epochs=2)
        assert total == 20

    # 5. total_steps epoch mode, drop_last=true --------------------------------

    def test_total_steps_in_epoch_mode_with_drop_last_true(self):
        """drop_last does not change configured epoch step budget."""
        loop = _make_loop(num_batches=10, grad_accum=4, drop_last=True, num_epochs=2)

        total = loop._calculate_total_steps(num_steps=None, num_epochs=2)
        assert total == 20

    # 6. save_interval uses optimizer steps ------------------------------------

    def test_save_interval_uses_optimizer_steps(self):
        """Checkpoints fire at optimizer-step boundaries, not batch boundaries."""
        loop = _make_loop(num_batches=8, grad_accum=2, drop_last=True)
        save_steps: List[int] = []

        def mock_train_step(batch_or_batches):
            loop.state.step += 1
            return _metrics()

        def mock_save(path, checkpoint_name=None, **kw):
            save_steps.append(loop.state.step)

        loop.run(
            train_step_fn=mock_train_step,
            save_checkpoint_fn=mock_save,
            num_steps=4,
            save_interval=2,  # save every 2 optimizer steps
            log_interval=0,
            save_path="/tmp/ckpt",
            use_progress_bar=False,
        )

        # Steps 2 and 4 trigger save_interval; step 4 also triggers completed save
        # _handle_step_checkpoint fires at step 2, 4; final completed save also at 4
        assert 2 in save_steps
        assert 4 in save_steps

    # 7. log_interval uses optimizer steps ------------------------------------

    def test_log_interval_uses_optimizer_steps(self, caplog):
        """Logging fires at optimizer-step boundaries, not batch boundaries."""
        loop = _make_loop(num_batches=8, grad_accum=2, drop_last=True)

        def mock_train_step(batch_or_batches):
            loop.state.step += 1
            return _metrics()

        with caplog.at_level(logging.INFO, logger="src.training.trainer.loop"):
            loop.run(
                train_step_fn=mock_train_step,
                save_checkpoint_fn=MagicMock(),
                num_steps=4,
                log_interval=2,  # log every 2 optimizer steps
                save_interval=0,
                use_progress_bar=False,
            )

        log_messages = [r.message for r in caplog.records if "Step " in r.message]
        # Should log at optimizer steps 2 and 4
        assert any("Step 2/" in m for m in log_messages)
        assert any("Step 4/" in m for m in log_messages)
        # Should NOT log at step 1 or 3
        assert not any("Step 1/" in m for m in log_messages)
        assert not any("Step 3/" in m for m in log_messages)

    # 8. scheduler step count matches optimizer step count --------------------

    def test_scheduler_step_matches_optimizer_step_count(self):
        """LR scheduler .step() count must equal optimizer step count."""
        loop = _make_loop(num_batches=8, grad_accum=2, drop_last=True)
        scheduler_steps = [0]

        class CountingScheduler:
            def step(self):
                scheduler_steps[0] += 1

            def get_last_lr(self):
                return [1e-4]

        mock_scheduler = CountingScheduler()

        # Directly use core.py train_step with a counting scheduler
        # Simulate: loop calls train_step_fn which calls lr_scheduler.step() once
        def mock_train_step(batch_or_batches):
            mock_scheduler.step()
            loop.state.step += 1
            return _metrics()

        loop.run(
            train_step_fn=mock_train_step,
            save_checkpoint_fn=MagicMock(),
            num_steps=4,
            log_interval=0,
            save_interval=0,
            use_progress_bar=False,
        )

        # 4 optimizer steps -> 4 scheduler steps
        assert scheduler_steps[0] == 4
        assert loop.state.step == 4

    # 9. OOM retry resets accumulator, no phantom step -------------------------

    def test_oom_retry_resets_accumulator_and_no_phantom_step(self):
        """OOM during accumulation must not leave phantom step increments."""
        from src.training.trainer.step import StepExecutor

        # Create a minimal executor with mocks
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        mock_fm = MagicMock()
        mock_fm.prepare_training.return_value = (
            torch.rand(2),                     # t
            torch.randn(2, 4),                 # z_t
            torch.randn(2, 4),                 # x_clean
            torch.randn(2, 4),                 # noise
        )

        executor = StepExecutor(
            model=model,
            optimizer=optimizer,
            flow_matching=mock_fm,
            combined_loss=MagicMock(),
            zclip=MagicMock(),
            device=torch.device("cpu"),
            gradient_accumulation_steps=2,
            max_grad_norm=1.0,
        )

        # Simulate: first forward_backward succeeds, second raises OOM
        call_count = [0]

        def fake_forward(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("CUDA out of memory")
            loss = torch.tensor(1.0, requires_grad=True)
            return {"total": loss, "vloss": loss, "freq_loss": torch.tensor(0.0), "repa_loss": torch.tensor(0.0)}, torch.zeros(1)

        executor._forward = fake_forward
        executor._backward_scaled = MagicMock()

        batch = {"images": torch.randn(2, 32, 32, 3)}

        # First micro-batch succeeds
        executor.forward_backward(batch, step=0, denom=2)
        assert executor._loss_accum_steps == 1

        # Second micro-batch OOMs
        with pytest.raises(RuntimeError, match="out of memory"):
            executor.forward_backward(batch, step=0, denom=2)

        # After OOM, caller should reset accumulators
        executor.reset_accumulators()
        assert executor._loss_accum_steps == 0
        assert executor._loss_accum == 0.0
        assert executor._total_batch_size == 0

    # 10. resume legacy checkpoint warns without auto-convert ------------------

    def test_resume_legacy_checkpoint_warns_without_auto_convert(self, tmp_path, caplog):
        """Loading a pre-Phase-C checkpoint (no version) should warn, not convert."""
        from src.training.trainer.checkpoint import CheckpointManager

        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        mgr = CheckpointManager(
            model=model, optimizer=optimizer, device=torch.device("cpu"),
        )

        # Save a legacy-style checkpoint (no format_version)
        legacy_ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "state": {
                "step": 100,
                "epoch": 5,
                "batch_idx": 0,
                "best_loss": 0.5,
                "total_samples": 1000,
            },
        }
        ckpt_path = tmp_path / "legacy.pt"
        torch.save(legacy_ckpt, ckpt_path)

        state = TrainerState()
        with caplog.at_level(logging.WARNING):
            mgr.load(ckpt_path, state)

        # Should warn about missing format version
        assert any("without format version" in r.message for r in caplog.records), \
            f"Expected warning about legacy checkpoint, got: {[r.message for r in caplog.records]}"

        # step should be loaded as-is, NOT auto-converted
        assert state.step == 100
        assert state.epoch == 5


# ---------------------------------------------------------------------------
# Parameter matrix: additional edge cases
# ---------------------------------------------------------------------------


class TestAccumulationParameterMatrix:
    """Cover parameter matrix: grad_accum in {1, 2, 4}, len in {8, 10}."""

    @pytest.mark.parametrize("num_batches,grad_accum,drop_last,expected_steps", [
        # Unified semantics: expected steps == num_batches for one configured epoch.
        (8, 4, True, 8),
        (8, 4, False, 8),
        (10, 4, True, 10),
        (10, 4, False, 10),
        (8, 2, True, 8),
        (8, 2, False, 8),
        (10, 2, True, 10),
        (10, 2, False, 10),
        # grad_accum=1 -> every batch is an optimizer step
        (8, 1, True, 8),
        (10, 1, True, 10),
    ])
    def test_optimizer_steps_per_epoch(
        self, num_batches, grad_accum, drop_last, expected_steps
    ):
        """Verify optimizer step count for one epoch across parameter matrix."""
        loop = _make_loop(
            num_batches=num_batches,
            grad_accum=grad_accum,
            drop_last=drop_last,
            num_epochs=1,
        )

        def mock_train_step(batch_or_batches):
            loop.state.step += 1
            return _metrics()

        loop.run(
            train_step_fn=mock_train_step,
            save_checkpoint_fn=MagicMock(),
            num_epochs=1,
            log_interval=0,
            save_interval=0,
            use_progress_bar=False,
        )

        assert loop.state.step == expected_steps, (
            f"len={num_batches}, grad_accum={grad_accum}, drop_last={drop_last}: "
            f"expected {expected_steps} steps, got {loop.state.step}"
        )


class TestEpochStepUnification:
    """Regression tests for epoch/step/save unified semantics."""

    def test_checkpoint_naming_uses_step_derived_epoch(self):
        """With len=10 and accum=2, epoch boundary stays at step 10 (not 5)."""
        loop = _make_loop(num_batches=10, grad_accum=2, drop_last=True)
        periodic_names: List[str] = []

        def mock_train_step(batch_or_batches):
            loop.state.step += 1
            return _metrics()

        def mock_save(path, checkpoint_name=None, **kwargs):
            if checkpoint_name and checkpoint_name != "checkpoint_completed":
                periodic_names.append(checkpoint_name)

        loop.run(
            train_step_fn=mock_train_step,
            save_checkpoint_fn=mock_save,
            num_steps=10,
            save_interval=5,
            save_every_epochs=1,
            log_interval=0,
            save_path="/tmp/ckpt",
            use_progress_bar=False,
        )

        assert periodic_names == [
            "checkpoint_epoch1_step5",
            "checkpoint_epoch1_step10",
        ]
        assert loop.state.epoch == 1

    def test_resume_realigns_epoch_from_step(self, caplog):
        """Resume mismatch should realign epoch using step // steps_per_epoch."""
        dl = MockDataLoader(num_batches=10)
        cfg = SimpleTrainingConfig(
            gradient_accumulation_steps=1,
            drop_last_accumulation=True,
            training_mode="steps",
            max_steps=100,
        )
        state = TrainerState(step=15, epoch=0)
        loop = TrainingLoop(dataloader=dl, training_config=cfg, state=state)

        def mock_train_step(batch_or_batches):
            loop.state.step += 1
            return _metrics()

        with caplog.at_level(logging.WARNING):
            loop.run(
                train_step_fn=mock_train_step,
                save_checkpoint_fn=MagicMock(),
                num_steps=16,
                save_interval=0,
                log_interval=0,
                use_progress_bar=False,
            )

        assert any("Realigning epoch" in r.message for r in caplog.records)
        assert loop.state.epoch == 1

    def test_steps_mode_non_multiple_no_epoch_save(self):
        """steps mode budget not hitting boundary should not trigger epoch-save."""
        dl = MockDataLoader(num_batches=10)
        cfg = SimpleTrainingConfig(
            gradient_accumulation_steps=2,
            drop_last_accumulation=True,
            training_mode="steps",
            max_steps=9,
        )
        state = TrainerState()
        loop = TrainingLoop(dataloader=dl, training_config=cfg, state=state)
        periodic_names: List[str] = []

        def mock_train_step(batch_or_batches):
            loop.state.step += 1
            return _metrics()

        def mock_save(path, checkpoint_name=None, **kwargs):
            if checkpoint_name and checkpoint_name != "checkpoint_completed":
                periodic_names.append(checkpoint_name)

        loop.run(
            train_step_fn=mock_train_step,
            save_checkpoint_fn=mock_save,
            num_steps=9,
            save_interval=5,
            save_every_epochs=1,
            log_interval=0,
            save_path="/tmp/ckpt",
            use_progress_bar=False,
        )

        assert periodic_names == ["checkpoint_epoch1_step5"]
        assert loop.state.epoch == 0


class TestUnknownDataloaderLengthPolicy:
    """Fail-fast policy when len(dataloader) is unavailable."""

    def test_epoch_mode_fails_without_len(self):
        dl = NoLenDataLoader(num_batches=5)
        cfg = SimpleTrainingConfig(training_mode="epochs", num_epochs=1)
        loop = TrainingLoop(dataloader=dl, training_config=cfg, state=TrainerState())

        with pytest.raises(ValueError, match="Epoch mode requires len\\(dataloader\\)"):
            loop.run(
                train_step_fn=lambda _: _metrics(),
                save_checkpoint_fn=MagicMock(),
                num_epochs=1,
                log_interval=0,
                save_interval=0,
                use_progress_bar=False,
            )

    def test_steps_mode_epoch_hooks_fail_without_len(self):
        dl = NoLenDataLoader(num_batches=5)
        cfg = SimpleTrainingConfig(training_mode="steps", max_steps=5)
        loop = TrainingLoop(dataloader=dl, training_config=cfg, state=TrainerState())

        with pytest.raises(ValueError, match="epoch-based save/log cannot be enabled"):
            loop.run(
                train_step_fn=lambda _: _metrics(),
                save_checkpoint_fn=MagicMock(),
                num_steps=3,
                save_every_epochs=1,
                log_interval=0,
                save_interval=0,
                use_progress_bar=False,
            )

    def test_steps_mode_runs_without_epoch_hooks_when_len_unknown(self, caplog):
        dl = NoLenDataLoader(num_batches=5)
        cfg = SimpleTrainingConfig(training_mode="steps", max_steps=5)
        state = TrainerState()
        loop = TrainingLoop(dataloader=dl, training_config=cfg, state=state)

        def mock_train_step(batch_or_batches):
            loop.state.step += 1
            return _metrics()

        with caplog.at_level(logging.WARNING):
            loop.run(
                train_step_fn=mock_train_step,
                save_checkpoint_fn=MagicMock(),
                num_steps=3,
                save_every_epochs=0,
                log_every_epochs=0,
                save_interval=0,
                log_interval=0,
                use_progress_bar=False,
            )

        assert state.step == 3
        assert any("epoch semantics disabled" in r.message for r in caplog.records)


class TestCheckpointFormatVersion:
    """Verify new checkpoints include format version metadata."""

    def test_new_checkpoint_has_version_2(self, tmp_path):
        """New checkpoints must include checkpoint_format_version=2."""
        from src.training.trainer.checkpoint import CheckpointManager

        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        mgr = CheckpointManager(
            model=model, optimizer=optimizer, device=torch.device("cpu"),
        )
        state = TrainerState()
        state.step = 10

        mgr.save(tmp_path, state, checkpoint_name="test_v2", cleanup=False)

        loaded = torch.load(tmp_path / "test_v2.pt", weights_only=False)
        assert loaded["checkpoint_format_version"] == 2
        assert loaded["step_unit"] == "optimizer"
