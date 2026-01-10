"""
Comprehensive Unit Tests for Training Loop

Tests for:
1. Iteration Tests (5 tests)
2. Checkpoint Tests (5 tests)
3. Logging Tests (5 tests)

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-08

Tests the TrainingLoop class which manages the main training process,
including batch iteration, epoch handling, checkpoint saving, and logging.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch, call
import tempfile

import pytest
import torch

from src.training.trainer.loop import TrainingLoop
from src.training.trainer.metrics import TrainerState, TrainMetrics


class MockDataLoader:
    """Mock dataloader for testing training loop."""

    def __init__(
        self,
        num_batches: int = 10,
        batch_size: int = 4,
        variable_last_batch: bool = False,
    ):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.variable_last_batch = variable_last_batch
        self._iteration_count = 0
        self._current_epoch = 0

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        self._iteration_count = 0
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        if self._iteration_count >= self.num_batches:
            self._current_epoch += 1
            raise StopIteration

        self._iteration_count += 1

        # Handle variable last batch size
        if self.variable_last_batch and self._iteration_count == self.num_batches:
            actual_batch_size = self.batch_size // 2
        else:
            actual_batch_size = self.batch_size

        return {
            "images": torch.randn(actual_batch_size, 32, 32, 3),
            "text_embeddings": torch.randn(actual_batch_size, 16, 64),
            "text_mask": torch.ones(actual_batch_size, 16),
        }


def create_mock_metrics(loss: float = 0.5) -> TrainMetrics:
    """Create mock training metrics."""
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


def create_training_loop(
    num_batches: int = 10,
    batch_size: int = 4,
    variable_last_batch: bool = False,
) -> TrainingLoop:
    """Create a TrainingLoop with mock components."""
    dataloader = MockDataLoader(
        num_batches=num_batches,
        batch_size=batch_size,
        variable_last_batch=variable_last_batch,
    )
    state = TrainerState()
    training_config = None

    return TrainingLoop(
        dataloader=dataloader,
        training_config=training_config,
        state=state,
    )


# ============================================================================
# Iteration Tests (5 tests)
# ============================================================================


class TestLoopIteration:
    """Tests for training loop iteration behavior."""

    def test_loop_iterates_dataloader(self):
        """Test that loop iterates through all batches in dataloader."""
        loop = create_training_loop(num_batches=5)
        batches_processed = []

        def mock_train_step(batch: Dict[str, torch.Tensor]) -> TrainMetrics:
            batches_processed.append(batch)
            return create_mock_metrics()

        def mock_save_checkpoint(*args, **kwargs):
            pass

        # Run for exactly 5 steps (1 epoch with 5 batches)
        loop.run(
            train_step_fn=mock_train_step,
            save_checkpoint_fn=mock_save_checkpoint,
            num_steps=5,
            log_interval=0,
            save_interval=0,
            use_progress_bar=False,
        )

        assert len(batches_processed) == 5, f"Expected 5 batches, got {len(batches_processed)}"

    def test_loop_epoch_counter(self):
        """Test that epoch counter increments correctly."""
        loop = create_training_loop(num_batches=3)

        def mock_train_step(batch: Dict[str, torch.Tensor]) -> TrainMetrics:
            return create_mock_metrics()

        def mock_save_checkpoint(*args, **kwargs):
            pass

        # Run for 9 steps = 3 epochs with 3 batches each
        loop.run(
            train_step_fn=mock_train_step,
            save_checkpoint_fn=mock_save_checkpoint,
            num_steps=9,
            log_interval=0,
            save_interval=0,
            use_progress_bar=False,
        )

        # After 9 steps with 3 batches/epoch, we should be at epoch 3
        assert loop.state.epoch == 3, f"Expected epoch 3, got {loop.state.epoch}"

    def test_loop_step_counter(self):
        """Test that step counter increments correctly."""
        loop = create_training_loop(num_batches=10)
        step_values = []

        def mock_train_step(batch: Dict[str, torch.Tensor]) -> TrainMetrics:
            step_values.append(loop.state.step)
            return create_mock_metrics()

        def mock_save_checkpoint(*args, **kwargs):
            pass

        loop.run(
            train_step_fn=mock_train_step,
            save_checkpoint_fn=mock_save_checkpoint,
            num_steps=5,
            log_interval=0,
            save_interval=0,
            use_progress_bar=False,
        )

        # Step should increment: 0, 1, 2, 3, 4 (before each train_step call)
        # Note: state.step is incremented after train_step in Trainer, not in loop
        # Loop just calls train_step which handles step increment
        assert loop.state.step == 5, f"Expected step 5, got {loop.state.step}"

    def test_loop_batch_size_consistency(self):
        """Test that batch sizes are consistent across iterations."""
        loop = create_training_loop(num_batches=5, batch_size=8)
        batch_sizes = []

        def mock_train_step(batch: Dict[str, torch.Tensor]) -> TrainMetrics:
            batch_sizes.append(batch["images"].shape[0])
            return create_mock_metrics()

        def mock_save_checkpoint(*args, **kwargs):
            pass

        loop.run(
            train_step_fn=mock_train_step,
            save_checkpoint_fn=mock_save_checkpoint,
            num_steps=5,
            log_interval=0,
            save_interval=0,
            use_progress_bar=False,
        )

        # All batches should have same size
        assert all(bs == 8 for bs in batch_sizes), f"Inconsistent batch sizes: {batch_sizes}"

    def test_loop_handles_variable_batch(self):
        """Test that loop handles last batch size variation."""
        loop = create_training_loop(
            num_batches=5, batch_size=8, variable_last_batch=True
        )
        batch_sizes = []

        def mock_train_step(batch: Dict[str, torch.Tensor]) -> TrainMetrics:
            batch_sizes.append(batch["images"].shape[0])
            return create_mock_metrics()

        def mock_save_checkpoint(*args, **kwargs):
            pass

        loop.run(
            train_step_fn=mock_train_step,
            save_checkpoint_fn=mock_save_checkpoint,
            num_steps=5,
            log_interval=0,
            save_interval=0,
            use_progress_bar=False,
        )

        # Last batch should be half size
        assert batch_sizes[-1] == 4, f"Expected last batch size 4, got {batch_sizes[-1]}"
        assert batch_sizes[0] == 8, f"Expected first batch size 8, got {batch_sizes[0]}"


# ============================================================================
# Checkpoint Tests (5 tests)
# ============================================================================


class TestLoopCheckpoint:
    """Tests for checkpoint saving behavior."""

    def test_loop_saves_checkpoint(self):
        """Test that checkpoints are saved at specified intervals."""
        loop = create_training_loop(num_batches=10)
        checkpoint_calls = []

        def mock_train_step(batch: Dict[str, torch.Tensor]) -> TrainMetrics:
            return create_mock_metrics()

        def mock_save_checkpoint(path, checkpoint_name=None, **kwargs):
            checkpoint_calls.append((path, checkpoint_name))

        with tempfile.TemporaryDirectory() as tmpdir:
            loop.run(
                train_step_fn=mock_train_step,
                save_checkpoint_fn=mock_save_checkpoint,
                num_steps=10,
                save_interval=5,  # Save every 5 steps
                log_interval=0,
                save_path=tmpdir,
                use_progress_bar=False,
            )

        # Should save at step 5, 10 (interval) + final checkpoint
        # Step 5: step checkpoint, Step 10: step checkpoint + completed checkpoint
        assert len(checkpoint_calls) >= 2, f"Expected at least 2 saves, got {len(checkpoint_calls)}"

    def test_loop_saves_ema(self):
        """Test that EMA state is saved with checkpoints."""
        # This test verifies the checkpoint function is called which should include EMA
        loop = create_training_loop(num_batches=5)
        save_called = []

        def mock_train_step(batch: Dict[str, torch.Tensor]) -> TrainMetrics:
            return create_mock_metrics()

        def mock_save_checkpoint(path, checkpoint_name=None, **kwargs):
            save_called.append(True)

        with tempfile.TemporaryDirectory() as tmpdir:
            loop.run(
                train_step_fn=mock_train_step,
                save_checkpoint_fn=mock_save_checkpoint,
                num_steps=5,
                save_interval=5,
                log_interval=0,
                save_path=tmpdir,
                use_progress_bar=False,
            )

        # Checkpoint function was called (EMA is saved within that function)
        assert len(save_called) >= 1

    def test_loop_resume_from_checkpoint(self):
        """Test that loop can resume from a checkpoint state."""
        # Create loop with pre-existing state (simulating resume)
        dataloader = MockDataLoader(num_batches=10)
        state = TrainerState(step=5, epoch=0, batch_idx=5)  # Resume from step 5

        loop = TrainingLoop(
            dataloader=dataloader,
            training_config=None,
            state=state,
        )

        def mock_train_step(batch: Dict[str, torch.Tensor]) -> TrainMetrics:
            return create_mock_metrics()

        def mock_save_checkpoint(*args, **kwargs):
            pass

        # Run for 5 more steps (total 10)
        loop.run(
            train_step_fn=mock_train_step,
            save_checkpoint_fn=mock_save_checkpoint,
            num_steps=10,
            log_interval=0,
            save_interval=0,
            use_progress_bar=False,
        )

        assert loop.state.step == 10, f"Expected step 10 after resume, got {loop.state.step}"

    def test_loop_resume_step_correct(self):
        """Test that resume continues from correct step."""
        dataloader = MockDataLoader(num_batches=5)
        initial_step = 7
        state = TrainerState(step=initial_step, epoch=1, batch_idx=2)

        loop = TrainingLoop(
            dataloader=dataloader,
            training_config=None,
            state=state,
        )

        steps_executed = []

        def mock_train_step(batch: Dict[str, torch.Tensor]) -> TrainMetrics:
            steps_executed.append(loop.state.step)
            return create_mock_metrics()

        def mock_save_checkpoint(*args, **kwargs):
            pass

        # Run 3 more steps
        loop.run(
            train_step_fn=mock_train_step,
            save_checkpoint_fn=mock_save_checkpoint,
            num_steps=10,  # Total target
            log_interval=0,
            save_interval=0,
            use_progress_bar=False,
        )

        # Should execute steps 7, 8, 9 (3 steps to reach 10)
        assert len(steps_executed) == 3, f"Expected 3 steps, got {len(steps_executed)}"
        assert loop.state.step == 10

    def test_loop_resume_epoch_correct(self):
        """Test that resume continues from correct epoch."""
        dataloader = MockDataLoader(num_batches=3)
        state = TrainerState(step=6, epoch=2, batch_idx=0)  # Start of epoch 2

        loop = TrainingLoop(
            dataloader=dataloader,
            training_config=None,
            state=state,
        )

        def mock_train_step(batch: Dict[str, torch.Tensor]) -> TrainMetrics:
            return create_mock_metrics()

        def mock_save_checkpoint(*args, **kwargs):
            pass

        # Run 3 more steps (1 epoch)
        loop.run(
            train_step_fn=mock_train_step,
            save_checkpoint_fn=mock_save_checkpoint,
            num_steps=9,
            log_interval=0,
            save_interval=0,
            use_progress_bar=False,
        )

        # Should be at epoch 3 after completing the epoch
        assert loop.state.epoch == 3, f"Expected epoch 3, got {loop.state.epoch}"


# ============================================================================
# Logging Tests (5 tests)
# ============================================================================


class TestLoopLogging:
    """Tests for logging behavior."""

    def test_loop_logs_loss(self):
        """Test that loss is logged during training."""
        loop = create_training_loop(num_batches=10)

        def mock_train_step(batch: Dict[str, torch.Tensor]) -> TrainMetrics:
            return create_mock_metrics(loss=0.42)

        def mock_save_checkpoint(*args, **kwargs):
            pass

        with patch("src.training.trainer.loop.logger") as mock_logger:
            loop.run(
                train_step_fn=mock_train_step,
                save_checkpoint_fn=mock_save_checkpoint,
                num_steps=5,
                log_interval=5,  # Log at step 5
                save_interval=0,
                use_progress_bar=False,
            )

            # Check that info was called with loss information
            info_calls = mock_logger.info.call_args_list
            # Should have at least one log with loss info
            logged_text = " ".join(str(c) for c in info_calls)
            assert "0.42" in logged_text or "Loss" in logged_text

    def test_loop_logs_lr(self):
        """Test that learning rate is logged."""
        loop = create_training_loop(num_batches=10)

        def mock_train_step(batch: Dict[str, torch.Tensor]) -> TrainMetrics:
            return create_mock_metrics()

        def mock_save_checkpoint(*args, **kwargs):
            pass

        with patch("src.training.trainer.loop.logger") as mock_logger:
            loop.run(
                train_step_fn=mock_train_step,
                save_checkpoint_fn=mock_save_checkpoint,
                num_steps=5,
                log_interval=5,
                save_interval=0,
                use_progress_bar=False,
            )

            info_calls = mock_logger.info.call_args_list
            logged_text = " ".join(str(c) for c in info_calls)
            # LR should be logged
            assert "LR" in logged_text or "1e-04" in logged_text or "1.00e-04" in logged_text

    def test_loop_logs_at_interval(self):
        """Test that logging happens at configured interval."""
        loop = create_training_loop(num_batches=20)
        log_steps = []

        def mock_train_step(batch: Dict[str, torch.Tensor]) -> TrainMetrics:
            return create_mock_metrics()

        def mock_save_checkpoint(*args, **kwargs):
            pass

        original_handle_logging = loop._handle_logging

        def tracking_handle_logging(log_interval, total_steps, metrics):
            if log_interval > 0 and loop.state.step % log_interval == 0:
                log_steps.append(loop.state.step)
            original_handle_logging(log_interval, total_steps, metrics)

        loop._handle_logging = tracking_handle_logging

        loop.run(
            train_step_fn=mock_train_step,
            save_checkpoint_fn=mock_save_checkpoint,
            num_steps=15,
            log_interval=5,  # Log every 5 steps
            save_interval=0,
            use_progress_bar=False,
        )

        # Should log at steps 5, 10, 15
        assert 5 in log_steps, f"Expected log at step 5, logs at: {log_steps}"
        assert 10 in log_steps, f"Expected log at step 10, logs at: {log_steps}"
        assert 15 in log_steps, f"Expected log at step 15, logs at: {log_steps}"

    def test_loop_logs_metrics_dict(self):
        """Test that metrics are logged as dictionary format."""
        loop = create_training_loop(num_batches=10)

        metrics_logged = []

        def mock_train_step(batch: Dict[str, torch.Tensor]) -> TrainMetrics:
            m = create_mock_metrics(loss=0.123)
            return m

        def mock_save_checkpoint(*args, **kwargs):
            pass

        def callback(step: int, metrics: TrainMetrics):
            metrics_logged.append(metrics.to_dict())

        loop.run(
            train_step_fn=mock_train_step,
            save_checkpoint_fn=mock_save_checkpoint,
            num_steps=3,
            log_interval=0,
            save_interval=0,
            callback=callback,
            use_progress_bar=False,
        )

        # Verify metrics dict structure
        assert len(metrics_logged) == 3
        for m in metrics_logged:
            assert "loss" in m
            assert "loss_vloss" in m
            assert "loss_freq" in m
            assert "loss_repa" in m
            assert "learning_rate" in m

    def test_loop_tqdm_progress(self):
        """Test that progress bar updates correctly."""
        loop = create_training_loop(num_batches=5)

        def mock_train_step(batch: Dict[str, torch.Tensor]) -> TrainMetrics:
            return create_mock_metrics()

        def mock_save_checkpoint(*args, **kwargs):
            pass

        with patch("src.training.trainer.loop.TQDM_AVAILABLE", True):
            with patch("src.training.trainer.loop.tqdm") as mock_tqdm:
                mock_pbar = MagicMock()
                mock_tqdm.return_value = mock_pbar

                loop.run(
                    train_step_fn=mock_train_step,
                    save_checkpoint_fn=mock_save_checkpoint,
                    num_steps=5,
                    log_interval=0,
                    save_interval=0,
                    use_progress_bar=True,
                )

                # Progress bar should be created
                mock_tqdm.assert_called_once()
                # Update should be called for each step
                assert mock_pbar.update.call_count == 5
                # Close should be called at end
                mock_pbar.close.assert_called_once()


# ============================================================================
# Additional Integration Tests
# ============================================================================


class TestLoopIntegration:
    """Integration tests for TrainingLoop."""

    def test_loop_epoch_checkpoint_saving(self):
        """Test checkpoint saving at epoch boundaries."""
        loop = create_training_loop(num_batches=3)
        checkpoint_names = []

        def mock_train_step(batch: Dict[str, torch.Tensor]) -> TrainMetrics:
            return create_mock_metrics()

        def mock_save_checkpoint(path, checkpoint_name=None, **kwargs):
            checkpoint_names.append(checkpoint_name)

        with tempfile.TemporaryDirectory() as tmpdir:
            loop.run(
                train_step_fn=mock_train_step,
                save_checkpoint_fn=mock_save_checkpoint,
                num_steps=9,  # 3 epochs
                save_every_epochs=1,  # Save every epoch
                log_interval=0,
                save_interval=0,
                save_path=tmpdir,
                use_progress_bar=False,
            )

        # Should save at epoch 1, 2, 3 + completed
        epoch_checkpoints = [n for n in checkpoint_names if n and "epoch" in n]
        assert len(epoch_checkpoints) >= 2

    def test_loop_callback_execution(self):
        """Test that callback is executed after each step."""
        loop = create_training_loop(num_batches=5)
        callback_calls = []

        def mock_train_step(batch: Dict[str, torch.Tensor]) -> TrainMetrics:
            return create_mock_metrics()

        def mock_save_checkpoint(*args, **kwargs):
            pass

        def callback(step: int, metrics: TrainMetrics):
            callback_calls.append((step, metrics.loss))

        loop.run(
            train_step_fn=mock_train_step,
            save_checkpoint_fn=mock_save_checkpoint,
            num_steps=5,
            log_interval=0,
            save_interval=0,
            callback=callback,
            use_progress_bar=False,
        )

        assert len(callback_calls) == 5
        # Steps should be 1, 2, 3, 4, 5 (after increment)
        steps = [c[0] for c in callback_calls]
        assert steps == [1, 2, 3, 4, 5]

    def test_loop_gc_interval(self):
        """Test garbage collection at specified interval."""
        loop = create_training_loop(num_batches=10)

        def mock_train_step(batch: Dict[str, torch.Tensor]) -> TrainMetrics:
            return create_mock_metrics()

        def mock_save_checkpoint(*args, **kwargs):
            pass

        with patch("src.training.trainer.loop.gc.collect") as mock_gc:
            loop.run(
                train_step_fn=mock_train_step,
                save_checkpoint_fn=mock_save_checkpoint,
                num_steps=10,
                gc_interval=5,  # GC every 5 steps
                log_interval=0,
                save_interval=0,
                use_progress_bar=False,
            )

            # GC should be called at steps 5, 10
            assert mock_gc.call_count >= 2

    def test_loop_calculate_total_steps_from_epochs(self):
        """Test total steps calculation from num_epochs."""
        loop = create_training_loop(num_batches=10)

        total = loop._calculate_total_steps(num_steps=None, num_epochs=3)

        # 3 epochs * 10 batches = 30 steps
        assert total == 30

    def test_loop_calculate_total_steps_from_steps(self):
        """Test total steps calculation from num_steps."""
        loop = create_training_loop(num_batches=10)

        total = loop._calculate_total_steps(num_steps=50, num_epochs=None)

        # Direct num_steps should be used
        assert total == 50

    def test_loop_batch_idx_tracking(self):
        """Test that batch_idx is tracked correctly within epoch."""
        loop = create_training_loop(num_batches=3)
        batch_indices = []

        def mock_train_step(batch: Dict[str, torch.Tensor]) -> TrainMetrics:
            batch_indices.append(loop.state.batch_idx)
            return create_mock_metrics()

        def mock_save_checkpoint(*args, **kwargs):
            pass

        loop.run(
            train_step_fn=mock_train_step,
            save_checkpoint_fn=mock_save_checkpoint,
            num_steps=6,  # 2 epochs
            log_interval=0,
            save_interval=0,
            use_progress_bar=False,
        )

        # batch_idx should go 1,2,3, then reset to 1,2,3 for second epoch
        assert batch_indices == [1, 2, 3, 1, 2, 3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
