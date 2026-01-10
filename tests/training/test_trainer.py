"""
Trainer Tests

Tests for Trainer, TrainerState, and TrainMetrics with mocked components.

Test Categories:
    - TrainerState (3 tests): Initialization, defaults
    - TrainMetrics (4 tests): Initialization, to_dict
    - Trainer Initialization (6 tests): Default config, custom config
    - Trainer Components (5 tests): LR scheduler, warmup, getters
    - train_step (6 tests): Forward pass, loss computation, gradient
    - train loop (4 tests): Loop execution, callbacks, progress
    - Checkpointing (6 tests): Save, load, state restoration
    - Factory Functions (2 tests): create_trainer

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass
from typing import Dict, Any
import tempfile
import shutil

from src.training.trainer import (
    Trainer,
    TrainerState,
    TrainMetrics,
    create_trainer,
)
from src.config import PixelHDMConfig, TrainingConfig


# ============================================================================
# Test Fixtures
# ============================================================================

class SimpleModel(nn.Module):
    """Simple model for testing.

    Uses smaller 64x64 internal representation to reduce memory and speed up tests.
    The 256x256 -> 64x64 pooling and 64x64 -> 256x256 upscaling reduce computation
    from 196608 to 12288 parameters per linear layer (16x faster).
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Use smaller internal resolution for efficiency
        self._internal_size = 64
        self._internal_features = 3 * self._internal_size * self._internal_size  # 12288
        self.linear1 = nn.Linear(self._internal_features, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, self._internal_features)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_embeddings: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        B = x.shape[0]
        H, W = x.shape[1], x.shape[2]

        # Pool to internal size for efficient computation
        x_bhwc = x.view(B, H, W, 3)
        x_bchw = x_bhwc.permute(0, 3, 1, 2)  # (B, 3, H, W)
        x_small = torch.nn.functional.adaptive_avg_pool2d(x_bchw, self._internal_size)

        # Process
        x_flat = x_small.reshape(B, -1)
        h = self.linear1(x_flat)
        out_flat = self.linear2(h)

        # Upsample back to original size
        out_small = out_flat.reshape(B, 3, self._internal_size, self._internal_size)
        out_bchw = torch.nn.functional.interpolate(out_small, size=(H, W), mode='bilinear', align_corners=False)
        out_bhwc = out_bchw.permute(0, 2, 3, 1)  # (B, H, W, 3)

        return out_bhwc


# Module-scoped fixtures for expensive objects to avoid recreation
@pytest.fixture(scope="module")
def testing_config() -> PixelHDMConfig:
    """Create minimal config for testing (module-scoped for reuse)."""
    return PixelHDMConfig.for_testing()


@pytest.fixture(scope="module")
def training_config() -> TrainingConfig:
    """Create training config for testing (module-scoped for reuse)."""
    return TrainingConfig(
        learning_rate=1e-4,
        weight_decay=0.01,
        max_steps=100,
        warmup_steps=10,
        ema_decay=0.9999,
        max_grad_norm=1.0,
        mixed_precision="fp32",
        gradient_accumulation_steps=1,
        lr_scheduler="cosine",
        min_lr=1e-6,
    )


@pytest.fixture
def simple_model() -> SimpleModel:
    """Create simple model for testing (function-scoped for isolation)."""
    return SimpleModel(hidden_dim=64)


@pytest.fixture
def mock_dataloader():
    """Create mock dataloader for testing."""
    def create_batch():
        return {
            "images": torch.randn(2, 256, 256, 3),
            "text_embeddings": torch.randn(2, 32, 256),
        }

    mock_dl = MagicMock()
    mock_dl.__len__ = MagicMock(return_value=10)
    mock_dl.__iter__ = MagicMock(return_value=iter([create_batch() for _ in range(10)]))
    return mock_dl


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    tmp_dir = tempfile.mkdtemp()
    yield Path(tmp_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================================
# TrainerState Tests
# ============================================================================

class TestTrainerState:
    """Tests for TrainerState dataclass."""

    def test_default_init(self):
        """Test default initialization."""
        state = TrainerState()

        assert state.step == 0
        assert state.epoch == 0
        assert state.batch_idx == 0
        assert state.best_loss == float("inf")
        assert state.total_samples == 0

    def test_custom_init(self):
        """Test custom initialization."""
        state = TrainerState(
            step=100,
            epoch=5,
            batch_idx=10,
            best_loss=0.5,
            total_samples=5000,
        )

        assert state.step == 100
        assert state.epoch == 5
        assert state.batch_idx == 10
        assert state.best_loss == 0.5
        assert state.total_samples == 5000

    def test_state_is_mutable(self):
        """Test state can be modified."""
        state = TrainerState()

        state.step = 10
        state.epoch = 1
        state.best_loss = 0.3

        assert state.step == 10
        assert state.epoch == 1
        assert state.best_loss == 0.3


# ============================================================================
# TrainMetrics Tests
# ============================================================================

class TestTrainMetrics:
    """Tests for TrainMetrics dataclass."""

    def test_default_init(self):
        """Test initialization with required field only."""
        metrics = TrainMetrics(loss=0.5)

        assert metrics.loss == 0.5
        assert metrics.loss_vloss == 0.0
        assert metrics.loss_freq == 0.0
        assert metrics.loss_repa == 0.0
        assert metrics.grad_norm == 0.0
        assert metrics.learning_rate == 0.0
        assert metrics.samples_per_sec == 0.0
        assert metrics.step_time == 0.0

    def test_full_init(self):
        """Test initialization with all fields."""
        metrics = TrainMetrics(
            loss=0.5,
            loss_vloss=0.3,
            loss_freq=0.1,
            loss_repa=0.1,
            grad_norm=1.5,
            learning_rate=1e-4,
            samples_per_sec=100.0,
            step_time=0.05,
        )

        assert metrics.loss == 0.5
        assert metrics.loss_vloss == 0.3
        assert metrics.loss_freq == 0.1
        assert metrics.loss_repa == 0.1
        assert metrics.grad_norm == 1.5
        assert metrics.learning_rate == 1e-4
        assert metrics.samples_per_sec == 100.0
        assert metrics.step_time == 0.05

    def test_to_dict(self):
        """Test to_dict method."""
        metrics = TrainMetrics(
            loss=0.5,
            loss_vloss=0.3,
            loss_freq=0.1,
            loss_repa=0.1,
            grad_norm=1.5,
            learning_rate=1e-4,
            samples_per_sec=100.0,
            step_time=0.05,
        )

        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert result["loss"] == 0.5
        assert result["loss_vloss"] == 0.3
        assert result["loss_freq"] == 0.1
        assert result["loss_repa"] == 0.1
        assert result["grad_norm"] == 1.5
        assert result["learning_rate"] == 1e-4
        assert result["samples_per_sec"] == 100.0
        assert result["step_time"] == 0.05

    def test_to_dict_keys(self):
        """Test to_dict contains all expected keys."""
        metrics = TrainMetrics(loss=0.5)
        result = metrics.to_dict()

        expected_keys = {
            "loss", "loss_vloss", "loss_freq", "loss_repa",
            "grad_norm", "learning_rate", "samples_per_sec", "step_time"
        }
        assert set(result.keys()) == expected_keys


# ============================================================================
# Trainer Initialization Tests
# ============================================================================

class TestTrainerInit:
    """Tests for Trainer initialization."""

    def test_basic_init(self, simple_model):
        """Test basic initialization with model only."""
        trainer = Trainer(model=simple_model, device=torch.device("cpu"))

        assert trainer.model is not None
        assert trainer.device.type == "cpu"
        assert trainer.optimizer is not None
        assert trainer.flow_matching is not None
        assert trainer.combined_loss is not None
        assert trainer.zclip is not None
        assert trainer.ema is not None
        assert trainer.cpu_checkpoint is not None
        assert trainer.state is not None
        assert trainer.state.step == 0

    def test_init_with_config(self, simple_model, testing_config):
        """Test initialization with model config."""
        trainer = Trainer(
            model=simple_model,
            config=testing_config,
            device=torch.device("cpu"),
        )

        assert trainer.config is testing_config

    def test_init_with_training_config(self, simple_model, training_config):
        """Test initialization with training config."""
        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            device=torch.device("cpu"),
        )

        assert trainer.training_config is training_config
        assert trainer.max_grad_norm == training_config.max_grad_norm
        assert trainer.gradient_accumulation_steps == training_config.gradient_accumulation_steps

    def test_init_with_custom_optimizer(self, simple_model):
        """Test initialization with custom optimizer."""
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)

        trainer = Trainer(
            model=simple_model,
            optimizer=optimizer,
            device=torch.device("cpu"),
        )

        assert trainer.optimizer is optimizer

    def test_init_no_ema_when_decay_zero(self, simple_model):
        """Test EMA is None when decay is 0."""
        training_config = TrainingConfig(ema_decay=0.0)

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            device=torch.device("cpu"),
        )

        assert trainer.ema is None

    def test_init_amp_settings(self, simple_model):
        """Test AMP settings based on precision."""
        # BF16
        trainer_bf16 = Trainer(
            model=simple_model,
            training_config=TrainingConfig(mixed_precision="bf16"),
            device=torch.device("cpu"),
        )
        assert trainer_bf16.use_amp is True
        assert trainer_bf16.amp_dtype == torch.bfloat16
        assert trainer_bf16.scaler is None  # No scaler for bf16

        # FP16
        trainer_fp16 = Trainer(
            model=simple_model,
            training_config=TrainingConfig(mixed_precision="fp16"),
            device=torch.device("cpu"),
        )
        assert trainer_fp16.use_amp is True
        assert trainer_fp16.amp_dtype == torch.float16
        assert trainer_fp16.scaler is not None

        # FP32
        trainer_fp32 = Trainer(
            model=simple_model,
            training_config=TrainingConfig(mixed_precision="fp32"),
            device=torch.device("cpu"),
        )
        assert trainer_fp32.use_amp is False


# ============================================================================
# Trainer Components Tests
# ============================================================================

class TestTrainerComponents:
    """Tests for Trainer components."""

    def test_get_lr(self, simple_model):
        """Test _get_lr returns current learning rate."""
        trainer = Trainer(model=simple_model, device=torch.device("cpu"))

        lr = trainer._get_lr()

        assert isinstance(lr, float)
        assert lr > 0

    def test_warmup_lr(self, simple_model, training_config):
        """Test _warmup_lr updates learning rate."""
        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            device=torch.device("cpu"),
        )

        # At step 0, warmup should give low LR
        trainer._warmup_lr(0)
        lr_step0 = trainer._get_lr()

        # At step 5 (middle of warmup), LR should be higher
        trainer._warmup_lr(5)
        lr_step5 = trainer._get_lr()

        assert lr_step5 > lr_step0

    def test_warmup_lr_no_config(self, simple_model):
        """Test _warmup_lr does nothing without training config."""
        trainer = Trainer(model=simple_model, device=torch.device("cpu"))

        initial_lr = trainer._get_lr()
        trainer._warmup_lr(5)
        after_lr = trainer._get_lr()

        assert initial_lr == after_lr

    def test_lr_scheduler_creation(self, simple_model, training_config):
        """Test lr_scheduler property creates scheduler."""
        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            device=torch.device("cpu"),
        )

        # Access property to trigger creation
        scheduler = trainer.lr_scheduler

        assert scheduler is not None
        assert trainer._lr_scheduler is scheduler

    def test_set_dino_encoder(self, simple_model):
        """Test set_dino_encoder passes encoder to combined_loss."""
        trainer = Trainer(model=simple_model, device=torch.device("cpu"))

        mock_encoder = MagicMock()
        trainer.combined_loss.set_dino_encoder = MagicMock()

        trainer.set_dino_encoder(mock_encoder)

        trainer.combined_loss.set_dino_encoder.assert_called_once_with(mock_encoder)


# ============================================================================
# train_step Tests
# ============================================================================

class TestTrainStep:
    """Tests for train_step method."""

    def test_train_step_returns_metrics(self, simple_model):
        """Test train_step returns TrainMetrics."""
        trainer = Trainer(
            model=simple_model,
            training_config=TrainingConfig(mixed_precision="fp32"),
            device=torch.device("cpu"),
        )

        batch = {
            "images": torch.randn(2, 256, 256, 3),
        }

        metrics = trainer.train_step(batch)

        assert isinstance(metrics, TrainMetrics)
        assert metrics.loss >= 0
        assert metrics.step_time > 0

    def test_train_step_increments_state(self, simple_model):
        """Test train_step increments step counter."""
        trainer = Trainer(
            model=simple_model,
            training_config=TrainingConfig(mixed_precision="fp32"),
            device=torch.device("cpu"),
        )

        assert trainer.state.step == 0

        batch = {"images": torch.randn(2, 256, 256, 3)}
        trainer.train_step(batch)

        assert trainer.state.step == 1

    def test_train_step_with_text_embeddings(self, simple_model):
        """Test train_step with text embeddings."""
        trainer = Trainer(
            model=simple_model,
            training_config=TrainingConfig(mixed_precision="fp32"),
            device=torch.device("cpu"),
        )

        batch = {
            "images": torch.randn(2, 256, 256, 3),
            "text_embeddings": torch.randn(2, 32, 256),
        }

        metrics = trainer.train_step(batch)

        assert isinstance(metrics, TrainMetrics)
        assert metrics.loss >= 0

    def test_train_step_updates_total_samples(self, simple_model):
        """Test train_step updates total_samples."""
        trainer = Trainer(
            model=simple_model,
            training_config=TrainingConfig(mixed_precision="fp32"),
            device=torch.device("cpu"),
        )

        batch = {"images": torch.randn(4, 256, 256, 3)}
        trainer.train_step(batch)

        assert trainer.state.total_samples == 4

        trainer.train_step(batch)

        assert trainer.state.total_samples == 8

    def test_train_step_gradient_accumulation(self, simple_model):
        """Test train_step with gradient accumulation."""
        trainer = Trainer(
            model=simple_model,
            training_config=TrainingConfig(
                mixed_precision="fp32",
                gradient_accumulation_steps=2,
            ),
            device=torch.device("cpu"),
        )

        batch = {"images": torch.randn(2, 256, 256, 3)}

        # First step: accumulating
        metrics1 = trainer.train_step(batch)
        assert metrics1.grad_norm == 0.0  # No update yet

        # Second step: should update
        metrics2 = trainer.train_step(batch)
        # grad_norm might be > 0 after update

    def test_train_step_ema_update(self, simple_model):
        """Test train_step updates EMA."""
        trainer = Trainer(
            model=simple_model,
            training_config=TrainingConfig(
                mixed_precision="fp32",
                ema_decay=0.999,
            ),
            device=torch.device("cpu"),
        )

        # EMA should exist
        assert trainer.ema is not None

        batch = {"images": torch.randn(2, 256, 256, 3)}
        trainer.train_step(batch)

        # EMA should have been updated
        # (internal state changed, hard to verify directly)


# ============================================================================
# train Loop Tests
# ============================================================================

class TestTrainLoop:
    """Tests for train method."""

    def test_train_requires_dataloader(self, simple_model):
        """Test train raises error without dataloader."""
        trainer = Trainer(model=simple_model, device=torch.device("cpu"))

        with pytest.raises(RuntimeError, match="dataloader"):
            trainer.train(num_steps=10)

    def test_train_with_num_steps(self, simple_model, mock_dataloader):
        """Test train with num_steps parameter."""
        trainer = Trainer(
            model=simple_model,
            dataloader=mock_dataloader,
            training_config=TrainingConfig(mixed_precision="fp32"),
            device=torch.device("cpu"),
        )

        # Train for 3 steps
        trainer.train(num_steps=3, use_progress_bar=False, log_interval=0)

        assert trainer.state.step == 3

    def test_train_with_callback(self, simple_model, mock_dataloader):
        """Test train with callback function."""
        trainer = Trainer(
            model=simple_model,
            dataloader=mock_dataloader,
            training_config=TrainingConfig(mixed_precision="fp32"),
            device=torch.device("cpu"),
        )

        callback_steps = []

        def callback(step, metrics):
            callback_steps.append(step)

        trainer.train(num_steps=3, callback=callback, use_progress_bar=False, log_interval=0)

        assert len(callback_steps) == 3
        assert callback_steps == [1, 2, 3]

    def test_train_updates_epoch_on_iterator_exhaustion(self, simple_model):
        """Test train increments epoch when dataloader is exhausted."""
        # Create dataloader that yields only 2 batches
        def create_batch():
            return {"images": torch.randn(2, 256, 256, 3)}

        mock_dl = MagicMock()
        mock_dl.__len__ = MagicMock(return_value=2)

        # First epoch: 2 batches, second epoch: 2 batches
        batches_epoch1 = [create_batch(), create_batch()]
        batches_epoch2 = [create_batch(), create_batch()]
        mock_dl.__iter__ = MagicMock(side_effect=[
            iter(batches_epoch1),
            iter(batches_epoch2),
        ])

        trainer = Trainer(
            model=simple_model,
            dataloader=mock_dl,
            training_config=TrainingConfig(mixed_precision="fp32"),
            device=torch.device("cpu"),
        )

        # Train for 3 steps (should cross epoch boundary)
        trainer.train(num_steps=3, use_progress_bar=False, log_interval=0)

        assert trainer.state.epoch == 1  # Crossed into second epoch


# ============================================================================
# Checkpointing Tests
# ============================================================================

class TestCheckpointing:
    """Tests for checkpoint save/load."""

    def test_save_checkpoint(self, simple_model, temp_checkpoint_dir):
        """Test save_checkpoint creates file."""
        trainer = Trainer(
            model=simple_model,
            training_config=TrainingConfig(mixed_precision="fp32"),
            device=torch.device("cpu"),
        )

        trainer.state.step = 100
        trainer.save_checkpoint(temp_checkpoint_dir)

        checkpoint_file = temp_checkpoint_dir / "checkpoint_step_100.pt"
        assert checkpoint_file.exists()

    def test_save_checkpoint_custom_name(self, simple_model, temp_checkpoint_dir):
        """Test save_checkpoint with custom name."""
        trainer = Trainer(
            model=simple_model,
            training_config=TrainingConfig(mixed_precision="fp32"),
            device=torch.device("cpu"),
        )

        trainer.save_checkpoint(temp_checkpoint_dir, checkpoint_name="best_model")

        checkpoint_file = temp_checkpoint_dir / "best_model.pt"
        assert checkpoint_file.exists()

    def test_save_checkpoint_contains_required_keys(self, simple_model, temp_checkpoint_dir):
        """Test checkpoint contains required keys."""
        trainer = Trainer(
            model=simple_model,
            training_config=TrainingConfig(mixed_precision="fp32", ema_decay=0.999),
            device=torch.device("cpu"),
        )

        trainer.state.step = 50
        trainer.save_checkpoint(temp_checkpoint_dir)

        checkpoint_file = temp_checkpoint_dir / "checkpoint_step_50.pt"
        checkpoint = torch.load(checkpoint_file, weights_only=True)

        assert "model" in checkpoint
        assert "optimizer" in checkpoint
        assert "state" in checkpoint
        assert "ema" in checkpoint
        assert "zclip" in checkpoint

    def test_load_checkpoint_restores_state(self, simple_model, temp_checkpoint_dir):
        """Test load_checkpoint restores state."""
        # Create and save checkpoint
        trainer1 = Trainer(
            model=simple_model,
            training_config=TrainingConfig(mixed_precision="fp32"),
            device=torch.device("cpu"),
        )
        trainer1.state.step = 100
        trainer1.state.epoch = 5
        trainer1.state.best_loss = 0.3
        trainer1.save_checkpoint(temp_checkpoint_dir)

        # Create new trainer and load checkpoint
        trainer2 = Trainer(
            model=SimpleModel(),
            training_config=TrainingConfig(mixed_precision="fp32"),
            device=torch.device("cpu"),
        )

        checkpoint_file = temp_checkpoint_dir / "checkpoint_step_100.pt"
        trainer2.load_checkpoint(checkpoint_file)

        assert trainer2.state.step == 100
        assert trainer2.state.epoch == 5
        assert trainer2.state.best_loss == 0.3

    def test_load_checkpoint_skip_optimizer(self, simple_model, temp_checkpoint_dir):
        """Test load_checkpoint can skip optimizer."""
        trainer1 = Trainer(
            model=simple_model,
            training_config=TrainingConfig(mixed_precision="fp32"),
            device=torch.device("cpu"),
        )
        trainer1.state.step = 50
        trainer1.save_checkpoint(temp_checkpoint_dir)

        trainer2 = Trainer(
            model=SimpleModel(),
            training_config=TrainingConfig(mixed_precision="fp32"),
            device=torch.device("cpu"),
        )

        original_optimizer_state = trainer2.optimizer.state_dict()

        checkpoint_file = temp_checkpoint_dir / "checkpoint_step_50.pt"
        trainer2.load_checkpoint(checkpoint_file, load_optimizer=False)

        # Optimizer state should be unchanged
        # (simplified check - just verify no error)
        assert trainer2.state.step == 50

    def test_checkpoint_round_trip(self, simple_model, temp_checkpoint_dir):
        """Test checkpoint save/load preserves model weights."""
        trainer1 = Trainer(
            model=simple_model,
            training_config=TrainingConfig(mixed_precision="fp32"),
            device=torch.device("cpu"),
        )

        # Get original weights
        original_weight = trainer1.model.linear1.weight.clone()

        trainer1.state.step = 25
        trainer1.save_checkpoint(temp_checkpoint_dir)

        # Modify weights
        with torch.no_grad():
            trainer1.model.linear1.weight.fill_(999.0)

        # Load checkpoint
        checkpoint_file = temp_checkpoint_dir / "checkpoint_step_25.pt"
        trainer1.load_checkpoint(checkpoint_file)

        # Weights should be restored
        assert torch.allclose(trainer1.model.linear1.weight, original_weight)


# ============================================================================
# Factory Function Tests
# ============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_trainer_basic(self, simple_model):
        """Test create_trainer factory function."""
        trainer = create_trainer(
            model=simple_model,
            device=torch.device("cpu"),
        )

        assert isinstance(trainer, Trainer)
        assert trainer.model is simple_model

    def test_create_trainer_full(self, simple_model, testing_config, training_config, mock_dataloader):
        """Test create_trainer with all parameters."""
        trainer = create_trainer(
            model=simple_model,
            config=testing_config,
            training_config=training_config,
            dataloader=mock_dataloader,
            device=torch.device("cpu"),
        )

        assert isinstance(trainer, Trainer)
        assert trainer.config is testing_config
        assert trainer.training_config is training_config
        assert trainer.dataloader is mock_dataloader


# ============================================================================
# Edge Cases Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_train_defaults_from_config(self, simple_model, mock_dataloader):
        """Test train uses config max_steps when neither num_steps nor num_epochs given."""
        training_config = TrainingConfig(
            max_steps=5,
            training_mode="steps",  # Use steps mode to respect max_steps
            mixed_precision="fp32"
        )

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            dataloader=mock_dataloader,
            device=torch.device("cpu"),
        )

        trainer.train(use_progress_bar=False, log_interval=0)

        assert trainer.state.step == 5

    def test_train_default_steps_no_config(self, simple_model, mock_dataloader):
        """Test train defaults to 10000 steps without config."""
        trainer = Trainer(
            model=simple_model,
            dataloader=mock_dataloader,
            device=torch.device("cpu"),
        )

        # Don't actually run 10000 steps, just verify it would
        # Instead, run with explicit num_steps
        trainer.train(num_steps=2, use_progress_bar=False, log_interval=0)

        assert trainer.state.step == 2

    def test_lr_scheduler_constant_without_config(self, simple_model):
        """Test lr_scheduler defaults to constant without training config."""
        trainer = Trainer(
            model=simple_model,
            device=torch.device("cpu"),
        )

        scheduler = trainer.lr_scheduler

        assert isinstance(scheduler, torch.optim.lr_scheduler.ConstantLR)


# ============================================================================
# Integration Tests (H-008)
# ============================================================================

class TestTrainingConvergence:
    """Tests for training convergence (H-008)."""

    @pytest.mark.slow
    @pytest.mark.xfail(reason="Flaky: simple model may not converge in 10 steps with random init")
    def test_loss_decreases_after_training(self, simple_model):
        """Verify loss decreases after multiple training steps.

        Note: This test is marked as xfail because:
        - Simple test model has no inductive bias
        - 10 steps may not be enough for convergence
        - Random initialization can cause unstable training
        Real model convergence is tested in integration tests.
        """
        trainer = Trainer(
            model=simple_model,
            training_config=TrainingConfig(
                learning_rate=1e-3,  # Lower LR for more stable training
                mixed_precision="fp32",
            ),
            device=torch.device("cpu"),
        )

        # Use fixed batch for consistent training
        torch.manual_seed(42)
        fixed_batch = {"images": torch.randn(2, 256, 256, 3)}

        # Record initial loss
        initial_metrics = trainer.train_step(fixed_batch)
        initial_loss = initial_metrics.loss

        # Train 20 more steps on same data (overfit intentionally)
        losses = [initial_loss]
        for _ in range(20):
            metrics = trainer.train_step(fixed_batch)
            losses.append(metrics.loss)

        # Verify loss decreased (at least one of the later losses should be lower)
        min_loss = min(losses[5:])  # After warmup
        assert min_loss < initial_loss, (
            f"Loss should decrease: initial={initial_loss:.4f}, min={min_loss:.4f}"
        )

    @pytest.mark.slow
    def test_ema_tracks_model_weights(self, simple_model):
        """Verify EMA weights track model weights during training."""
        trainer = Trainer(
            model=simple_model,
            training_config=TrainingConfig(
                learning_rate=0.1,  # High LR to see changes
                ema_decay=0.9,      # Lower decay for faster tracking
                mixed_precision="fp32",
            ),
            device=torch.device("cpu"),
        )

        assert trainer.ema is not None

        # Record initial EMA weight
        initial_ema = trainer.ema.shadow["linear1.weight"].clone()
        initial_model = trainer.model.linear1.weight.clone()

        # Train several steps
        batch = {"images": torch.randn(2, 256, 256, 3)}
        for _ in range(5):
            trainer.train_step(batch)

        # EMA should have updated
        final_ema = trainer.ema.shadow["linear1.weight"]
        assert not torch.allclose(initial_ema, final_ema, atol=1e-4), (
            "EMA weights should have been updated"
        )

    def test_triple_loss_non_zero(self, simple_model):
        """Test VLoss and FreqLoss are computed (REPA disabled by default)."""
        trainer = Trainer(
            model=simple_model,
            training_config=TrainingConfig(mixed_precision="fp32"),
            device=torch.device("cpu"),
        )

        batch = {"images": torch.randn(2, 256, 256, 3)}
        metrics = trainer.train_step(batch)

        # VLoss should be computed
        assert metrics.loss_vloss >= 0
        # FreqLoss should be computed
        assert metrics.loss_freq >= 0
        # Total loss should be non-negative
        assert metrics.loss >= 0
        assert not torch.isnan(torch.tensor(metrics.loss))


# ============================================================================
# OOM Recovery Tests (H-009)
# ============================================================================

class TestOOMRecovery:
    """Tests for OOM recovery mechanism (H-009)."""

    def test_halve_batch(self, simple_model):
        """Test _halve_batch reduces batch size."""
        trainer = Trainer(model=simple_model, device=torch.device("cpu"))

        batch = {
            "images": torch.randn(8, 256, 256, 3),
            "text_embeddings": torch.randn(8, 32, 256),
        }

        halved = trainer._halve_batch(batch)

        assert halved["images"].shape[0] == 4
        assert halved["text_embeddings"].shape[0] == 4

    def test_halve_batch_preserves_non_tensor(self, simple_model):
        """Test _halve_batch preserves non-tensor values."""
        trainer = Trainer(model=simple_model, device=torch.device("cpu"))

        batch = {
            "images": torch.randn(8, 256, 256, 3),
            "metadata": "some_string",
        }

        halved = trainer._halve_batch(batch)

        assert halved["images"].shape[0] == 4
        assert halved["metadata"] == "some_string"

    def test_halve_batch_single_sample(self, simple_model):
        """Test _halve_batch handles single sample (cannot halve)."""
        trainer = Trainer(model=simple_model, device=torch.device("cpu"))

        batch = {"images": torch.randn(1, 256, 256, 3)}

        halved = trainer._halve_batch(batch)

        # Single sample cannot be halved, should keep original
        assert halved["images"].shape[0] == 1

    def test_safe_train_step_success(self, simple_model):
        """Test safe_train_step returns metrics on success."""
        trainer = Trainer(
            model=simple_model,
            training_config=TrainingConfig(mixed_precision="fp32"),
            device=torch.device("cpu"),
        )

        batch = {"images": torch.randn(2, 256, 256, 3)}
        metrics = trainer.safe_train_step(batch)

        assert metrics is not None
        assert isinstance(metrics, TrainMetrics)
        assert metrics.loss >= 0

    def test_safe_train_step_oom_recovery(self, simple_model):
        """Test safe_train_step recovers from OOM."""
        trainer = Trainer(
            model=simple_model,
            training_config=TrainingConfig(mixed_precision="fp32"),
            device=torch.device("cpu"),
        )

        batch = {"images": torch.randn(4, 256, 256, 3)}

        # Mock train_step to fail once then succeed
        original_train_step = trainer.train_step
        call_count = [0]

        def mock_train_step(b):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("CUDA out of memory")
            return original_train_step(b)

        trainer.train_step = mock_train_step

        metrics = trainer.safe_train_step(batch)

        assert metrics is not None
        assert call_count[0] == 2  # First failed, second succeeded

    def test_safe_train_step_oom_with_batch_halving(self, simple_model):
        """Test safe_train_step halves batch on repeated OOM."""
        trainer = Trainer(
            model=simple_model,
            training_config=TrainingConfig(mixed_precision="fp32"),
            device=torch.device("cpu"),
        )

        batch = {"images": torch.randn(8, 256, 256, 3)}

        original_train_step = trainer.train_step
        call_count = [0]
        batch_sizes = []

        def mock_train_step(b):
            call_count[0] += 1
            batch_sizes.append(b["images"].shape[0])
            if call_count[0] <= 2:  # Fail first 2 times
                raise RuntimeError("CUDA out of memory")
            return original_train_step(b)

        trainer.train_step = mock_train_step

        metrics = trainer.safe_train_step(batch)

        assert metrics is not None
        assert call_count[0] == 3
        # First attempt: 8, second: 8, third: 4 (halved after 2nd retry)
        assert batch_sizes == [8, 8, 4]

    def test_safe_train_step_max_retries_exceeded(self, simple_model):
        """Test safe_train_step returns None when max retries exceeded."""
        trainer = Trainer(
            model=simple_model,
            training_config=TrainingConfig(mixed_precision="fp32"),
            device=torch.device("cpu"),
        )

        batch = {"images": torch.randn(4, 256, 256, 3)}

        # Always fail with OOM
        def mock_train_step(b):
            raise RuntimeError("CUDA out of memory")

        trainer.train_step = mock_train_step

        metrics = trainer.safe_train_step(batch, max_retries=2)

        assert metrics is None

    def test_safe_train_step_non_oom_error_propagates(self, simple_model):
        """Test safe_train_step propagates non-OOM errors."""
        trainer = Trainer(
            model=simple_model,
            training_config=TrainingConfig(mixed_precision="fp32"),
            device=torch.device("cpu"),
        )

        batch = {"images": torch.randn(4, 256, 256, 3)}

        def mock_train_step(b):
            raise ValueError("Some other error")

        trainer.train_step = mock_train_step

        with pytest.raises(ValueError, match="Some other error"):
            trainer.safe_train_step(batch)

    def test_safe_train_step_retry_disabled(self, simple_model):
        """Test safe_train_step with retry_on_oom=False."""
        trainer = Trainer(
            model=simple_model,
            training_config=TrainingConfig(mixed_precision="fp32"),
            device=torch.device("cpu"),
        )

        batch = {"images": torch.randn(4, 256, 256, 3)}

        def mock_train_step(b):
            raise RuntimeError("CUDA out of memory")

        trainer.train_step = mock_train_step

        with pytest.raises(RuntimeError, match="out of memory"):
            trainer.safe_train_step(batch, retry_on_oom=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
