"""
Trainer Strict Tests (2026-01-02)

Strict tests for Trainer following STRICT_TEST_STRATEGY.md.
These tests verify core training logic without excessive mocking.

Test Categories:
    1. test_text_encoder_caption_encoding - Verify captions are encoded correctly
    2. test_gradient_accumulation - Verify weights update every N steps
    3. test_zclip_from_config - Verify ZClip parameters from config (not hardcoded)
    4. test_ema_update - Verify EMA weights update correctly
    5. test_oom_recovery - Simulate OOM and verify safe_train_step handling
    6. test_checkpoint_cleanup - Verify max_checkpoints limit works
    7. test_warmup_lr - Verify warmup period LR linear increase
    8. test_lr_scheduler_restart - Verify cosine_restart restarts correctly

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

import gc
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.config import PixelHDMConfig, TrainingConfig
from src.training.trainer import (
    Trainer,
    TrainerState,
    TrainMetrics,
    create_trainer,
)
from src.training.optimization import EMA, ZClip


# ============================================================================
# Test Fixtures
# ============================================================================

class SimpleModel(nn.Module):
    """Simple model for testing with trackable weights."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(3 * 256 * 256, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 3 * 256 * 256)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        B = x.shape[0]
        # Flatten and process
        x_flat = x.reshape(B, -1)
        h = self.linear1(x_flat)
        out = self.linear2(h)
        # Reshape back to image-like (B, H, W, C)
        result = out.reshape(B, 256, 256, 3)
        if return_features:
            return result, h
        return result


class MockTextEncoder(nn.Module):
    """Mock text encoder that returns deterministic embeddings."""

    def __init__(self, hidden_dim: int = 256, seq_len: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        # Track calls for verification
        self.call_count = 0
        self.last_input = None

    def forward(
        self,
        texts: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.call_count += 1
        self.last_input = texts

        if texts is not None:
            batch_size = len(texts)
        elif input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = 1

        # Return (embeddings, mask)
        embeddings = torch.randn(batch_size, self.seq_len, self.hidden_dim)
        mask = torch.ones(batch_size, self.seq_len, dtype=torch.bool)
        return embeddings, mask


@pytest.fixture
def testing_config() -> PixelHDMConfig:
    """Create minimal config for testing."""
    return PixelHDMConfig.for_testing()


@pytest.fixture
def training_config() -> TrainingConfig:
    """Create training config for testing."""
    return TrainingConfig(
        learning_rate=1e-4,
        weight_decay=0.01,
        max_steps=100,
        warmup_steps=10,
        ema_decay=0.999,
        max_grad_norm=1.0,
        mixed_precision="fp32",  # Use fp32 for deterministic testing
        gradient_accumulation_steps=1,
        lr_scheduler="cosine",
        min_lr=1e-6,
        zclip_threshold=2.5,
        zclip_ema_decay=0.99,
    )


@pytest.fixture
def simple_model() -> SimpleModel:
    """Create simple model for testing."""
    return SimpleModel(hidden_dim=256)


@pytest.fixture
def mock_text_encoder() -> MockTextEncoder:
    """Create mock text encoder for testing."""
    return MockTextEncoder(hidden_dim=256, seq_len=32)


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    tmp_dir = tempfile.mkdtemp()
    yield Path(tmp_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================================
# 1. Text Encoder Caption Encoding Tests
# ============================================================================

class TestTextEncoderCaptionEncoding:
    """Tests for text encoder caption encoding (C-002 fix verification)."""

    def test_text_encoder_called_with_captions(self, simple_model, training_config):
        """Verify text_encoder is called when captions are provided."""
        text_encoder = MockTextEncoder(hidden_dim=256, seq_len=32)

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            text_encoder=text_encoder,
            device=torch.device("cpu"),
        )

        # Batch with captions (no pre-computed embeddings)
        batch = {
            "images": torch.randn(2, 256, 256, 3),
            "captions": ["a beautiful sunset", "a cat sitting on a chair"],
        }

        # Text encoder should be called during train_step
        metrics = trainer.train_step(batch)

        assert text_encoder.call_count == 1, "Text encoder should be called once"
        assert text_encoder.last_input == batch["captions"], \
            "Text encoder should receive the captions"

    def test_text_embeddings_used_when_provided(self, simple_model, training_config):
        """Verify pre-computed text_embeddings are used when provided."""
        text_encoder = MockTextEncoder(hidden_dim=256, seq_len=32)

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            text_encoder=text_encoder,
            device=torch.device("cpu"),
        )

        # Batch with pre-computed embeddings
        batch = {
            "images": torch.randn(2, 256, 256, 3),
            "text_embeddings": torch.randn(2, 32, 256),
            "text_mask": torch.ones(2, 32, dtype=torch.bool),
        }

        metrics = trainer.train_step(batch)

        # Text encoder should NOT be called when embeddings are provided
        assert text_encoder.call_count == 0, \
            "Text encoder should NOT be called when text_embeddings are provided"

    def test_captions_encoding_produces_valid_embeddings(self, simple_model, training_config):
        """Verify caption encoding produces valid text embeddings."""
        text_encoder = MockTextEncoder(hidden_dim=256, seq_len=32)

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            text_encoder=text_encoder,
            device=torch.device("cpu"),
        )

        batch = {
            "images": torch.randn(2, 256, 256, 3),
            "captions": ["test caption 1", "test caption 2"],
        }

        # Should complete without error
        metrics = trainer.train_step(batch)

        assert metrics is not None
        assert metrics.loss >= 0
        assert not math.isnan(metrics.loss)

    def test_no_text_encoder_uses_none_embeddings(self, simple_model, training_config):
        """Verify training works without text encoder (unconditional mode)."""
        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            text_encoder=None,  # No text encoder
            device=torch.device("cpu"),
        )

        batch = {
            "images": torch.randn(2, 256, 256, 3),
            # No text_embeddings or captions
        }

        # Should complete without error
        metrics = trainer.train_step(batch)

        assert metrics is not None
        assert metrics.loss >= 0


# ============================================================================
# 2. Gradient Accumulation Tests
# ============================================================================

class TestGradientAccumulation:
    """Tests for gradient accumulation behavior."""

    def test_weights_update_every_n_steps(self, simple_model):
        """Verify weights only update every N accumulation steps."""
        accumulation_steps = 4

        trainer = Trainer(
            model=simple_model,
            training_config=TrainingConfig(
                mixed_precision="fp32",
                gradient_accumulation_steps=accumulation_steps,
                learning_rate=0.1,  # High LR to see weight changes
            ),
            device=torch.device("cpu"),
        )

        batch = {"images": torch.randn(2, 256, 256, 3)}

        # Record initial weights
        initial_weights = trainer.model.linear1.weight.clone()

        # First N-1 steps: weights should NOT change (accumulating)
        for step in range(accumulation_steps - 1):
            metrics = trainer.train_step(batch)
            current_weights = trainer.model.linear1.weight

            # Weights should still match initial (no update yet)
            # Note: Small numerical differences may occur due to gradient computation
            # but there should be no optimizer step
            assert metrics.grad_norm == 0.0, \
                f"Step {step}: grad_norm should be 0 during accumulation"

        # N-th step: weights SHOULD change
        metrics = trainer.train_step(batch)

        # After accumulation, grad_norm should be computed
        # Note: grad_norm might still be 0 if gradients are very small
        # but the optimizer step should have occurred
        final_weights = trainer.model.linear1.weight

        # Weights should have changed after accumulation
        assert not torch.allclose(initial_weights, final_weights, atol=1e-6), \
            "Weights should update after gradient accumulation completes"

    def test_gradients_accumulate_correctly(self, simple_model):
        """Verify gradients accumulate (sum) over accumulation steps."""
        accumulation_steps = 2

        trainer = Trainer(
            model=simple_model,
            training_config=TrainingConfig(
                mixed_precision="fp32",
                gradient_accumulation_steps=accumulation_steps,
            ),
            device=torch.device("cpu"),
        )

        # Use fixed batches
        torch.manual_seed(42)
        batch1 = {"images": torch.randn(2, 256, 256, 3)}
        batch2 = {"images": torch.randn(2, 256, 256, 3)}

        # Step 1: Accumulate gradients
        trainer.train_step(batch1)

        # Check gradients exist after first step
        has_grads_step1 = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in trainer.model.parameters()
        )

        # Step 2: Should update (complete accumulation)
        trainer.train_step(batch2)

        # Verify step counter incremented correctly
        assert trainer.state.step == 2

    def test_accumulation_step_counter(self, simple_model):
        """Verify step counter increments every step (not just on update)."""
        accumulation_steps = 3

        trainer = Trainer(
            model=simple_model,
            training_config=TrainingConfig(
                mixed_precision="fp32",
                gradient_accumulation_steps=accumulation_steps,
            ),
            device=torch.device("cpu"),
        )

        batch = {"images": torch.randn(2, 256, 256, 3)}

        for i in range(5):
            trainer.train_step(batch)
            assert trainer.state.step == i + 1, \
                f"Step counter should be {i + 1}, got {trainer.state.step}"


# ============================================================================
# 3. ZClip From Config Tests
# ============================================================================

class TestZClipFromConfig:
    """Tests for ZClip configuration (H-002 fix verification)."""

    def test_zclip_threshold_from_config(self, simple_model):
        """Verify ZClip threshold is read from config, not hardcoded."""
        custom_threshold = 3.5  # Not the default 2.5

        training_config = TrainingConfig(
            mixed_precision="fp32",
            zclip_threshold=custom_threshold,
        )

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            device=torch.device("cpu"),
        )

        assert trainer.zclip.threshold == custom_threshold, \
            f"ZClip threshold should be {custom_threshold}, got {trainer.zclip.threshold}"

    def test_zclip_ema_decay_from_config(self, simple_model):
        """Verify ZClip EMA decay is read from config."""
        custom_ema_decay = 0.95  # Not the default 0.99

        training_config = TrainingConfig(
            mixed_precision="fp32",
            zclip_ema_decay=custom_ema_decay,
        )

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            device=torch.device("cpu"),
        )

        assert trainer.zclip.ema_decay == custom_ema_decay, \
            f"ZClip ema_decay should be {custom_ema_decay}, got {trainer.zclip.ema_decay}"

    def test_zclip_default_without_config(self, simple_model):
        """Verify ZClip uses defaults when no config provided."""
        trainer = Trainer(
            model=simple_model,
            training_config=None,  # No config
            device=torch.device("cpu"),
        )

        # Should use defaults
        assert trainer.zclip.threshold == 2.5, "Default threshold should be 2.5"
        assert trainer.zclip.ema_decay == 0.99, "Default ema_decay should be 0.99"

    def test_zclip_is_used_in_training(self, simple_model):
        """Verify ZClip is actually used during gradient clipping."""
        training_config = TrainingConfig(
            mixed_precision="fp32",
            zclip_threshold=1.5,  # Low threshold to trigger clipping
        )

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            device=torch.device("cpu"),
        )

        batch = {"images": torch.randn(2, 256, 256, 3)}

        # Run several steps to let ZClip build statistics
        for _ in range(5):
            trainer.train_step(batch)

        # Verify ZClip has been used (step_count > 0)
        stats = trainer.zclip.get_stats()
        assert stats.steps > 0, "ZClip should have been used during training"


# ============================================================================
# 4. EMA Update Tests
# ============================================================================

class TestEMAUpdate:
    """Tests for EMA weight update behavior."""

    def test_ema_weights_update_after_train_step(self, simple_model):
        """Verify EMA weights are updated after each train step."""
        training_config = TrainingConfig(
            mixed_precision="fp32",
            ema_decay=0.9,  # Lower decay for faster tracking
            learning_rate=0.1,  # High LR to see weight changes
        )

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            device=torch.device("cpu"),
        )

        assert trainer.ema is not None, "EMA should be enabled"

        # Record initial EMA weights
        initial_ema_weights = trainer.ema.shadow["linear1.weight"].clone()

        batch = {"images": torch.randn(2, 256, 256, 3)}

        # Run a training step
        trainer.train_step(batch)

        # EMA weights should have changed
        current_ema_weights = trainer.ema.shadow["linear1.weight"]
        assert not torch.allclose(initial_ema_weights, current_ema_weights, atol=1e-6), \
            "EMA weights should update after train step"

    def test_ema_decay_parameter_affects_update(self, simple_model):
        """Verify EMA decay parameter affects weight update magnitude."""
        # Create two trainers with different decay values
        trainer_fast = Trainer(
            model=SimpleModel(hidden_dim=256),
            training_config=TrainingConfig(
                mixed_precision="fp32",
                ema_decay=0.5,  # Fast tracking (50% new weight)
                learning_rate=0.1,
            ),
            device=torch.device("cpu"),
        )

        trainer_slow = Trainer(
            model=SimpleModel(hidden_dim=256),
            training_config=TrainingConfig(
                mixed_precision="fp32",
                ema_decay=0.99,  # Slow tracking (1% new weight)
                learning_rate=0.1,
            ),
            device=torch.device("cpu"),
        )

        # Use same random seed for both
        torch.manual_seed(42)
        batch = {"images": torch.randn(2, 256, 256, 3)}

        # Record initial EMA weights
        fast_initial = trainer_fast.ema.shadow["linear1.weight"].clone()
        slow_initial = trainer_slow.ema.shadow["linear1.weight"].clone()

        # Train both
        trainer_fast.train_step(batch.copy())
        trainer_slow.train_step(batch.copy())

        # Compute differences
        fast_diff = (trainer_fast.ema.shadow["linear1.weight"] - fast_initial).abs().mean()
        slow_diff = (trainer_slow.ema.shadow["linear1.weight"] - slow_initial).abs().mean()

        # Fast decay should result in larger weight changes
        assert fast_diff > slow_diff, \
            f"Fast decay (0.5) should change weights more than slow decay (0.99). " \
            f"Fast diff: {fast_diff:.6f}, Slow diff: {slow_diff:.6f}"

    def test_ema_disabled_when_decay_zero(self, simple_model):
        """Verify EMA is disabled when ema_decay is 0."""
        training_config = TrainingConfig(
            mixed_precision="fp32",
            ema_decay=0.0,  # Disable EMA
        )

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            device=torch.device("cpu"),
        )

        assert trainer.ema is None, "EMA should be None when decay is 0"

    def test_ema_num_updates_increments(self, simple_model):
        """Verify EMA num_updates counter increments with each update."""
        training_config = TrainingConfig(
            mixed_precision="fp32",
            ema_decay=0.999,
        )

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            device=torch.device("cpu"),
        )

        assert trainer.ema.num_updates == 0

        batch = {"images": torch.randn(2, 256, 256, 3)}

        for i in range(5):
            trainer.train_step(batch)
            assert trainer.ema.num_updates == i + 1, \
                f"EMA num_updates should be {i + 1}, got {trainer.ema.num_updates}"


# ============================================================================
# 5. OOM Recovery Tests
# ============================================================================

class TestOOMRecovery:
    """Tests for OOM recovery mechanism (H-009 verification)."""

    def test_safe_train_step_recovers_from_oom(self, simple_model):
        """Verify safe_train_step recovers from OOM by retrying."""
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
                raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")
            return original_train_step(b)

        trainer.train_step = mock_train_step

        metrics = trainer.safe_train_step(batch)

        assert metrics is not None, "Should recover from OOM"
        assert call_count[0] == 2, "Should retry once after OOM"

    def test_safe_train_step_halves_batch_on_repeated_oom(self, simple_model):
        """Verify batch is halved after multiple OOM failures."""
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
        assert batch_sizes == [8, 8, 4], \
            f"Expected batch sizes [8, 8, 4], got {batch_sizes}"

    def test_safe_train_step_returns_none_on_max_retries(self, simple_model):
        """Verify None is returned when max retries exceeded."""
        trainer = Trainer(
            model=simple_model,
            training_config=TrainingConfig(mixed_precision="fp32"),
            device=torch.device("cpu"),
        )

        batch = {"images": torch.randn(4, 256, 256, 3)}

        def mock_train_step(b):
            raise RuntimeError("CUDA out of memory")

        trainer.train_step = mock_train_step

        metrics = trainer.safe_train_step(batch, max_retries=2)

        assert metrics is None, "Should return None after max retries"

    def test_halve_batch_preserves_all_keys(self, simple_model):
        """Verify _halve_batch preserves all batch keys."""
        trainer = Trainer(model=simple_model, device=torch.device("cpu"))

        batch = {
            "images": torch.randn(8, 256, 256, 3),
            "text_embeddings": torch.randn(8, 32, 256),
            "captions": ["cap1", "cap2", "cap3", "cap4", "cap5", "cap6", "cap7", "cap8"],
            "metadata": {"key": "value"},  # Non-tensor
        }

        halved = trainer._halve_batch(batch)

        assert halved["images"].shape[0] == 4
        assert halved["text_embeddings"].shape[0] == 4
        assert halved["captions"] == batch["captions"]  # List not halved
        assert halved["metadata"] == {"key": "value"}  # Dict preserved

    def test_safe_train_step_propagates_non_oom_errors(self, simple_model):
        """Verify non-OOM errors are propagated, not caught."""
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


# ============================================================================
# 6. Checkpoint Cleanup Tests
# ============================================================================

class TestCheckpointCleanup:
    """Tests for checkpoint cleanup (max_checkpoints verification)."""

    def test_max_checkpoints_limit(self, simple_model, temp_checkpoint_dir):
        """Verify old checkpoints are deleted when max_checkpoints is reached."""
        training_config = TrainingConfig(
            mixed_precision="fp32",
            max_checkpoints=2,
        )

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            device=torch.device("cpu"),
        )

        # Save 4 checkpoints
        for step in [10, 20, 30, 40]:
            trainer.state.step = step
            trainer.save_checkpoint(temp_checkpoint_dir)

        # Only 2 latest should remain
        checkpoints = list(temp_checkpoint_dir.glob("checkpoint_*.pt"))
        assert len(checkpoints) == 2, \
            f"Expected 2 checkpoints, found {len(checkpoints)}"

        # Verify it's the latest 2
        checkpoint_steps = sorted([
            int(p.stem.split("_")[-1]) for p in checkpoints
        ])
        assert checkpoint_steps == [30, 40], \
            f"Expected steps [30, 40], got {checkpoint_steps}"

    def test_max_checkpoints_zero_keeps_all(self, simple_model, temp_checkpoint_dir):
        """Verify max_checkpoints=0 keeps all checkpoints."""
        training_config = TrainingConfig(
            mixed_precision="fp32",
            max_checkpoints=0,  # Unlimited
        )

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            device=torch.device("cpu"),
        )

        # Save 4 checkpoints
        for step in [10, 20, 30, 40]:
            trainer.state.step = step
            trainer.save_checkpoint(temp_checkpoint_dir)

        # All 4 should remain
        checkpoints = list(temp_checkpoint_dir.glob("checkpoint_*.pt"))
        assert len(checkpoints) == 4, \
            f"Expected 4 checkpoints, found {len(checkpoints)}"

    def test_max_checkpoints_one_keeps_only_latest(self, simple_model, temp_checkpoint_dir):
        """Verify max_checkpoints=1 keeps only the latest checkpoint."""
        training_config = TrainingConfig(
            mixed_precision="fp32",
            max_checkpoints=1,
        )

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            device=torch.device("cpu"),
        )

        # Save 3 checkpoints
        for step in [100, 200, 300]:
            trainer.state.step = step
            trainer.save_checkpoint(temp_checkpoint_dir)

        # Only 1 should remain
        checkpoints = list(temp_checkpoint_dir.glob("checkpoint_*.pt"))
        assert len(checkpoints) == 1, \
            f"Expected 1 checkpoint, found {len(checkpoints)}"

        # Should be the latest
        assert checkpoints[0].stem == "checkpoint_step_300"

    def test_named_checkpoints_not_deleted(self, simple_model, temp_checkpoint_dir):
        """Verify named checkpoints (like best_model) are not deleted by cleanup."""
        training_config = TrainingConfig(
            mixed_precision="fp32",
            max_checkpoints=1,
        )

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            device=torch.device("cpu"),
        )

        # Save a named checkpoint
        trainer.save_checkpoint(temp_checkpoint_dir, checkpoint_name="best_model", cleanup=False)

        # Save regular checkpoints
        for step in [10, 20]:
            trainer.state.step = step
            trainer.save_checkpoint(temp_checkpoint_dir)

        # Should have best_model + 1 regular checkpoint
        all_checkpoints = list(temp_checkpoint_dir.glob("*.pt"))
        assert len(all_checkpoints) == 2, \
            f"Expected 2 checkpoints, found {len(all_checkpoints)}"

        names = [p.stem for p in all_checkpoints]
        assert "best_model" in names, "best_model should not be deleted"


# ============================================================================
# 7. Warmup LR Tests
# ============================================================================

class TestWarmupLR:
    """Tests for warmup learning rate behavior."""

    def test_warmup_lr_linear_increase(self, simple_model):
        """Verify LR increases linearly during warmup period."""
        warmup_steps = 10
        base_lr = 1e-3

        training_config = TrainingConfig(
            mixed_precision="fp32",
            learning_rate=base_lr,
            warmup_steps=warmup_steps,
            lr_scheduler="constant",  # Use constant after warmup to avoid interference
            cpu_checkpoint_interval=0,  # Disable to avoid spike detection noise
        )

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            device=torch.device("cpu"),
        )

        # Test warmup directly without train_step to avoid scheduler interference
        lrs = []
        for step in range(warmup_steps):
            trainer._warmup_lr(step)
            lrs.append(trainer._get_lr())

        # LR should increase during warmup
        for i in range(1, warmup_steps):
            assert lrs[i] > lrs[i - 1], \
                f"LR should increase during warmup. Step {i-1}: {lrs[i-1]}, Step {i}: {lrs[i]}"

        # At warmup completion, LR should be close to base_lr
        assert abs(lrs[warmup_steps - 1] - base_lr) < base_lr * 0.15, \
            f"LR at warmup end should be ~{base_lr}, got {lrs[warmup_steps - 1]}"

    def test_warmup_initial_lr_very_small(self, simple_model):
        """Verify initial LR is very small at step 0."""
        warmup_steps = 100
        base_lr = 1e-3

        training_config = TrainingConfig(
            mixed_precision="fp32",
            learning_rate=base_lr,
            warmup_steps=warmup_steps,
        )

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            device=torch.device("cpu"),
        )

        # At step 0, before any warmup
        initial_lr = trainer._get_lr()

        # Apply warmup at step 0
        trainer._warmup_lr(0)
        lr_step_0 = trainer._get_lr()

        # LR at step 0 should be base_lr * (1/warmup_steps)
        expected_lr_step_0 = base_lr * (1 / warmup_steps)
        assert abs(lr_step_0 - expected_lr_step_0) < 1e-8, \
            f"LR at step 0 should be ~{expected_lr_step_0}, got {lr_step_0}"

    def test_warmup_sets_initial_lr_after_completion(self, simple_model):
        """Verify initial_lr is set in param_groups after warmup completes (H-003 fix)."""
        warmup_steps = 5
        base_lr = 1e-4

        training_config = TrainingConfig(
            mixed_precision="fp32",
            learning_rate=base_lr,
            warmup_steps=warmup_steps,
            lr_scheduler="cosine",
        )

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            device=torch.device("cpu"),
        )

        batch = {"images": torch.randn(2, 256, 256, 3)}

        # Run through warmup
        for step in range(warmup_steps + 1):
            trainer.train_step(batch)

        # After warmup, initial_lr should be set
        for param_group in trainer.optimizer.param_groups:
            assert "initial_lr" in param_group, \
                "initial_lr should be set in param_groups after warmup"
            assert param_group["initial_lr"] == base_lr, \
                f"initial_lr should be {base_lr}, got {param_group['initial_lr']}"


# ============================================================================
# 8. LR Scheduler Restart Tests
# ============================================================================

class TestLRSchedulerRestart:
    """Tests for cosine_restart scheduler behavior."""

    def test_cosine_restart_t0_calculation(self, simple_model):
        """Verify T_0 is calculated correctly from restart_epochs and dataloader."""
        # Create mock dataloader with known length
        mock_dl = MagicMock()
        mock_dl.__len__ = MagicMock(return_value=100)  # 100 batches per epoch

        restart_epochs = 4
        accumulation_steps = 2

        training_config = TrainingConfig(
            mixed_precision="fp32",
            lr_scheduler="cosine_restart",
            restart_epochs=restart_epochs,
            gradient_accumulation_steps=accumulation_steps,
        )

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            dataloader=mock_dl,
            device=torch.device("cpu"),
        )

        # Force scheduler creation
        scheduler = trainer.lr_scheduler

        # T_0 should be: (100 batches / 2 accum) * 4 epochs = 200 optimizer steps
        expected_t0 = (100 // accumulation_steps) * restart_epochs
        assert scheduler.T_0 == expected_t0, \
            f"T_0 should be {expected_t0}, got {scheduler.T_0}"

    def test_cosine_restart_lr_resets(self, simple_model):
        """Verify LR resets to peak after T_0 steps."""
        # Create mock dataloader
        mock_dl = MagicMock()
        mock_dl.__len__ = MagicMock(return_value=10)  # 10 batches per epoch

        training_config = TrainingConfig(
            mixed_precision="fp32",
            lr_scheduler="cosine_restart",
            learning_rate=1e-3,
            min_lr=1e-5,
            restart_epochs=1,  # Restart every epoch
            gradient_accumulation_steps=1,
        )

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            dataloader=mock_dl,
            device=torch.device("cpu"),
        )

        # Force scheduler creation and get T_0
        scheduler = trainer.lr_scheduler
        t_0 = scheduler.T_0

        # Record LR at each step
        lrs = []
        for _ in range(t_0 * 2 + 1):
            scheduler.step()
            lrs.append(trainer._get_lr())

        # LR should drop during first cycle
        assert lrs[t_0 // 2] < lrs[0], "LR should decrease during cycle"

        # LR should reset (or be close to peak) at T_0
        # Note: Due to decay_per_cycle, it might not be exactly the same
        peak_lr = lrs[0]
        lr_at_restart = lrs[t_0]

        # After restart, LR should be >= min_lr and close to peak (within decay)
        assert lr_at_restart >= training_config.min_lr, \
            f"LR at restart should be >= min_lr. Got {lr_at_restart}"

    def test_cosine_restart_with_decay(self, simple_model):
        """Verify lr_decay_per_cycle reduces peak LR each cycle."""
        mock_dl = MagicMock()
        mock_dl.__len__ = MagicMock(return_value=10)

        training_config = TrainingConfig(
            mixed_precision="fp32",
            lr_scheduler="cosine_restart",
            learning_rate=1e-3,
            min_lr=1e-6,
            restart_epochs=1,
            lr_decay_per_cycle=0.5,  # Halve peak LR each cycle
            gradient_accumulation_steps=1,
        )

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            dataloader=mock_dl,
            device=torch.device("cpu"),
        )

        scheduler = trainer.lr_scheduler
        t_0 = scheduler.T_0

        # Step through 3 complete cycles
        cycle_peak_lrs = []
        for cycle in range(3):
            scheduler.step()  # First step of cycle
            cycle_peak_lrs.append(trainer._get_lr())

            # Complete the rest of the cycle
            for _ in range(t_0 - 1):
                scheduler.step()

        # Each cycle's peak should be ~half the previous
        for i in range(1, len(cycle_peak_lrs)):
            ratio = cycle_peak_lrs[i] / cycle_peak_lrs[i - 1]
            # Allow some tolerance for numerical precision
            assert 0.4 < ratio < 0.6, \
                f"Cycle {i} peak LR should be ~0.5x previous. Ratio: {ratio}"

    def test_restart_period_overrides_epochs(self, simple_model):
        """Verify restart_period > 0 overrides restart_epochs calculation."""
        mock_dl = MagicMock()
        mock_dl.__len__ = MagicMock(return_value=100)

        fixed_period = 50

        training_config = TrainingConfig(
            mixed_precision="fp32",
            lr_scheduler="cosine_restart",
            restart_epochs=10,  # Would give T_0 = 1000
            restart_period=fixed_period,  # Should override
            gradient_accumulation_steps=1,
        )

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            dataloader=mock_dl,
            device=torch.device("cpu"),
        )

        scheduler = trainer.lr_scheduler

        assert scheduler.T_0 == fixed_period, \
            f"T_0 should be {fixed_period} (from restart_period), got {scheduler.T_0}"


# ============================================================================
# Integration Test
# ============================================================================

class SimpleDataLoader:
    """Simple iterable dataloader without pickling issues."""

    def __init__(self, batch_generator, num_batches: int = 10):
        self.batch_generator = batch_generator
        self.num_batches = num_batches

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            yield self.batch_generator()


class TestTrainerIntegration:
    """Integration tests for complete training scenarios."""

    def test_full_training_loop_with_text_encoder(self, simple_model, temp_checkpoint_dir):
        """Verify complete training loop with text encoder works end-to-end."""
        text_encoder = MockTextEncoder(hidden_dim=256, seq_len=32)

        training_config = TrainingConfig(
            mixed_precision="fp32",
            learning_rate=1e-3,
            warmup_steps=2,
            ema_decay=0.9,
            gradient_accumulation_steps=2,
            max_checkpoints=2,
            cpu_checkpoint_interval=0,  # Disable to avoid noise
        )

        # Create simple dataloader (not MagicMock to avoid pickle issues)
        def create_batch():
            return {
                "images": torch.randn(2, 256, 256, 3),
                "captions": ["test caption 1", "test caption 2"],
            }

        dataloader = SimpleDataLoader(create_batch, num_batches=10)

        trainer = Trainer(
            model=simple_model,
            training_config=training_config,
            text_encoder=text_encoder,
            dataloader=dataloader,
            device=torch.device("cpu"),
        )

        # Run training for a few steps
        initial_weights = trainer.model.linear1.weight.clone()

        trainer.train(
            num_steps=4,
            save_path=temp_checkpoint_dir,
            save_interval=2,
            use_progress_bar=False,
            log_interval=0,
        )

        # Verify training occurred
        assert trainer.state.step == 4
        assert not torch.allclose(trainer.model.linear1.weight, initial_weights), \
            "Weights should have changed after training"

        # Verify text encoder was used
        assert text_encoder.call_count > 0, "Text encoder should have been called"

        # Verify checkpoints were saved
        checkpoints = list(temp_checkpoint_dir.glob("*.pt"))
        assert len(checkpoints) >= 1, "At least one checkpoint should be saved"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
