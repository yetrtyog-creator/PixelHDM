"""
CPU Memory Checkpoint tests.

Tests for:
- Checkpoint initialization
- Checkpoint saving
- Checkpoint loading
- CPU offload mechanism
- Checkpoint statistics

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import warnings

from src.training.optimization.checkpoint import CPUMemoryCheckpoint, CheckpointStats
from src.training.optimization.ema import EMA


class SimpleModel(nn.Module):
    """Simple model for checkpoint testing."""

    def __init__(self, in_features: int = 10, out_features: int = 10):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestCheckpointInit:
    """Tests for CPUMemoryCheckpoint initialization."""

    def test_checkpoint_init_default(self):
        """Test checkpoint initializes with default parameters."""
        ckpt = CPUMemoryCheckpoint(save_interval=100, spike_threshold=5.0)

        assert ckpt.save_interval == 100
        assert ckpt.spike_threshold == 5.0
        assert ckpt._cpu_model_state is None
        assert ckpt._cpu_optimizer_state is None
        assert ckpt._restore_count == 0

    def test_checkpoint_init_custom(self):
        """Test checkpoint with custom parameters."""
        ckpt = CPUMemoryCheckpoint(save_interval=50, spike_threshold=3.0)

        assert ckpt.save_interval == 50
        assert ckpt.spike_threshold == 3.0

    def test_checkpoint_init_invalid_save_interval(self):
        """Test checkpoint rejects invalid save_interval."""
        with pytest.raises(ValueError, match="save_interval"):
            CPUMemoryCheckpoint(save_interval=0)

        with pytest.raises(ValueError, match="save_interval"):
            CPUMemoryCheckpoint(save_interval=-10)

    def test_checkpoint_init_invalid_spike_threshold(self):
        """Test checkpoint rejects invalid spike_threshold."""
        with pytest.raises(ValueError, match="spike_threshold"):
            CPUMemoryCheckpoint(save_interval=100, spike_threshold=1.0)

        with pytest.raises(ValueError, match="spike_threshold"):
            CPUMemoryCheckpoint(save_interval=100, spike_threshold=0.5)


class TestCheckpointSave:
    """Tests for checkpoint saving."""

    def test_checkpoint_saves_at_interval(self):
        """Test checkpoint saves at correct intervals."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        ckpt = CPUMemoryCheckpoint(save_interval=10, spike_threshold=5.0)

        # Step 0 should save
        ckpt.step(model, optimizer, None, 1.0, global_step=0)
        assert ckpt._cpu_model_state is not None
        assert ckpt._last_save_step == 0

        # Step 5 should not save
        ckpt._cpu_model_state = None  # Clear to test
        ckpt.step(model, optimizer, None, 1.0, global_step=5)
        assert ckpt._cpu_model_state is None

        # Step 10 should save
        ckpt.step(model, optimizer, None, 1.0, global_step=10)
        assert ckpt._cpu_model_state is not None
        assert ckpt._last_save_step == 10

    def test_checkpoint_saves_model_state(self):
        """Test checkpoint correctly saves model state."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        ckpt = CPUMemoryCheckpoint(save_interval=1, spike_threshold=5.0)

        # Set known values
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(42.0)

        ckpt.step(model, optimizer, None, 1.0, global_step=0)

        # Check saved state has correct values
        for name, saved_param in ckpt._cpu_model_state.items():
            # All values should be 42.0
            assert torch.allclose(saved_param, torch.full_like(saved_param, 42.0))

    def test_checkpoint_saves_optimizer_state(self):
        """Test checkpoint correctly saves optimizer state."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        ckpt = CPUMemoryCheckpoint(save_interval=1, spike_threshold=5.0)

        # Do a training step to create optimizer state
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        ckpt.step(model, optimizer, None, 1.0, global_step=0)

        assert ckpt._cpu_optimizer_state is not None
        assert "state" in ckpt._cpu_optimizer_state
        assert "param_groups" in ckpt._cpu_optimizer_state

    def test_checkpoint_saves_ema_state(self):
        """Test checkpoint correctly saves EMA state."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        ema = EMA(model, decay=0.999)
        ckpt = CPUMemoryCheckpoint(save_interval=1, spike_threshold=5.0)

        ckpt.step(model, optimizer, ema, 1.0, global_step=0)

        assert ckpt._cpu_ema_state is not None
        # Check EMA shadow params are saved
        for name in ema.shadow:
            assert name in ckpt._cpu_ema_state


class TestCheckpointLoad:
    """Tests for checkpoint loading."""

    def test_checkpoint_restores_on_spike(self):
        """Test checkpoint restores model on loss spike."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        # Use save_interval=10 so only step 0 saves
        ckpt = CPUMemoryCheckpoint(save_interval=10, spike_threshold=2.0)

        # Save checkpoint with known values at step 0
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(1.0)

        # Step 0: saves checkpoint (step 0 % 10 == 0), sets _last_loss = 1.0
        ckpt.step(model, optimizer, None, 1.0, global_step=0)

        # Step 1: does NOT save (step 1 % 10 != 0), sets _last_loss = 1.5
        ckpt.step(model, optimizer, None, 1.5, global_step=1)

        # Modify model after checkpoint was saved at step 0
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(999.0)

        # Step 2: spike detected (loss=10.0 > 1.5 * 2.0 = 3.0)
        # This should restore from checkpoint saved at step 0 (with value 1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore the spike warning
            restored = ckpt.step(model, optimizer, None, 10.0, global_step=2)

        assert restored is True

        # Model should be restored to 1.0 (from step 0 checkpoint)
        for param in model.parameters():
            assert torch.allclose(param.data, torch.full_like(param.data, 1.0))

    def test_checkpoint_no_restore_without_saved(self):
        """Test no restore happens if no checkpoint saved."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        ckpt = CPUMemoryCheckpoint(save_interval=100, spike_threshold=5.0)

        # First step at non-save interval, then spike
        # No checkpoint saved yet, so restore should fail gracefully
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Step 1 doesn't save (interval=100), loss=1.0
            ckpt.step(model, optimizer, None, 1.0, global_step=1)
            # Step 2 with spike, but no checkpoint
            restored = ckpt.step(model, optimizer, None, 10.0, global_step=2)

        assert restored is False

    def test_checkpoint_restores_ema(self):
        """Test checkpoint restores EMA state."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        ema = EMA(model, decay=0.999)
        # Use save_interval=10 so only step 0 saves
        ckpt = CPUMemoryCheckpoint(save_interval=10, spike_threshold=2.0)

        # Save state at step 0
        original_shadow = {k: v.clone() for k, v in ema.shadow.items()}
        ckpt.step(model, optimizer, ema, 1.0, global_step=0)

        # Step 1: does NOT save (step 1 % 10 != 0), sets _last_loss = 1.5
        ckpt.step(model, optimizer, ema, 1.5, global_step=1)

        # Modify EMA shadow after checkpoint was saved at step 0
        with torch.no_grad():
            for name in ema.shadow:
                ema.shadow[name].fill_(999.0)

        # Step 2: spike detected (10.0 > 1.5 * 2.0 = 3.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            restored = ckpt.step(model, optimizer, ema, 10.0, global_step=2)

        assert restored is True

        # EMA should be restored to original values (from step 0 checkpoint)
        for name, shadow in ema.shadow.items():
            assert torch.allclose(shadow, original_shadow[name])


class TestCheckpointCPUOffload:
    """Tests for CPU offload mechanism."""

    def test_checkpoint_state_on_cpu(self):
        """Test saved state is on CPU."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        ckpt = CPUMemoryCheckpoint(save_interval=1, spike_threshold=5.0)

        ckpt.step(model, optimizer, None, 1.0, global_step=0)

        # All saved tensors should be on CPU
        for name, tensor in ckpt._cpu_model_state.items():
            assert tensor.device == torch.device("cpu")

    def test_checkpoint_clear(self):
        """Test clear() removes all saved state."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        ema = EMA(model, decay=0.999)
        ckpt = CPUMemoryCheckpoint(save_interval=1, spike_threshold=5.0)

        # Save state
        ckpt.step(model, optimizer, ema, 1.0, global_step=0)

        assert ckpt._cpu_model_state is not None
        assert ckpt._cpu_optimizer_state is not None
        assert ckpt._cpu_ema_state is not None

        # Clear
        ckpt.clear()

        assert ckpt._cpu_model_state is None
        assert ckpt._cpu_optimizer_state is None
        assert ckpt._cpu_ema_state is None

    def test_checkpoint_invalid_loss_rejected(self):
        """Test invalid loss values are rejected."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        ckpt = CPUMemoryCheckpoint(save_interval=1, spike_threshold=5.0)

        with pytest.raises(ValueError, match="無效的 loss"):
            ckpt.step(model, optimizer, None, float('nan'), global_step=0)

        with pytest.raises(ValueError, match="無效的 loss"):
            ckpt.step(model, optimizer, None, float('inf'), global_step=0)

        with pytest.raises(ValueError, match="無效的 loss"):
            ckpt.step(model, optimizer, None, float('-inf'), global_step=0)


class TestCheckpointStats:
    """Tests for checkpoint statistics."""

    def test_checkpoint_get_stats(self):
        """Test get_stats returns correct information."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        ckpt = CPUMemoryCheckpoint(save_interval=10, spike_threshold=5.0)

        # Initial stats
        stats = ckpt.get_stats()
        assert stats.restore_count == 0
        assert stats.last_save_step == 0
        assert stats.has_checkpoint is False

        # After save
        ckpt.step(model, optimizer, None, 1.0, global_step=0)
        stats = ckpt.get_stats()
        assert stats.has_checkpoint is True
        assert stats.last_save_step == 0

    def test_checkpoint_restore_count_increments(self):
        """Test restore count increments on each restore."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        ckpt = CPUMemoryCheckpoint(save_interval=1, spike_threshold=2.0)

        # Save
        ckpt.step(model, optimizer, None, 1.0, global_step=0)

        # Trigger multiple restores
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ckpt.step(model, optimizer, None, 3.0, global_step=1)  # Spike
            ckpt.step(model, optimizer, None, 1.0, global_step=2)  # Normal, re-save
            ckpt.step(model, optimizer, None, 3.0, global_step=3)  # Spike again

        stats = ckpt.get_stats()
        assert stats.restore_count == 2

    def test_checkpoint_state_dict_round_trip(self):
        """Test state_dict save and load."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        ckpt = CPUMemoryCheckpoint(save_interval=10, spike_threshold=5.0)

        # Build up state
        ckpt.step(model, optimizer, None, 1.0, global_step=0)
        ckpt.step(model, optimizer, None, 1.5, global_step=10)

        # Save state
        state = ckpt.state_dict()

        # Create new checkpoint and load
        ckpt2 = CPUMemoryCheckpoint(save_interval=1, spike_threshold=2.0)
        ckpt2.load_state_dict(state)

        # Values should match
        assert ckpt2.save_interval == 10
        assert ckpt2.spike_threshold == 5.0
        assert ckpt2._last_save_step == 10
        assert ckpt2._last_loss == 1.5

    def test_checkpointstats_dataclass(self):
        """Test CheckpointStats dataclass fields."""
        stats = CheckpointStats(
            restore_count=5,
            last_save_step=100,
            has_checkpoint=True,
        )

        assert stats.restore_count == 5
        assert stats.last_save_step == 100
        assert stats.has_checkpoint is True
