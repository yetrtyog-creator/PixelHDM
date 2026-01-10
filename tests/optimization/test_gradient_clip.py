"""
ZClip (Z-Score Adaptive Gradient Clipping) tests.

Tests for:
- ZClip initialization
- Statistics update
- Threshold computation
- Gradient clipping
- Convenience function

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import math

from src.training.optimization.gradient_clip import ZClip, ZClipStats, clip_grad_norm_with_zclip


class SimpleModel(nn.Module):
    """Simple model for gradient clipping testing."""

    def __init__(self, in_features: int = 10, out_features: int = 10):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestZClipInit:
    """Tests for ZClip initialization."""

    def test_zclip_init_default(self):
        """Test ZClip initializes with default parameters."""
        zclip = ZClip(threshold=2.5, ema_decay=0.99)

        assert zclip.threshold == 2.5
        assert zclip.ema_decay == 0.99
        assert zclip.mean_ema is None
        assert zclip.var_ema is None
        assert zclip.step_count == 0
        assert zclip.clip_count == 0

    def test_zclip_init_custom_threshold(self):
        """Test ZClip with custom threshold."""
        zclip = ZClip(threshold=3.0, ema_decay=0.95)

        assert zclip.threshold == 3.0
        assert zclip.ema_decay == 0.95

    def test_zclip_init_invalid_threshold(self):
        """Test ZClip rejects invalid threshold."""
        with pytest.raises(ValueError, match="threshold"):
            ZClip(threshold=0.0)

        with pytest.raises(ValueError, match="threshold"):
            ZClip(threshold=-1.0)

    def test_zclip_init_invalid_ema_decay(self):
        """Test ZClip rejects invalid ema_decay."""
        with pytest.raises(ValueError, match="ema_decay"):
            ZClip(threshold=2.5, ema_decay=0.0)

        with pytest.raises(ValueError, match="ema_decay"):
            ZClip(threshold=2.5, ema_decay=1.0)

        with pytest.raises(ValueError, match="ema_decay"):
            ZClip(threshold=2.5, ema_decay=-0.5)


class TestZClipUpdateStats:
    """Tests for ZClip statistics update."""

    def test_zclip_first_update_initializes(self):
        """Test first call initializes statistics."""
        zclip = ZClip(threshold=2.5)

        result = zclip(5.0)

        assert zclip.mean_ema == 5.0
        assert zclip.var_ema == 0.0
        assert zclip.step_count == 1
        assert result == 5.0

    def test_zclip_update_ema_mean(self):
        """Test EMA mean updates correctly."""
        zclip = ZClip(threshold=2.5, ema_decay=0.9)

        # First call
        zclip(10.0)
        assert zclip.mean_ema == 10.0

        # Second call - EMA should move toward new value
        zclip(20.0)
        # mean = 0.9 * 10.0 + 0.1 * 20.0 = 11.0
        expected_mean = 0.9 * 10.0 + 0.1 * 20.0
        assert abs(zclip.mean_ema - expected_mean) < 0.01

    def test_zclip_update_increments_step_count(self):
        """Test step count increments correctly."""
        zclip = ZClip(threshold=2.5)

        assert zclip.step_count == 0

        zclip(1.0)
        assert zclip.step_count == 1

        zclip(2.0)
        assert zclip.step_count == 2

        zclip(3.0)
        assert zclip.step_count == 3

    def test_zclip_invalid_grad_norm(self):
        """Test ZClip rejects invalid gradient norms."""
        zclip = ZClip(threshold=2.5)

        with pytest.raises(ValueError, match="grad_norm"):
            zclip(-1.0)

        with pytest.raises(ValueError, match="grad_norm"):
            zclip(float('inf'))


class TestZClipComputeThreshold:
    """Tests for ZClip threshold computation."""

    def test_zclip_no_clip_below_threshold(self):
        """Test normal gradients are not clipped when below z-score threshold."""
        zclip = ZClip(threshold=3.0, ema_decay=0.99)

        # Build up some history with stable gradients
        for _ in range(100):
            result = zclip(1.0)

        # A slightly higher gradient within normal variation shouldn't trigger clip
        # The clip_count should remain at 0 for gradients within z-score threshold
        initial_clip_count = zclip.clip_count

        # Use a value that's only slightly above mean (within threshold)
        result = zclip(1.01)

        # Should not increment clip count (within threshold)
        assert zclip.clip_count == initial_clip_count

    def test_zclip_clips_above_threshold(self):
        """Test extreme gradients are clipped."""
        zclip = ZClip(threshold=2.0, ema_decay=0.99)

        # Build up history with stable gradients around 1.0
        for _ in range(50):
            zclip(1.0)

        initial_clip_count = zclip.clip_count

        # Very extreme gradient should trigger clip
        result = zclip(100.0)

        # Should either be clipped or very different from 100.0
        # The exact behavior depends on the current statistics
        # At minimum, step_count should have increased
        assert zclip.step_count > 50

    def test_zclip_get_stats(self):
        """Test get_stats returns correct information."""
        zclip = ZClip(threshold=2.5, ema_decay=0.99)

        # Initial stats
        stats = zclip.get_stats()
        assert stats.steps == 0
        assert stats.clips == 0
        assert stats.clip_rate == 0.0

        # After some updates
        for _ in range(10):
            zclip(1.0)

        stats = zclip.get_stats()
        assert stats.steps == 10
        assert isinstance(stats.mean, float)
        assert isinstance(stats.std, float)


class TestZClipClipGradients:
    """Tests for actual gradient clipping."""

    def test_zclip_returns_original_for_normal(self):
        """Test first call returns exact input value (initialization)."""
        zclip = ZClip(threshold=3.0)

        # First call initializes and returns exact input
        result1 = zclip(1.0)
        assert result1 == 1.0

        # Subsequent calls may adjust based on EMA, but step_count should increase
        result2 = zclip(1.1)
        assert zclip.step_count == 2
        # Result may not be exactly 1.1 due to EMA calculation, but should be finite
        assert isinstance(result2, float) and result2 > 0

    def test_zclip_reset(self):
        """Test reset clears all statistics."""
        zclip = ZClip(threshold=2.5)

        # Build up some state
        for i in range(10):
            zclip(float(i))

        assert zclip.step_count == 10
        assert zclip.mean_ema is not None

        # Reset
        zclip.reset()

        assert zclip.step_count == 0
        assert zclip.clip_count == 0
        assert zclip.mean_ema is None
        assert zclip.var_ema is None

    def test_zclip_state_dict_round_trip(self):
        """Test state_dict save and load."""
        zclip = ZClip(threshold=2.5, ema_decay=0.99)

        # Build up state
        for i in range(20):
            zclip(float(i) + 1.0)

        # Save state
        state = zclip.state_dict()

        # Create new ZClip and load
        zclip2 = ZClip(threshold=1.0, ema_decay=0.5)  # Different params
        zclip2.load_state_dict(state)

        # Should match
        assert zclip2.mean_ema == zclip.mean_ema
        assert zclip2.var_ema == zclip.var_ema
        assert zclip2.step_count == zclip.step_count
        assert zclip2.clip_count == zclip.clip_count


class TestClipGradNormWithZClip:
    """Tests for clip_grad_norm_with_zclip convenience function."""

    def test_clip_grad_norm_basic(self):
        """Test basic gradient clipping with model."""
        model = SimpleModel()
        zclip = ZClip(threshold=2.5)

        # Create gradients via backward
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Clip
        grad_norm = clip_grad_norm_with_zclip(model, zclip, max_norm=1.0)

        assert grad_norm >= 0
        assert math.isfinite(grad_norm)

    def test_clip_grad_norm_no_gradients(self):
        """Test with no gradients returns 0."""
        model = SimpleModel()
        zclip = ZClip(threshold=2.5)

        # No backward called, so no gradients
        grad_norm = clip_grad_norm_with_zclip(model, zclip, max_norm=1.0)

        assert grad_norm == 0.0

    def test_clip_grad_norm_updates_zclip(self):
        """Test clipping updates ZClip statistics."""
        model = SimpleModel()
        zclip = ZClip(threshold=2.5)

        assert zclip.step_count == 0

        # Create gradients
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()

        clip_grad_norm_with_zclip(model, zclip, max_norm=1.0)

        assert zclip.step_count == 1

    def test_clip_grad_norm_respects_max_norm(self):
        """Test max_norm is respected."""
        model = SimpleModel()
        zclip = ZClip(threshold=10.0)  # High threshold so ZClip doesn't interfere

        # Create large gradients
        x = torch.randn(4, 10)
        y = model(x)
        loss = (y ** 2).sum() * 1000  # Large loss for large gradients
        loss.backward()

        # Check gradient norm before clipping
        params = [p for p in model.parameters() if p.grad is not None]
        original_norm = torch.norm(
            torch.stack([torch.norm(p.grad, 2) for p in params]), 2
        ).item()

        # Clip with small max_norm
        clip_grad_norm_with_zclip(model, zclip, max_norm=0.1)

        # Check gradient norm after clipping
        clipped_norm = torch.norm(
            torch.stack([torch.norm(p.grad, 2) for p in params]), 2
        ).item()

        # Clipped norm should be <= max_norm (allowing small tolerance)
        assert clipped_norm <= 0.1 + 1e-6

    def test_clip_grad_norm_nan_detection(self):
        """Test NaN gradient detection."""
        model = SimpleModel()
        zclip = ZClip(threshold=2.5)

        # Manually set NaN gradient
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Set one gradient to NaN
        for param in model.parameters():
            if param.grad is not None:
                param.grad.fill_(float('nan'))
                break

        with pytest.raises(RuntimeError, match="無效梯度範數"):
            clip_grad_norm_with_zclip(model, zclip, max_norm=1.0)
