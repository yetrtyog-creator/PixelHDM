"""
Tests for M-1: Freq loss operates in image space (not velocity space)

After fix, combined_loss passes x_pred and x_clean to freq_loss,
not v_pred_out and v_target.

Phase 3 of audit fix plan (FIX_PLAN_INDEX_2026-02-06.md).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
import torch


class TestFreqLossSpace:
    """M-1: Freq loss should operate in image space."""

    def test_combined_loss_freq_receives_image(self):
        """After fix, freq_loss receives x_pred=v_pred+noise, x_clean."""
        from src.training.losses.combined_loss import CombinedLoss

        B, C, H, W = 2, 3, 32, 32
        noise = torch.randn(B, C, H, W)
        x_clean = torch.randn(B, C, H, W)
        v_target = x_clean - noise
        v_pred = v_target + torch.randn_like(v_target) * 0.01

        # Build CombinedLoss with freq enabled
        loss_fn = CombinedLoss(lambda_freq=1.0, freq_quality=90)

        # Spy on freq_loss forward
        original_forward = loss_fn.freq_loss.forward
        captured_args = {}

        def spy_forward(pred, target):
            captured_args["pred"] = pred
            captured_args["target"] = target
            return original_forward(pred, target)

        loss_fn.freq_loss.forward = spy_forward

        _ = loss_fn(v_pred, x_clean, noise)

        # Verify freq_loss received image-space tensors
        assert "pred" in captured_args
        assert "target" in captured_args
        # target should be x_clean, not v_target
        assert torch.allclose(captured_args["target"], x_clean, atol=1e-5), \
            "freq_loss target should be x_clean (image space)"

    def test_freq_loss_param_names_updated(self):
        """FrequencyLoss.forward() should use 'pred'/'target' param names."""
        import inspect
        from src.training.losses.freq.core import FrequencyLoss
        sig = inspect.signature(FrequencyLoss.forward)
        param_names = list(sig.parameters.keys())
        assert "pred" in param_names, f"Expected 'pred' in params, got {param_names}"
        assert "target" in param_names, f"Expected 'target' in params, got {param_names}"
        assert "v_pred" not in param_names, "Old param name 'v_pred' should be removed"
        assert "v_target" not in param_names, "Old param name 'v_target' should be removed"
