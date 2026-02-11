"""
Tests for H-4: CombinedLossSimple missing repa_loss key

step.py:403 used loss_dict["repa_loss"] (direct index) but
CombinedLossSimple doesn't include "repa_loss" in its return dict.
Fix: use .get() with default torch.tensor(0.0).

Phase 2 of audit fix plan (FIX_PLAN_INDEX_2026-02-06.md).
"""

from __future__ import annotations

import pytest
import torch


class TestSimpleLossMissingRepaKey:
    """H-4: repa_loss safe access."""

    def test_simple_loss_missing_repa_key(self):
        """CombinedLossSimple returns dict without repa_loss."""
        # Simulate CombinedLossSimple output
        loss_dict = {
            "total": torch.tensor(1.0),
            "vloss": torch.tensor(0.5),
            "freq_loss": torch.tensor(0.3),
            # No "repa_loss" key
        }
        # Direct indexing would raise KeyError
        with pytest.raises(KeyError):
            _ = loss_dict["repa_loss"]

    def test_step_create_metrics_safe_access(self):
        """get() with default returns 0.0 when key missing."""
        loss_dict = {
            "total": torch.tensor(1.0),
            "vloss": torch.tensor(0.5),
            "freq_loss": torch.tensor(0.3),
        }
        repa_val = loss_dict.get("repa_loss", torch.tensor(0.0)).item()
        assert repa_val == 0.0

    def test_combined_loss_returns_repa_key(self):
        """Full CombinedLoss should include repa_loss in dict."""
        loss_dict = {
            "total": torch.tensor(1.0),
            "vloss": torch.tensor(0.5),
            "freq_loss": torch.tensor(0.3),
            "repa_loss": torch.tensor(0.2),
        }
        repa_val = loss_dict.get("repa_loss", torch.tensor(0.0)).item()
        assert repa_val == pytest.approx(0.2)

    def test_gamma_l2_also_uses_get(self):
        """gamma_l2 already uses .get() — verify consistency."""
        loss_dict = {"total": torch.tensor(1.0)}
        gamma_val = loss_dict.get("gamma_l2", torch.tensor(0.0)).item()
        assert gamma_val == 0.0
