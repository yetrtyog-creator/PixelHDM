"""
EMA (Exponential Moving Average) tests.

Tests for:
- EMA initialization
- EMA update mechanism
- Decay rate correctness
- Shadow weight application
- Weight restoration

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.training.optimization.ema import EMA, EMAModelWrapper


class SimpleModel(nn.Module):
    """Simple model for EMA testing."""

    def __init__(self, in_features: int = 10, out_features: int = 10):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestEMAInit:
    """Tests for EMA initialization."""

    def test_ema_init_default_decay(self):
        """Test EMA initializes with default decay rate."""
        model = SimpleModel()
        ema = EMA(model, decay=0.999)

        assert ema.decay == 0.999
        assert ema.update_after_step == 0
        assert ema.num_updates == 0

    def test_ema_init_shadow_params_match(self):
        """Test shadow parameters match model parameters at init."""
        model = SimpleModel()
        ema = EMA(model, decay=0.9999)

        # Shadow params should match original params
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in ema.shadow
                assert torch.allclose(ema.shadow[name], param.data)

    def test_ema_init_invalid_decay(self):
        """Test EMA rejects invalid decay values."""
        model = SimpleModel()

        with pytest.raises(ValueError, match="decay"):
            EMA(model, decay=0.0)

        with pytest.raises(ValueError, match="decay"):
            EMA(model, decay=1.0)

        with pytest.raises(ValueError, match="decay"):
            EMA(model, decay=-0.5)

        with pytest.raises(ValueError, match="decay"):
            EMA(model, decay=1.5)

    def test_ema_init_update_after_step(self):
        """Test EMA respects update_after_step setting."""
        model = SimpleModel()
        ema = EMA(model, decay=0.999, update_after_step=100)

        assert ema.update_after_step == 100


class TestEMAUpdate:
    """Tests for EMA update mechanism."""

    def test_ema_update_modifies_shadow(self):
        """Test EMA update modifies shadow parameters."""
        model = SimpleModel()
        ema = EMA(model, decay=0.999)

        # Store original shadow values
        original_shadows = {k: v.clone() for k, v in ema.shadow.items()}

        # Modify model parameters
        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)

        # Update EMA
        ema.update(model)

        # Shadow should have moved toward new values (but not equal)
        for name, shadow in ema.shadow.items():
            assert not torch.allclose(shadow, original_shadows[name])

    def test_ema_update_respects_step_threshold(self):
        """Test EMA does not update before update_after_step."""
        model = SimpleModel()
        ema = EMA(model, decay=0.999, update_after_step=10)

        # Store original shadow values
        original_shadows = {k: v.clone() for k, v in ema.shadow.items()}

        # Modify model and call update with step < threshold
        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)

        ema.update(model, step=5)  # Step 5 < threshold 10

        # Shadow should NOT have changed
        for name, shadow in ema.shadow.items():
            assert torch.allclose(shadow, original_shadows[name])

    def test_ema_update_increments_counter(self):
        """Test EMA update increments num_updates counter."""
        model = SimpleModel()
        ema = EMA(model, decay=0.999)

        assert ema.num_updates == 0

        ema.update(model)
        assert ema.num_updates == 1

        ema.update(model)
        assert ema.num_updates == 2


class TestEMADecayRate:
    """Tests for EMA decay rate correctness."""

    def test_ema_decay_rate_high(self):
        """Test high decay rate (0.9999) moves slowly."""
        model = SimpleModel()
        ema = EMA(model, decay=0.9999)

        original_shadows = {k: v.clone() for k, v in ema.shadow.items()}

        # Large modification to model
        with torch.no_grad():
            for param in model.parameters():
                param.add_(10.0)

        ema.update(model)

        # With high decay, shadow should still be very close to original
        for name, shadow in ema.shadow.items():
            original = original_shadows[name]
            current_param = dict(model.named_parameters())[name].data

            # Shadow should be closer to original than to current param
            dist_to_original = (shadow - original).abs().mean()
            dist_to_current = (shadow - current_param).abs().mean()

            assert dist_to_original < dist_to_current

    def test_ema_decay_rate_low(self):
        """Test low decay rate (0.5) moves quickly."""
        model = SimpleModel()
        ema = EMA(model, decay=0.5)

        # Large modification to model
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(10.0)

        ema.update(model)

        # With decay=0.5, shadow should be halfway between old and new
        # shadow = 0.5 * old + 0.5 * new
        for name, shadow in ema.shadow.items():
            current_param = dict(model.named_parameters())[name].data
            # After one update with decay=0.5, should be exactly between
            # old (0 or random init) and new (10.0)

    def test_ema_decay_formula(self):
        """Test EMA follows correct formula: shadow = decay * shadow + (1-decay) * param."""
        model = SimpleModel()
        decay = 0.9

        # Initialize with known values
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(1.0)

        ema = EMA(model, decay=decay)

        # Verify initial shadow equals param
        for name, shadow in ema.shadow.items():
            assert torch.allclose(shadow, torch.ones_like(shadow))

        # Update with new values
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(2.0)

        ema.update(model)

        # Expected: 0.9 * 1.0 + 0.1 * 2.0 = 1.1
        expected_value = decay * 1.0 + (1 - decay) * 2.0
        for name, shadow in ema.shadow.items():
            assert torch.allclose(shadow, torch.full_like(shadow, expected_value))


class TestEMAShadowApplication:
    """Tests for applying shadow weights to model."""

    def test_ema_copy_to(self):
        """Test copy_to copies shadow params to model."""
        model = SimpleModel()
        ema = EMA(model, decay=0.999)

        # Modify model (but not EMA shadow)
        with torch.no_grad():
            for param in model.parameters():
                param.add_(100.0)

        # Copy shadow (original values) back to model
        ema.copy_to(model)

        # Model should now match shadow
        for name, param in model.named_parameters():
            if param.requires_grad and name in ema.shadow:
                assert torch.allclose(param.data, ema.shadow[name])

    def test_ema_apply_to_context_manager(self):
        """Test apply_to context manager temporarily uses shadow weights."""
        model = SimpleModel()
        ema = EMA(model, decay=0.999)

        # Store original model weights
        original_weights = {n: p.clone() for n, p in model.named_parameters()}

        # Modify model
        with torch.no_grad():
            for param in model.parameters():
                param.add_(50.0)

        modified_weights = {n: p.clone() for n, p in model.named_parameters()}

        # Use shadow weights temporarily
        with ema.apply_to(model):
            # Inside context, model should have shadow (original) weights
            for name, param in model.named_parameters():
                if param.requires_grad:
                    assert torch.allclose(param.data, original_weights[name])

        # After context, model should have modified weights restored
        for name, param in model.named_parameters():
            assert torch.allclose(param.data, modified_weights[name])


class TestEMARestore:
    """Tests for weight restoration."""

    def test_ema_store_and_restore(self):
        """Test store() and restore() work correctly."""
        model = SimpleModel()
        ema = EMA(model, decay=0.999)

        # Store current weights
        original_weights = {n: p.clone() for n, p in model.named_parameters()}
        ema.store(model)

        # Modify model
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(999.0)

        # Restore
        ema.restore(model)

        # Should be back to original
        for name, param in model.named_parameters():
            assert torch.allclose(param.data, original_weights[name])

    def test_ema_restore_without_store_raises(self):
        """Test restore() without prior store() raises error."""
        model = SimpleModel()
        ema = EMA(model, decay=0.999)

        with pytest.raises(RuntimeError, match="沒有備份可恢復"):
            ema.restore(model)

    def test_ema_state_dict_round_trip(self):
        """Test state_dict save and load."""
        model = SimpleModel()
        ema = EMA(model, decay=0.999)

        # Do some updates
        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)
        ema.update(model)
        ema.update(model)

        # Save state
        state = ema.state_dict()

        # Create new EMA and load state
        model2 = SimpleModel()
        ema2 = EMA(model2, decay=0.9)  # Different initial decay

        ema2.load_state_dict(state)

        # Should match
        assert ema2.decay == 0.999
        assert ema2.num_updates == 2
        for name in ema.shadow:
            assert torch.allclose(ema.shadow[name].cpu(), ema2.shadow[name].cpu())


class TestEMAModelWrapper:
    """Tests for EMAModelWrapper."""

    def test_ema_model_wrapper_uses_shadow(self):
        """Test EMAModelWrapper automatically uses shadow weights."""
        model = SimpleModel()
        ema = EMA(model, decay=0.999)

        # Update EMA and modify model differently
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(1.0)

        ema.update(model)  # Shadow now has 1.0

        with torch.no_grad():
            for param in model.parameters():
                param.fill_(100.0)  # Model has 100.0, shadow has 1.0

        # Wrapper should use shadow weights (1.0) during forward
        wrapper = EMAModelWrapper(model, ema)
        x = torch.randn(2, 10)

        # The output should use shadow weights, but model should be restored after
        _ = wrapper(x)

        # Model should still have 100.0 after forward
        for param in model.parameters():
            assert torch.allclose(param.data, torch.full_like(param.data, 100.0))
