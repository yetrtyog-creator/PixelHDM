"""
Tests for Classifier-Free Guidance (CFG) implementations.

This module tests:
- StandardCFG formula and application
- RescaledCFG with variance correction
- CFGWithInterval time-based control
- CFGScheduler for dynamic CFG weights
- Utility functions for CFG computation

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn
from typing import Optional

from src.inference.cfg import (
    BaseCFG,
    StandardCFG,
    RescaledCFG,
    CFGWithInterval,
    PerplexityCFG,
    CFGScheduler,
    CFGScheduleType,
    CFGConfig,
    CFGWrapper,
    apply_cfg,
    compute_guidance_scale_schedule,
    create_cfg,
    create_cfg_scheduler,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def batch_tensors():
    """Create paired conditional and unconditional tensors for testing."""
    torch.manual_seed(42)
    x_cond = torch.randn(2, 3, 64, 64)
    x_uncond = torch.randn(2, 3, 64, 64)
    return x_cond, x_uncond


@pytest.fixture
def standard_cfg():
    """Create StandardCFG instance."""
    return StandardCFG()


@pytest.fixture
def rescaled_cfg():
    """Create RescaledCFG instance with default rescale factor."""
    return RescaledCFG(rescale_factor=0.7)


@pytest.fixture
def interval_cfg():
    """Create CFGWithInterval instance."""
    return CFGWithInterval(
        guidance_scale=7.5,
        rescale_factor=0.0,
        interval_start=0.0,
        interval_end=0.75,
    )


# ============================================================================
# Test: StandardCFG
# ============================================================================

class TestStandardCFG:
    """Tests for StandardCFG class."""

    def test_standard_cfg_apply(self, standard_cfg, batch_tensors):
        """Test basic CFG application."""
        x_cond, x_uncond = batch_tensors
        guidance_scale = 7.5

        result = standard_cfg.apply(x_cond, x_uncond, guidance_scale)

        assert result.shape == x_cond.shape
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_standard_cfg_formula(self, standard_cfg, batch_tensors):
        """Test that CFG follows the correct formula: out = uncond + scale * (cond - uncond)."""
        x_cond, x_uncond = batch_tensors
        guidance_scale = 7.5

        result = standard_cfg.apply(x_cond, x_uncond, guidance_scale)

        # Manual computation
        expected = x_uncond + guidance_scale * (x_cond - x_uncond)

        assert torch.allclose(result, expected, rtol=1e-5, atol=1e-6)

    def test_standard_cfg_scale_1_returns_cond(self, standard_cfg, batch_tensors):
        """Test that CFG with scale=1.0 returns conditional output."""
        x_cond, x_uncond = batch_tensors

        result = standard_cfg.apply(x_cond, x_uncond, guidance_scale=1.0)

        # With scale=1: out = uncond + 1*(cond - uncond) = cond
        assert torch.allclose(result, x_cond, rtol=1e-5, atol=1e-6)

    def test_standard_cfg_scale_0_returns_uncond(self, standard_cfg, batch_tensors):
        """Test that CFG with scale=0.0 returns unconditional output."""
        x_cond, x_uncond = batch_tensors

        result = standard_cfg.apply(x_cond, x_uncond, guidance_scale=0.0)

        # With scale=0: out = uncond + 0*(cond - uncond) = uncond
        assert torch.allclose(result, x_uncond, rtol=1e-5, atol=1e-6)

    def test_standard_cfg_different_scales(self, standard_cfg, batch_tensors):
        """Test CFG with various guidance scales."""
        x_cond, x_uncond = batch_tensors

        for scale in [1.0, 3.0, 7.5, 15.0]:
            result = standard_cfg.apply(x_cond, x_uncond, scale)
            expected = x_uncond + scale * (x_cond - x_uncond)
            assert torch.allclose(result, expected, rtol=1e-5, atol=1e-6)


# ============================================================================
# Test: RescaledCFG
# ============================================================================

class TestRescaledCFG:
    """Tests for RescaledCFG class."""

    def test_rescaled_cfg_apply(self, rescaled_cfg, batch_tensors):
        """Test rescaled CFG application."""
        x_cond, x_uncond = batch_tensors
        guidance_scale = 7.5

        result = rescaled_cfg.apply(x_cond, x_uncond, guidance_scale)

        assert result.shape == x_cond.shape
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_rescaled_cfg_formula(self, batch_tensors):
        """Test rescaled CFG formula correctness."""
        x_cond, x_uncond = batch_tensors
        rescale_factor = 0.7
        guidance_scale = 7.5

        cfg = RescaledCFG(rescale_factor=rescale_factor)
        result = cfg.apply(x_cond, x_uncond, guidance_scale)

        # Manual computation
        x_cfg = x_uncond + guidance_scale * (x_cond - x_uncond)

        # Compute standard deviations
        std_cond = x_cond.std(dim=list(range(1, x_cond.dim())), keepdim=True)
        std_cfg = x_cfg.std(dim=list(range(1, x_cfg.dim())), keepdim=True)

        # Rescale
        factor = std_cond / (std_cfg + 1e-8)
        rescaled = x_cfg * factor

        # Mix
        expected = rescale_factor * rescaled + (1 - rescale_factor) * x_cfg

        assert torch.allclose(result, expected, rtol=1e-4, atol=1e-5)

    def test_rescaled_cfg_zero_factor_equals_standard(self, batch_tensors):
        """Test that rescale_factor=0 produces standard CFG."""
        x_cond, x_uncond = batch_tensors
        guidance_scale = 7.5

        rescaled = RescaledCFG(rescale_factor=0.0)
        standard = StandardCFG()

        result_rescaled = rescaled.apply(x_cond, x_uncond, guidance_scale)
        result_standard = standard.apply(x_cond, x_uncond, guidance_scale)

        assert torch.allclose(result_rescaled, result_standard, rtol=1e-5, atol=1e-6)

    def test_rescaled_cfg_reduces_variance(self, batch_tensors):
        """Test that rescaling helps control output variance."""
        x_cond, x_uncond = batch_tensors
        guidance_scale = 15.0  # High guidance

        rescaled = RescaledCFG(rescale_factor=0.7)
        standard = StandardCFG()

        result_rescaled = rescaled.apply(x_cond, x_uncond, guidance_scale)
        result_standard = standard.apply(x_cond, x_uncond, guidance_scale)

        # Rescaled should have variance closer to original conditional
        var_cond = x_cond.var()
        var_rescaled = result_rescaled.var()
        var_standard = result_standard.var()

        # Rescaled variance should be closer to conditional variance
        assert abs(var_rescaled - var_cond) < abs(var_standard - var_cond)


# ============================================================================
# Test: CFGWithInterval
# ============================================================================

class TestCFGWithInterval:
    """Tests for CFGWithInterval class."""

    def test_cfg_with_interval_should_apply(self, interval_cfg):
        """Test time interval check."""
        # Within interval [0.0, 0.75)
        assert interval_cfg.should_apply_cfg(0.0) == True
        assert interval_cfg.should_apply_cfg(0.5) == True
        assert interval_cfg.should_apply_cfg(0.74) == True

        # Outside interval
        assert interval_cfg.should_apply_cfg(0.75) == False
        assert interval_cfg.should_apply_cfg(0.8) == False
        assert interval_cfg.should_apply_cfg(1.0) == False

    def test_cfg_with_interval_apply(self, interval_cfg, batch_tensors):
        """Test CFG application within interval."""
        x_cond, x_uncond = batch_tensors
        guidance_scale = 7.5

        result = interval_cfg.apply(x_cond, x_uncond, guidance_scale)

        # Should apply standard CFG formula
        expected = x_uncond + guidance_scale * (x_cond - x_uncond)
        assert torch.allclose(result, expected, rtol=1e-5, atol=1e-6)

    def test_cfg_with_interval_skip(self, interval_cfg, batch_tensors):
        """Test that CFG is skipped outside interval."""
        x_cond, x_uncond = batch_tensors

        # t=0.8 is outside [0.0, 0.75) interval
        result = interval_cfg.apply_with_interval_check(x_cond, x_uncond, t=0.8)

        # Should return conditional output (CFG skipped)
        assert torch.allclose(result, x_cond, rtol=1e-5, atol=1e-6)

    def test_cfg_with_interval_apply_inside(self, interval_cfg, batch_tensors):
        """Test apply_with_interval_check inside interval."""
        x_cond, x_uncond = batch_tensors

        # t=0.5 is inside [0.0, 0.75) interval
        result = interval_cfg.apply_with_interval_check(x_cond, x_uncond, t=0.5)

        # Should apply CFG
        expected = x_uncond + interval_cfg.guidance_scale * (x_cond - x_uncond)
        assert torch.allclose(result, expected, rtol=1e-5, atol=1e-6)

    def test_cfg_with_interval_get_effective_scale(self, interval_cfg):
        """Test effective scale retrieval."""
        # Inside interval
        assert interval_cfg.get_effective_scale(0.5) == 7.5

        # Outside interval
        assert interval_cfg.get_effective_scale(0.8) == 1.0

    def test_cfg_with_interval_invalid_raises(self):
        """Test that invalid interval raises error."""
        with pytest.raises(ValueError, match="Invalid interval"):
            CFGWithInterval(
                guidance_scale=7.5,
                interval_start=0.8,
                interval_end=0.5,  # end < start
            )

        with pytest.raises(ValueError, match="Invalid interval"):
            CFGWithInterval(
                guidance_scale=7.5,
                interval_start=-0.1,  # < 0
                interval_end=0.75,
            )

    def test_cfg_with_interval_rescale(self, batch_tensors):
        """Test CFGWithInterval with rescale factor."""
        x_cond, x_uncond = batch_tensors

        cfg = CFGWithInterval(
            guidance_scale=7.5,
            rescale_factor=0.5,
            interval_start=0.0,
            interval_end=0.75,
        )

        result = cfg.apply(x_cond, x_uncond, 7.5)

        # Should have rescaling applied
        assert result.shape == x_cond.shape


# ============================================================================
# Test: CFGScheduler
# ============================================================================

class TestCFGScheduler:
    """Tests for CFGScheduler class."""

    def test_cfg_scheduler_constant(self):
        """Test constant schedule."""
        scheduler = CFGScheduler(
            schedule_type="constant",
            min_scale=1.0,
            max_scale=7.5,
        )

        # Should always return max_scale
        assert scheduler.get_scale(0.0) == 7.5
        assert scheduler.get_scale(0.5) == 7.5
        assert scheduler.get_scale(1.0) == 7.5

    def test_cfg_scheduler_linear(self):
        """Test linear schedule."""
        scheduler = CFGScheduler(
            schedule_type="linear",
            min_scale=1.0,
            max_scale=7.5,
        )

        # At t=0: max_scale
        assert scheduler.get_scale(0.0) == 7.5

        # At t=1: min_scale
        assert scheduler.get_scale(1.0) == 1.0

        # At t=0.5: midpoint
        mid = scheduler.get_scale(0.5)
        assert abs(mid - 4.25) < 0.01  # (7.5 + 1.0) / 2

    def test_cfg_scheduler_cosine(self):
        """Test cosine schedule."""
        scheduler = CFGScheduler(
            schedule_type="cosine",
            min_scale=1.0,
            max_scale=7.5,
        )

        # At t=0: max_scale (cos(0) = 1)
        scale_0 = scheduler.get_scale(0.0)
        assert abs(scale_0 - 7.5) < 0.01

        # At t=1: min_scale (cos(pi) = -1)
        scale_1 = scheduler.get_scale(1.0)
        assert abs(scale_1 - 1.0) < 0.01

        # At t=0.5: midpoint (cos(pi/2) = 0)
        scale_mid = scheduler.get_scale(0.5)
        assert abs(scale_mid - 4.25) < 0.01

    def test_cfg_scheduler_quadratic(self):
        """Test quadratic schedule."""
        scheduler = CFGScheduler(
            schedule_type="quadratic",
            min_scale=1.0,
            max_scale=7.5,
        )

        # At t=0: max_scale
        assert scheduler.get_scale(0.0) == 7.5

        # At t=1: min_scale
        scale_1 = scheduler.get_scale(1.0)
        assert abs(scale_1 - 1.0) < 0.01

        # Quadratic: scale = max - (max - min) * t^2
        # At t=0.5: 7.5 - 6.5 * 0.25 = 5.875
        scale_mid = scheduler.get_scale(0.5)
        expected = 7.5 - (7.5 - 1.0) * 0.25
        assert abs(scale_mid - expected) < 0.01


# ============================================================================
# Test: Utility Functions
# ============================================================================

class TestCFGUtilities:
    """Tests for CFG utility functions."""

    def test_apply_cfg_function(self, batch_tensors):
        """Test apply_cfg convenience function."""
        x_cond, x_uncond = batch_tensors

        result = apply_cfg(x_cond, x_uncond, guidance_scale=7.5)

        expected = x_uncond + 7.5 * (x_cond - x_uncond)
        assert torch.allclose(result, expected, rtol=1e-5, atol=1e-6)

    def test_apply_cfg_with_rescale(self, batch_tensors):
        """Test apply_cfg with rescale factor."""
        x_cond, x_uncond = batch_tensors

        result = apply_cfg(
            x_cond, x_uncond,
            guidance_scale=7.5,
            rescale_factor=0.5,
        )

        assert result.shape == x_cond.shape
        assert not torch.isnan(result).any()

    def test_compute_guidance_scale_schedule_constant(self):
        """Test schedule computation for constant type."""
        scales = compute_guidance_scale_schedule(
            num_steps=10,
            start_scale=7.5,
            end_scale=1.0,
            schedule_type="constant",
        )

        assert scales.shape == (10,)
        assert torch.all(scales == 7.5)

    def test_compute_guidance_scale_schedule_linear(self):
        """Test schedule computation for linear type."""
        scales = compute_guidance_scale_schedule(
            num_steps=11,
            start_scale=10.0,
            end_scale=0.0,
            schedule_type="linear",
        )

        assert scales.shape == (11,)
        assert abs(scales[0].item() - 10.0) < 0.01
        assert abs(scales[-1].item() - 0.0) < 0.01
        # Linear decrease
        assert abs(scales[5].item() - 5.0) < 0.01

    def test_compute_guidance_scale_schedule_invalid(self):
        """Test that invalid schedule type raises error."""
        with pytest.raises(ValueError, match="未知的調度類型"):
            compute_guidance_scale_schedule(
                num_steps=10,
                schedule_type="invalid",
            )


# ============================================================================
# Test: Factory Functions
# ============================================================================

class TestCFGFactories:
    """Tests for CFG factory functions."""

    def test_create_cfg_standard(self):
        """Test creating standard CFG."""
        cfg = create_cfg(method="standard")
        assert isinstance(cfg, StandardCFG)

    def test_create_cfg_rescaled(self):
        """Test creating rescaled CFG."""
        cfg = create_cfg(method="rescaled", rescale_factor=0.7)
        assert isinstance(cfg, RescaledCFG)
        assert cfg.rescale_factor == 0.7

    def test_create_cfg_perplexity(self):
        """Test creating perplexity CFG."""
        cfg = create_cfg(method="perplexity", temperature=2.0)
        assert isinstance(cfg, PerplexityCFG)
        assert cfg.temperature == 2.0

    def test_create_cfg_interval(self):
        """Test creating interval CFG."""
        cfg = create_cfg(
            method="interval",
            guidance_scale=5.0,
            interval_start=0.1,
            interval_end=0.8,
        )
        assert isinstance(cfg, CFGWithInterval)
        assert cfg.guidance_scale == 5.0
        assert cfg.interval_start == 0.1
        assert cfg.interval_end == 0.8

    def test_create_cfg_invalid(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="未知的 CFG 方法"):
            create_cfg(method="invalid_method")

    def test_create_cfg_scheduler(self):
        """Test creating CFG scheduler."""
        scheduler = create_cfg_scheduler(
            schedule_type="cosine",
            min_scale=2.0,
            max_scale=10.0,
        )

        assert isinstance(scheduler, CFGScheduler)
        assert scheduler.min_scale == 2.0
        assert scheduler.max_scale == 10.0


# ============================================================================
# Test: PerplexityCFG
# ============================================================================

class TestPerplexityCFG:
    """Tests for PerplexityCFG class."""

    def test_perplexity_cfg_apply(self, batch_tensors):
        """Test perplexity CFG application."""
        x_cond, x_uncond = batch_tensors
        cfg = PerplexityCFG(temperature=1.0)

        result = cfg.apply(x_cond, x_uncond, guidance_scale=7.5)

        assert result.shape == x_cond.shape
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_perplexity_cfg_adaptive(self, batch_tensors):
        """Test that perplexity CFG adapts based on uncertainty."""
        x_cond, x_uncond = batch_tensors
        cfg = PerplexityCFG(temperature=1.0)

        result = cfg.apply(x_cond, x_uncond, guidance_scale=7.5)

        # Result should be somewhere between uncond and full CFG
        full_cfg = x_uncond + 7.5 * (x_cond - x_uncond)

        # Adaptive result should have smaller magnitude than full CFG
        assert result.abs().mean() <= full_cfg.abs().mean() + 1.0


# ============================================================================
# Test: CFGConfig
# ============================================================================

class TestCFGConfig:
    """Tests for CFGConfig dataclass."""

    def test_cfg_config_defaults(self):
        """Test default CFG config values."""
        config = CFGConfig()

        assert config.guidance_scale == 7.5
        assert config.guidance_rescale == 0.0
        assert config.schedule_type == CFGScheduleType.CONSTANT
        assert config.use_negative_prompt == True

    def test_cfg_config_custom(self):
        """Test custom CFG config."""
        config = CFGConfig(
            guidance_scale=10.0,
            guidance_rescale=0.5,
            schedule_type=CFGScheduleType.LINEAR,
            min_guidance_scale=2.0,
        )

        assert config.guidance_scale == 10.0
        assert config.guidance_rescale == 0.5
        assert config.schedule_type == CFGScheduleType.LINEAR
        assert config.min_guidance_scale == 2.0


# ============================================================================
# Test: CFGWrapper
# ============================================================================

class DummyModelForWrapper(nn.Module):
    """Dummy model that accepts text_embed kwargs for CFGWrapper testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        # Apply linear to last dim, broadcasting over spatial dims
        return self.linear(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class TestCFGWrapper:
    """Tests for CFGWrapper class."""

    def test_cfg_wrapper_forward_no_cfg(self, batch_tensors):
        """Test wrapper forward without CFG."""
        x_cond, x_uncond = batch_tensors

        # Simple model that accepts text_embed
        model = DummyModelForWrapper()
        wrapper = CFGWrapper(model, default_scale=1.0)

        x_t = torch.randn(2, 3, 64, 64)
        t = torch.tensor([0.5, 0.5])

        result = wrapper(x_t, t)

        assert result.shape == x_t.shape

    def test_cfg_wrapper_with_scheduler(self, batch_tensors):
        """Test wrapper with scheduler."""
        model = DummyModelForWrapper()
        scheduler = CFGScheduler("linear", min_scale=1.0, max_scale=10.0)
        wrapper = CFGWrapper(model, scheduler=scheduler)

        x_t = torch.randn(2, 3, 64, 64)
        t = torch.tensor([0.5, 0.5])

        # Should use scheduler to determine scale
        result = wrapper(x_t, t)
        assert result.shape == x_t.shape
