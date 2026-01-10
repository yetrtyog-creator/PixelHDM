"""
BaseSampler and SamplerConfig Unit Tests.

Validates base class contract and configuration:
- SamplerConfig dataclass
- SamplerMethod enum
- Base class interface
- Factory function creation

Test Count: 15 test cases

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-08
"""

import pytest
import torch
from typing import Optional

from src.inference.sampler.base import BaseSampler, SamplerConfig, SamplerMethod
from src.inference.sampler import (
    EulerSampler,
    HeunSampler,
    DPMPPSampler,
    UnifiedSampler,
    create_sampler,
    create_sampler_from_config,
)


# =============================================================================
# Test Class: SamplerMethod Enum
# =============================================================================


class TestSamplerMethod:
    """Test SamplerMethod enum."""

    def test_enum_values(self):
        """Test enum has expected values."""
        assert SamplerMethod.EULER.value == "euler"
        assert SamplerMethod.HEUN.value == "heun"
        assert SamplerMethod.DPM_PP.value == "dpm_pp"

    def test_enum_from_string(self):
        """Test creating enum from string."""
        assert SamplerMethod("euler") == SamplerMethod.EULER
        assert SamplerMethod("heun") == SamplerMethod.HEUN
        assert SamplerMethod("dpm_pp") == SamplerMethod.DPM_PP

    def test_invalid_method_raises(self):
        """Test invalid method raises ValueError."""
        with pytest.raises(ValueError):
            SamplerMethod("invalid_method")


# =============================================================================
# Test Class: SamplerConfig
# =============================================================================


class TestSamplerConfig:
    """Test SamplerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SamplerConfig()

        assert config.num_steps == 50
        assert config.method == SamplerMethod.HEUN
        assert config.t_eps == 0.05
        assert config.guidance_scale == 7.5
        assert config.guidance_rescale == 0.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SamplerConfig(
            num_steps=100,
            method=SamplerMethod.EULER,
            t_eps=0.1,
            guidance_scale=15.0,
        )

        assert config.num_steps == 100
        assert config.method == SamplerMethod.EULER
        assert config.t_eps == 0.1
        assert config.guidance_scale == 15.0

    def test_config_immutability(self):
        """Test config values can be changed (mutable dataclass)."""
        config = SamplerConfig()
        config.num_steps = 100

        assert config.num_steps == 100


# =============================================================================
# Test Class: Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Test sampler factory functions."""

    def test_create_sampler_euler(self):
        """Test creating Euler sampler via factory."""
        sampler = create_sampler(method="euler", num_steps=50)

        assert isinstance(sampler, UnifiedSampler)
        assert sampler.method == SamplerMethod.EULER

    def test_create_sampler_heun(self):
        """Test creating Heun sampler via factory."""
        sampler = create_sampler(method="heun", num_steps=50)

        assert isinstance(sampler, UnifiedSampler)
        assert sampler.method == SamplerMethod.HEUN

    def test_create_sampler_dpm(self):
        """Test creating DPM++ sampler via factory."""
        sampler = create_sampler(method="dpm_pp", num_steps=50)

        assert isinstance(sampler, UnifiedSampler)
        assert sampler.method == SamplerMethod.DPM_PP

    def test_create_sampler_from_config(self):
        """Test creating sampler from PixelHDMConfig."""
        from src.config import PixelHDMConfig

        config = PixelHDMConfig.for_testing()
        sampler = create_sampler_from_config(config, method="heun")

        assert isinstance(sampler, UnifiedSampler)

    def test_create_sampler_custom_params(self):
        """Test factory with custom parameters."""
        sampler = create_sampler(
            method="euler",
            num_steps=100,
            t_eps=0.1,
        )

        assert sampler.num_steps == 100
        assert sampler.t_eps == 0.1


# =============================================================================
# Test Class: UnifiedSampler Interface
# =============================================================================


class TestUnifiedSamplerInterface:
    """Test UnifiedSampler unified interface."""

    def test_sample_method_exists(self):
        """Test sample method exists."""
        sampler = create_sampler(method="euler")
        assert hasattr(sampler, "sample")
        assert callable(sampler.sample)

    def test_sample_progressive_exists(self):
        """Test sample_progressive method exists."""
        sampler = create_sampler(method="euler")
        assert hasattr(sampler, "sample_progressive")
        assert callable(sampler.sample_progressive)

    def test_count_nfe_exists(self):
        """Test count_nfe method exists."""
        sampler = create_sampler(method="euler")
        assert hasattr(sampler, "count_nfe")
        assert callable(sampler.count_nfe)

    def test_count_nfe_euler(self):
        """Test NFE count for Euler sampler."""
        sampler = create_sampler(method="euler", num_steps=50)

        # Euler: 1 evaluation per step
        nfe_no_cfg = sampler.count_nfe(use_cfg=False)
        assert nfe_no_cfg == 50

        # With CFG: 2 evaluations per step
        nfe_with_cfg = sampler.count_nfe(use_cfg=True)
        assert nfe_with_cfg == 100

    def test_count_nfe_heun(self):
        """Test NFE count for Heun sampler."""
        sampler = create_sampler(method="heun", num_steps=50)

        # Heun: 2 evaluations per step (except last)
        nfe_no_cfg = sampler.count_nfe(use_cfg=False)
        assert nfe_no_cfg == 2 * 50 - 1  # 99

        # With CFG (default: not full_heun_cfg): doubles steps
        nfe_with_cfg = sampler.count_nfe(use_cfg=True)
        assert nfe_with_cfg == 50 * 2  # 100

        # With full Heun CFG: doubles base NFE
        nfe_full_heun = sampler.count_nfe(use_cfg=True, full_heun_cfg=True)
        assert nfe_full_heun == (2 * 50 - 1) * 2  # 198

    def test_count_nfe_dpm(self):
        """Test NFE count for DPM++ sampler."""
        sampler = create_sampler(method="dpm_pp", num_steps=50)

        # DPM++: 1 evaluation per step (uses history)
        nfe_no_cfg = sampler.count_nfe(use_cfg=False)
        assert nfe_no_cfg == 50


# =============================================================================
# Test Class: Sampler Interchangeability
# =============================================================================


class TestSamplerInterchangeability:
    """Test that different samplers are interchangeable."""

    def test_all_samplers_same_interface(self):
        """Test all samplers have same interface."""
        methods = ["euler", "heun", "dpm_pp"]

        for method in methods:
            sampler = create_sampler(method=method, num_steps=50)

            # Check common interface
            assert hasattr(sampler, "sample")
            assert hasattr(sampler, "sample_progressive")
            assert hasattr(sampler, "count_nfe")

    def test_all_samplers_produce_output(self):
        """Test all samplers produce valid output."""
        import torch.nn as nn

        class DummyModel(nn.Module):
            def forward(self, z, t, **kwargs):
                return torch.zeros_like(z)

        model = DummyModel()
        z_0 = torch.randn(1, 32, 32, 3)

        methods = ["euler", "heun", "dpm_pp"]

        for method in methods:
            sampler = create_sampler(method=method, num_steps=10)

            output = sampler.sample(
                model=model,
                z_0=z_0,
                num_steps=10,
                guidance_scale=1.0,
            )

            assert output.shape == z_0.shape, f"{method} output shape mismatch"
            assert not torch.isnan(output).any(), f"{method} produced NaN"
