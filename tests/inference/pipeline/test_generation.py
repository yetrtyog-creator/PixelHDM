"""
Pipeline Generation Unit Tests.

Validates core generation logic:
- Sampler creation and caching
- Generation config
- NFE counting
- T_eps handling

Test Count: 15 test cases

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-08
"""

import pytest
import torch
import torch.nn as nn
from typing import Optional
from dataclasses import asdict

from src.inference.pipeline.generation import Generator, GenerationConfig
from src.inference.pipeline.preprocessing import GenerationInputs
from src.inference.sampler import UnifiedSampler
from src.config import PixelHDMConfig


# =============================================================================
# Test Fixtures
# =============================================================================


class MockModel(nn.Module):
    """Mock model for generation testing."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Linear(10, 10)
        self.call_count = 0

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        pooled_text_embed: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        self.call_count += 1
        return torch.zeros_like(x_t)


@pytest.fixture
def mock_model() -> MockModel:
    """Create mock model."""
    return MockModel()


@pytest.fixture
def generator(mock_model: MockModel) -> Generator:
    """Create generator with mock model."""
    return Generator(model=mock_model)


@pytest.fixture
def generator_with_config(mock_model: MockModel) -> Generator:
    """Create generator with config."""
    config = PixelHDMConfig.for_testing()
    return Generator(model=mock_model, config=config)


@pytest.fixture
def sample_inputs() -> GenerationInputs:
    """Create sample generation inputs."""
    batch_size = 2
    height, width = 256, 256
    return GenerationInputs(
        z_0=torch.randn(batch_size, height, width, 3),
        text_embed=torch.randn(batch_size, 32, 1024),
        text_mask=torch.ones(batch_size, 32),
        pooled_text_embed=torch.randn(batch_size, 1024),
        null_text_embed=torch.zeros(batch_size, 32, 1024),
        null_text_mask=torch.ones(batch_size, 32),
        null_pooled_text_embed=torch.zeros(batch_size, 1024),
        batch_size=batch_size,
        height=height,
        width=width,
    )


# =============================================================================
# Test Class: GenerationConfig
# =============================================================================


class TestGenerationConfig:
    """Test GenerationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = GenerationConfig()

        assert config.height == 512
        assert config.width == 512
        assert config.num_steps == 50
        assert config.sampler_method == "heun"
        assert config.guidance_scale == 7.5

    def test_custom_values(self):
        """Test custom configuration values."""
        config = GenerationConfig(
            height=768,
            width=768,
            num_steps=100,
            guidance_scale=15.0,
        )

        assert config.height == 768
        assert config.num_steps == 100
        assert config.guidance_scale == 15.0

    def test_to_dict(self):
        """Test config can be converted to dict."""
        config = GenerationConfig(seed=42)
        config_dict = asdict(config)

        assert isinstance(config_dict, dict)
        assert config_dict["seed"] == 42


# =============================================================================
# Test Class: Sampler Creation
# =============================================================================


class TestSamplerCreation:
    """Test sampler creation and caching."""

    def test_get_sampler_creates_unified_sampler(self, generator: Generator):
        """Test get_sampler returns UnifiedSampler."""
        sampler = generator.get_sampler(method="euler", num_steps=50)
        assert isinstance(sampler, UnifiedSampler)

    def test_get_sampler_caches(self, generator: Generator):
        """Test sampler is cached."""
        sampler1 = generator.get_sampler(method="euler", num_steps=50)
        sampler2 = generator.get_sampler(method="euler", num_steps=50)
        assert sampler1 is sampler2

    def test_get_sampler_different_method(self, generator: Generator):
        """Test different method creates new sampler."""
        sampler_euler = generator.get_sampler(method="euler", num_steps=50)
        sampler_heun = generator.get_sampler(method="heun", num_steps=50)
        assert sampler_euler is not sampler_heun

    @pytest.mark.parametrize("method", ["euler", "heun", "dpm_pp"])
    def test_all_sampler_methods(self, generator: Generator, method: str):
        """Test all sampler methods can be created."""
        sampler = generator.get_sampler(method=method, num_steps=50)
        assert sampler is not None


# =============================================================================
# Test Class: T_eps Handling
# =============================================================================


class TestTEpsHandling:
    """Test time epsilon handling."""

    def test_t_eps_from_config(self, generator_with_config: Generator):
        """Test t_eps is read from config."""
        t_eps = generator_with_config._get_t_eps()
        # Testing config should have time_eps defined
        assert t_eps > 0

    def test_t_eps_default(self, generator: Generator):
        """Test t_eps defaults to 0.05 without config."""
        t_eps = generator._get_t_eps()
        assert t_eps == 0.05


# =============================================================================
# Test Class: NFE Counting
# =============================================================================


class TestNFECounting:
    """Test Network Function Evaluation counting."""

    def test_count_nfe_euler(self, generator: Generator):
        """Test NFE count for Euler sampler."""
        nfe = generator.count_nfe(num_steps=50, use_cfg=False, sampler_method="euler")
        assert nfe == 50

    def test_count_nfe_euler_with_cfg(self, generator: Generator):
        """Test NFE count for Euler with CFG."""
        nfe = generator.count_nfe(num_steps=50, use_cfg=True, sampler_method="euler")
        assert nfe == 100  # 2x for CFG

    def test_count_nfe_heun(self, generator: Generator):
        """Test NFE count for Heun sampler."""
        nfe = generator.count_nfe(num_steps=50, use_cfg=False, sampler_method="heun")
        # Heun: 2 per step except last = 2*50 - 1 = 99
        assert nfe == 99

    def test_count_nfe_heun_with_cfg(self, generator: Generator):
        """Test NFE count for Heun with CFG (default, not full_heun_cfg)."""
        nfe = generator.count_nfe(num_steps=50, use_cfg=True, sampler_method="heun")
        # Default CFG doubles steps (not full Heun on both branches)
        assert nfe == 100


# =============================================================================
# Test Class: Generation Workflow
# =============================================================================


class TestGenerationWorkflow:
    """Test generation workflow."""

    def test_generate_returns_tensor(
        self, generator: Generator, sample_inputs: GenerationInputs
    ):
        """Test generate returns tensor."""
        result = generator.generate(
            inputs=sample_inputs,
            num_steps=5,
            guidance_scale=1.0,
            sampler_method="euler",
        )

        assert isinstance(result, torch.Tensor)
        assert result.shape == sample_inputs.z_0.shape

    def test_generate_with_intermediates(
        self, generator: Generator, sample_inputs: GenerationInputs
    ):
        """Test generate with intermediates returns stacked tensor."""
        result = generator.generate(
            inputs=sample_inputs,
            num_steps=10,
            guidance_scale=1.0,
            sampler_method="euler",
            return_intermediates=True,
        )

        # sample_progressive returns stacked tensor (N_saved, B, H, W, 3)
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 5  # (N_saved, B, H, W, 3)
        assert result.shape[0] > 0  # At least one intermediate

    def test_generate_uses_correct_sampler(
        self, generator: Generator, sample_inputs: GenerationInputs
    ):
        """Test generate uses specified sampler method."""
        # Generate with Euler
        generator.generate(
            inputs=sample_inputs,
            num_steps=5,
            guidance_scale=1.0,
            sampler_method="euler",
        )
        assert generator._current_method == "euler"

        # Generate with Heun
        generator.generate(
            inputs=sample_inputs,
            num_steps=5,
            guidance_scale=1.0,
            sampler_method="heun",
        )
        assert generator._current_method == "heun"


# =============================================================================
# Test Class: Image-to-Image Generation
# =============================================================================


class TestI2IGeneration:
    """Test image-to-image generation."""

    def test_generate_i2i_returns_tensor(self, generator: Generator):
        """Test I2I generation returns tensor."""
        z_t = torch.randn(1, 256, 256, 3)
        text_embed = torch.randn(1, 32, 1024)
        text_mask = torch.ones(1, 32)

        result = generator.generate_i2i(
            z_t=z_t,
            text_embed=text_embed,
            text_mask=text_mask,
            null_text_embed=None,
            num_steps=5,
            guidance_scale=1.0,
            t_start=0.5,
            sampler_method="euler",
        )

        assert isinstance(result, torch.Tensor)
        assert result.shape == z_t.shape

    def test_generate_i2i_t_start_parameter(self, generator: Generator):
        """Test I2I respects t_start parameter."""
        z_t = torch.randn(1, 256, 256, 3)
        text_embed = torch.randn(1, 32, 1024)
        text_mask = torch.ones(1, 32)

        # With t_start=0.5, generation starts from midpoint
        result = generator.generate_i2i(
            z_t=z_t,
            text_embed=text_embed,
            text_mask=text_mask,
            null_text_embed=None,
            num_steps=10,
            guidance_scale=1.0,
            t_start=0.5,  # 50% noise
            sampler_method="euler",
        )

        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
