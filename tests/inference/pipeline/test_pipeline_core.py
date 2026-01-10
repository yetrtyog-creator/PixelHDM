"""
Pipeline Core Unit Tests.

Validates the main PixelHDMPipeline class:
- Pipeline initialization
- Device and dtype handling
- Text encoder property
- Generator creation
- Output creation

Test Count: 15 test cases

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-08
"""

import pytest
import torch
import torch.nn as nn
from typing import Optional, List
from unittest.mock import Mock, MagicMock, patch
from PIL import Image
import numpy as np

from src.inference.pipeline.core import PixelHDMPipeline
from src.inference.pipeline.postprocessing import PipelineOutput


# =============================================================================
# Test Fixtures
# =============================================================================


class MockModel(nn.Module):
    """Mock model for pipeline testing."""

    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.dummy = nn.Linear(10, 10)
        self.config = MagicMock()
        self.config.patch_size = 16
        self.config.hidden_dim = hidden_dim
        self.config.time_eps = 0.05
        self.config.max_patches = 1024
        self._text_encoder = None
        self.call_count = 0

    @property
    def text_encoder(self):
        return self._text_encoder

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


class MockTextEncoder:
    """Mock text encoder for testing."""

    def __init__(self, hidden_dim: int = 1024):
        self.hidden_dim = hidden_dim

    def __call__(self, texts: List[str], return_pooled: bool = False):
        batch_size = len(texts)
        hidden_states = torch.randn(batch_size, 77, self.hidden_dim)
        mask = torch.ones(batch_size, 77)
        pooled = torch.randn(batch_size, self.hidden_dim) if return_pooled else None
        return hidden_states, mask, pooled


@pytest.fixture
def mock_model() -> MockModel:
    """Create mock model."""
    return MockModel()


@pytest.fixture
def mock_text_encoder() -> MockTextEncoder:
    """Create mock text encoder."""
    return MockTextEncoder()


@pytest.fixture
def pipeline(mock_model: MockModel, mock_text_encoder: MockTextEncoder) -> PixelHDMPipeline:
    """Create pipeline with mocks."""
    return PixelHDMPipeline(
        model=mock_model,
        text_encoder=mock_text_encoder,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


# =============================================================================
# Test Class: Pipeline Initialization
# =============================================================================


class TestPipelineInitialization:
    """Test pipeline initialization."""

    def test_pipeline_creation(self, mock_model: MockModel, mock_text_encoder: MockTextEncoder):
        """Test pipeline can be created."""
        pipeline = PixelHDMPipeline(
            model=mock_model,
            text_encoder=mock_text_encoder,
        )
        assert pipeline is not None

    def test_pipeline_stores_model(self, mock_model: MockModel, mock_text_encoder: MockTextEncoder):
        """Test pipeline stores model reference."""
        pipeline = PixelHDMPipeline(model=mock_model, text_encoder=mock_text_encoder)
        assert pipeline.model is mock_model

    def test_pipeline_stores_dtype(self, mock_model: MockModel, mock_text_encoder: MockTextEncoder):
        """Test pipeline stores dtype."""
        pipeline = PixelHDMPipeline(
            model=mock_model,
            text_encoder=mock_text_encoder,
            dtype=torch.float16,
        )
        assert pipeline.dtype == torch.float16

    def test_pipeline_infers_device(self, mock_model: MockModel, mock_text_encoder: MockTextEncoder):
        """Test pipeline infers device from model."""
        pipeline = PixelHDMPipeline(
            model=mock_model,
            text_encoder=mock_text_encoder,
            device=None,  # Should infer from model
        )
        # Model is on CPU by default
        assert pipeline.device == torch.device("cpu")


# =============================================================================
# Test Class: Text Encoder Property
# =============================================================================


class TestTextEncoderProperty:
    """Test text encoder property behavior."""

    def test_text_encoder_from_constructor(self, mock_model: MockModel, mock_text_encoder: MockTextEncoder):
        """Test text encoder from constructor takes precedence."""
        pipeline = PixelHDMPipeline(
            model=mock_model,
            text_encoder=mock_text_encoder,
        )
        assert pipeline.text_encoder is mock_text_encoder

    def test_text_encoder_from_model(self, mock_model: MockModel):
        """Test text encoder falls back to model."""
        model_encoder = MockTextEncoder()
        mock_model._text_encoder = model_encoder

        pipeline = PixelHDMPipeline(
            model=mock_model,
            text_encoder=None,  # No encoder provided
        )
        assert pipeline.text_encoder is model_encoder

    def test_text_encoder_none(self, mock_model: MockModel):
        """Test text encoder is None when not available."""
        mock_model._text_encoder = None

        pipeline = PixelHDMPipeline(
            model=mock_model,
            text_encoder=None,
        )
        assert pipeline.text_encoder is None


# =============================================================================
# Test Class: Generator Creation
# =============================================================================


class TestGeneratorCreation:
    """Test random generator creation."""

    def test_create_generator_with_seed(self, pipeline: PixelHDMPipeline):
        """Test generator creation with seed."""
        generator = pipeline._create_generator(seed=42)

        assert generator is not None
        assert isinstance(generator, torch.Generator)

    def test_create_generator_none_without_seed(self, pipeline: PixelHDMPipeline):
        """Test generator is None without seed."""
        generator = pipeline._create_generator(seed=None)
        assert generator is None

    def test_generator_produces_reproducible_results(self, pipeline: PixelHDMPipeline):
        """Test same seed produces same results."""
        gen1 = pipeline._create_generator(seed=42)
        vals1 = torch.randn(10, generator=gen1)

        gen2 = pipeline._create_generator(seed=42)
        vals2 = torch.randn(10, generator=gen2)

        assert torch.allclose(vals1, vals2)


# =============================================================================
# Test Class: Output Creation
# =============================================================================


class TestOutputCreation:
    """Test PipelineOutput creation."""

    def test_create_output_basic(self, pipeline: PixelHDMPipeline):
        """Test basic output creation."""
        images = torch.randn(2, 256, 256, 3)
        inputs = MagicMock()
        inputs.z_0 = torch.randn(2, 256, 256, 3)

        output = pipeline._create_output(
            result=images,
            inputs=inputs,
            prompt="a cat",
            negative_prompt=None,
            height=256,
            width=256,
            num_steps=50,
            guidance_scale=7.5,
            seed=42,
            sampler_method="heun",
            output_type="pil",
            return_intermediates=False,
        )

        assert isinstance(output, PipelineOutput)

    def test_create_output_with_intermediates(self, pipeline: PixelHDMPipeline):
        """Test output creation with intermediates."""
        intermediates = [torch.randn(2, 256, 256, 3) for _ in range(5)]
        inputs = MagicMock()
        inputs.z_0 = torch.randn(2, 256, 256, 3)

        output = pipeline._create_output(
            result=intermediates,
            inputs=inputs,
            prompt="a cat",
            negative_prompt=None,
            height=256,
            width=256,
            num_steps=50,
            guidance_scale=7.5,
            seed=42,
            sampler_method="heun",
            output_type="pil",
            return_intermediates=True,
        )

        assert output.intermediates is not None


# =============================================================================
# Test Class: Component Initialization
# =============================================================================


class TestComponentInitialization:
    """Test internal component initialization."""

    def test_validator_initialized(self, pipeline: PixelHDMPipeline):
        """Test validator is initialized."""
        assert pipeline._validator is not None

    def test_preprocessor_initialized(self, pipeline: PixelHDMPipeline):
        """Test preprocessor is initialized."""
        assert pipeline._preprocessor is not None

    def test_postprocessor_initialized(self, pipeline: PixelHDMPipeline):
        """Test postprocessor is initialized."""
        assert pipeline._postprocessor is not None

    def test_generator_initialized(self, pipeline: PixelHDMPipeline):
        """Test generator is initialized."""
        assert pipeline._generator is not None


# =============================================================================
# Test Class: Compilation State
# =============================================================================


class TestCompilationState:
    """Test compilation state tracking."""

    def test_initial_compilation_state(self, pipeline: PixelHDMPipeline):
        """Test pipeline starts uncompiled."""
        assert pipeline._compiled is False


# =============================================================================
# Test Class: Config Access
# =============================================================================


class TestConfigAccess:
    """Test configuration access."""

    def test_config_from_model(self, pipeline: PixelHDMPipeline):
        """Test config is accessed from model."""
        assert pipeline.config is not None

    def test_config_is_model_config(self, mock_model: MockModel, mock_text_encoder: MockTextEncoder):
        """Test config is same as model config."""
        pipeline = PixelHDMPipeline(
            model=mock_model,
            text_encoder=mock_text_encoder,
        )
        assert pipeline.config is mock_model.config


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
