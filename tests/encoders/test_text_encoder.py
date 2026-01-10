"""
Text Encoder Tests

Tests for Qwen3TextEncoder, TextProjector, CaptionEmbedder, and NullTextEncoder
with mocked model loading.

Test Categories:
    - Qwen3TextEncoder Init (5 tests): Config, model variants
    - Qwen3TextEncoder Forward (5 tests): Mocked forward pass
    - TextProjector (5 tests): Shape transformation
    - NullTextEncoder (4 tests): Null embedding generation
    - CaptionEmbedder (3 tests): Integration
    - Factory Functions (4 tests): Creation methods

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import Dict

from src.models.encoders.text_encoder import (
    Qwen3TextEncoder,
    TextProjector,
    CaptionEmbedder,
    NullTextEncoder,
    create_text_encoder,
    create_text_encoder_from_config,
    create_caption_embedder,
    create_null_text_encoder,
)
from src.config import PixelHDMConfig


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def testing_config() -> PixelHDMConfig:
    """Create minimal config for testing."""
    return PixelHDMConfig.for_testing()


@pytest.fixture
def mock_qwen_output():
    """Create mock Qwen3 model output."""
    class MockOutput:
        def __init__(self, batch_size: int, seq_len: int, hidden_size: int):
            self.last_hidden_state = torch.randn(batch_size, seq_len, hidden_size)
            self.hidden_states = (self.last_hidden_state,)
    return MockOutput


# ============================================================================
# Qwen3TextEncoder Initialization Tests
# ============================================================================

class TestQwen3TextEncoderInit:
    """Tests for Qwen3TextEncoder initialization."""

    def test_default_init(self):
        """Test default initialization without loading model."""
        encoder = Qwen3TextEncoder(
            model_name="Qwen/Qwen3-0.6B",
            max_length=256,
            freeze=True,
        )

        assert encoder.model_name == "Qwen/Qwen3-0.6B"
        assert encoder.max_length == 256
        assert encoder.freeze is True
        assert encoder.hidden_size == 1024
        assert encoder._loaded is False
        assert encoder.model is None

    def test_init_with_config(self, testing_config):
        """Test initialization from config."""
        encoder = Qwen3TextEncoder(config=testing_config)

        assert encoder.model_name == testing_config.text_encoder_name
        assert encoder.max_length == testing_config.text_max_length
        assert encoder.freeze == testing_config.text_encoder_frozen

    def test_supported_models_hidden_sizes(self):
        """Test hidden sizes for supported models."""
        expected_sizes = {
            "Qwen/Qwen3-0.6B": 1024,
            "Qwen/Qwen3-1.7B": 2048,
            "Qwen/Qwen3-4B": 2560,
        }

        for model_name, expected_size in expected_sizes.items():
            encoder = Qwen3TextEncoder(model_name=model_name)
            assert encoder.hidden_size == expected_size, \
                f"{model_name} should have hidden_size={expected_size}"

    def test_unknown_model_defaults_to_1024(self):
        """Test unknown model defaults to hidden_size=1024."""
        encoder = Qwen3TextEncoder(model_name="unknown/model")
        assert encoder.hidden_size == 1024

    def test_device_map_parameter(self):
        """Test device_map parameter is stored."""
        encoder = Qwen3TextEncoder(device_map="auto")
        assert encoder.device_map == "auto"


# ============================================================================
# Qwen3TextEncoder Forward Pass Tests (Mocked)
# ============================================================================

class TestQwen3TextEncoderForward:
    """Tests for Qwen3TextEncoder forward pass with mocked model."""

    def test_forward_with_mock_model(self, mock_qwen_output):
        """Test forward pass with mocked model."""
        encoder = Qwen3TextEncoder()

        B, T = 2, 77
        hidden_size = 1024

        # Create mock model
        mock_model = MagicMock()
        mock_model.return_value = mock_qwen_output(B, T, hidden_size)
        mock_model.eval = MagicMock()
        mock_model.parameters = MagicMock(return_value=iter([]))

        # Inject mock
        encoder.model = mock_model
        encoder._loaded = True

        # Forward pass
        input_ids = torch.randint(0, 1000, (B, T))
        attention_mask = torch.ones(B, T, dtype=torch.long)

        hidden_states, mask, pooled = encoder(input_ids=input_ids, attention_mask=attention_mask)

        assert hidden_states.shape == (B, T, hidden_size)
        assert mask.shape == (B, T)
        assert pooled.shape == (B, hidden_size)

    def test_forward_creates_default_mask(self, mock_qwen_output):
        """Test forward creates default attention mask if not provided."""
        encoder = Qwen3TextEncoder()

        B, T = 2, 77
        hidden_size = 1024

        mock_model = MagicMock()
        mock_model.return_value = mock_qwen_output(B, T, hidden_size)
        mock_model.eval = MagicMock()
        mock_model.parameters = MagicMock(return_value=iter([]))

        encoder.model = mock_model
        encoder._loaded = True

        input_ids = torch.randint(0, 1000, (B, T))
        hidden_states, mask, pooled = encoder(input_ids=input_ids)

        assert mask.shape == (B, T)
        assert (mask == 1).all()

    def test_forward_return_dict(self, mock_qwen_output):
        """Test forward with return_dict=True."""
        encoder = Qwen3TextEncoder(use_pooler=True)

        B, T = 2, 77
        hidden_size = 1024

        mock_model = MagicMock()
        mock_model.return_value = mock_qwen_output(B, T, hidden_size)
        mock_model.eval = MagicMock()
        mock_model.parameters = MagicMock(return_value=iter([]))

        encoder.model = mock_model
        encoder._loaded = True

        input_ids = torch.randint(0, 1000, (B, T))
        attention_mask = torch.ones(B, T, dtype=torch.long)

        result = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        assert isinstance(result, dict)
        assert "hidden_states" in result
        assert "attention_mask" in result
        assert "pooled_output" in result
        assert result["pooled_output"].shape == (B, hidden_size)

    def test_forward_no_input_raises(self):
        """Test forward without input raises error."""
        encoder = Qwen3TextEncoder()
        encoder._loaded = True
        encoder.model = MagicMock()

        with pytest.raises(ValueError, match="必須提供"):
            encoder()

    def test_get_output_dim(self):
        """Test get_output_dim method."""
        encoder = Qwen3TextEncoder(model_name="Qwen/Qwen3-0.6B")
        assert encoder.get_output_dim() == 1024

        encoder_large = Qwen3TextEncoder(model_name="Qwen/Qwen3-1.7B")
        assert encoder_large.get_output_dim() == 2048


# ============================================================================
# TextProjector Tests
# ============================================================================

class TestTextProjector:
    """Tests for TextProjector."""

    def test_projector_init_same_dim(self):
        """Test projector with same input/output dim uses identity."""
        proj = TextProjector(input_dim=1024, output_dim=1024)

        assert proj.input_dim == 1024
        assert proj.output_dim == 1024
        assert isinstance(proj.proj, nn.Identity)

    def test_projector_init_different_dim(self):
        """Test projector with different dims uses linear."""
        proj = TextProjector(input_dim=2048, output_dim=1024)

        assert proj.input_dim == 2048
        assert proj.output_dim == 1024
        assert isinstance(proj.proj, nn.Linear)

    def test_projector_init_from_config(self, testing_config):
        """Test projector initialization from config."""
        proj = TextProjector(config=testing_config)

        assert proj.input_dim == testing_config.text_hidden_size
        assert proj.output_dim == testing_config.hidden_dim

    def test_projector_forward_identity(self):
        """Test forward with identity projection."""
        proj = TextProjector(input_dim=1024, output_dim=1024)

        x = torch.randn(2, 77, 1024)
        output = proj(x)

        assert output.shape == (2, 77, 1024)
        assert torch.allclose(output, x)

    def test_projector_forward_linear(self):
        """Test forward with linear projection."""
        proj = TextProjector(input_dim=2048, output_dim=1024)

        x = torch.randn(2, 77, 2048)
        output = proj(x)

        assert output.shape == (2, 77, 1024)
        assert not torch.isnan(output).any()


# ============================================================================
# NullTextEncoder Tests
# ============================================================================

class TestNullTextEncoder:
    """Tests for NullTextEncoder."""

    def test_null_encoder_init(self):
        """Test NullTextEncoder initialization."""
        encoder = NullTextEncoder(hidden_dim=1024, max_length=1)

        assert encoder.hidden_dim == 1024
        assert encoder.max_length == 1
        assert encoder.null_embedding.shape == (1, 1, 1024)

    def test_null_encoder_init_from_config(self, testing_config):
        """Test NullTextEncoder from config."""
        encoder = NullTextEncoder(config=testing_config)

        assert encoder.hidden_dim == testing_config.hidden_dim
        assert encoder.max_length == 1

    def test_null_encoder_forward(self):
        """Test NullTextEncoder forward pass."""
        encoder = NullTextEncoder(hidden_dim=1024, max_length=1)

        batch_size = 4
        embeddings, mask = encoder(batch_size=batch_size)

        assert embeddings.shape == (batch_size, 1, 1024)
        assert mask.shape == (batch_size, 1)
        assert (mask == 1).all()

    def test_null_encoder_forward_with_device(self):
        """Test NullTextEncoder forward with device specification."""
        encoder = NullTextEncoder(hidden_dim=1024, max_length=1)

        embeddings, mask = encoder(batch_size=2, device=torch.device("cpu"))

        assert embeddings.device.type == "cpu"
        assert mask.device.type == "cpu"

    def test_null_encoder_get_output_dim(self):
        """Test get_output_dim method."""
        encoder = NullTextEncoder(hidden_dim=512)
        assert encoder.get_output_dim() == 512


# ============================================================================
# CaptionEmbedder Tests
# ============================================================================

class TestCaptionEmbedder:
    """Tests for CaptionEmbedder."""

    def test_caption_embedder_init(self, testing_config):
        """Test CaptionEmbedder initialization."""
        embedder = CaptionEmbedder(config=testing_config)

        assert isinstance(embedder.encoder, Qwen3TextEncoder)
        assert isinstance(embedder.projector, TextProjector)

    def test_caption_embedder_with_custom_encoder(self, testing_config):
        """Test CaptionEmbedder with custom encoder."""
        custom_encoder = Qwen3TextEncoder(config=testing_config)
        embedder = CaptionEmbedder(config=testing_config, encoder=custom_encoder)

        assert embedder.encoder is custom_encoder

    def test_caption_embedder_get_output_dim(self, testing_config):
        """Test get_output_dim method."""
        embedder = CaptionEmbedder(config=testing_config)
        assert embedder.get_output_dim() == testing_config.hidden_dim


# ============================================================================
# Factory Function Tests
# ============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_text_encoder(self):
        """Test create_text_encoder factory."""
        encoder = create_text_encoder(
            model_name="Qwen/Qwen3-0.6B",
            max_length=256,
            freeze=True,
        )

        assert isinstance(encoder, Qwen3TextEncoder)
        assert encoder.model_name == "Qwen/Qwen3-0.6B"
        assert encoder.max_length == 256

    def test_create_text_encoder_from_config(self, testing_config):
        """Test create_text_encoder_from_config factory."""
        encoder = create_text_encoder_from_config(testing_config)

        assert isinstance(encoder, Qwen3TextEncoder)
        assert encoder.model_name == testing_config.text_encoder_name

    def test_create_caption_embedder(self, testing_config):
        """Test create_caption_embedder factory."""
        embedder = create_caption_embedder(config=testing_config)

        assert isinstance(embedder, CaptionEmbedder)

    def test_create_null_text_encoder(self):
        """Test create_null_text_encoder factory."""
        encoder = create_null_text_encoder(hidden_dim=1024, max_length=1)

        assert isinstance(encoder, NullTextEncoder)
        assert encoder.hidden_dim == 1024


# ============================================================================
# Extra Repr Tests
# ============================================================================

class TestExtraRepr:
    """Tests for extra_repr methods."""

    def test_qwen3_extra_repr(self):
        """Test Qwen3TextEncoder extra_repr."""
        encoder = Qwen3TextEncoder(
            model_name="Qwen/Qwen3-0.6B",
            max_length=256,
            freeze=True,
        )
        repr_str = encoder.extra_repr()

        assert "Qwen/Qwen3-0.6B" in repr_str
        assert "1024" in repr_str
        assert "256" in repr_str
        assert "True" in repr_str

    def test_text_projector_extra_repr(self):
        """Test TextProjector extra_repr."""
        proj = TextProjector(input_dim=2048, output_dim=1024)
        repr_str = proj.extra_repr()

        assert "2048" in repr_str
        assert "1024" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
