"""
Pipeline Preprocessing Unit Tests.

Validates text encoding and input preparation:
- Latent preparation
- Text encoding workflow
- Null embedding caching
- Multi-image batch handling

Test Count: 15 test cases

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-08
"""

import pytest
import torch
from typing import Optional, Tuple, List
from unittest.mock import Mock, MagicMock

from src.inference.pipeline.preprocessing import Preprocessor, GenerationInputs
from src.inference.pipeline.validation import InputValidator


# =============================================================================
# Test Fixtures
# =============================================================================


class MockTextEncoder:
    """Mock text encoder for testing."""

    def __init__(self, hidden_dim: int = 1024, seq_len: int = 77):
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.call_count = 0

    def __call__(
        self, texts: List[str], return_pooled: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        self.call_count += 1
        batch_size = len(texts)
        hidden_states = torch.randn(batch_size, self.seq_len, self.hidden_dim)
        attention_mask = torch.ones(batch_size, self.seq_len)
        pooled_output = torch.randn(batch_size, self.hidden_dim) if return_pooled else None
        return hidden_states, attention_mask, pooled_output


@pytest.fixture
def mock_text_encoder() -> MockTextEncoder:
    """Create mock text encoder."""
    return MockTextEncoder()


@pytest.fixture
def preprocessor(mock_text_encoder: MockTextEncoder) -> Preprocessor:
    """Create preprocessor with mock encoder."""
    validator = InputValidator()
    return Preprocessor(
        text_encoder=mock_text_encoder,
        validator=validator,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


# =============================================================================
# Test Class: Latent Preparation
# =============================================================================


class TestLatentPreparation:
    """Test latent preparation functionality."""

    def test_prepare_latents_shape(self, preprocessor: Preprocessor):
        """Test latent tensor has correct shape."""
        batch_size = 2
        height, width = 512, 512

        latents = preprocessor.prepare_latents(batch_size, height, width, None)

        assert latents.shape == (batch_size, height, width, 3)

    def test_prepare_latents_dtype(self, preprocessor: Preprocessor):
        """Test latent tensor has correct dtype."""
        latents = preprocessor.prepare_latents(2, 256, 256, None)
        assert latents.dtype == torch.float32

    def test_prepare_latents_device(self, preprocessor: Preprocessor):
        """Test latent tensor is on correct device."""
        latents = preprocessor.prepare_latents(2, 256, 256, None)
        assert latents.device == torch.device("cpu")

    def test_prepare_latents_with_generator(self, preprocessor: Preprocessor):
        """Test latent preparation with generator produces reproducible results."""
        generator = torch.Generator().manual_seed(42)
        latents1 = preprocessor.prepare_latents(2, 256, 256, generator)

        generator = torch.Generator().manual_seed(42)
        latents2 = preprocessor.prepare_latents(2, 256, 256, generator)

        assert torch.allclose(latents1, latents2)

    def test_prepare_latents_different_seeds(self, preprocessor: Preprocessor):
        """Test different seeds produce different latents."""
        generator1 = torch.Generator().manual_seed(42)
        latents1 = preprocessor.prepare_latents(2, 256, 256, generator1)

        generator2 = torch.Generator().manual_seed(123)
        latents2 = preprocessor.prepare_latents(2, 256, 256, generator2)

        assert not torch.allclose(latents1, latents2)


# =============================================================================
# Test Class: Text Encoding
# =============================================================================


class TestTextEncoding:
    """Test text encoding workflow."""

    def test_encode_single_prompt(self, preprocessor: Preprocessor):
        """Test encoding single prompt."""
        prompts = ["a cat sitting on a chair"]

        text_embed, text_mask, pooled, null_embed, null_mask, null_pooled = \
            preprocessor.encode_prompt(prompts, None, num_images_per_prompt=1)

        assert text_embed.shape[0] == 1
        assert text_mask.shape[0] == 1

    def test_encode_multiple_prompts(self, preprocessor: Preprocessor):
        """Test encoding multiple prompts."""
        prompts = ["a cat", "a dog", "a bird"]

        text_embed, text_mask, pooled, null_embed, null_mask, null_pooled = \
            preprocessor.encode_prompt(prompts, None, num_images_per_prompt=1)

        assert text_embed.shape[0] == 3

    def test_encode_with_num_images_per_prompt(self, preprocessor: Preprocessor):
        """Test encoding with multiple images per prompt."""
        prompts = ["a cat"]

        text_embed, text_mask, pooled, null_embed, null_mask, null_pooled = \
            preprocessor.encode_prompt(prompts, None, num_images_per_prompt=4)

        # Should be repeated 4 times
        assert text_embed.shape[0] == 4

    def test_encode_with_negative_prompt(self, preprocessor: Preprocessor):
        """Test encoding with negative prompt."""
        prompts = ["a cat"]
        negative = "blurry, bad quality"

        text_embed, text_mask, pooled, null_embed, null_mask, null_pooled = \
            preprocessor.encode_prompt(prompts, negative, num_images_per_prompt=1)

        # Null embed should come from negative prompt, not cached
        assert null_embed is not None
        assert null_embed.shape[0] == 1


# =============================================================================
# Test Class: Null Embedding Caching
# =============================================================================


class TestNullEmbeddingCaching:
    """Test null embedding caching functionality."""

    def test_null_embed_cached(self, preprocessor: Preprocessor, mock_text_encoder: MockTextEncoder):
        """Test null embedding is cached."""
        prompts = ["a cat"]

        # First call
        preprocessor.encode_prompt(prompts, None, num_images_per_prompt=1)
        first_call_count = mock_text_encoder.call_count

        # Second call should use cached null embed
        preprocessor.encode_prompt(prompts, None, num_images_per_prompt=1)
        second_call_count = mock_text_encoder.call_count

        # Should only encode the prompt again, not null text
        assert second_call_count == first_call_count + 1

    def test_null_embed_batch_expansion(self, preprocessor: Preprocessor):
        """Test null embedding is correctly expanded for batch."""
        prompts = ["a cat", "a dog"]

        text_embed, text_mask, pooled, null_embed, null_mask, null_pooled = \
            preprocessor.encode_prompt(prompts, None, num_images_per_prompt=1)

        assert null_embed.shape[0] == 2

    def test_clear_cache(self, preprocessor: Preprocessor, mock_text_encoder: MockTextEncoder):
        """Test cache clearing."""
        prompts = ["a cat"]

        preprocessor.encode_prompt(prompts, None, num_images_per_prompt=1)
        preprocessor.clear_cache()

        # After clearing, null embed should be None
        assert preprocessor._null_text_embed is None


# =============================================================================
# Test Class: GenerationInputs
# =============================================================================


class TestGenerationInputs:
    """Test GenerationInputs dataclass."""

    def test_generation_inputs_creation(self, preprocessor: Preprocessor):
        """Test GenerationInputs is created correctly."""
        inputs = preprocessor.prepare_inputs(
            prompt="a cat",
            negative_prompt=None,
            height=256,
            width=256,
            num_images_per_prompt=1,
            generator=None,
        )

        assert isinstance(inputs, GenerationInputs)
        assert inputs.batch_size == 1
        assert inputs.height == 256
        assert inputs.width == 256

    def test_generation_inputs_shapes(self, preprocessor: Preprocessor):
        """Test all inputs have correct shapes."""
        inputs = preprocessor.prepare_inputs(
            prompt="a cat",
            negative_prompt=None,
            height=512,
            width=512,
            num_images_per_prompt=2,
            generator=None,
        )

        # z_0 shape: (batch, H, W, 3)
        assert inputs.z_0.shape == (2, 512, 512, 3)

        # text_embed shape: (batch, seq_len, hidden_dim)
        assert inputs.text_embed.shape[0] == 2

    def test_prepare_inputs_validates_resolution(self, preprocessor: Preprocessor):
        """Test prepare_inputs validates resolution."""
        with pytest.raises(ValueError):
            preprocessor.prepare_inputs(
                prompt="a cat",
                negative_prompt=None,
                height=517,  # Not divisible by 16
                width=512,
                num_images_per_prompt=1,
                generator=None,
            )


# =============================================================================
# Test Class: Error Handling
# =============================================================================


class TestPreprocessorErrors:
    """Test preprocessor error handling."""

    def test_no_text_encoder_raises(self):
        """Test error when no text encoder is available."""
        validator = InputValidator()
        preprocessor = Preprocessor(
            text_encoder=None,
            validator=validator,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        with pytest.raises(RuntimeError, match="文本編碼器"):
            preprocessor.encode_prompt(["a cat"], None, num_images_per_prompt=1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
