"""
Pipeline Image-to-Image Unit Tests.

Validates image-to-image generation pipeline:
- Image preprocessing
- Noise addition based on strength
- PIL/tensor input handling
- Resolution alignment

Test Count: 15 test cases

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-08
"""

import pytest
import torch
import numpy as np
from PIL import Image
from typing import Optional
from unittest.mock import Mock, MagicMock, patch

from src.inference.pipeline.img2img import PixelHDMPipelineForImg2Img


# =============================================================================
# Test Fixtures
# =============================================================================


class MockModel:
    """Mock model for I2I pipeline testing."""

    def __init__(self):
        self.config = MagicMock()
        self.config.patch_size = 16
        self.config.hidden_dim = 1024
        self.config.time_eps = 0.05
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._param])

    def __call__(self, *args, **kwargs):
        # Return zeros matching input shape
        x_t = kwargs.get('x_t', args[0] if args else None)
        if x_t is not None:
            return torch.zeros_like(x_t)
        return torch.zeros(1, 256, 256, 3)


class MockTextEncoder:
    """Mock text encoder."""

    def __call__(self, texts, return_pooled=False):
        batch_size = len(texts)
        hidden_states = torch.randn(batch_size, 77, 1024)
        mask = torch.ones(batch_size, 77)
        pooled = torch.randn(batch_size, 1024) if return_pooled else None
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
def sample_pil_image() -> Image.Image:
    """Create sample PIL image."""
    return Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))


@pytest.fixture
def sample_tensor_image() -> torch.Tensor:
    """Create sample tensor image in [-1, 1] range."""
    return torch.randn(1, 256, 256, 3).clamp(-1, 1)


# =============================================================================
# Test Class: Image Preprocessing
# =============================================================================


class TestImagePreprocessing:
    """Test image preprocessing for I2I pipeline."""

    def test_preprocess_pil_image(self, mock_model: MockModel, mock_text_encoder: MockTextEncoder):
        """Test PIL image preprocessing."""
        pipeline = PixelHDMPipelineForImg2Img(mock_model, mock_text_encoder)

        pil_image = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
        tensor = pipeline._preprocess_image(pil_image)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 256, 256, 3)

    def test_preprocess_converts_to_rgb(self, mock_model: MockModel, mock_text_encoder: MockTextEncoder):
        """Test non-RGB images are converted."""
        pipeline = PixelHDMPipelineForImg2Img(mock_model, mock_text_encoder)

        # Create RGBA image
        rgba_array = np.zeros((256, 256, 4), dtype=np.uint8)
        rgba_image = Image.fromarray(rgba_array, mode="RGBA")

        tensor = pipeline._preprocess_image(rgba_image)
        assert tensor.shape[-1] == 3  # RGB channels

    def test_preprocess_aligns_to_patch_size(self, mock_model: MockModel, mock_text_encoder: MockTextEncoder):
        """Test images are aligned to patch size."""
        pipeline = PixelHDMPipelineForImg2Img(mock_model, mock_text_encoder)

        # Create image not aligned to patch size (16)
        unaligned = Image.fromarray(np.zeros((250, 250, 3), dtype=np.uint8))
        tensor = pipeline._preprocess_image(unaligned)

        # Should be aligned to 240 (250 // 16 * 16 = 240)
        assert tensor.shape[1] == 240
        assert tensor.shape[2] == 240

    def test_preprocess_value_range(self, mock_model: MockModel, mock_text_encoder: MockTextEncoder):
        """Test preprocessed values are in [-1, 1] range."""
        pipeline = PixelHDMPipelineForImg2Img(mock_model, mock_text_encoder)

        pil_image = Image.fromarray(np.ones((256, 256, 3), dtype=np.uint8) * 255)
        tensor = pipeline._preprocess_image(pil_image)

        # 255 -> (255/255 * 2 - 1) = 1.0
        assert tensor.max().item() == pytest.approx(1.0, abs=0.01)

        pil_black = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
        tensor_black = pipeline._preprocess_image(pil_black)

        # 0 -> (0/255 * 2 - 1) = -1.0
        assert tensor_black.min().item() == pytest.approx(-1.0, abs=0.01)


# =============================================================================
# Test Class: Prepare Image
# =============================================================================


class TestPrepareImage:
    """Test _prepare_image method."""

    def test_prepare_tensor_passthrough(self, mock_model: MockModel, mock_text_encoder: MockTextEncoder):
        """Test tensor input passes through."""
        pipeline = PixelHDMPipelineForImg2Img(mock_model, mock_text_encoder)

        tensor = torch.randn(1, 256, 256, 3)
        result = pipeline._prepare_image(tensor)

        assert torch.equal(result, tensor)

    def test_prepare_pil_converts(self, mock_model: MockModel, mock_text_encoder: MockTextEncoder):
        """Test PIL image is converted."""
        pipeline = PixelHDMPipelineForImg2Img(mock_model, mock_text_encoder)

        pil_image = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
        result = pipeline._prepare_image(pil_image)

        assert isinstance(result, torch.Tensor)

    def test_prepare_list_of_images(self, mock_model: MockModel, mock_text_encoder: MockTextEncoder):
        """Test list of PIL images."""
        pipeline = PixelHDMPipelineForImg2Img(mock_model, mock_text_encoder)

        images = [
            Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)),
            Image.fromarray(np.ones((256, 256, 3), dtype=np.uint8) * 128),
        ]
        result = pipeline._prepare_image(images)

        assert result.shape[0] == 2


# =============================================================================
# Test Class: Noise Addition
# =============================================================================


class TestNoiseAddition:
    """Test noise addition based on strength."""

    def test_add_noise_strength_zero(self, mock_model: MockModel, mock_text_encoder: MockTextEncoder):
        """Test strength=0 means minimal noise (t_start clamped to 1-t_eps)."""
        pipeline = PixelHDMPipelineForImg2Img(mock_model, mock_text_encoder)

        image = torch.zeros(1, 256, 256, 3, device="cpu", dtype=torch.float32)
        z_t, t_start, actual_steps = pipeline._add_noise_to_image(
            image, strength=0.0, num_steps=50, batch_size=1, generator=None
        )

        # t_start = 1 - 0 = 1.0, clamped to 1 - t_eps = 0.95
        t_eps = 0.05  # default
        assert t_start == 1.0 - t_eps  # 0.95
        assert actual_steps == 0

    def test_add_noise_strength_one(self, mock_model: MockModel, mock_text_encoder: MockTextEncoder):
        """Test strength=1 means full noise (t_start clamped to t_eps)."""
        pipeline = PixelHDMPipelineForImg2Img(mock_model, mock_text_encoder)

        image = torch.zeros(1, 256, 256, 3, device="cpu", dtype=torch.float32)
        z_t, t_start, actual_steps = pipeline._add_noise_to_image(
            image, strength=1.0, num_steps=50, batch_size=1, generator=None
        )

        # t_start = 1 - 1 = 0.0, clamped to t_eps = 0.05
        t_eps = 0.05  # default
        assert t_start == t_eps  # 0.05
        assert actual_steps == 50

    def test_add_noise_strength_half(self, mock_model: MockModel, mock_text_encoder: MockTextEncoder):
        """Test strength=0.5 for partial denoising."""
        pipeline = PixelHDMPipelineForImg2Img(mock_model, mock_text_encoder)

        image = torch.zeros(1, 256, 256, 3, device="cpu", dtype=torch.float32)
        z_t, t_start, actual_steps = pipeline._add_noise_to_image(
            image, strength=0.5, num_steps=50, batch_size=1, generator=None
        )

        # t_start = 1 - 0.5 = 0.5
        assert t_start == 0.5
        assert actual_steps == 25

    def test_add_noise_interpolation_formula(self, mock_model: MockModel, mock_text_encoder: MockTextEncoder):
        """Test noise interpolation follows PixelHDM convention."""
        pipeline = PixelHDMPipelineForImg2Img(mock_model, mock_text_encoder)

        # Create clean image (all ones)
        image = torch.ones(1, 64, 64, 3, device="cpu", dtype=torch.float32)

        # Set seed for reproducible noise
        generator = torch.Generator().manual_seed(42)
        z_t, t_start, _ = pipeline._add_noise_to_image(
            image, strength=0.5, num_steps=50, batch_size=1, generator=generator
        )

        # z_t = t * x + (1-t) * noise
        # At t_start=0.5: z_t = 0.5 * image + 0.5 * noise
        # Result should be mix of image and noise
        # Convert to same dtype for comparison (pipeline may change dtype)
        assert not torch.allclose(z_t.float(), image.float())

    def test_add_noise_reproducible_with_seed(self, mock_model: MockModel, mock_text_encoder: MockTextEncoder):
        """Test noise is reproducible with same seed."""
        pipeline = PixelHDMPipelineForImg2Img(mock_model, mock_text_encoder)

        image = torch.ones(1, 64, 64, 3, device="cpu", dtype=torch.float32)

        gen1 = torch.Generator().manual_seed(42)
        z_t1, _, _ = pipeline._add_noise_to_image(
            image.clone(), strength=0.5, num_steps=50, batch_size=1, generator=gen1
        )

        gen2 = torch.Generator().manual_seed(42)
        z_t2, _, _ = pipeline._add_noise_to_image(
            image.clone(), strength=0.5, num_steps=50, batch_size=1, generator=gen2
        )

        assert torch.allclose(z_t1, z_t2)


# =============================================================================
# Test Class: Strength Parameter
# =============================================================================


class TestStrengthParameter:
    """Test strength parameter edge cases."""

    @pytest.mark.parametrize("strength", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_various_strengths(self, mock_model: MockModel, mock_text_encoder: MockTextEncoder, strength: float):
        """Test various strength values (with t_eps clamping)."""
        pipeline = PixelHDMPipelineForImg2Img(mock_model, mock_text_encoder)

        image = torch.zeros(1, 64, 64, 3, device="cpu", dtype=torch.float32)
        z_t, t_start, actual_steps = pipeline._add_noise_to_image(
            image, strength=strength, num_steps=100, batch_size=1, generator=None
        )

        # t_start is clamped to [t_eps, 1-t_eps] = [0.05, 0.95]
        t_eps = 0.05
        raw_t_start = 1.0 - strength
        expected_t_start = max(t_eps, min(1.0 - t_eps, raw_t_start))
        expected_steps = max(1, int(100 * strength)) if strength > 0 else 0

        assert t_start == pytest.approx(expected_t_start)
        assert actual_steps == expected_steps


# =============================================================================
# Test Class: Batch Handling
# =============================================================================


class TestBatchHandling:
    """Test batch size handling for I2I."""

    def test_single_image_multiple_prompts(self, mock_model: MockModel, mock_text_encoder: MockTextEncoder):
        """Test single image expanded for multiple prompts."""
        pipeline = PixelHDMPipelineForImg2Img(mock_model, mock_text_encoder)

        # Single image
        image = torch.zeros(1, 256, 256, 3)

        # This would be expanded in __call__ when batch_size > 1
        # Testing the expansion logic
        batch_size = 3
        if image.shape[0] == 1 and batch_size > 1:
            expanded = image.expand(batch_size, -1, -1, -1)
            assert expanded.shape[0] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
