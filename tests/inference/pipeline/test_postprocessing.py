"""
Pipeline Postprocessing Unit Tests.

Validates output processing and format conversion:
- Tensor to PIL conversion
- Tensor to numpy conversion
- Value clamping
- PipelineOutput creation

Test Count: 15 test cases

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-08
"""

import pytest
import torch
import numpy as np
from PIL import Image

from src.inference.pipeline.postprocessing import Postprocessor, PipelineOutput


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def postprocessor() -> Postprocessor:
    """Create postprocessor instance."""
    return Postprocessor()


@pytest.fixture
def sample_images() -> torch.Tensor:
    """Create sample image tensor in [-1, 1] range."""
    # (B, H, W, 3) format
    return torch.randn(2, 256, 256, 3).clamp(-1, 1)


# =============================================================================
# Test Class: PIL Conversion
# =============================================================================


class TestPILConversion:
    """Test tensor to PIL conversion."""

    def test_to_pil_returns_list(self, postprocessor: Postprocessor, sample_images: torch.Tensor):
        """Test PIL output returns list."""
        result = postprocessor.process(sample_images, output_type="pil")
        assert isinstance(result, list)

    def test_to_pil_correct_count(self, postprocessor: Postprocessor, sample_images: torch.Tensor):
        """Test PIL output has correct count."""
        result = postprocessor.process(sample_images, output_type="pil")
        assert len(result) == 2

    def test_to_pil_image_type(self, postprocessor: Postprocessor, sample_images: torch.Tensor):
        """Test each item is PIL Image."""
        result = postprocessor.process(sample_images, output_type="pil")
        for img in result:
            assert isinstance(img, Image.Image)

    def test_to_pil_image_size(self, postprocessor: Postprocessor, sample_images: torch.Tensor):
        """Test PIL images have correct size."""
        result = postprocessor.process(sample_images, output_type="pil")
        for img in result:
            assert img.size == (256, 256)

    def test_to_pil_image_mode(self, postprocessor: Postprocessor, sample_images: torch.Tensor):
        """Test PIL images are RGB."""
        result = postprocessor.process(sample_images, output_type="pil")
        for img in result:
            assert img.mode == "RGB"


# =============================================================================
# Test Class: Numpy Conversion
# =============================================================================


class TestNumpyConversion:
    """Test tensor to numpy conversion."""

    def test_to_numpy_returns_ndarray(self, postprocessor: Postprocessor, sample_images: torch.Tensor):
        """Test numpy output returns ndarray."""
        result = postprocessor.process(sample_images, output_type="numpy")
        assert isinstance(result, np.ndarray)

    def test_to_numpy_shape(self, postprocessor: Postprocessor, sample_images: torch.Tensor):
        """Test numpy output has correct shape."""
        result = postprocessor.process(sample_images, output_type="numpy")
        assert result.shape == (2, 256, 256, 3)

    def test_to_numpy_range(self, postprocessor: Postprocessor, sample_images: torch.Tensor):
        """Test numpy output is in [0, 1] range."""
        result = postprocessor.process(sample_images, output_type="numpy")
        assert result.min() >= 0.0
        assert result.max() <= 1.0


# =============================================================================
# Test Class: Tensor Output
# =============================================================================


class TestTensorOutput:
    """Test tensor output mode."""

    def test_to_tensor_returns_tensor(self, postprocessor: Postprocessor, sample_images: torch.Tensor):
        """Test tensor output returns tensor."""
        result = postprocessor.process(sample_images, output_type="tensor")
        assert isinstance(result, torch.Tensor)

    def test_to_tensor_shape_preserved(self, postprocessor: Postprocessor, sample_images: torch.Tensor):
        """Test tensor output shape is preserved."""
        result = postprocessor.process(sample_images, output_type="tensor")
        assert result.shape == sample_images.shape

    def test_to_tensor_clamped(self, postprocessor: Postprocessor):
        """Test tensor output is clamped."""
        images = torch.randn(2, 256, 256, 3) * 10  # Out of range
        result = postprocessor.process(images, output_type="tensor")
        assert result.min() >= -1.0
        assert result.max() <= 1.0


# =============================================================================
# Test Class: Value Clamping
# =============================================================================


class TestValueClamping:
    """Test value clamping for extreme values."""

    def test_clamp_high_values(self, postprocessor: Postprocessor):
        """Test high values are clamped."""
        images = torch.ones(1, 64, 64, 3) * 5.0
        result = postprocessor.process(images, output_type="tensor")
        assert result.max() == 1.0

    def test_clamp_low_values(self, postprocessor: Postprocessor):
        """Test low values are clamped."""
        images = torch.ones(1, 64, 64, 3) * -5.0
        result = postprocessor.process(images, output_type="tensor")
        assert result.min() == -1.0

    def test_clamp_mixed_values(self, postprocessor: Postprocessor):
        """Test mixed values are clamped correctly."""
        images = torch.tensor([[[[3.0, -3.0, 0.5]]]])  # (1, 1, 1, 3)
        result = postprocessor.process(images, output_type="tensor")
        expected = torch.tensor([[[[1.0, -1.0, 0.5]]]])
        assert torch.allclose(result, expected)


# =============================================================================
# Test Class: PipelineOutput
# =============================================================================


class TestPipelineOutput:
    """Test PipelineOutput dataclass."""

    def test_create_output_basic(self, postprocessor: Postprocessor):
        """Test basic PipelineOutput creation."""
        images = [Image.new("RGB", (64, 64))]
        output = postprocessor.create_output(images=images)

        assert isinstance(output, PipelineOutput)
        assert output.images == images
        assert output.latents is None
        assert output.intermediates is None
        assert output.metadata == {}

    def test_create_output_with_latents(self, postprocessor: Postprocessor):
        """Test PipelineOutput with latents."""
        images = [Image.new("RGB", (64, 64))]
        latents = torch.randn(1, 64, 64, 3)

        output = postprocessor.create_output(images=images, latents=latents)

        assert output.latents is latents

    def test_create_output_with_metadata(self, postprocessor: Postprocessor):
        """Test PipelineOutput with metadata."""
        images = [Image.new("RGB", (64, 64))]
        metadata = {"prompt": "a cat", "seed": 42}

        output = postprocessor.create_output(images=images, metadata=metadata)

        assert output.metadata == metadata

    def test_create_output_with_intermediates(self, postprocessor: Postprocessor):
        """Test PipelineOutput with intermediates."""
        images = [Image.new("RGB", (64, 64))]
        intermediates = torch.randn(10, 1, 64, 64, 3)

        output = postprocessor.create_output(images=images, intermediates=intermediates)

        assert output.intermediates is intermediates


# =============================================================================
# Test Class: Dtype Handling
# =============================================================================


class TestDtypeHandling:
    """Test different input dtypes."""

    def test_float16_input(self, postprocessor: Postprocessor):
        """Test float16 input is handled."""
        images = torch.randn(1, 64, 64, 3, dtype=torch.float16).clamp(-1, 1)
        result = postprocessor.process(images, output_type="pil")
        assert len(result) == 1

    def test_float32_input(self, postprocessor: Postprocessor):
        """Test float32 input is handled."""
        images = torch.randn(1, 64, 64, 3, dtype=torch.float32).clamp(-1, 1)
        result = postprocessor.process(images, output_type="pil")
        assert len(result) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
