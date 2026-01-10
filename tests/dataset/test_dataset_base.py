"""
Dataset Base Module Tests

Tests for BaseImageTextDataset and utility functions.

Test Categories:
    - IMAGE_EXTENSIONS (2 tests): Constant validation
    - _load_caption (4 tests): Caption loading
    - _load_image (4 tests): Image loading with mocks
    - _apply_random_flip (2 tests): Random flip logic
    - _image_to_tensor (3 tests): Tensor conversion
    - _apply_caption_dropout (3 tests): Dropout logic
    - find_images (3 tests): Directory scanning

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from PIL import Image
import tempfile
import shutil
import os

from src.training.dataset.base import (
    IMAGE_EXTENSIONS,
    BaseImageTextDataset,
    find_images,
)


# ============================================================================
# Test Fixtures
# ============================================================================

class ConcreteDataset(BaseImageTextDataset):
    """Concrete implementation for testing abstract base class."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.items = []

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    tmp_dir = tempfile.mkdtemp()
    yield Path(tmp_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def dataset():
    """Create concrete dataset instance for testing."""
    return ConcreteDataset(
        caption_dropout=0.1,
        default_caption="default",
        use_random_crop=True,
        use_random_flip=True,
    )


@pytest.fixture
def sample_pil_image():
    """Create sample PIL image for testing."""
    return Image.new("RGB", (256, 256), color=(128, 128, 128))


# ============================================================================
# IMAGE_EXTENSIONS Tests
# ============================================================================

class TestImageExtensions:
    """Tests for IMAGE_EXTENSIONS constant."""

    def test_extensions_is_set(self):
        """Test IMAGE_EXTENSIONS is a set."""
        assert isinstance(IMAGE_EXTENSIONS, set)

    def test_common_extensions_included(self):
        """Test common image extensions are included."""
        common = {".jpg", ".jpeg", ".png", ".webp"}
        for ext in common:
            assert ext in IMAGE_EXTENSIONS, f"{ext} should be in IMAGE_EXTENSIONS"


# ============================================================================
# _load_caption Tests
# ============================================================================

class TestLoadCaption:
    """Tests for _load_caption method."""

    def test_load_caption_existing_file(self, dataset, temp_dir):
        """Test loading caption from existing file."""
        # Create image and caption files
        image_path = temp_dir / "test.png"
        caption_path = temp_dir / "test.txt"

        image_path.touch()
        caption_path.write_text("Test caption content", encoding="utf-8")

        caption = dataset._load_caption(image_path)

        assert caption == "Test caption content"

    def test_load_caption_missing_file(self, dataset, temp_dir):
        """Test default caption when file is missing."""
        image_path = temp_dir / "test.png"
        image_path.touch()

        caption = dataset._load_caption(image_path)

        assert caption == "default"

    def test_load_caption_empty_file(self, dataset, temp_dir):
        """Test default caption when file is empty."""
        image_path = temp_dir / "test.png"
        caption_path = temp_dir / "test.txt"

        image_path.touch()
        caption_path.write_text("", encoding="utf-8")

        caption = dataset._load_caption(image_path)

        assert caption == "default"

    def test_load_caption_strips_whitespace(self, dataset, temp_dir):
        """Test caption whitespace is stripped."""
        image_path = temp_dir / "test.png"
        caption_path = temp_dir / "test.txt"

        image_path.touch()
        caption_path.write_text("  caption with spaces  \n", encoding="utf-8")

        caption = dataset._load_caption(image_path)

        assert caption == "caption with spaces"


# ============================================================================
# _load_image Tests
# ============================================================================

class TestLoadImage:
    """Tests for _load_image method."""

    def test_load_valid_image(self, dataset, temp_dir):
        """Test loading valid image."""
        image_path = temp_dir / "test.png"

        # Create actual image
        img = Image.new("RGB", (100, 100), color="red")
        img.save(image_path)

        loaded = dataset._load_image(image_path)

        assert loaded is not None
        assert isinstance(loaded, Image.Image)
        assert loaded.mode == "RGB"
        assert loaded.size == (100, 100)

    def test_load_image_converts_to_rgb(self, dataset, temp_dir):
        """Test RGBA image is converted to RGB."""
        image_path = temp_dir / "test.png"

        # Create RGBA image
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        img.save(image_path)

        loaded = dataset._load_image(image_path)

        assert loaded is not None
        assert loaded.mode == "RGB"

    def test_load_image_invalid_returns_none(self, dataset, temp_dir):
        """Test invalid image returns None."""
        image_path = temp_dir / "test.png"

        # Create invalid file
        image_path.write_text("not an image")

        loaded = dataset._load_image(image_path)

        assert loaded is None

    def test_load_image_nonexistent_returns_none(self, dataset, temp_dir):
        """Test nonexistent file returns None."""
        image_path = temp_dir / "nonexistent.png"

        loaded = dataset._load_image(image_path)

        assert loaded is None


# ============================================================================
# _apply_random_flip Tests
# ============================================================================

class TestApplyRandomFlip:
    """Tests for _apply_random_flip method."""

    def test_flip_disabled(self, sample_pil_image):
        """Test flip is not applied when disabled."""
        dataset = ConcreteDataset(use_random_flip=False)

        # Get original pixels
        original_pixels = list(sample_pil_image.getdata())

        result = dataset._apply_random_flip(sample_pil_image)
        result_pixels = list(result.getdata())

        # Should be identical
        assert original_pixels == result_pixels

    def test_flip_enabled_probabilistic(self, sample_pil_image):
        """Test flip is applied probabilistically when enabled."""
        dataset = ConcreteDataset(use_random_flip=True)

        # Run multiple times to check probabilistic behavior
        results = []
        for _ in range(100):
            # Create fresh image each time
            img = Image.new("RGB", (10, 10), color="red")
            img.putpixel((0, 0), (0, 255, 0))  # Mark top-left green

            result = dataset._apply_random_flip(img)
            # Check if flipped by looking at top-right pixel
            results.append(result.getpixel((9, 0)) == (0, 255, 0))

        # Should have some flipped and some not
        num_flipped = sum(results)
        assert 0 < num_flipped < 100, "Flip should be probabilistic"


# ============================================================================
# _image_to_tensor Tests
# ============================================================================

class TestImageToTensor:
    """Tests for _image_to_tensor method."""

    def test_tensor_shape(self, dataset, sample_pil_image):
        """Test output tensor shape is [C, H, W]."""
        tensor = dataset._image_to_tensor(sample_pil_image)

        assert tensor.shape == (3, 256, 256)

    def test_tensor_range(self, dataset, sample_pil_image):
        """Test output tensor values are in [-1, 1]."""
        tensor = dataset._image_to_tensor(sample_pil_image)

        assert tensor.min() >= -1.0
        assert tensor.max() <= 1.0

    def test_tensor_dtype(self, dataset, sample_pil_image):
        """Test output tensor dtype is float32."""
        tensor = dataset._image_to_tensor(sample_pil_image)

        assert tensor.dtype == torch.float32


# ============================================================================
# _apply_caption_dropout Tests
# ============================================================================

class TestApplyCaptionDropout:
    """Tests for _apply_caption_dropout method."""

    def test_dropout_zero(self):
        """Test no dropout when rate is 0."""
        dataset = ConcreteDataset(caption_dropout=0.0)

        for _ in range(100):
            result = dataset._apply_caption_dropout("test caption")
            assert result == "test caption"

    def test_dropout_one(self):
        """Test always dropout when rate is 1."""
        dataset = ConcreteDataset(caption_dropout=1.0)

        for _ in range(100):
            result = dataset._apply_caption_dropout("test caption")
            assert result == ""

    def test_dropout_probabilistic(self):
        """Test dropout is probabilistic."""
        dataset = ConcreteDataset(caption_dropout=0.5)

        results = [dataset._apply_caption_dropout("test") for _ in range(100)]
        empty_count = sum(1 for r in results if r == "")

        # Should have some empty and some not
        assert 10 < empty_count < 90, "Dropout should be probabilistic"


# ============================================================================
# find_images Tests
# ============================================================================

class TestFindImages:
    """Tests for find_images function."""

    def test_find_images_empty_dir(self, temp_dir):
        """Test finding images in empty directory."""
        images = find_images(temp_dir)

        assert len(images) == 0

    def test_find_images_with_images(self, temp_dir):
        """Test finding images in directory with images."""
        # Create some image files
        (temp_dir / "image1.png").touch()
        (temp_dir / "image2.jpg").touch()
        (temp_dir / "image3.JPEG").touch()
        (temp_dir / "not_image.txt").touch()

        images = find_images(temp_dir)

        assert len(images) == 3
        assert all(p.suffix.lower() in IMAGE_EXTENSIONS for p in images)

    def test_find_images_recursive(self, temp_dir):
        """Test finding images recursively."""
        # Create nested directory
        subdir = temp_dir / "subdir"
        subdir.mkdir()

        (temp_dir / "root.png").touch()
        (subdir / "nested.png").touch()

        images = find_images(temp_dir)

        assert len(images) == 2


# ============================================================================
# BaseImageTextDataset Abstract Methods Tests
# ============================================================================

class TestBaseImageTextDatasetInit:
    """Tests for BaseImageTextDataset initialization."""

    def test_init_default_values(self):
        """Test default initialization values."""
        dataset = ConcreteDataset()

        assert dataset.caption_dropout == 0.1
        assert dataset.default_caption == ""
        assert dataset.use_random_crop is True
        assert dataset.use_random_flip is True

    def test_init_custom_values(self):
        """Test custom initialization values."""
        dataset = ConcreteDataset(
            caption_dropout=0.5,
            default_caption="custom default",
            use_random_crop=False,
            use_random_flip=False,
        )

        assert dataset.caption_dropout == 0.5
        assert dataset.default_caption == "custom default"
        assert dataset.use_random_crop is False
        assert dataset.use_random_flip is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
