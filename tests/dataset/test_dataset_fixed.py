"""
Fixed Resolution Dataset Tests

Tests for ImageTextDataset and collate_fn.

Test Categories:
    - ImageTextDataset Init (5 tests): Initialization validation
    - ImageTextDataset __len__ (1 test): Length method
    - ImageTextDataset __getitem__ (5 tests): Sample retrieval
    - _preprocess_image (4 tests): Image preprocessing
    - collate_fn (4 tests): Batch collation

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile
import shutil

from src.training.dataset.fixed import (
    ImageTextDataset,
    collate_fn,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    tmp_dir = tempfile.mkdtemp()
    yield Path(tmp_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def temp_dir_with_images(temp_dir):
    """Create temporary directory with test images."""
    # Create valid images
    for i in range(5):
        img = Image.new("RGB", (512, 512), color=(i * 50, i * 50, i * 50))
        img.save(temp_dir / f"image_{i}.png")
        (temp_dir / f"image_{i}.txt").write_text(f"Caption for image {i}", encoding="utf-8")

    return temp_dir


@pytest.fixture
def temp_dir_with_various_sizes(temp_dir):
    """Create temporary directory with images of various sizes."""
    sizes = [
        (256, 256),
        (512, 512),
        (1024, 768),
        (768, 1024),
        (100, 100),  # Too small
    ]

    for i, (w, h) in enumerate(sizes):
        img = Image.new("RGB", (w, h), color=(i * 50, i * 50, i * 50))
        img.save(temp_dir / f"image_{i}.png")
        (temp_dir / f"image_{i}.txt").write_text(f"Caption {i}", encoding="utf-8")

    return temp_dir


# ============================================================================
# ImageTextDataset Initialization Tests
# ============================================================================

class TestImageTextDatasetInit:
    """Tests for ImageTextDataset initialization."""

    def test_init_basic(self, temp_dir_with_images):
        """Test basic initialization."""
        dataset = ImageTextDataset(
            root_dir=temp_dir_with_images,
            target_resolution=256,
            patch_size=16,
        )

        assert dataset.root_dir == temp_dir_with_images
        assert dataset.target_resolution == 256
        assert dataset.patch_size == 16
        assert len(dataset) == 5

    def test_init_custom_params(self, temp_dir_with_images):
        """Test initialization with custom parameters."""
        dataset = ImageTextDataset(
            root_dir=temp_dir_with_images,
            target_resolution=512,
            patch_size=16,
            max_resolution=1024,
            min_resolution=256,
            caption_dropout=0.2,
            default_caption="custom default",
            use_random_crop=False,
            use_random_flip=False,
        )

        assert dataset.target_resolution == 512
        assert dataset.max_resolution == 1024
        assert dataset.min_resolution == 256
        assert dataset.caption_dropout == 0.2
        assert dataset.default_caption == "custom default"
        assert dataset.use_random_crop is False
        assert dataset.use_random_flip is False

    def test_init_invalid_resolution_raises(self, temp_dir_with_images):
        """Test invalid resolution (not multiple of patch_size) raises error."""
        with pytest.raises(ValueError, match="multiple of patch_size"):
            ImageTextDataset(
                root_dir=temp_dir_with_images,
                target_resolution=250,  # Not multiple of 16
                patch_size=16,
            )

    def test_init_empty_dir_raises(self, temp_dir):
        """Test empty directory raises error."""
        with pytest.raises(RuntimeError, match="No images found"):
            ImageTextDataset(root_dir=temp_dir, target_resolution=256)

    def test_init_nonexistent_dir_raises(self, temp_dir):
        """Test nonexistent directory raises error."""
        with pytest.raises(RuntimeError, match="No images found"):
            ImageTextDataset(
                root_dir=temp_dir / "nonexistent",
                target_resolution=256,
            )


# ============================================================================
# ImageTextDataset __len__ Tests
# ============================================================================

class TestImageTextDatasetLen:
    """Tests for ImageTextDataset __len__."""

    def test_len_returns_correct_count(self, temp_dir_with_images):
        """Test __len__ returns correct image count."""
        dataset = ImageTextDataset(
            root_dir=temp_dir_with_images,
            target_resolution=256,
        )

        assert len(dataset) == 5


# ============================================================================
# ImageTextDataset __getitem__ Tests
# ============================================================================

class TestImageTextDatasetGetItem:
    """Tests for ImageTextDataset __getitem__."""

    def test_getitem_returns_dict(self, temp_dir_with_images):
        """Test __getitem__ returns dict with correct keys."""
        dataset = ImageTextDataset(
            root_dir=temp_dir_with_images,
            target_resolution=256,
            caption_dropout=0.0,  # Disable dropout for consistent testing
        )

        sample = dataset[0]

        assert isinstance(sample, dict)
        assert "image" in sample
        assert "caption" in sample
        assert "image_path" in sample

    def test_getitem_image_shape(self, temp_dir_with_images):
        """Test __getitem__ returns correct image shape."""
        dataset = ImageTextDataset(
            root_dir=temp_dir_with_images,
            target_resolution=256,
        )

        sample = dataset[0]

        assert sample["image"].shape == (3, 256, 256)
        assert sample["image"].dtype == torch.float32

    def test_getitem_image_range(self, temp_dir_with_images):
        """Test __getitem__ returns image in [-1, 1] range."""
        dataset = ImageTextDataset(
            root_dir=temp_dir_with_images,
            target_resolution=256,
        )

        sample = dataset[0]

        assert sample["image"].min() >= -1.0
        assert sample["image"].max() <= 1.0

    def test_getitem_caption(self, temp_dir_with_images):
        """Test __getitem__ returns caption."""
        dataset = ImageTextDataset(
            root_dir=temp_dir_with_images,
            target_resolution=256,
            caption_dropout=0.0,
        )

        sample = dataset[0]

        assert isinstance(sample["caption"], str)
        assert "Caption" in sample["caption"]

    def test_getitem_path(self, temp_dir_with_images):
        """Test __getitem__ returns correct path type."""
        dataset = ImageTextDataset(
            root_dir=temp_dir_with_images,
            target_resolution=256,
        )

        sample = dataset[0]

        assert isinstance(sample["image_path"], Path)


# ============================================================================
# _preprocess_image Tests
# ============================================================================

class TestPreprocessImage:
    """Tests for _preprocess_image method."""

    def test_preprocess_skips_small_image(self, temp_dir_with_images):
        """Test preprocessing skips images smaller than target resolution."""
        dataset = ImageTextDataset(
            root_dir=temp_dir_with_images,
            target_resolution=256,
        )

        small_img = Image.new("RGB", (100, 100), color="red")
        result = dataset._preprocess_image(small_img)

        assert result is None

    def test_preprocess_scales_large_image(self, temp_dir_with_images):
        """Test preprocessing scales down large images."""
        dataset = ImageTextDataset(
            root_dir=temp_dir_with_images,
            target_resolution=256,
            max_resolution=512,
        )

        large_img = Image.new("RGB", (2048, 2048), color="blue")
        result = dataset._preprocess_image(large_img)

        assert result is not None
        assert result.shape == (3, 256, 256)

    def test_preprocess_crops_to_target(self, temp_dir_with_images):
        """Test preprocessing crops to target resolution."""
        dataset = ImageTextDataset(
            root_dir=temp_dir_with_images,
            target_resolution=256,
            use_random_crop=False,  # Use center crop
        )

        img = Image.new("RGB", (512, 512), color="green")
        result = dataset._preprocess_image(img)

        assert result is not None
        assert result.shape == (3, 256, 256)

    def test_preprocess_output_dtype(self, temp_dir_with_images):
        """Test preprocessing outputs float32 tensor."""
        dataset = ImageTextDataset(
            root_dir=temp_dir_with_images,
            target_resolution=256,
        )

        img = Image.new("RGB", (512, 512), color="yellow")
        result = dataset._preprocess_image(img)

        assert result.dtype == torch.float32


# ============================================================================
# collate_fn Tests
# ============================================================================

class TestCollateFn:
    """Tests for collate_fn function."""

    def test_collate_valid_batch(self):
        """Test collating valid batch."""
        batch = [
            {
                "image": torch.randn(3, 256, 256),
                "caption": "Caption 1",
                "image_path": Path("/test/1.png"),
            },
            {
                "image": torch.randn(3, 256, 256),
                "caption": "Caption 2",
                "image_path": Path("/test/2.png"),
            },
        ]

        result = collate_fn(batch)

        assert "images" in result
        assert "captions" in result
        assert "image_paths" in result
        assert result["images"].shape == (2, 3, 256, 256)
        assert len(result["captions"]) == 2
        assert len(result["image_paths"]) == 2

    def test_collate_empty_batch_raises(self):
        """Test collating empty batch raises error."""
        with pytest.raises(ValueError, match="Empty batch"):
            collate_fn([])

    def test_collate_inconsistent_shapes_raises(self):
        """Test collating batch with inconsistent shapes raises error."""
        batch = [
            {
                "image": torch.randn(3, 256, 256),
                "caption": "Caption 1",
                "image_path": Path("/test/1.png"),
            },
            {
                "image": torch.randn(3, 512, 512),  # Different shape
                "caption": "Caption 2",
                "image_path": Path("/test/2.png"),
            },
        ]

        with pytest.raises(ValueError, match="Inconsistent image shapes"):
            collate_fn(batch)

    def test_collate_preserves_captions(self):
        """Test collating preserves caption content."""
        batch = [
            {
                "image": torch.randn(3, 256, 256),
                "caption": "First caption",
                "image_path": Path("/test/1.png"),
            },
            {
                "image": torch.randn(3, 256, 256),
                "caption": "Second caption",
                "image_path": Path("/test/2.png"),
            },
        ]

        result = collate_fn(batch)

        assert result["captions"] == ["First caption", "Second caption"]


# ============================================================================
# Integration Tests
# ============================================================================

class TestImageTextDatasetIntegration:
    """Integration tests for ImageTextDataset."""

    def test_full_pipeline(self, temp_dir_with_images):
        """Test full pipeline from dataset to batch."""
        dataset = ImageTextDataset(
            root_dir=temp_dir_with_images,
            target_resolution=256,
            caption_dropout=0.0,
        )

        # Get multiple samples
        samples = [dataset[i] for i in range(3)]

        # Collate into batch
        batch = collate_fn(samples)

        assert batch["images"].shape == (3, 3, 256, 256)
        assert len(batch["captions"]) == 3
        assert len(batch["image_paths"]) == 3

    def test_different_resolutions(self, temp_dir_with_images):
        """Test with different target resolutions."""
        for resolution in [256, 512]:
            dataset = ImageTextDataset(
                root_dir=temp_dir_with_images,
                target_resolution=resolution,
            )

            sample = dataset[0]
            assert sample["image"].shape == (3, resolution, resolution)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
