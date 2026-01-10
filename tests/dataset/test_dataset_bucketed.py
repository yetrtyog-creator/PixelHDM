"""
Bucketed Dataset Tests

Tests for BucketImageTextDataset and bucket_collate_fn.

Test Categories:
    - BucketImageTextDataset Init (5 tests): Initialization and serialization
    - BucketImageTextDataset Serialization (4 tests): Path/bucket_id serialization
    - BucketImageTextDataset __getitem__ (5 tests): Sample retrieval
    - _preprocess_image (3 tests): Image preprocessing for buckets
    - bucket_collate_fn (5 tests): Batch collation

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock
from PIL import Image
import tempfile
import shutil

from src.training.dataset.bucketed import (
    BucketImageTextDataset,
    bucket_collate_fn,
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
    for i in range(5):
        img = Image.new("RGB", (512, 512), color=(i * 50, i * 50, i * 50))
        img.save(temp_dir / f"image_{i}.png")
        (temp_dir / f"image_{i}.txt").write_text(f"Caption for image {i}", encoding="utf-8")

    return temp_dir


@pytest.fixture
def mock_bucket_manager():
    """Create mock AspectRatioBucket."""
    mock = MagicMock()
    mock.get_bucket_resolution.return_value = (256, 256)
    return mock


@pytest.fixture
def sample_paths(temp_dir_with_images):
    """Create sample paths list."""
    return [temp_dir_with_images / f"image_{i}.png" for i in range(5)]


@pytest.fixture
def sample_bucket_ids():
    """Create sample bucket_ids list."""
    return [0, 0, 1, 1, 2]


# ============================================================================
# BucketImageTextDataset Initialization Tests
# ============================================================================

class TestBucketImageTextDatasetInit:
    """Tests for BucketImageTextDataset initialization."""

    def test_init_basic(self, sample_paths, sample_bucket_ids, mock_bucket_manager):
        """Test basic initialization."""
        dataset = BucketImageTextDataset(
            image_paths=sample_paths,
            bucket_ids=sample_bucket_ids,
            bucket_manager=mock_bucket_manager,
        )

        assert len(dataset) == 5
        assert dataset.bucket_manager is mock_bucket_manager

    def test_init_custom_params(self, sample_paths, sample_bucket_ids, mock_bucket_manager):
        """Test initialization with custom parameters."""
        dataset = BucketImageTextDataset(
            image_paths=sample_paths,
            bucket_ids=sample_bucket_ids,
            bucket_manager=mock_bucket_manager,
            caption_dropout=0.2,
            default_caption="custom default",
            use_random_crop=False,
            use_random_flip=False,
        )

        assert dataset.caption_dropout == 0.2
        assert dataset.default_caption == "custom default"
        assert dataset.use_random_crop is False
        assert dataset.use_random_flip is False

    def test_init_mismatched_lengths_raises(self, sample_paths, mock_bucket_manager):
        """Test mismatched lengths raises error."""
        with pytest.raises(ValueError, match="same length"):
            BucketImageTextDataset(
                image_paths=sample_paths,
                bucket_ids=[0, 1],  # Wrong length
                bucket_manager=mock_bucket_manager,
            )

    def test_init_empty_paths(self, mock_bucket_manager):
        """Test empty paths initialization."""
        dataset = BucketImageTextDataset(
            image_paths=[],
            bucket_ids=[],
            bucket_manager=mock_bucket_manager,
        )

        assert len(dataset) == 0

    def test_init_serializes_data(self, sample_paths, sample_bucket_ids, mock_bucket_manager):
        """Test data is serialized to tensors."""
        dataset = BucketImageTextDataset(
            image_paths=sample_paths,
            bucket_ids=sample_bucket_ids,
            bucket_manager=mock_bucket_manager,
        )

        # Check serialization artifacts exist
        assert hasattr(dataset, "_path_data")
        assert hasattr(dataset, "_path_addr")
        assert hasattr(dataset, "_bucket_ids")
        assert isinstance(dataset._bucket_ids, torch.Tensor)


# ============================================================================
# BucketImageTextDataset Serialization Tests
# ============================================================================

class TestBucketImageTextDatasetSerialization:
    """Tests for path and bucket_id serialization."""

    def test_get_path_returns_correct_path(self, sample_paths, sample_bucket_ids, mock_bucket_manager):
        """Test _get_path returns correct Path object."""
        dataset = BucketImageTextDataset(
            image_paths=sample_paths,
            bucket_ids=sample_bucket_ids,
            bucket_manager=mock_bucket_manager,
        )

        for i, expected_path in enumerate(sample_paths):
            retrieved_path = dataset._get_path(i)
            assert retrieved_path == expected_path

    def test_get_bucket_id_returns_correct_id(self, sample_paths, sample_bucket_ids, mock_bucket_manager):
        """Test _get_bucket_id returns correct bucket ID."""
        dataset = BucketImageTextDataset(
            image_paths=sample_paths,
            bucket_ids=sample_bucket_ids,
            bucket_manager=mock_bucket_manager,
        )

        for i, expected_id in enumerate(sample_bucket_ids):
            retrieved_id = dataset._get_bucket_id(i)
            assert retrieved_id == expected_id

    def test_serialization_preserves_order(self, sample_paths, sample_bucket_ids, mock_bucket_manager):
        """Test serialization preserves order of paths and bucket_ids."""
        dataset = BucketImageTextDataset(
            image_paths=sample_paths,
            bucket_ids=sample_bucket_ids,
            bucket_manager=mock_bucket_manager,
        )

        # Verify all paths and bucket_ids are accessible in order
        retrieved_paths = [dataset._get_path(i) for i in range(len(sample_paths))]
        retrieved_ids = [dataset._get_bucket_id(i) for i in range(len(sample_bucket_ids))]

        assert retrieved_paths == sample_paths
        assert retrieved_ids == sample_bucket_ids

    def test_bucket_ids_tensor_dtype(self, sample_paths, sample_bucket_ids, mock_bucket_manager):
        """Test bucket_ids tensor has correct dtype."""
        dataset = BucketImageTextDataset(
            image_paths=sample_paths,
            bucket_ids=sample_bucket_ids,
            bucket_manager=mock_bucket_manager,
        )

        assert dataset._bucket_ids.dtype == torch.int32


# ============================================================================
# BucketImageTextDataset __getitem__ Tests
# ============================================================================

class TestBucketImageTextDatasetGetItem:
    """Tests for BucketImageTextDataset __getitem__."""

    def test_getitem_returns_dict(self, sample_paths, sample_bucket_ids, mock_bucket_manager):
        """Test __getitem__ returns dict with correct keys."""
        dataset = BucketImageTextDataset(
            image_paths=sample_paths,
            bucket_ids=sample_bucket_ids,
            bucket_manager=mock_bucket_manager,
            caption_dropout=0.0,
        )

        sample = dataset[0]

        assert isinstance(sample, dict)
        assert "image" in sample
        assert "caption" in sample
        assert "image_path" in sample
        assert "bucket_id" in sample
        assert "resolution" in sample

    def test_getitem_image_shape(self, sample_paths, sample_bucket_ids, mock_bucket_manager):
        """Test __getitem__ returns correct image shape."""
        mock_bucket_manager.get_bucket_resolution.return_value = (256, 256)

        dataset = BucketImageTextDataset(
            image_paths=sample_paths,
            bucket_ids=sample_bucket_ids,
            bucket_manager=mock_bucket_manager,
        )

        sample = dataset[0]

        assert sample["image"].shape == (3, 256, 256)
        assert sample["image"].dtype == torch.float32

    def test_getitem_bucket_info(self, sample_paths, sample_bucket_ids, mock_bucket_manager):
        """Test __getitem__ returns correct bucket info."""
        mock_bucket_manager.get_bucket_resolution.return_value = (512, 384)

        dataset = BucketImageTextDataset(
            image_paths=sample_paths,
            bucket_ids=sample_bucket_ids,
            bucket_manager=mock_bucket_manager,
        )

        sample = dataset[0]

        assert sample["bucket_id"] == 0
        assert sample["resolution"] == (512, 384)

    def test_getitem_image_range(self, sample_paths, sample_bucket_ids, mock_bucket_manager):
        """Test __getitem__ returns image in [-1, 1] range."""
        dataset = BucketImageTextDataset(
            image_paths=sample_paths,
            bucket_ids=sample_bucket_ids,
            bucket_manager=mock_bucket_manager,
        )

        sample = dataset[0]

        assert sample["image"].min() >= -1.0
        assert sample["image"].max() <= 1.0

    def test_getitem_different_resolutions(self, sample_paths, sample_bucket_ids, mock_bucket_manager):
        """Test __getitem__ handles different bucket resolutions."""
        # Different resolutions for different bucket_ids
        def get_resolution(bucket_id):
            resolutions = {
                0: (256, 256),
                1: (384, 256),
                2: (256, 384),
            }
            return resolutions.get(bucket_id, (256, 256))

        mock_bucket_manager.get_bucket_resolution.side_effect = get_resolution

        dataset = BucketImageTextDataset(
            image_paths=sample_paths,
            bucket_ids=sample_bucket_ids,
            bucket_manager=mock_bucket_manager,
        )

        # Test each bucket resolution
        sample0 = dataset[0]  # bucket_id=0 -> (256, 256)
        sample2 = dataset[2]  # bucket_id=1 -> (384, 256)

        assert sample0["image"].shape == (3, 256, 256)
        assert sample2["image"].shape == (3, 256, 384)  # H, W


# ============================================================================
# _preprocess_image Tests
# ============================================================================

class TestBucketPreprocessImage:
    """Tests for _preprocess_image method."""

    def test_preprocess_scales_to_target(self, sample_paths, sample_bucket_ids, mock_bucket_manager):
        """Test preprocessing scales image to target resolution."""
        dataset = BucketImageTextDataset(
            image_paths=sample_paths,
            bucket_ids=sample_bucket_ids,
            bucket_manager=mock_bucket_manager,
            use_random_crop=False,
        )

        img = Image.new("RGB", (1024, 1024), color="red")
        result = dataset._preprocess_image(img, 256, 256)

        assert result.shape == (3, 256, 256)

    def test_preprocess_handles_aspect_ratio(self, sample_paths, sample_bucket_ids, mock_bucket_manager):
        """Test preprocessing handles non-square target resolution."""
        dataset = BucketImageTextDataset(
            image_paths=sample_paths,
            bucket_ids=sample_bucket_ids,
            bucket_manager=mock_bucket_manager,
            use_random_crop=False,
        )

        img = Image.new("RGB", (512, 512), color="blue")
        result = dataset._preprocess_image(img, 384, 256)

        assert result.shape == (3, 256, 384)

    def test_preprocess_output_dtype(self, sample_paths, sample_bucket_ids, mock_bucket_manager):
        """Test preprocessing outputs float32 tensor."""
        dataset = BucketImageTextDataset(
            image_paths=sample_paths,
            bucket_ids=sample_bucket_ids,
            bucket_manager=mock_bucket_manager,
        )

        img = Image.new("RGB", (512, 512), color="green")
        result = dataset._preprocess_image(img, 256, 256)

        assert result.dtype == torch.float32


# ============================================================================
# bucket_collate_fn Tests
# ============================================================================

class TestBucketCollateFn:
    """Tests for bucket_collate_fn function."""

    def test_collate_valid_batch(self):
        """Test collating valid batch."""
        batch = [
            {
                "image": torch.randn(3, 256, 256),
                "caption": "Caption 1",
                "image_path": Path("/test/1.png"),
                "bucket_id": 0,
                "resolution": (256, 256),
            },
            {
                "image": torch.randn(3, 256, 256),
                "caption": "Caption 2",
                "image_path": Path("/test/2.png"),
                "bucket_id": 0,
                "resolution": (256, 256),
            },
        ]

        result = bucket_collate_fn(batch)

        assert "images" in result
        assert "captions" in result
        assert "image_paths" in result
        assert "bucket_ids" in result
        assert "resolution" in result
        assert result["images"].shape == (2, 3, 256, 256)
        assert len(result["captions"]) == 2
        assert len(result["bucket_ids"]) == 2
        assert result["resolution"] == (256, 256)

    def test_collate_empty_batch_raises(self):
        """Test collating empty batch raises error."""
        with pytest.raises(ValueError, match="Empty batch"):
            bucket_collate_fn([])

    def test_collate_mixed_resolutions_raises(self):
        """Test collating batch with mixed resolutions raises error."""
        batch = [
            {
                "image": torch.randn(3, 256, 256),
                "caption": "Caption 1",
                "image_path": Path("/test/1.png"),
                "bucket_id": 0,
                "resolution": (256, 256),
            },
            {
                "image": torch.randn(3, 512, 512),
                "caption": "Caption 2",
                "image_path": Path("/test/2.png"),
                "bucket_id": 1,
                "resolution": (512, 512),  # Different resolution
            },
        ]

        with pytest.raises(ValueError, match="Mixed resolutions"):
            bucket_collate_fn(batch)

    def test_collate_inconsistent_shapes_raises(self):
        """Test collating batch with inconsistent shapes raises error."""
        batch = [
            {
                "image": torch.randn(3, 256, 256),
                "caption": "Caption 1",
                "image_path": Path("/test/1.png"),
                "bucket_id": 0,
                "resolution": (256, 256),
            },
            {
                "image": torch.randn(3, 256, 384),  # Different shape
                "caption": "Caption 2",
                "image_path": Path("/test/2.png"),
                "bucket_id": 0,
                "resolution": (256, 256),  # Same resolution (bug scenario)
            },
        ]

        with pytest.raises(ValueError, match="Inconsistent image shapes"):
            bucket_collate_fn(batch)

    def test_collate_preserves_bucket_ids(self):
        """Test collating preserves bucket IDs."""
        batch = [
            {
                "image": torch.randn(3, 256, 256),
                "caption": "Caption 1",
                "image_path": Path("/test/1.png"),
                "bucket_id": 5,
                "resolution": (256, 256),
            },
            {
                "image": torch.randn(3, 256, 256),
                "caption": "Caption 2",
                "image_path": Path("/test/2.png"),
                "bucket_id": 5,
                "resolution": (256, 256),
            },
        ]

        result = bucket_collate_fn(batch)

        assert result["bucket_ids"] == [5, 5]


# ============================================================================
# Integration Tests
# ============================================================================

class TestBucketImageTextDatasetIntegration:
    """Integration tests for BucketImageTextDataset."""

    def test_full_pipeline(self, sample_paths, sample_bucket_ids, mock_bucket_manager):
        """Test full pipeline from dataset to batch."""
        mock_bucket_manager.get_bucket_resolution.return_value = (256, 256)

        dataset = BucketImageTextDataset(
            image_paths=sample_paths,
            bucket_ids=[0] * len(sample_paths),  # All same bucket
            bucket_manager=mock_bucket_manager,
            caption_dropout=0.0,
        )

        # Get multiple samples from same bucket
        samples = [dataset[i] for i in range(3)]

        # Collate into batch
        batch = bucket_collate_fn(samples)

        assert batch["images"].shape == (3, 3, 256, 256)
        assert len(batch["captions"]) == 3
        assert batch["resolution"] == (256, 256)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
