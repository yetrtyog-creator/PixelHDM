"""
Dataset Factory Tests

Tests for create_dataloader, create_bucket_dataloader, and related factory functions.

Test Categories:
    - create_dataloader (5 tests): Fixed resolution dataloader
    - create_dataloader_from_config (2 tests): Config-based creation
    - create_bucket_dataloader (5 tests): Bucketed dataloader
    - create_dataloader_from_config_v2 (4 tests): V2 config-based creation

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import pytest
import torch
from pathlib import Path
from PIL import Image
import tempfile
import shutil
from torch.utils.data import DataLoader

from src.training.dataset.factory import (
    create_dataloader,
    create_dataloader_from_config,
    create_bucket_dataloader,
    create_dataloader_from_config_v2,
)
from src.training.dataset.fixed import ImageTextDataset
from src.training.dataset.bucketed import BucketImageTextDataset
from src.config import PixelHDMConfig, DataConfig


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
    for i in range(10):
        img = Image.new("RGB", (512, 512), color=(i * 25, i * 25, i * 25))
        img.save(temp_dir / f"image_{i}.png")
        (temp_dir / f"image_{i}.txt").write_text(f"Caption for image {i}", encoding="utf-8")

    return temp_dir


@pytest.fixture
def temp_dir_with_various_sizes(temp_dir):
    """Create temporary directory with images of various sizes.

    Creates multiple images per aspect ratio to ensure buckets have enough samples.
    """
    # Create multiple images of the same size to ensure full batches
    sizes = [
        (512, 512),
        (512, 512),
        (512, 512),
        (512, 512),
        (768, 512),
        (768, 512),
        (512, 768),
        (512, 768),
    ]

    for i, (w, h) in enumerate(sizes):
        img = Image.new("RGB", (w, h), color=(i * 30, i * 30, i * 30))
        img.save(temp_dir / f"image_{i}.png")
        (temp_dir / f"image_{i}.txt").write_text(f"Caption {i}: {w}x{h}", encoding="utf-8")

    return temp_dir


@pytest.fixture
def testing_config():
    """Create minimal config for testing."""
    return PixelHDMConfig.for_testing()


@pytest.fixture
def data_config():
    """Create DataConfig for testing."""
    return DataConfig()


# ============================================================================
# create_dataloader Tests
# ============================================================================

class TestCreateDataloader:
    """Tests for create_dataloader function."""

    def test_create_dataloader_basic(self, temp_dir_with_images):
        """Test basic dataloader creation."""
        dataloader = create_dataloader(
            root_dir=temp_dir_with_images,
            batch_size=2,
            target_resolution=256,
            num_workers=0,  # For testing
        )

        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 2

    def test_create_dataloader_returns_batches(self, temp_dir_with_images):
        """Test dataloader returns proper batches."""
        dataloader = create_dataloader(
            root_dir=temp_dir_with_images,
            batch_size=2,
            target_resolution=256,
            num_workers=0,
        )

        batch = next(iter(dataloader))

        assert "images" in batch
        assert "captions" in batch
        assert "image_paths" in batch
        assert batch["images"].shape == (2, 3, 256, 256)

    def test_create_dataloader_custom_params(self, temp_dir_with_images):
        """Test dataloader with custom parameters."""
        dataloader = create_dataloader(
            root_dir=temp_dir_with_images,
            batch_size=4,
            target_resolution=512,
            patch_size=16,
            caption_dropout=0.2,
            use_random_crop=False,
            use_random_flip=False,
            num_workers=0,
            shuffle=False,
        )

        batch = next(iter(dataloader))
        assert batch["images"].shape == (4, 3, 512, 512)

    def test_create_dataloader_empty_dir_raises(self, temp_dir):
        """Test empty directory raises error."""
        with pytest.raises(RuntimeError, match="No images found"):
            create_dataloader(
                root_dir=temp_dir,
                batch_size=2,
                target_resolution=256,
                num_workers=0,
            )

    def test_create_dataloader_drop_last(self, temp_dir_with_images):
        """Test dataloader drops last incomplete batch."""
        dataloader = create_dataloader(
            root_dir=temp_dir_with_images,
            batch_size=3,  # 10 images / 3 = 3 full batches + 1 dropped
            target_resolution=256,
            num_workers=0,
        )

        batches = list(dataloader)
        # Should have 3 batches (9 images), last incomplete batch dropped
        assert len(batches) == 3


# ============================================================================
# create_dataloader_from_config Tests
# ============================================================================

class TestCreateDataloaderFromConfig:
    """Tests for create_dataloader_from_config function."""

    def test_create_from_config_basic(self, temp_dir_with_images, testing_config):
        """Test dataloader creation from config."""
        dataloader = create_dataloader_from_config(
            root_dir=temp_dir_with_images,
            config=testing_config,
            batch_size=2,
            num_workers=0,
        )

        assert isinstance(dataloader, DataLoader)

    def test_create_from_config_uses_patch_size(self, temp_dir_with_images, testing_config):
        """Test config's patch_size is used."""
        dataloader = create_dataloader_from_config(
            root_dir=temp_dir_with_images,
            config=testing_config,
            batch_size=2,
            num_workers=0,
        )

        # patch_size is used for validation (target_resolution must be multiple)
        assert isinstance(dataloader.dataset, ImageTextDataset)
        assert dataloader.dataset.patch_size == testing_config.patch_size


# ============================================================================
# create_bucket_dataloader Tests
# ============================================================================

class TestCreateBucketDataloader:
    """Tests for create_bucket_dataloader function."""

    def test_create_bucket_dataloader_basic(self, temp_dir_with_various_sizes):
        """Test basic bucket dataloader creation."""
        dataloader = create_bucket_dataloader(
            root_dir=temp_dir_with_various_sizes,
            batch_size=2,
            min_resolution=256,
            max_resolution=1024,
            num_workers=0,
        )

        assert isinstance(dataloader, DataLoader)
        assert isinstance(dataloader.dataset, BucketImageTextDataset)

    def test_create_bucket_dataloader_returns_batches(self, temp_dir_with_various_sizes):
        """Test bucket dataloader returns proper batches."""
        dataloader = create_bucket_dataloader(
            root_dir=temp_dir_with_various_sizes,
            batch_size=2,
            min_resolution=256,
            max_resolution=1024,
            num_workers=0,
        )

        batch = next(iter(dataloader))

        assert "images" in batch
        assert "captions" in batch
        assert "bucket_ids" in batch
        assert "resolution" in batch

    def test_create_bucket_dataloader_sampler_modes(self, temp_dir_with_various_sizes):
        """Test different sampler modes."""
        for mode in ["random", "sequential", "buffered_shuffle"]:
            dataloader = create_bucket_dataloader(
                root_dir=temp_dir_with_various_sizes,
                batch_size=2,
                sampler_mode=mode,
                num_workers=0,
            )

            assert isinstance(dataloader, DataLoader)

    def test_create_bucket_dataloader_empty_dir_raises(self, temp_dir):
        """Test empty directory raises error."""
        with pytest.raises(RuntimeError, match="No valid images"):
            create_bucket_dataloader(
                root_dir=temp_dir,
                batch_size=2,
                num_workers=0,
            )

    def test_create_bucket_dataloader_optimization_mode(self, temp_dir_with_various_sizes):
        """Test optimization_mode backward compatibility."""
        dataloader = create_bucket_dataloader(
            root_dir=temp_dir_with_various_sizes,
            batch_size=2,
            optimization_mode=True,  # Should use sequential sampler
            num_workers=0,
        )

        assert isinstance(dataloader, DataLoader)


# ============================================================================
# create_dataloader_from_config_v2 Tests
# ============================================================================

class TestCreateDataloaderFromConfigV2:
    """Tests for create_dataloader_from_config_v2 function."""

    def test_create_v2_with_bucketing(self, temp_dir_with_various_sizes, testing_config, data_config):
        """Test V2 creation with bucketing enabled."""
        dataloader = create_dataloader_from_config_v2(
            root_dir=temp_dir_with_various_sizes,
            model_config=testing_config,
            data_config=data_config,
            batch_size=2,
            num_workers=0,
            use_bucketing=True,
        )

        assert isinstance(dataloader, DataLoader)
        assert isinstance(dataloader.dataset, BucketImageTextDataset)

    def test_create_v2_without_bucketing(self, temp_dir_with_images, testing_config, data_config):
        """Test V2 creation without bucketing."""
        dataloader = create_dataloader_from_config_v2(
            root_dir=temp_dir_with_images,
            model_config=testing_config,
            data_config=data_config,
            batch_size=2,
            num_workers=0,
            use_bucketing=False,
        )

        assert isinstance(dataloader, DataLoader)
        assert isinstance(dataloader.dataset, ImageTextDataset)

    def test_create_v2_overrides_config(self, temp_dir_with_various_sizes, testing_config, data_config):
        """Test V2 parameter overrides config values."""
        dataloader = create_dataloader_from_config_v2(
            root_dir=temp_dir_with_various_sizes,
            model_config=testing_config,
            data_config=data_config,
            batch_size=4,  # Override
            num_workers=0,  # Override
            use_bucketing=True,
        )

        assert isinstance(dataloader, DataLoader)

    def test_create_v2_without_data_config(self, temp_dir_with_various_sizes, testing_config):
        """Test V2 creation without DataConfig uses defaults."""
        dataloader = create_dataloader_from_config_v2(
            root_dir=temp_dir_with_various_sizes,
            model_config=testing_config,
            data_config=None,
            batch_size=2,
            num_workers=0,
        )

        assert isinstance(dataloader, DataLoader)


# ============================================================================
# Integration Tests
# ============================================================================

class TestFactoryIntegration:
    """Integration tests for factory functions."""

    def test_fixed_dataloader_iteration(self, temp_dir_with_images):
        """Test iterating through fixed dataloader."""
        dataloader = create_dataloader(
            root_dir=temp_dir_with_images,
            batch_size=2,
            target_resolution=256,
            num_workers=0,
        )

        total_samples = 0
        for batch in dataloader:
            total_samples += batch["images"].shape[0]
            assert batch["images"].shape[1:] == (3, 256, 256)

        # 10 images / 2 batch_size = 5 batches (drop_last=True)
        assert total_samples == 10

    def test_bucket_dataloader_iteration(self, temp_dir_with_various_sizes):
        """Test iterating through bucket dataloader."""
        dataloader = create_bucket_dataloader(
            root_dir=temp_dir_with_various_sizes,
            batch_size=2,
            num_workers=0,
        )

        total_batches = 0
        for batch in dataloader:
            total_batches += 1
            # All images in batch should have same resolution
            assert batch["images"].shape[0] <= 2

        assert total_batches > 0

    def test_dataloader_tensor_dtype(self, temp_dir_with_images):
        """Test dataloader returns float32 tensors."""
        dataloader = create_dataloader(
            root_dir=temp_dir_with_images,
            batch_size=2,
            target_resolution=256,
            num_workers=0,
        )

        batch = next(iter(dataloader))
        assert batch["images"].dtype == torch.float32

    def test_dataloader_value_range(self, temp_dir_with_images):
        """Test dataloader returns values in [-1, 1]."""
        dataloader = create_dataloader(
            root_dir=temp_dir_with_images,
            batch_size=2,
            target_resolution=256,
            num_workers=0,
        )

        batch = next(iter(dataloader))
        assert batch["images"].min() >= -1.0
        assert batch["images"].max() <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
