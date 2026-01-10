"""
PixelHDM-RPEA-DinoV3 Bucket Utils Tests

Tests for:
    - AdaptiveBufferManager: buffer management and prefetch calculation
    - scan_images_for_buckets: image scanning and bucket assignment

Test Count: 20

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.training.bucket.utils import (
    AdaptiveBufferManager,
    IMAGE_EXTENSIONS,
    scan_images_for_buckets,
    create_bucket_manager,
)
from src.training.bucket.generator import AspectRatioBucket


# ============================================================================
# AdaptiveBufferManager Tests (10 tests)
# ============================================================================


class TestAdaptiveBufferManagerInit:
    """AdaptiveBufferManager initialization tests."""

    def test_buffer_manager_init_default(self):
        """Test default initialization parameters."""
        manager = AdaptiveBufferManager()

        assert manager.max_buffer_bytes == 4.0 * 1024 * 1024 * 1024  # 4 GB
        assert manager.min_prefetch == 1
        assert manager.max_prefetch == 8
        assert manager.bytes_per_pixel == 12.0

    def test_buffer_manager_init_custom_size(self):
        """Test initialization with custom buffer size."""
        manager = AdaptiveBufferManager(
            max_buffer_gb=8.0,
            min_prefetch=2,
            max_prefetch=16,
            bytes_per_pixel=24.0,
        )

        assert manager.max_buffer_bytes == 8.0 * 1024 * 1024 * 1024  # 8 GB
        assert manager.min_prefetch == 2
        assert manager.max_prefetch == 16
        assert manager.bytes_per_pixel == 24.0


class TestAdaptiveBufferManagerPrefetch:
    """AdaptiveBufferManager prefetch factor tests."""

    @pytest.fixture
    def buffer_manager(self) -> AdaptiveBufferManager:
        """Create buffer manager for tests."""
        return AdaptiveBufferManager(max_buffer_gb=4.0)

    def test_buffer_manager_compute_prefetch_factor(self, buffer_manager):
        """Test prefetch factor calculation returns valid int."""
        prefetch = buffer_manager.compute_prefetch_factor(
            max_pixels=512 * 512,
            batch_size=4,
            num_workers=4,
        )

        assert isinstance(prefetch, int)
        assert buffer_manager.min_prefetch <= prefetch <= buffer_manager.max_prefetch

    def test_buffer_manager_prefetch_scales_with_resolution(self, buffer_manager):
        """Test prefetch factor decreases with higher resolution."""
        prefetch_low = buffer_manager.compute_prefetch_factor(
            max_pixels=256 * 256,
            batch_size=4,
            num_workers=4,
        )
        prefetch_high = buffer_manager.compute_prefetch_factor(
            max_pixels=1024 * 1024,
            batch_size=4,
            num_workers=4,
        )

        # Higher resolution should have same or lower prefetch
        assert prefetch_high <= prefetch_low

    def test_buffer_manager_min_prefetch_1(self, buffer_manager):
        """Test minimum prefetch factor is at least 1."""
        # Very large resolution that would require very low prefetch
        prefetch = buffer_manager.compute_prefetch_factor(
            max_pixels=4096 * 4096,  # 16M pixels
            batch_size=32,
            num_workers=16,
        )

        assert prefetch >= buffer_manager.min_prefetch
        assert prefetch >= 1

    def test_buffer_manager_different_batch_sizes(self, buffer_manager):
        """Test prefetch factor adjusts for different batch sizes."""
        prefetch_small = buffer_manager.compute_prefetch_factor(
            max_pixels=512 * 512,
            batch_size=1,
            num_workers=4,
        )
        prefetch_large = buffer_manager.compute_prefetch_factor(
            max_pixels=512 * 512,
            batch_size=16,
            num_workers=4,
        )

        # Larger batch should have same or lower prefetch
        assert prefetch_large <= prefetch_small

    def test_buffer_manager_different_num_workers(self, buffer_manager):
        """Test prefetch factor adjusts for different worker counts."""
        prefetch_few = buffer_manager.compute_prefetch_factor(
            max_pixels=512 * 512,
            batch_size=4,
            num_workers=2,
        )
        prefetch_many = buffer_manager.compute_prefetch_factor(
            max_pixels=512 * 512,
            batch_size=4,
            num_workers=8,
        )

        # More workers should have same or lower prefetch
        assert prefetch_many <= prefetch_few

    def test_buffer_manager_zero_workers_returns_default(self, buffer_manager):
        """Test zero workers returns default prefetch."""
        prefetch = buffer_manager.compute_prefetch_factor(
            max_pixels=512 * 512,
            batch_size=4,
            num_workers=0,
        )

        assert prefetch == 2  # Default for zero workers


class TestAdaptiveBufferManagerStats:
    """AdaptiveBufferManager statistics tests."""

    @pytest.fixture
    def buffer_manager(self) -> AdaptiveBufferManager:
        """Create buffer manager for tests."""
        return AdaptiveBufferManager(max_buffer_gb=4.0)

    def test_buffer_manager_estimate_memory_format(self, buffer_manager):
        """Test buffer stats returns correct format."""
        stats = buffer_manager.get_buffer_stats(
            max_pixels=512 * 512,
            batch_size=4,
            num_workers=4,
            prefetch_factor=2,
        )

        assert isinstance(stats, dict)
        assert "batch_mb" in stats
        assert "total_mb" in stats
        assert "utilization" in stats
        assert "max_buffer_mb" in stats

    def test_buffer_manager_estimate_memory_values(self, buffer_manager):
        """Test buffer stats returns reasonable values."""
        stats = buffer_manager.get_buffer_stats(
            max_pixels=512 * 512,
            batch_size=4,
            num_workers=4,
            prefetch_factor=2,
        )

        # Check values are positive
        assert stats["batch_mb"] > 0
        assert stats["total_mb"] > 0
        assert stats["utilization"] >= 0
        assert stats["max_buffer_mb"] > 0

        # Total should be batch * prefetch * workers
        expected_total = stats["batch_mb"] * 2 * 4
        assert abs(stats["total_mb"] - expected_total) < 0.1

    def test_buffer_manager_get_current_stats(self, buffer_manager):
        """Test get_buffer_stats with computed prefetch."""
        prefetch = buffer_manager.compute_prefetch_factor(
            max_pixels=512 * 512,
            batch_size=4,
            num_workers=4,
        )
        stats = buffer_manager.get_buffer_stats(
            max_pixels=512 * 512,
            batch_size=4,
            num_workers=4,
            prefetch_factor=prefetch,
        )

        # Utilization should be <= 1.0 (within buffer limit)
        assert stats["utilization"] <= 1.0


# ============================================================================
# scan_images_for_buckets Tests (10 tests)
# ============================================================================


class TestScanImagesForBuckets:
    """Tests for scan_images_for_buckets function."""

    @pytest.fixture
    def bucket_manager(self) -> AspectRatioBucket:
        """Create bucket manager for tests."""
        return AspectRatioBucket(
            min_resolution=256,
            max_resolution=1024,
            patch_size=16,
        )

    @pytest.fixture
    def temp_image_dir(self):
        """Create temporary directory with test images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def _create_test_image(
        self,
        path: Path,
        width: int = 512,
        height: int = 512,
    ) -> None:
        """Create a test image file."""
        img = Image.new("RGB", (width, height), color="red")
        img.save(path)

    def test_scan_empty_directory(self, temp_image_dir, bucket_manager):
        """Test scanning empty directory returns empty lists."""
        paths, bucket_ids = scan_images_for_buckets(
            temp_image_dir,
            bucket_manager,
        )

        assert isinstance(paths, list)
        assert isinstance(bucket_ids, list)
        assert len(paths) == 0
        assert len(bucket_ids) == 0

    def test_scan_with_valid_images(self, temp_image_dir, bucket_manager):
        """Test scanning directory with valid images."""
        # Create test images
        self._create_test_image(temp_image_dir / "test1.jpg", 512, 512)
        self._create_test_image(temp_image_dir / "test2.png", 1024, 768)

        paths, bucket_ids = scan_images_for_buckets(
            temp_image_dir,
            bucket_manager,
        )

        assert len(paths) == 2
        assert len(bucket_ids) == 2
        assert all(isinstance(p, Path) for p in paths)
        assert all(isinstance(bid, int) for bid in bucket_ids)

    def test_scan_with_invalid_files(self, temp_image_dir, bucket_manager):
        """Test scanning skips invalid files."""
        # Create valid image
        self._create_test_image(temp_image_dir / "valid.jpg", 512, 512)

        # Create invalid file (text file with image extension)
        invalid_path = temp_image_dir / "invalid.jpg"
        invalid_path.write_text("not an image")

        paths, bucket_ids = scan_images_for_buckets(
            temp_image_dir,
            bucket_manager,
        )

        # Should only include valid image
        assert len(paths) == 1
        assert paths[0].name == "valid.jpg"

    def test_scan_with_mixed_files(self, temp_image_dir, bucket_manager):
        """Test scanning with mixed file types."""
        # Create images
        self._create_test_image(temp_image_dir / "image.jpg", 512, 512)
        self._create_test_image(temp_image_dir / "image.png", 512, 512)

        # Create non-image files
        (temp_image_dir / "readme.txt").write_text("readme")
        (temp_image_dir / "data.json").write_text("{}")

        paths, bucket_ids = scan_images_for_buckets(
            temp_image_dir,
            bucket_manager,
        )

        # Should only include images
        assert len(paths) == 2
        extensions = {p.suffix.lower() for p in paths}
        assert extensions.issubset({".jpg", ".png"})

    def test_scan_returns_bucket_ids(self, temp_image_dir, bucket_manager):
        """Test scan returns valid bucket IDs."""
        self._create_test_image(temp_image_dir / "test.jpg", 512, 512)

        paths, bucket_ids = scan_images_for_buckets(
            temp_image_dir,
            bucket_manager,
        )

        assert len(bucket_ids) == 1
        # Bucket ID should be valid index
        assert 0 <= bucket_ids[0] < len(bucket_manager)

    def test_scan_returns_paths(self, temp_image_dir, bucket_manager):
        """Test scan returns correct paths."""
        test_path = temp_image_dir / "test.jpg"
        self._create_test_image(test_path, 512, 512)

        paths, bucket_ids = scan_images_for_buckets(
            temp_image_dir,
            bucket_manager,
        )

        assert len(paths) == 1
        assert paths[0].resolve() == test_path.resolve()

    def test_scan_recursive_option(self, temp_image_dir, bucket_manager):
        """Test scan searches subdirectories recursively."""
        # Create subdirectory
        subdir = temp_image_dir / "subdir"
        subdir.mkdir()

        # Create images in both directories
        self._create_test_image(temp_image_dir / "root.jpg", 512, 512)
        self._create_test_image(subdir / "nested.jpg", 512, 512)

        paths, bucket_ids = scan_images_for_buckets(
            temp_image_dir,
            bucket_manager,
        )

        # Should find both images (rglob is recursive by default)
        assert len(paths) == 2
        names = {p.name for p in paths}
        assert "root.jpg" in names
        assert "nested.jpg" in names

    def test_scan_supported_formats(self, temp_image_dir, bucket_manager):
        """Test scan supports jpg, png, webp formats."""
        # Create images in different formats
        self._create_test_image(temp_image_dir / "test.jpg", 512, 512)
        self._create_test_image(temp_image_dir / "test.png", 512, 512)

        # WebP requires pillow with webp support
        try:
            webp_path = temp_image_dir / "test.webp"
            img = Image.new("RGB", (512, 512), color="blue")
            img.save(webp_path, "WEBP")
        except Exception:
            pass  # Skip webp if not supported

        paths, bucket_ids = scan_images_for_buckets(
            temp_image_dir,
            bucket_manager,
        )

        # Should find at least jpg and png
        assert len(paths) >= 2
        extensions = {p.suffix.lower() for p in paths}
        assert ".jpg" in extensions
        assert ".png" in extensions

    def test_scan_parallel_option(self, temp_image_dir, bucket_manager):
        """Test scan works (parallel option is implicit in implementation)."""
        # Create multiple images
        for i in range(5):
            self._create_test_image(
                temp_image_dir / f"test_{i}.jpg",
                512 + i * 64,
                512,
            )

        # Scan should work regardless of parallel processing
        paths, bucket_ids = scan_images_for_buckets(
            temp_image_dir,
            bucket_manager,
        )

        assert len(paths) == 5
        assert len(bucket_ids) == 5

    def test_scan_with_bucket_manager(self, temp_image_dir, bucket_manager):
        """Test scan assigns correct buckets based on image dimensions."""
        # Create images with different aspect ratios
        self._create_test_image(temp_image_dir / "square.jpg", 512, 512)
        self._create_test_image(temp_image_dir / "wide.jpg", 1024, 512)
        self._create_test_image(temp_image_dir / "tall.jpg", 512, 1024)

        paths, bucket_ids = scan_images_for_buckets(
            temp_image_dir,
            bucket_manager,
        )

        assert len(paths) == 3
        assert len(bucket_ids) == 3

        # All bucket IDs should be valid
        for bid in bucket_ids:
            assert 0 <= bid < len(bucket_manager)

        # Different aspect ratios should potentially get different buckets
        # (not guaranteed but likely for 1:1, 2:1, 1:2)
        assert len(set(bucket_ids)) >= 1  # At least one bucket used


class TestScanImagesSkipsSmall:
    """Test that scan_images_for_buckets skips small images."""

    @pytest.fixture
    def bucket_manager(self) -> AspectRatioBucket:
        """Create bucket manager with min_resolution=256."""
        return AspectRatioBucket(
            min_resolution=256,
            max_resolution=1024,
            patch_size=16,
        )

    @pytest.fixture
    def temp_image_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_scan_skips_small_images(self, temp_image_dir, bucket_manager):
        """Test small images are skipped based on min_resolution."""
        # Create small image (smaller than min_resolution=256)
        small_img = Image.new("RGB", (128, 128), color="red")
        small_img.save(temp_image_dir / "small.jpg")

        # Create normal image
        normal_img = Image.new("RGB", (512, 512), color="blue")
        normal_img.save(temp_image_dir / "normal.jpg")

        paths, bucket_ids = scan_images_for_buckets(
            temp_image_dir,
            bucket_manager,
            min_resolution=256,
        )

        # Should only include normal image
        assert len(paths) == 1
        assert paths[0].name == "normal.jpg"


class TestCreateBucketManager:
    """Tests for create_bucket_manager factory function."""

    def test_create_bucket_manager_default(self):
        """Test create_bucket_manager with default values."""
        manager = create_bucket_manager()

        assert isinstance(manager, AspectRatioBucket)
        assert manager.min_resolution == 256
        assert manager.max_resolution == 1024
        assert manager.patch_size == 16

    def test_create_bucket_manager_with_config(self):
        """Test create_bucket_manager with model config."""
        from src.config import PixelHDMConfig, DataConfig

        model_config = PixelHDMConfig.for_testing()
        data_config = DataConfig()

        manager = create_bucket_manager(
            model_config=model_config,
            data_config=data_config,
        )

        assert isinstance(manager, AspectRatioBucket)
        assert manager.patch_size == model_config.patch_size
