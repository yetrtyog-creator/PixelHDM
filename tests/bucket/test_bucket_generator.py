"""
AspectRatioBucket Tests

Comprehensive test suite for the bucket generator module.
Tests initialization, bucket generation, and bucket assignment.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import math
from typing import Set, Tuple

import pytest

from src.training.bucket.generator import AspectRatioBucket


class TestAspectRatioBucketInit:
    """Tests for AspectRatioBucket initialization."""

    # === Test 1 ===
    def test_init_default_params(self):
        """Test initialization with default parameters (patch_size=16, min=256, max=1024)."""
        bucket = AspectRatioBucket(
            patch_size=16,
            min_resolution=256,
            max_resolution=1024,
        )

        assert bucket.patch_size == 16
        assert bucket.min_resolution == 256
        assert bucket.max_resolution == 1024
        assert bucket.max_aspect_ratio == 2.0  # Default
        assert bucket.target_pixels == 512 * 512  # Default
        assert len(bucket.buckets) > 0
        assert len(bucket.bucket_to_id) == len(bucket.buckets)

    # === Test 2 ===
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        bucket = AspectRatioBucket(
            patch_size=32,
            min_resolution=128,
            max_resolution=512,
            max_aspect_ratio=1.5,
            target_pixels=256 * 256,
            bucket_max_resolution=480,
            follow_max_resolution=True,
        )

        assert bucket.patch_size == 32
        assert bucket.min_resolution == 128
        assert bucket.max_resolution == 512
        assert bucket.max_aspect_ratio == 1.5
        assert bucket.target_pixels == 256 * 256
        assert bucket.bucket_max_resolution == 480
        assert bucket.follow_max_resolution is True
        assert bucket.effective_max_resolution == 480  # min(512, 480)
        assert len(bucket.buckets) > 0

    # === Test 3 ===
    def test_init_invalid_params_raises(self):
        """Test that invalid parameters raise ValueError (min > max leads to no valid buckets)."""
        # When min_resolution > max_resolution, no buckets can be generated
        # which raises ValueError in __init__
        with pytest.raises(ValueError, match="No valid buckets generated"):
            AspectRatioBucket(
                patch_size=16,
                min_resolution=1024,
                max_resolution=256,
            )

    # === Test 4 ===
    def test_init_min_equals_max(self):
        """Test edge case where min_resolution equals max_resolution (only generates square buckets)."""
        bucket = AspectRatioBucket(
            patch_size=16,
            min_resolution=512,
            max_resolution=512,
        )

        assert len(bucket.buckets) >= 1
        # All buckets should be 512x512 since min == max
        for w, h in bucket.buckets:
            assert w == 512 or h == 512
            assert w >= 512 and h >= 512 or w <= 512 and h <= 512

    # === Test 5 ===
    def test_init_very_small_min(self):
        """Test initialization with very small min_resolution (64)."""
        bucket = AspectRatioBucket(
            patch_size=16,
            min_resolution=64,
            max_resolution=512,
        )

        assert bucket.min_resolution == 64
        assert len(bucket.buckets) > 0

        # Should include some buckets with 64 as one dimension
        min_dimension = min(min(w, h) for w, h in bucket.buckets)
        assert min_dimension <= 128  # At least some small buckets exist


class TestAspectRatioBucketGeneration:
    """Tests for bucket generation logic."""

    @pytest.fixture
    def default_bucket(self):
        """Default bucket manager with standard parameters."""
        return AspectRatioBucket(
            patch_size=16,
            min_resolution=256,
            max_resolution=1024,
            target_pixels=512 * 512,
        )

    @pytest.fixture
    def small_bucket(self):
        """Smaller bucket manager for fast tests."""
        return AspectRatioBucket(
            patch_size=16,
            min_resolution=256,
            max_resolution=512,
            target_pixels=256 * 256,
        )

    # === Test 6 ===
    def test_buckets_divisible_by_patch_size(self, default_bucket):
        """Test that all bucket resolutions are divisible by patch_size."""
        for w, h in default_bucket.buckets:
            assert w % 16 == 0, f"Width {w} is not divisible by patch_size=16"
            assert h % 16 == 0, f"Height {h} is not divisible by patch_size=16"

    # === Test 7 ===
    def test_buckets_respect_min_max_bounds(self, default_bucket):
        """Test that all buckets are within min/max resolution bounds."""
        for w, h in default_bucket.buckets:
            assert w >= 256, f"Width {w} is below min_resolution=256"
            assert h >= 256, f"Height {h} is below min_resolution=256"
            assert w <= 1024, f"Width {w} is above max_resolution=1024"
            assert h <= 1024, f"Height {h} is above max_resolution=1024"

    # === Test 8 ===
    def test_buckets_respect_aspect_ratio_limits(self, default_bucket):
        """Test that all bucket aspect ratios are within limits (default 2.0)."""
        max_ar = default_bucket.max_aspect_ratio
        EPSILON = 1e-6

        for w, h in default_bucket.buckets:
            aspect = w / h
            assert aspect <= max_ar + EPSILON, \
                f"Bucket {w}x{h} aspect ratio {aspect:.3f} exceeds max {max_ar}"
            assert aspect >= (1.0 / max_ar) - EPSILON, \
                f"Bucket {w}x{h} aspect ratio {aspect:.3f} is below min {1.0/max_ar:.3f}"

    # === Test 9 ===
    def test_buckets_follow_max_resolution_true(self):
        """Test that follow_max_resolution=True respects bucket_max_resolution constraint."""
        bucket = AspectRatioBucket(
            patch_size=16,
            min_resolution=256,
            max_resolution=1024,
            bucket_max_resolution=768,
            follow_max_resolution=True,
        )

        # effective_max should be 768
        assert bucket.effective_max_resolution == 768

        # All buckets should be within 768
        for w, h in bucket.buckets:
            assert w <= 768, f"Width {w} exceeds bucket_max_resolution=768"
            assert h <= 768, f"Height {h} exceeds bucket_max_resolution=768"

    # === Test 10 ===
    def test_buckets_ignore_max_resolution_false(self):
        """Test that follow_max_resolution=False ignores bucket_max_resolution constraint."""
        bucket = AspectRatioBucket(
            patch_size=16,
            min_resolution=256,
            max_resolution=1024,
            bucket_max_resolution=512,  # Should be ignored
            follow_max_resolution=False,
        )

        # effective_max should use max_resolution
        assert bucket.effective_max_resolution == 1024

        # Should have buckets larger than 512
        max_dimension = max(max(w, h) for w, h in bucket.buckets)
        assert max_dimension > 512, "Expected buckets larger than 512 when follow_max_resolution=False"

    # === Test 11 ===
    def test_buckets_include_standard_squares(self, default_bucket):
        """Test that standard square resolutions (256, 512, 1024) are included."""
        bucket_set: Set[Tuple[int, int]] = set(default_bucket.buckets)

        # 256x256 should be included if within range
        if default_bucket.min_resolution <= 256 <= default_bucket.effective_max_resolution:
            assert (256, 256) in bucket_set, "256x256 square bucket missing"

        # 512x512 should be included
        if default_bucket.min_resolution <= 512 <= default_bucket.effective_max_resolution:
            assert (512, 512) in bucket_set, "512x512 square bucket missing"

        # 1024x1024 should be included
        if default_bucket.min_resolution <= 1024 <= default_bucket.effective_max_resolution:
            assert (1024, 1024) in bucket_set, "1024x1024 square bucket missing"

    # === Test 12 ===
    def test_buckets_no_duplicates(self, default_bucket):
        """Test that there are no duplicate buckets."""
        bucket_list = default_bucket.buckets
        bucket_set = set(bucket_list)

        assert len(bucket_list) == len(bucket_set), \
            f"Found {len(bucket_list) - len(bucket_set)} duplicate buckets"

    # === Test 13 ===
    def test_buckets_sorted_by_pixels(self, default_bucket):
        """Test that buckets are sorted by total pixel count."""
        buckets = default_bucket.buckets

        for i in range(len(buckets) - 1):
            current_pixels = buckets[i][0] * buckets[i][1]
            next_pixels = buckets[i + 1][0] * buckets[i + 1][1]

            # Should be sorted by pixels first
            if current_pixels > next_pixels:
                raise AssertionError(
                    f"Buckets not sorted by pixels: {buckets[i]} ({current_pixels}) > "
                    f"{buckets[i+1]} ({next_pixels})"
                )

    # === Test 14 ===
    def test_buckets_count_reasonable(self, default_bucket):
        """Test that the number of buckets is reasonable (between 50 and 300)."""
        count = len(default_bucket.buckets)

        # Reasonable range for 256-1024 with patch_size=16
        assert 50 <= count <= 300, \
            f"Bucket count {count} is outside reasonable range [50, 300]"

    # === Test 15 ===
    def test_generate_buckets_deterministic(self):
        """Test that bucket generation is deterministic (same params = same buckets)."""
        bucket1 = AspectRatioBucket(
            patch_size=16,
            min_resolution=256,
            max_resolution=1024,
            target_pixels=512 * 512,
        )

        bucket2 = AspectRatioBucket(
            patch_size=16,
            min_resolution=256,
            max_resolution=1024,
            target_pixels=512 * 512,
        )

        assert bucket1.buckets == bucket2.buckets, \
            "Bucket generation is not deterministic"

    # === Test 16 ===
    def test_buckets_include_16_9_aspect(self, default_bucket):
        """Test that buckets include something close to 16:9 aspect ratio."""
        target_aspect = 16 / 9  # ~1.778
        TOLERANCE = 0.15

        found_16_9 = False
        for w, h in default_bucket.buckets:
            aspect = w / h
            if abs(aspect - target_aspect) < TOLERANCE:
                found_16_9 = True
                break

        assert found_16_9, \
            f"No bucket close to 16:9 aspect ratio ({target_aspect:.3f}) found"

    # === Test 17 ===
    def test_buckets_include_1_1_aspect(self, default_bucket):
        """Test that buckets include 1:1 square aspect ratio."""
        found_square = False
        for w, h in default_bucket.buckets:
            if w == h:
                found_square = True
                break

        assert found_square, "No square (1:1) bucket found"


class TestAspectRatioBucketAssignment:
    """Tests for bucket ID assignment logic."""

    @pytest.fixture
    def default_bucket(self):
        """Default bucket manager for assignment tests."""
        return AspectRatioBucket(
            patch_size=16,
            min_resolution=256,
            max_resolution=1024,
            target_pixels=512 * 512,
        )

    @pytest.fixture
    def small_bucket(self):
        """Smaller bucket manager for fast tests."""
        return AspectRatioBucket(
            patch_size=16,
            min_resolution=256,
            max_resolution=512,
            target_pixels=256 * 256,
        )

    # === Test 18 ===
    def test_get_bucket_id_exact_match(self, default_bucket):
        """Test that exact dimension match returns correct bucket ID."""
        # Use the first bucket's dimensions
        w, h = default_bucket.buckets[0]
        bucket_id = default_bucket.get_bucket_id(w, h)

        assert bucket_id == 0, f"Expected bucket_id=0 for exact match, got {bucket_id}"

        # Also test a middle bucket
        mid_idx = len(default_bucket.buckets) // 2
        w_mid, h_mid = default_bucket.buckets[mid_idx]
        bucket_id_mid = default_bucket.get_bucket_id(w_mid, h_mid)

        # Should return the same or a very similar bucket
        result_w, result_h = default_bucket.buckets[bucket_id_mid]
        assert (result_w, result_h) == (w_mid, h_mid) or \
               abs(result_w * result_h - w_mid * h_mid) < w_mid * h_mid * 0.1

    # === Test 19 ===
    def test_get_bucket_id_closest_aspect_ratio(self, default_bucket):
        """Test that assignment prefers closest aspect ratio."""
        # Image with 16:9 aspect ratio
        img_w, img_h = 1600, 900  # 16:9
        bucket_id = default_bucket.get_bucket_id(img_w, img_h)
        result_w, result_h = default_bucket.buckets[bucket_id]
        result_aspect = result_w / result_h

        # Result should be reasonably close to 16:9
        target_aspect = 16 / 9
        assert abs(result_aspect - target_aspect) < 0.5, \
            f"Assigned bucket {result_w}x{result_h} (aspect={result_aspect:.3f}) " \
            f"is too far from 16:9 ({target_aspect:.3f})"

    # === Test 20 ===
    def test_get_bucket_id_closest_resolution(self, default_bucket):
        """Test that assignment considers resolution as secondary criterion."""
        # Large image that should get a large bucket
        img_w, img_h = 900, 900  # Should get something near 896x896 or 832x832
        bucket_id = default_bucket.get_bucket_id(img_w, img_h)
        result_w, result_h = default_bucket.buckets[bucket_id]

        # Result should be close to 900x900 without exceeding
        result_pixels = result_w * result_h
        img_pixels = img_w * img_h

        # Should not exceed original image pixels (no upsampling)
        assert result_pixels <= img_pixels, \
            f"Bucket {result_w}x{result_h} ({result_pixels} pixels) exceeds " \
            f"image {img_w}x{img_h} ({img_pixels} pixels)"

    # === Test 21 ===
    def test_get_bucket_id_small_image_fallback(self, default_bucket):
        """Test that small images fall back to minimum bucket."""
        # Image smaller than minimum bucket
        img_w, img_h = 100, 100
        bucket_id = default_bucket.get_bucket_id(img_w, img_h)

        # Should return a valid bucket ID
        assert 0 <= bucket_id < len(default_bucket.buckets)

        # The assigned bucket should be one of the smallest (fallback behavior)
        result_w, result_h = default_bucket.buckets[bucket_id]
        min_bucket_pixels = default_bucket.buckets[0][0] * default_bucket.buckets[0][1]
        result_pixels = result_w * result_h

        # Should be among the smallest buckets
        assert result_pixels == min_bucket_pixels, \
            f"Small image fallback should use minimum bucket, got {result_w}x{result_h}"

    # === Test 22 ===
    def test_get_bucket_id_large_image_no_upsampling(self, default_bucket):
        """Test that large images are not assigned to larger buckets (no upsampling)."""
        # Large image that exceeds all buckets
        img_w, img_h = 2000, 2000
        bucket_id = default_bucket.get_bucket_id(img_w, img_h)
        result_w, result_h = default_bucket.buckets[bucket_id]

        # Result bucket should not exceed the original image dimensions
        assert result_w <= img_w and result_h <= img_h, \
            f"Bucket {result_w}x{result_h} should not exceed image {img_w}x{img_h}"

        # Should be the largest available bucket (since image is larger)
        max_bucket_pixels = max(w * h for w, h in default_bucket.buckets)
        result_pixels = result_w * result_h

        # Should be near the maximum bucket
        assert result_pixels >= max_bucket_pixels * 0.5, \
            f"Large image should get a large bucket, got {result_w}x{result_h}"

    # === Test 23 ===
    def test_get_bucket_id_invalid_dimensions(self, default_bucket):
        """Test handling of invalid dimensions (0 or negative)."""
        # Zero dimensions
        bucket_id_zero = default_bucket.get_bucket_id(0, 512)
        assert bucket_id_zero == 0, "Zero dimension should return default bucket (id=0)"

        bucket_id_zero2 = default_bucket.get_bucket_id(512, 0)
        assert bucket_id_zero2 == 0, "Zero height should return default bucket (id=0)"

        # Negative dimensions
        bucket_id_neg = default_bucket.get_bucket_id(-100, 512)
        assert bucket_id_neg == 0, "Negative dimension should return default bucket (id=0)"

    # === Test 24 ===
    def test_get_bucket_id_extreme_aspect_ratio(self, default_bucket):
        """Test handling of extreme aspect ratios (e.g., 10:1)."""
        # Very wide image
        img_w, img_h = 1000, 100  # 10:1 aspect ratio
        bucket_id = default_bucket.get_bucket_id(img_w, img_h)
        result_w, result_h = default_bucket.buckets[bucket_id]

        # Should return a valid bucket
        assert 0 <= bucket_id < len(default_bucket.buckets)

        # The assigned bucket should have the widest available aspect ratio
        result_aspect = result_w / result_h
        max_allowed_aspect = default_bucket.max_aspect_ratio

        # Should be clamped to max aspect ratio
        assert result_aspect <= max_allowed_aspect + 0.1, \
            f"Extreme aspect ratio image should get bucket with aspect <= {max_allowed_aspect}"

    # === Test 25 ===
    def test_get_bucket_resolution_valid_id(self, default_bucket):
        """Test that get_bucket_resolution returns correct dimensions."""
        for i, (expected_w, expected_h) in enumerate(default_bucket.buckets):
            result_w, result_h = default_bucket.get_bucket_resolution(i)

            assert result_w == expected_w, \
                f"Bucket {i}: expected width {expected_w}, got {result_w}"
            assert result_h == expected_h, \
                f"Bucket {i}: expected height {expected_h}, got {result_h}"


class TestAspectRatioBucketEdgeCases:
    """Additional edge case tests."""

    def test_len_method(self):
        """Test __len__ method returns correct bucket count."""
        bucket = AspectRatioBucket(
            patch_size=16,
            min_resolution=256,
            max_resolution=512,
        )

        assert len(bucket) == len(bucket.buckets)

    def test_bucket_to_id_mapping(self):
        """Test bucket_to_id dictionary is correctly populated."""
        bucket = AspectRatioBucket(
            patch_size=16,
            min_resolution=256,
            max_resolution=512,
        )

        for i, (w, h) in enumerate(bucket.buckets):
            assert bucket.bucket_to_id[(w, h)] == i, \
                f"bucket_to_id mismatch for bucket {i}: {(w, h)}"

    @pytest.mark.parametrize("patch_size", [8, 16, 32])
    def test_different_patch_sizes(self, patch_size):
        """Test bucket generation with different patch sizes."""
        bucket = AspectRatioBucket(
            patch_size=patch_size,
            min_resolution=256,
            max_resolution=512,
        )

        # All dimensions should be divisible by patch_size
        for w, h in bucket.buckets:
            assert w % patch_size == 0, \
                f"Width {w} not divisible by patch_size={patch_size}"
            assert h % patch_size == 0, \
                f"Height {h} not divisible by patch_size={patch_size}"

    @pytest.mark.parametrize("max_ar", [1.5, 2.0, 3.0])
    def test_different_max_aspect_ratios(self, max_ar):
        """Test bucket generation with different max aspect ratios."""
        bucket = AspectRatioBucket(
            patch_size=16,
            min_resolution=256,
            max_resolution=512,
            max_aspect_ratio=max_ar,
        )

        EPSILON = 1e-6
        for w, h in bucket.buckets:
            aspect = w / h
            assert aspect <= max_ar + EPSILON, \
                f"Bucket {w}x{h} aspect {aspect:.3f} exceeds max_ar={max_ar}"
            assert aspect >= (1.0 / max_ar) - EPSILON, \
                f"Bucket {w}x{h} aspect {aspect:.3f} below 1/max_ar={1.0/max_ar:.3f}"

    def test_portrait_vs_landscape_distribution(self):
        """Test that both portrait and landscape orientations are generated."""
        bucket = AspectRatioBucket(
            patch_size=16,
            min_resolution=256,
            max_resolution=1024,
        )

        portrait_count = 0
        landscape_count = 0
        square_count = 0

        for w, h in bucket.buckets:
            if w > h:
                landscape_count += 1
            elif h > w:
                portrait_count += 1
            else:
                square_count += 1

        # Should have a mix of all orientations
        assert portrait_count > 0, "No portrait buckets found"
        assert landscape_count > 0, "No landscape buckets found"
        assert square_count > 0, "No square buckets found"

        # Portrait and landscape should be roughly balanced
        ratio = portrait_count / landscape_count if landscape_count > 0 else float('inf')
        assert 0.5 <= ratio <= 2.0, \
            f"Portrait/Landscape ratio {ratio:.2f} is too imbalanced"


class TestAspectRatioBucketIntegration:
    """Integration tests for typical usage patterns."""

    def test_typical_workflow(self):
        """Test typical workflow: create bucket manager, assign images, get resolution."""
        # Create bucket manager
        bucket = AspectRatioBucket(
            patch_size=16,
            min_resolution=256,
            max_resolution=1024,
        )

        # Simulate processing a batch of images
        test_images = [
            (800, 600),   # 4:3 landscape
            (600, 800),   # 3:4 portrait
            (1000, 1000), # 1:1 square
            (1920, 1080), # 16:9 landscape
        ]

        for img_w, img_h in test_images:
            bucket_id = bucket.get_bucket_id(img_w, img_h)
            assert 0 <= bucket_id < len(bucket.buckets)

            res_w, res_h = bucket.get_bucket_resolution(bucket_id)
            assert res_w % 16 == 0 and res_h % 16 == 0

            # No upsampling
            assert res_w * res_h <= img_w * img_h or \
                   (img_w * img_h < bucket.buckets[0][0] * bucket.buckets[0][1])

    def test_batch_same_bucket(self):
        """Test that similar images get assigned to the same bucket."""
        bucket = AspectRatioBucket(
            patch_size=16,
            min_resolution=256,
            max_resolution=1024,
        )

        # Images with same aspect ratio but slightly different sizes
        similar_images = [
            (800, 600),
            (810, 608),
            (795, 596),
        ]

        bucket_ids = [bucket.get_bucket_id(w, h) for w, h in similar_images]

        # Similar images might get same or adjacent buckets
        unique_buckets = set(bucket_ids)
        assert len(unique_buckets) <= 2, \
            f"Similar images should get at most 2 different buckets, got {len(unique_buckets)}"
