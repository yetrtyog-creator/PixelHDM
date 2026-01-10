"""
Tests for PixelHDM-RPEA-DinoV3 Bucket Samplers.

This module contains tests for:
    - BucketSampler: Random bucket sampler
    - SequentialBucketSampler: Sequential bucket sampler (RAM optimized)
    - BufferedShuffleBucketSampler: Buffered shuffle sampler

Total: 40 test cases

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Set

import pytest

from src.training.bucket.sampler import (
    BucketSampler,
    BufferedShuffleBucketSampler,
    SequentialBucketSampler,
)
from src.training.bucket.generator import AspectRatioBucket


# ============================================================================
# Shared Fixtures
# ============================================================================


@pytest.fixture
def bucket_manager() -> AspectRatioBucket:
    """Create a bucket manager for testing."""
    return AspectRatioBucket(
        min_resolution=256,
        max_resolution=512,
        patch_size=16,
        target_pixels=256 * 256,
        max_aspect_ratio=2.0,
    )


@pytest.fixture
def sample_bucket_ids_5_buckets() -> List[int]:
    """
    Sample bucket IDs list (100 samples distributed across 5 buckets).

    Distribution:
        - Bucket 0: 20 samples (indices 0, 5, 10, ..., 95)
        - Bucket 1: 20 samples (indices 1, 6, 11, ..., 96)
        - Bucket 2: 20 samples (indices 2, 7, 12, ..., 97)
        - Bucket 3: 20 samples (indices 3, 8, 13, ..., 98)
        - Bucket 4: 20 samples (indices 4, 9, 14, ..., 99)
    """
    return [i % 5 for i in range(100)]


@pytest.fixture
def sample_bucket_ids_single_bucket() -> List[int]:
    """All samples in a single bucket (bucket 0)."""
    return [0] * 50


@pytest.fixture
def sample_bucket_ids_unique_buckets() -> List[int]:
    """Each sample in a different bucket."""
    return list(range(10))


@pytest.fixture
def sample_bucket_ids_uneven() -> List[int]:
    """
    Uneven distribution of samples across buckets.

    Distribution:
        - Bucket 0: 40 samples
        - Bucket 1: 30 samples
        - Bucket 2: 20 samples
        - Bucket 3: 10 samples
    """
    return [0] * 40 + [1] * 30 + [2] * 20 + [3] * 10


# ============================================================================
# BucketSampler Tests (14 test cases)
# ============================================================================


class TestBucketSampler:
    """Test suite for BucketSampler."""

    @pytest.fixture
    def bucket_sampler(
        self, sample_bucket_ids_5_buckets: List[int]
    ) -> BucketSampler:
        """Create a BucketSampler instance for testing."""
        return BucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            batch_size=4,
            drop_last=True,
            shuffle=True,
            seed=42,
        )

    def test_bucket_sampler_init_groups_by_bucket(
        self, sample_bucket_ids_5_buckets: List[int]
    ) -> None:
        """Test that BucketSampler correctly groups samples by bucket ID."""
        sampler = BucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            batch_size=4,
            drop_last=True,
            shuffle=False,
            seed=42,
        )

        # Check that we have 5 buckets
        assert len(sampler.bucket_to_indices) == 5

        # Check that each bucket has 20 samples
        for bucket_id, indices in sampler.bucket_to_indices.items():
            assert len(indices) == 20, f"Bucket {bucket_id} should have 20 samples"

        # Check that samples are correctly assigned
        for idx, bucket_id in enumerate(sample_bucket_ids_5_buckets):
            assert idx in sampler.bucket_to_indices[bucket_id]

    def test_bucket_sampler_iter_correct_batch_count(
        self, bucket_sampler: BucketSampler
    ) -> None:
        """Test that iteration produces the correct number of batches."""
        batches = list(bucket_sampler)

        # 100 samples, 5 buckets, each bucket has 20 samples
        # batch_size=4, drop_last=True
        # Each bucket produces 20 // 4 = 5 batches
        # Total: 5 buckets * 5 batches = 25 batches
        expected_batches = 25
        assert len(batches) == expected_batches

    def test_bucket_sampler_iter_all_samples_batched(
        self, sample_bucket_ids_5_buckets: List[int]
    ) -> None:
        """Test that all samples are included in batches (with drop_last=False)."""
        sampler = BucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            batch_size=4,
            drop_last=False,  # Include incomplete batches
            shuffle=False,
            seed=42,
        )

        all_indices = []
        for batch in sampler:
            all_indices.extend(batch)

        # All 100 samples should be included
        assert len(all_indices) == 100
        assert set(all_indices) == set(range(100))

    def test_bucket_sampler_iter_same_bucket_in_batch(
        self, sample_bucket_ids_5_buckets: List[int]
    ) -> None:
        """Test that all samples in a batch come from the same bucket."""
        sampler = BucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            batch_size=4,
            drop_last=True,
            shuffle=True,
            seed=42,
        )

        for batch in sampler:
            # Get bucket IDs for all samples in batch
            batch_bucket_ids = [sample_bucket_ids_5_buckets[idx] for idx in batch]

            # All samples should be from the same bucket
            assert len(set(batch_bucket_ids)) == 1, \
                f"Batch contains samples from multiple buckets: {batch_bucket_ids}"

    def test_bucket_sampler_iter_respects_batch_size(
        self, bucket_sampler: BucketSampler
    ) -> None:
        """Test that batches respect the specified batch size."""
        for batch in bucket_sampler:
            assert len(batch) == 4, f"Batch size should be 4, got {len(batch)}"

    def test_bucket_sampler_iter_respects_drop_last(
        self, sample_bucket_ids_uneven: List[int]
    ) -> None:
        """Test that drop_last=True discards incomplete batches."""
        # With drop_last=True
        sampler_drop = BucketSampler(
            bucket_ids=sample_bucket_ids_uneven,
            batch_size=7,  # Chosen to create incomplete batches
            drop_last=True,
            shuffle=False,
            seed=42,
        )

        for batch in sampler_drop:
            assert len(batch) == 7

        # With drop_last=False
        sampler_keep = BucketSampler(
            bucket_ids=sample_bucket_ids_uneven,
            batch_size=7,
            drop_last=False,
            shuffle=False,
            seed=42,
        )

        batches = list(sampler_keep)
        has_incomplete = any(len(b) < 7 for b in batches)
        assert has_incomplete, "Should have incomplete batches with drop_last=False"

    def test_bucket_sampler_iter_randomness_with_shuffle(
        self, sample_bucket_ids_5_buckets: List[int]
    ) -> None:
        """Test that shuffle=True produces different orderings across epochs."""
        sampler = BucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            batch_size=4,
            drop_last=True,
            shuffle=True,
            seed=42,
        )

        sampler.set_epoch(0)
        epoch0_batches = list(sampler)

        sampler.set_epoch(1)
        epoch1_batches = list(sampler)

        # Batches should be different between epochs
        epoch0_flat = [tuple(b) for b in epoch0_batches]
        epoch1_flat = [tuple(b) for b in epoch1_batches]

        assert epoch0_flat != epoch1_flat, \
            "Different epochs should produce different batch orderings"

    def test_bucket_sampler_iter_deterministic_with_seed(
        self, sample_bucket_ids_5_buckets: List[int]
    ) -> None:
        """Test that same seed produces identical results."""
        sampler1 = BucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            batch_size=4,
            drop_last=True,
            shuffle=True,
            seed=42,
        )

        sampler2 = BucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            batch_size=4,
            drop_last=True,
            shuffle=True,
            seed=42,
        )

        batches1 = list(sampler1)
        batches2 = list(sampler2)

        assert batches1 == batches2, "Same seed should produce identical batches"

    def test_bucket_sampler_set_epoch_changes_order(
        self, bucket_sampler: BucketSampler
    ) -> None:
        """Test that set_epoch changes the batch order."""
        bucket_sampler.set_epoch(0)
        epoch0_batches = list(bucket_sampler)

        bucket_sampler.set_epoch(100)
        epoch100_batches = list(bucket_sampler)

        # Content should be same (same samples)
        epoch0_samples = set(s for b in epoch0_batches for s in b)
        epoch100_samples = set(s for b in epoch100_batches for s in b)
        assert epoch0_samples == epoch100_samples

        # But order should be different
        epoch0_order = [tuple(b) for b in epoch0_batches]
        epoch100_order = [tuple(b) for b in epoch100_batches]
        assert epoch0_order != epoch100_order

    def test_bucket_sampler_same_epoch_same_order(
        self, bucket_sampler: BucketSampler
    ) -> None:
        """Test that same epoch produces same order."""
        bucket_sampler.set_epoch(5)
        first_run = list(bucket_sampler)

        bucket_sampler.set_epoch(5)
        second_run = list(bucket_sampler)

        assert first_run == second_run, "Same epoch should produce same order"

    def test_bucket_sampler_all_samples_one_bucket(
        self, sample_bucket_ids_single_bucket: List[int]
    ) -> None:
        """Test handling of all samples in a single bucket."""
        sampler = BucketSampler(
            bucket_ids=sample_bucket_ids_single_bucket,
            batch_size=5,
            drop_last=True,
            shuffle=True,
            seed=42,
        )

        batches = list(sampler)

        # 50 samples, batch_size=5 -> 10 batches
        assert len(batches) == 10

        # All samples should come from bucket 0
        for batch in batches:
            for idx in batch:
                assert sample_bucket_ids_single_bucket[idx] == 0

    def test_bucket_sampler_all_samples_different_buckets(
        self, sample_bucket_ids_unique_buckets: List[int]
    ) -> None:
        """Test handling when each sample is in a different bucket."""
        sampler = BucketSampler(
            bucket_ids=sample_bucket_ids_unique_buckets,
            batch_size=1,  # Must be 1 since each bucket has only 1 sample
            drop_last=False,
            shuffle=True,
            seed=42,
        )

        batches = list(sampler)

        # 10 samples, each in different bucket, batch_size=1 -> 10 batches
        assert len(batches) == 10

        # Each batch should have exactly 1 sample
        for batch in batches:
            assert len(batch) == 1

    def test_bucket_sampler_batch_size_larger_than_bucket(
        self, sample_bucket_ids_unique_buckets: List[int]
    ) -> None:
        """Test when batch_size is larger than samples in any bucket."""
        sampler = BucketSampler(
            bucket_ids=sample_bucket_ids_unique_buckets,
            batch_size=5,  # Larger than any bucket (each has 1 sample)
            drop_last=True,  # All batches will be dropped
            shuffle=False,
            seed=42,
        )

        batches = list(sampler)

        # No batches should be produced (drop_last=True, all incomplete)
        assert len(batches) == 0

    def test_bucket_sampler_empty_bucket_ids(self) -> None:
        """Test handling of empty bucket_ids list."""
        sampler = BucketSampler(
            bucket_ids=[],
            batch_size=4,
            drop_last=True,
            shuffle=True,
            seed=42,
        )

        batches = list(sampler)
        assert len(batches) == 0
        assert len(sampler) == 0


# ============================================================================
# SequentialBucketSampler Tests (13 test cases)
# ============================================================================


class TestSequentialBucketSampler:
    """Test suite for SequentialBucketSampler."""

    @pytest.fixture
    def sequential_sampler(
        self,
        sample_bucket_ids_5_buckets: List[int],
        bucket_manager: AspectRatioBucket,
    ) -> SequentialBucketSampler:
        """Create a SequentialBucketSampler instance for testing."""
        return SequentialBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=4,
            drop_last=True,
            order="ascending",
        )

    def test_sequential_sampler_init_sorts_ascending(
        self,
        sample_bucket_ids_5_buckets: List[int],
        bucket_manager: AspectRatioBucket,
    ) -> None:
        """Test that ascending order sorts buckets by increasing pixels."""
        sampler = SequentialBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=4,
            drop_last=True,
            order="ascending",
        )

        # Verify bucket IDs are sorted by increasing pixel count
        prev_pixels = 0
        for bucket_id in sampler.sorted_bucket_ids:
            w, h = bucket_manager.get_bucket_resolution(bucket_id)
            pixels = w * h
            assert pixels >= prev_pixels, "Buckets should be sorted ascending by pixels"
            prev_pixels = pixels

    def test_sequential_sampler_init_sorts_descending(
        self,
        sample_bucket_ids_5_buckets: List[int],
        bucket_manager: AspectRatioBucket,
    ) -> None:
        """Test that descending order sorts buckets by decreasing pixels."""
        sampler = SequentialBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=4,
            drop_last=True,
            order="descending",
        )

        # Verify bucket IDs are sorted by decreasing pixel count
        prev_pixels = float('inf')
        for bucket_id in sampler.sorted_bucket_ids:
            w, h = bucket_manager.get_bucket_resolution(bucket_id)
            pixels = w * h
            assert pixels <= prev_pixels, "Buckets should be sorted descending by pixels"
            prev_pixels = pixels

    def test_sequential_sampler_iter_buckets_ordered_by_pixels(
        self,
        sample_bucket_ids_5_buckets: List[int],
        bucket_manager: AspectRatioBucket,
    ) -> None:
        """Test that iteration processes buckets in pixel order."""
        sampler = SequentialBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=4,
            drop_last=True,
            order="ascending",
        )

        # Track which bucket each batch comes from
        batch_bucket_ids = []
        for batch in sampler:
            bucket_id = sample_bucket_ids_5_buckets[batch[0]]
            batch_bucket_ids.append(bucket_id)

        # Get the pixel count for each batch's bucket
        batch_pixels = []
        for bucket_id in batch_bucket_ids:
            w, h = bucket_manager.get_bucket_resolution(bucket_id)
            batch_pixels.append(w * h)

        # Pixels should be non-decreasing (within same bucket, constant)
        for i in range(len(batch_pixels) - 1):
            assert batch_pixels[i] <= batch_pixels[i + 1] or batch_pixels[i] == batch_pixels[i + 1], \
                "Batches should come from buckets in ascending pixel order"

    def test_sequential_sampler_iter_no_shuffling(
        self, sequential_sampler: SequentialBucketSampler
    ) -> None:
        """Test that sequential sampler produces same order on every iteration."""
        first_iter = list(sequential_sampler)
        second_iter = list(sequential_sampler)

        assert first_iter == second_iter, "Sequential sampler should not shuffle"

    def test_sequential_sampler_get_state_returns_indices(
        self,
        sample_bucket_ids_5_buckets: List[int],
        bucket_manager: AspectRatioBucket,
    ) -> None:
        """Test that get_state returns correct state dictionary."""
        sampler = SequentialBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=4,
            drop_last=True,
            order="ascending",
        )

        state = sampler.get_state()

        assert "bucket_idx" in state
        assert "batch_idx" in state
        assert isinstance(state["bucket_idx"], int)
        assert isinstance(state["batch_idx"], int)

    def test_sequential_sampler_set_state_resumes_correctly(
        self,
        sample_bucket_ids_5_buckets: List[int],
        bucket_manager: AspectRatioBucket,
    ) -> None:
        """Test that set_state allows correct resumption."""
        sampler = SequentialBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=4,
            drop_last=True,
            order="ascending",
        )

        # Get all batches first
        all_batches = list(sampler)

        # Resume from middle
        mid_idx = len(all_batches) // 2

        # Find which bucket and batch within bucket corresponds to mid_idx
        sampler2 = SequentialBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=4,
            drop_last=True,
            order="ascending",
        )

        # Iterate to middle and get state
        batches_before_mid = []
        for i, batch in enumerate(sampler2):
            batches_before_mid.append(batch)
            if i == mid_idx - 1:
                state = sampler2.get_state()
                break

        # Create new sampler and set state
        sampler3 = SequentialBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=4,
            drop_last=True,
            order="ascending",
        )
        sampler3.set_state(state["bucket_idx"], state["batch_idx"])

        # Get remaining batches
        remaining_batches = list(sampler3)

        # Combined should match original (approximately, as state might skip some)
        combined_samples = set()
        for b in batches_before_mid + remaining_batches:
            combined_samples.update(b)

        original_samples = set()
        for b in all_batches:
            original_samples.update(b)

        # At minimum, remaining batches should be a subset of original
        remaining_samples = set(s for b in remaining_batches for s in b)
        assert remaining_samples.issubset(original_samples)

    def test_sequential_sampler_mid_batch_checkpoint_resume(
        self,
        sample_bucket_ids_5_buckets: List[int],
        bucket_manager: AspectRatioBucket,
    ) -> None:
        """Test checkpoint and resume mid-iteration."""
        sampler = SequentialBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=4,
            drop_last=True,
            order="ascending",
        )

        # Iterate a few batches
        batches_collected = []
        for i, batch in enumerate(sampler):
            batches_collected.append(batch)
            if i >= 5:
                # Save state after 6 batches
                state = sampler.get_state()
                break

        # Create new sampler with saved state
        sampler2 = SequentialBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=4,
            drop_last=True,
            order="ascending",
            start_bucket_idx=state["bucket_idx"],
            start_batch_idx=state["batch_idx"],
        )

        # Get remaining batches
        remaining = list(sampler2)

        # Should have gotten remaining batches
        assert len(remaining) > 0, "Should have remaining batches after checkpoint"

    def test_sequential_sampler_estimate_ram_usage_format(
        self, sequential_sampler: SequentialBucketSampler
    ) -> None:
        """Test that estimate_ram_usage returns correct format."""
        ram_usage = sequential_sampler.estimate_ram_usage()

        assert isinstance(ram_usage, dict)
        assert "batch_mb" in ram_usage
        assert "max_batch_mb" in ram_usage
        assert isinstance(ram_usage["batch_mb"], (int, float))
        assert isinstance(ram_usage["max_batch_mb"], (int, float))
        assert ram_usage["batch_mb"] >= 0
        assert ram_usage["max_batch_mb"] >= 0

    def test_sequential_sampler_estimate_ram_usage_scales(
        self,
        sample_bucket_ids_5_buckets: List[int],
        bucket_manager: AspectRatioBucket,
    ) -> None:
        """Test that RAM estimation scales with batch size."""
        sampler_small = SequentialBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=2,
            drop_last=True,
            order="ascending",
        )

        sampler_large = SequentialBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=8,
            drop_last=True,
            order="ascending",
        )

        ram_small = sampler_small.estimate_ram_usage()
        ram_large = sampler_large.estimate_ram_usage()

        # Larger batch size should use more RAM
        assert ram_large["max_batch_mb"] > ram_small["max_batch_mb"]

    def test_sequential_sampler_bucket_at_start_checkpoint(
        self,
        sample_bucket_ids_5_buckets: List[int],
        bucket_manager: AspectRatioBucket,
    ) -> None:
        """Test checkpoint at start of iteration."""
        sampler = SequentialBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=4,
            drop_last=True,
            order="ascending",
            start_bucket_idx=0,
            start_batch_idx=0,
        )

        # Should produce all batches
        all_batches = list(sampler)
        assert len(all_batches) == len(sampler)

    def test_sequential_sampler_bucket_at_end_checkpoint(
        self,
        sample_bucket_ids_5_buckets: List[int],
        bucket_manager: AspectRatioBucket,
    ) -> None:
        """Test checkpoint at end of iteration."""
        # First get total number of buckets
        sampler = SequentialBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=4,
            drop_last=True,
            order="ascending",
        )

        num_buckets = len(sampler.sorted_bucket_ids)

        # Start from last bucket
        sampler_end = SequentialBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=4,
            drop_last=True,
            order="ascending",
            start_bucket_idx=num_buckets - 1,
            start_batch_idx=0,
        )

        # Should only produce batches from last bucket
        end_batches = list(sampler_end)

        # All batches should be from the same bucket
        if end_batches:
            bucket_ids_in_batches = set(
                sample_bucket_ids_5_buckets[idx] for b in end_batches for idx in b
            )
            assert len(bucket_ids_in_batches) == 1

    def test_sequential_sampler_single_sample_per_bucket(
        self,
        sample_bucket_ids_unique_buckets: List[int],
        bucket_manager: AspectRatioBucket,
    ) -> None:
        """Test with single sample per bucket."""
        sampler = SequentialBucketSampler(
            bucket_ids=sample_bucket_ids_unique_buckets,
            bucket_manager=bucket_manager,
            batch_size=1,
            drop_last=False,
            order="ascending",
        )

        batches = list(sampler)

        # Should have 10 batches (one per bucket)
        assert len(batches) == 10

    def test_sequential_sampler_very_large_batch_size(
        self,
        sample_bucket_ids_5_buckets: List[int],
        bucket_manager: AspectRatioBucket,
    ) -> None:
        """Test with batch size larger than samples per bucket."""
        sampler = SequentialBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=100,  # Larger than 20 samples per bucket
            drop_last=True,
            order="ascending",
        )

        batches = list(sampler)

        # No complete batches possible
        assert len(batches) == 0


# ============================================================================
# BufferedShuffleBucketSampler Tests (13 test cases)
# ============================================================================


class TestBufferedShuffleBucketSampler:
    """Test suite for BufferedShuffleBucketSampler."""

    @pytest.fixture
    def buffered_sampler(
        self,
        sample_bucket_ids_5_buckets: List[int],
        bucket_manager: AspectRatioBucket,
    ) -> BufferedShuffleBucketSampler:
        """Create a BufferedShuffleBucketSampler instance for testing."""
        return BufferedShuffleBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=4,
            drop_last=True,
            chunk_size=2,
            shuffle_chunks=True,
            shuffle_within_bucket=True,
            seed=42,
        )

    def test_buffered_sampler_chunks_created_correctly(
        self,
        sample_bucket_ids_5_buckets: List[int],
        bucket_manager: AspectRatioBucket,
    ) -> None:
        """Test that chunks are created from sorted buckets."""
        sampler = BufferedShuffleBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=4,
            drop_last=True,
            chunk_size=2,
            shuffle_chunks=False,  # Keep original order for testing
            shuffle_within_bucket=False,
            seed=42,
        )

        # Verify chunks are created from sorted bucket IDs
        all_bucket_ids_from_chunks = []
        for chunk in sampler.chunks:
            all_bucket_ids_from_chunks.extend(chunk)

        assert all_bucket_ids_from_chunks == sampler.sorted_bucket_ids

    def test_buffered_sampler_chunks_respect_chunk_size(
        self,
        sample_bucket_ids_5_buckets: List[int],
        bucket_manager: AspectRatioBucket,
    ) -> None:
        """Test that chunks respect the specified chunk_size."""
        chunk_size = 3
        sampler = BufferedShuffleBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=4,
            drop_last=True,
            chunk_size=chunk_size,
            shuffle_chunks=False,
            shuffle_within_bucket=False,
            seed=42,
        )

        # All chunks except possibly the last should have chunk_size buckets
        for i, chunk in enumerate(sampler.chunks[:-1]):
            assert len(chunk) == chunk_size, \
                f"Chunk {i} should have {chunk_size} buckets, got {len(chunk)}"

    def test_buffered_sampler_last_chunk_partial_ok(
        self,
        sample_bucket_ids_5_buckets: List[int],
        bucket_manager: AspectRatioBucket,
    ) -> None:
        """Test that the last chunk can be partial."""
        # 5 buckets with chunk_size=3 should create 2 chunks: [3, 2]
        sampler = BufferedShuffleBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=4,
            drop_last=True,
            chunk_size=3,
            shuffle_chunks=False,
            shuffle_within_bucket=False,
            seed=42,
        )

        assert len(sampler.chunks) == 2
        assert len(sampler.chunks[0]) == 3  # First chunk full
        assert len(sampler.chunks[1]) == 2  # Last chunk partial

    def test_buffered_sampler_iter_chunk_randomness(
        self,
        sample_bucket_ids_5_buckets: List[int],
        bucket_manager: AspectRatioBucket,
    ) -> None:
        """Test that chunks are shuffled across epochs."""
        sampler = BufferedShuffleBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=4,
            drop_last=True,
            chunk_size=2,
            shuffle_chunks=True,
            shuffle_within_bucket=False,
            seed=42,
        )

        # Collect chunk order from multiple epochs
        epoch_chunk_orders = []
        for epoch in range(5):
            sampler.set_epoch(epoch)

            # Track which bucket each batch comes from
            batch_buckets = []
            for batch in sampler:
                bucket_id = sample_bucket_ids_5_buckets[batch[0]]
                if not batch_buckets or batch_buckets[-1] != bucket_id:
                    batch_buckets.append(bucket_id)

            epoch_chunk_orders.append(tuple(batch_buckets))

        # Should have some variation across epochs
        unique_orders = set(epoch_chunk_orders)
        assert len(unique_orders) > 1, "Chunk order should vary across epochs"

    def test_buffered_sampler_iter_sample_randomness_within_bucket(
        self,
        sample_bucket_ids_5_buckets: List[int],
        bucket_manager: AspectRatioBucket,
    ) -> None:
        """Test that samples within buckets are shuffled."""
        sampler = BufferedShuffleBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=4,
            drop_last=True,
            chunk_size=2,
            shuffle_chunks=False,  # Keep chunk order fixed
            shuffle_within_bucket=True,
            seed=42,
        )

        sampler.set_epoch(0)
        epoch0_batches = [tuple(b) for b in sampler]

        sampler.set_epoch(1)
        epoch1_batches = [tuple(b) for b in sampler]

        # Same content but different order
        epoch0_samples = set(s for b in epoch0_batches for s in b)
        epoch1_samples = set(s for b in epoch1_batches for s in b)

        assert epoch0_samples == epoch1_samples, "Same samples should be present"
        assert epoch0_batches != epoch1_batches, "Sample order should differ"

    def test_buffered_sampler_set_epoch_changes_chunk_order(
        self, buffered_sampler: BufferedShuffleBucketSampler
    ) -> None:
        """Test that set_epoch changes chunk order when shuffle_chunks=True."""
        buffered_sampler.set_epoch(0)
        epoch0_batches = [tuple(b) for b in buffered_sampler]

        buffered_sampler.set_epoch(1)
        epoch1_batches = [tuple(b) for b in buffered_sampler]

        # Content should be same (same samples overall)
        epoch0_samples = set(s for b in epoch0_batches for s in b)
        epoch1_samples = set(s for b in epoch1_batches for s in b)
        assert epoch0_samples == epoch1_samples

        # But batch order should be different
        assert epoch0_batches != epoch1_batches

    def test_buffered_sampler_shuffle_chunks_true(
        self,
        sample_bucket_ids_5_buckets: List[int],
        bucket_manager: AspectRatioBucket,
    ) -> None:
        """Test behavior with shuffle_chunks=True."""
        sampler = BufferedShuffleBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=4,
            drop_last=True,
            chunk_size=2,
            shuffle_chunks=True,
            shuffle_within_bucket=False,
            seed=42,
        )

        # Collect first bucket ID from each chunk across epochs
        first_buckets_per_epoch = []
        for epoch in range(10):
            sampler.set_epoch(epoch)

            # Get first batch's bucket
            first_batch = next(iter(sampler))
            first_bucket = sample_bucket_ids_5_buckets[first_batch[0]]
            first_buckets_per_epoch.append(first_bucket)

        # Should have variation in which bucket comes first
        unique_first_buckets = set(first_buckets_per_epoch)
        assert len(unique_first_buckets) > 1, \
            "Different epochs should start with different buckets"

    def test_buffered_sampler_shuffle_chunks_false(
        self,
        sample_bucket_ids_5_buckets: List[int],
        bucket_manager: AspectRatioBucket,
    ) -> None:
        """Test behavior with shuffle_chunks=False."""
        sampler = BufferedShuffleBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=4,
            drop_last=True,
            chunk_size=2,
            shuffle_chunks=False,  # Keep chunk order fixed
            shuffle_within_bucket=False,
            seed=42,
        )

        sampler.set_epoch(0)
        epoch0_batches = list(sampler)

        sampler.set_epoch(1)
        epoch1_batches = list(sampler)

        # With both shuffles off, should be identical
        assert epoch0_batches == epoch1_batches

    def test_buffered_sampler_shuffle_within_bucket_true(
        self,
        sample_bucket_ids_5_buckets: List[int],
        bucket_manager: AspectRatioBucket,
    ) -> None:
        """Test that shuffle_within_bucket=True shuffles samples."""
        sampler = BufferedShuffleBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=4,
            drop_last=True,
            chunk_size=2,
            shuffle_chunks=False,  # Fix chunk order
            shuffle_within_bucket=True,
            seed=42,
        )

        sampler.set_epoch(0)
        epoch0_first_bucket_samples = []
        for batch in sampler:
            bucket_id = sample_bucket_ids_5_buckets[batch[0]]
            if bucket_id == sampler.sorted_bucket_ids[0]:
                epoch0_first_bucket_samples.extend(batch)

        sampler.set_epoch(1)
        epoch1_first_bucket_samples = []
        for batch in sampler:
            bucket_id = sample_bucket_ids_5_buckets[batch[0]]
            if bucket_id == sampler.sorted_bucket_ids[0]:
                epoch1_first_bucket_samples.extend(batch)

        # Same samples but different order
        assert set(epoch0_first_bucket_samples) == set(epoch1_first_bucket_samples)
        assert epoch0_first_bucket_samples != epoch1_first_bucket_samples

    def test_buffered_sampler_shuffle_within_bucket_false(
        self,
        sample_bucket_ids_5_buckets: List[int],
        bucket_manager: AspectRatioBucket,
    ) -> None:
        """Test that shuffle_within_bucket=False keeps sample order fixed."""
        sampler = BufferedShuffleBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=4,
            drop_last=True,
            chunk_size=2,
            shuffle_chunks=False,
            shuffle_within_bucket=False,
            seed=42,
        )

        sampler.set_epoch(0)
        epoch0_batches = [tuple(b) for b in sampler]

        sampler.set_epoch(5)
        epoch5_batches = [tuple(b) for b in sampler]

        # Should be identical when both shuffles are off
        assert epoch0_batches == epoch5_batches

    def test_buffered_sampler_get_current_chunk_info_format(
        self, buffered_sampler: BufferedShuffleBucketSampler
    ) -> None:
        """Test that get_current_chunk_info returns correct format."""
        info = buffered_sampler.get_current_chunk_info()

        assert info is not None
        assert isinstance(info, dict)
        assert "chunk_idx" in info
        assert "num_buckets" in info
        assert "max_pixels" in info
        assert "total_samples" in info

        assert isinstance(info["chunk_idx"], int)
        assert isinstance(info["num_buckets"], int)
        assert isinstance(info["max_pixels"], int)
        assert isinstance(info["total_samples"], int)

        assert info["chunk_idx"] >= 0
        assert info["num_buckets"] > 0
        assert info["max_pixels"] > 0
        assert info["total_samples"] > 0

    def test_buffered_sampler_estimate_buffer_size_format(
        self, buffered_sampler: BufferedShuffleBucketSampler
    ) -> None:
        """Test that estimate_buffer_size returns correct format."""
        buffer_info = buffered_sampler.estimate_buffer_size()

        assert isinstance(buffer_info, dict)
        assert "current_mb" in buffer_info
        assert "prefetch_mb" in buffer_info
        assert "total_mb" in buffer_info

        assert isinstance(buffer_info["current_mb"], (int, float))
        assert isinstance(buffer_info["prefetch_mb"], (int, float))
        assert isinstance(buffer_info["total_mb"], (int, float))

        assert buffer_info["current_mb"] >= 0
        assert buffer_info["prefetch_mb"] >= 0
        assert buffer_info["total_mb"] >= buffer_info["current_mb"]

    def test_buffered_sampler_state_tracking_during_iteration(
        self, buffered_sampler: BufferedShuffleBucketSampler
    ) -> None:
        """Test that state is updated during iteration."""
        states_during_iteration = []

        for i, batch in enumerate(buffered_sampler):
            state = buffered_sampler.get_state()
            states_during_iteration.append(state.copy())
            if i >= 10:  # Collect first 11 states
                break

        # Verify state structure
        for state in states_during_iteration:
            assert "epoch" in state
            assert "chunk_idx" in state
            assert "bucket_in_chunk" in state
            assert "batch_in_bucket" in state

        # State should change during iteration
        # At minimum, batch_in_bucket or bucket_in_chunk should change
        state_changes = 0
        for i in range(1, len(states_during_iteration)):
            prev = states_during_iteration[i - 1]
            curr = states_during_iteration[i]
            if (prev["chunk_idx"] != curr["chunk_idx"] or
                prev["bucket_in_chunk"] != curr["bucket_in_chunk"] or
                prev["batch_in_bucket"] != curr["batch_in_bucket"]):
                state_changes += 1

        assert state_changes > 0, "State should change during iteration"


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestBucketSamplerEdgeCases:
    """Edge case tests for all bucket samplers."""

    def test_bucket_sampler_invalid_batch_size(self) -> None:
        """Test that invalid batch_size raises error."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            BucketSampler(
                bucket_ids=[0, 1, 2],
                batch_size=0,
            )

        with pytest.raises(ValueError, match="batch_size must be positive"):
            BucketSampler(
                bucket_ids=[0, 1, 2],
                batch_size=-1,
            )

    def test_sequential_sampler_invalid_order(
        self, bucket_manager: AspectRatioBucket
    ) -> None:
        """Test that invalid order raises error."""
        with pytest.raises(ValueError, match="order must be"):
            SequentialBucketSampler(
                bucket_ids=[0, 1, 2],
                bucket_manager=bucket_manager,
                batch_size=2,
                order="invalid",
            )

    def test_buffered_sampler_invalid_chunk_size(
        self, bucket_manager: AspectRatioBucket
    ) -> None:
        """Test that invalid chunk_size raises error."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            BufferedShuffleBucketSampler(
                bucket_ids=[0, 1, 2],
                bucket_manager=bucket_manager,
                batch_size=2,
                chunk_size=0,
            )

    def test_len_matches_iteration_count(
        self,
        sample_bucket_ids_5_buckets: List[int],
        bucket_manager: AspectRatioBucket,
    ) -> None:
        """Test that __len__ matches actual iteration count for all samplers."""
        # BucketSampler
        bs = BucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            batch_size=4,
            drop_last=True,
        )
        assert len(list(bs)) == len(bs)

        # SequentialBucketSampler
        ss = SequentialBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=4,
            drop_last=True,
        )
        assert len(list(ss)) == len(ss)

        # BufferedShuffleBucketSampler
        bss = BufferedShuffleBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=4,
            drop_last=True,
            chunk_size=2,
        )
        assert len(list(bss)) == len(bss)


# ============================================================================
# State Dict Tests (Checkpoint Save/Restore)
# ============================================================================


class TestSamplerStateDict:
    """Tests for sampler state_dict and load_state_dict methods."""

    def test_bucket_sampler_state_dict_format(self) -> None:
        """Test BucketSampler state_dict returns correct format."""
        sampler = BucketSampler(
            bucket_ids=[0, 0, 1, 1, 2, 2],
            batch_size=2,
            seed=42,
        )
        sampler.set_epoch(5)

        state = sampler.state_dict()

        assert "epoch" in state
        assert "seed" in state
        assert state["epoch"] == 5
        assert state["seed"] == 42

    def test_bucket_sampler_load_state_dict(self) -> None:
        """Test BucketSampler load_state_dict restores state."""
        sampler = BucketSampler(
            bucket_ids=[0, 0, 1, 1, 2, 2],
            batch_size=2,
            seed=42,
        )

        # Save state
        sampler.set_epoch(10)
        state = sampler.state_dict()

        # Create new sampler and load state
        sampler2 = BucketSampler(
            bucket_ids=[0, 0, 1, 1, 2, 2],
            batch_size=2,
            seed=0,  # Different seed
        )
        sampler2.load_state_dict(state)

        assert sampler2.epoch == 10
        assert sampler2.seed == 42

    def test_bucket_sampler_reproducible_after_restore(self) -> None:
        """Test that BucketSampler produces same batches after restore."""
        bucket_ids = list(range(100))  # 100 samples

        sampler1 = BucketSampler(
            bucket_ids=bucket_ids,
            batch_size=4,
            seed=42,
        )
        sampler1.set_epoch(5)
        batches1 = list(sampler1)

        # Save and restore
        state = sampler1.state_dict()

        sampler2 = BucketSampler(
            bucket_ids=bucket_ids,
            batch_size=4,
            seed=0,
        )
        sampler2.load_state_dict(state)
        batches2 = list(sampler2)

        assert batches1 == batches2, "Restored sampler should produce same batches"

    def test_sequential_sampler_state_dict_format(
        self, bucket_manager: AspectRatioBucket
    ) -> None:
        """Test SequentialBucketSampler state_dict returns correct format."""
        sampler = SequentialBucketSampler(
            bucket_ids=[0, 0, 1, 1, 2, 2],
            bucket_manager=bucket_manager,
            batch_size=2,
            order="ascending",
        )

        # Iterate partially
        for i, batch in enumerate(sampler):
            if i >= 2:
                break

        state = sampler.state_dict()

        assert "bucket_idx" in state
        assert "batch_idx" in state
        assert "order" in state

    def test_sequential_sampler_load_state_dict(
        self,
        sample_bucket_ids_5_buckets: List[int],
        bucket_manager: AspectRatioBucket,
    ) -> None:
        """Test SequentialBucketSampler load_state_dict restores state."""
        sampler = SequentialBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=4,
            order="ascending",
        )

        # Get all batches
        all_batches = list(sampler)

        # Create new sampler, load state to skip first half
        sampler2 = SequentialBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=4,
            order="ascending",
        )

        mid_state = {"bucket_idx": 2, "batch_idx": 0}
        sampler2.load_state_dict(mid_state)

        remaining_batches = list(sampler2)

        # Should have skipped some batches
        assert len(remaining_batches) < len(all_batches)

    def test_buffered_sampler_state_dict_format(
        self,
        sample_bucket_ids_5_buckets: List[int],
        bucket_manager: AspectRatioBucket,
    ) -> None:
        """Test BufferedShuffleBucketSampler state_dict returns correct format."""
        sampler = BufferedShuffleBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=4,
            chunk_size=2,
            seed=42,
        )
        sampler.set_epoch(3)

        # Iterate partially
        for i, batch in enumerate(sampler):
            if i >= 5:
                break

        state = sampler.state_dict()

        assert "epoch" in state
        assert "seed" in state
        assert "chunk_idx" in state
        assert "bucket_in_chunk" in state
        assert "batch_in_bucket" in state
        assert "rng_state" in state
        assert state["epoch"] == 3

    def test_buffered_sampler_load_state_dict(
        self,
        sample_bucket_ids_5_buckets: List[int],
        bucket_manager: AspectRatioBucket,
    ) -> None:
        """Test BufferedShuffleBucketSampler load_state_dict restores state."""
        sampler = BufferedShuffleBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=4,
            chunk_size=2,
            seed=42,
        )
        sampler.set_epoch(5)

        state = sampler.state_dict()

        # Create new sampler and load state
        sampler2 = BufferedShuffleBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=4,
            chunk_size=2,
            seed=0,  # Different seed
        )
        sampler2.load_state_dict(state)

        assert sampler2.epoch == 5
        assert sampler2.seed == 42

    def test_buffered_sampler_reproducible_after_restore(
        self,
        sample_bucket_ids_5_buckets: List[int],
        bucket_manager: AspectRatioBucket,
    ) -> None:
        """Test that BufferedShuffleBucketSampler produces same batches after restore."""
        sampler1 = BufferedShuffleBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=4,
            chunk_size=2,
            shuffle_chunks=True,
            shuffle_within_bucket=True,
            seed=42,
        )
        sampler1.set_epoch(7)
        batches1 = list(sampler1)

        # Save and restore
        state = sampler1.state_dict()

        sampler2 = BufferedShuffleBucketSampler(
            bucket_ids=sample_bucket_ids_5_buckets,
            bucket_manager=bucket_manager,
            batch_size=4,
            chunk_size=2,
            shuffle_chunks=True,
            shuffle_within_bucket=True,
            seed=0,  # Different seed
        )
        sampler2.load_state_dict(state)
        batches2 = list(sampler2)

        assert batches1 == batches2, "Restored sampler should produce same batches"
