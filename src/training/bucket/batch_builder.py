"""
Bucket Batch Builder

Provides batch building logic for bucket-based samplers.
Handles bucket grouping, sorting, chunking, and batch generation.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .generator import AspectRatioBucket


class BucketGrouper:
    """Groups samples by bucket ID."""

    def __init__(self, bucket_ids: List[int]) -> None:
        """
        Initialize bucket grouper.

        Args:
            bucket_ids: List of bucket IDs for each sample
        """
        self.bucket_ids = bucket_ids
        self.bucket_to_indices: Dict[int, List[int]] = defaultdict(list)

        for idx, bucket_id in enumerate(bucket_ids):
            self.bucket_to_indices[bucket_id].append(idx)

    @property
    def num_buckets(self) -> int:
        """Return number of buckets."""
        return len(self.bucket_to_indices)

    @property
    def num_samples(self) -> int:
        """Return total number of samples."""
        return len(self.bucket_ids)

    def get_bucket_indices(self, bucket_id: int) -> List[int]:
        """Get sample indices for a specific bucket."""
        return self.bucket_to_indices.get(bucket_id, [])


class BucketSorter:
    """Sorts buckets by pixel count."""

    def __init__(
        self,
        bucket_manager: "AspectRatioBucket",
        bucket_ids: List[int],
        order: str = "ascending",
    ) -> None:
        """
        Initialize bucket sorter.

        Args:
            bucket_manager: AspectRatioBucket instance
            bucket_ids: List of unique bucket IDs to sort
            order: Sort order ("ascending" or "descending")
        """
        self.bucket_manager = bucket_manager
        self.order = order
        self.sorted_bucket_ids = self._sort_buckets(bucket_ids)

    def _sort_buckets(self, bucket_ids: List[int]) -> List[int]:
        """Sort bucket IDs by pixel count."""
        unique_ids = list(set(bucket_ids))
        bucket_ids_with_pixels = []

        for bucket_id in unique_ids:
            w, h = self.bucket_manager.get_bucket_resolution(bucket_id)
            pixels = w * h
            bucket_ids_with_pixels.append((bucket_id, pixels))

        reverse = (self.order == "descending")
        bucket_ids_with_pixels.sort(key=lambda x: x[1], reverse=reverse)
        return [bid for bid, _ in bucket_ids_with_pixels]


class ChunkBuilder:
    """Builds chunks (super-blocks) from sorted buckets."""

    def __init__(self, sorted_bucket_ids: List[int], chunk_size: int) -> None:
        """
        Initialize chunk builder.

        Args:
            sorted_bucket_ids: List of bucket IDs sorted by pixel count
            chunk_size: Number of buckets per chunk
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        self.chunk_size = chunk_size
        self.chunks = self._build_chunks(sorted_bucket_ids)

    def _build_chunks(self, sorted_bucket_ids: List[int]) -> List[List[int]]:
        """Split sorted bucket IDs into chunks."""
        chunks = []
        for i in range(0, len(sorted_bucket_ids), self.chunk_size):
            chunk = sorted_bucket_ids[i:i + self.chunk_size]
            chunks.append(chunk)
        return chunks

    @property
    def num_chunks(self) -> int:
        """Return number of chunks."""
        return len(self.chunks)

    def get_chunk(self, idx: int) -> List[int]:
        """Get bucket IDs in a specific chunk."""
        if 0 <= idx < len(self.chunks):
            return self.chunks[idx]
        return []


class BatchCounter:
    """Counts batches per bucket."""

    def __init__(
        self,
        bucket_to_indices: Dict[int, List[int]],
        batch_size: int,
        drop_last: bool = True,
    ) -> None:
        """
        Initialize batch counter.

        Args:
            bucket_to_indices: Mapping from bucket ID to sample indices
            batch_size: Batch size
            drop_last: Whether to drop incomplete batches
        """
        self.batch_size = batch_size
        self.drop_last = drop_last
        self._counts = self._compute_counts(bucket_to_indices)
        self._total = sum(self._counts.values())

    def _compute_counts(
        self, bucket_to_indices: Dict[int, List[int]]
    ) -> Dict[int, int]:
        """Compute batch count per bucket."""
        counts = {}
        for bucket_id, indices in bucket_to_indices.items():
            n_batches = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size > 0:
                n_batches += 1
            counts[bucket_id] = n_batches
        return counts

    def get_count(self, bucket_id: int) -> int:
        """Get batch count for a bucket."""
        return self._counts.get(bucket_id, 0)

    @property
    def total_batches(self) -> int:
        """Return total batch count across all buckets."""
        return self._total


class BatchGenerator:
    """Generates batches from bucket indices."""

    def __init__(self, batch_size: int, drop_last: bool = True) -> None:
        """
        Initialize batch generator.

        Args:
            batch_size: Batch size
            drop_last: Whether to drop incomplete batches
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.batch_size = batch_size
        self.drop_last = drop_last

    def generate_batches(
        self,
        indices: List[int],
        shuffle: bool = False,
        rng: Optional[random.Random] = None,
    ) -> Iterator[List[int]]:
        """
        Generate batches from indices.

        Args:
            indices: List of sample indices
            shuffle: Whether to shuffle indices
            rng: Random number generator for shuffling

        Yields:
            Batches of indices
        """
        indices_copy = indices.copy()

        if shuffle and rng is not None:
            rng.shuffle(indices_copy)

        for i in range(0, len(indices_copy), self.batch_size):
            batch = indices_copy[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch


class RAMEstimator:
    """Estimates RAM usage for bucket operations."""

    DEFAULT_BYTES_PER_PIXEL = 12.0  # RGB float32

    def __init__(
        self,
        bucket_manager: "AspectRatioBucket",
        batch_size: int,
    ) -> None:
        """
        Initialize RAM estimator.

        Args:
            bucket_manager: AspectRatioBucket instance
            batch_size: Batch size
        """
        self.bucket_manager = bucket_manager
        self.batch_size = batch_size

    def estimate_bucket_batch_mb(
        self,
        bucket_id: int,
        bytes_per_pixel: float = DEFAULT_BYTES_PER_PIXEL,
    ) -> float:
        """Estimate RAM usage for a batch from a bucket."""
        w, h = self.bucket_manager.get_bucket_resolution(bucket_id)
        pixels = w * h
        batch_bytes = pixels * bytes_per_pixel * self.batch_size
        return batch_bytes / (1024 * 1024)

    def find_max_batch_mb(
        self,
        bucket_ids: List[int],
        bytes_per_pixel: float = DEFAULT_BYTES_PER_PIXEL,
    ) -> float:
        """Find maximum batch RAM usage across buckets."""
        max_pixels = 0
        for bucket_id in bucket_ids:
            w, h = self.bucket_manager.get_bucket_resolution(bucket_id)
            pixels = w * h
            if pixels > max_pixels:
                max_pixels = pixels

        max_batch_bytes = max_pixels * bytes_per_pixel * self.batch_size
        return max_batch_bytes / (1024 * 1024)

    def get_bucket_info(self, bucket_id: int) -> Dict[str, int]:
        """Get information about a bucket."""
        w, h = self.bucket_manager.get_bucket_resolution(bucket_id)
        return {
            "bucket_id": bucket_id,
            "width": w,
            "height": h,
            "pixels": w * h,
        }


__all__ = [
    "BucketGrouper",
    "BucketSorter",
    "ChunkBuilder",
    "BatchCounter",
    "BatchGenerator",
    "RAMEstimator",
]
