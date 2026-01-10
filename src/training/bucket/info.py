"""
Bucket Sampler Info and Estimation

Provides information and RAM estimation utilities for bucket samplers.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .batch_builder import BatchCounter, BucketGrouper, RAMEstimator, ChunkBuilder
    from .state import SequentialSamplerState, BufferedShuffleSamplerState


class SequentialSamplerInfo:
    """Provides info and RAM estimation for SequentialBucketSampler."""

    def __init__(
        self,
        grouper: "BucketGrouper",
        counter: "BatchCounter",
        ram_estimator: "RAMEstimator",
        sorted_bucket_ids: List[int],
        state: "SequentialSamplerState",
    ) -> None:
        self._grouper = grouper
        self._counter = counter
        self._ram_estimator = ram_estimator
        self._sorted_ids = sorted_bucket_ids
        self._state = state

    def get_current_bucket_info(self) -> Optional[Dict[str, Any]]:
        """Get current bucket info for RAM estimation."""
        if self._state.current_bucket_idx >= len(self._sorted_ids):
            return None

        bucket_id = self._sorted_ids[self._state.current_bucket_idx]
        info = self._ram_estimator.get_bucket_info(bucket_id)
        info["num_samples"] = len(self._grouper.bucket_to_indices[bucket_id])
        remaining = self._counter.get_count(bucket_id) - self._state.current_batch_in_bucket
        info["remaining_batches"] = remaining
        info["resolution"] = (info["width"], info["height"])
        return info

    def estimate_ram_usage(self, bytes_per_pixel: float = 12.0) -> Dict[str, float]:
        """Estimate RAM usage."""
        if not self._sorted_ids:
            return {"batch_mb": 0, "max_batch_mb": 0}

        max_batch_mb = self._ram_estimator.find_max_batch_mb(
            self._sorted_ids, bytes_per_pixel
        )

        info = self.get_current_bucket_info()
        if info:
            current_mb = self._ram_estimator.estimate_bucket_batch_mb(
                info["bucket_id"], bytes_per_pixel
            )
        else:
            current_mb = max_batch_mb

        return {
            "batch_mb": max_batch_mb,
            "current_batch_mb": current_mb,
            "max_batch_mb": max_batch_mb,
        }


class BufferedShuffleSamplerInfo:
    """Provides info and RAM estimation for BufferedShuffleBucketSampler."""

    def __init__(
        self,
        grouper: "BucketGrouper",
        chunker: "ChunkBuilder",
        ram_estimator: "RAMEstimator",
        state: "BufferedShuffleSamplerState",
        batch_size: int,
        prefetch_chunks: int,
    ) -> None:
        self._grouper = grouper
        self._chunker = chunker
        self._ram_estimator = ram_estimator
        self._state = state
        self._batch_size = batch_size
        self._prefetch_chunks = prefetch_chunks

    def get_current_chunk_info(self) -> Optional[Dict[str, Any]]:
        """Get current chunk info for buffer planning."""
        chunks = self._chunker.chunks
        if self._state.current_chunk_idx >= len(chunks):
            return None

        chunk = chunks[self._state.current_chunk_idx]
        max_pixels = 0
        total_samples = 0

        for bucket_id in chunk:
            info = self._ram_estimator.get_bucket_info(bucket_id)
            if info["pixels"] > max_pixels:
                max_pixels = info["pixels"]
            total_samples += len(self._grouper.bucket_to_indices[bucket_id])

        return {
            "chunk_idx": self._state.current_chunk_idx,
            "num_buckets": len(chunk),
            "max_pixels": max_pixels,
            "total_samples": total_samples,
        }

    def get_prefetch_info(self) -> List[Dict[str, Any]]:
        """Get prefetch chunk info for buffer planning."""
        chunks = self._chunker.chunks
        infos = []

        for i in range(self._prefetch_chunks):
            chunk_idx = self._state.current_chunk_idx + i
            if chunk_idx >= len(chunks):
                break

            chunk = chunks[chunk_idx]
            max_pixels = 0
            for bucket_id in chunk:
                info = self._ram_estimator.get_bucket_info(bucket_id)
                if info["pixels"] > max_pixels:
                    max_pixels = info["pixels"]

            infos.append({"chunk_idx": chunk_idx, "max_pixels": max_pixels})

        return infos

    def estimate_buffer_size(self, bytes_per_pixel: float = 12.0) -> Dict[str, float]:
        """Estimate buffer size for prefetch planning."""
        current_info = self.get_current_chunk_info()
        prefetch_infos = self.get_prefetch_info()

        if current_info is None:
            return {"current_mb": 0, "prefetch_mb": 0, "total_mb": 0}

        current_bytes = current_info["max_pixels"] * bytes_per_pixel * self._batch_size
        current_mb = current_bytes / (1024 * 1024)

        prefetch_mb = 0
        for info in prefetch_infos:
            prefetch_bytes = info["max_pixels"] * bytes_per_pixel * self._batch_size
            prefetch_mb += prefetch_bytes / (1024 * 1024)

        return {
            "current_mb": current_mb,
            "prefetch_mb": prefetch_mb,
            "total_mb": current_mb + prefetch_mb,
        }


__all__ = [
    "SequentialSamplerInfo",
    "BufferedShuffleSamplerInfo",
]
