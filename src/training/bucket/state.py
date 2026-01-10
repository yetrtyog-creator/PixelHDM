"""
Bucket Sampler State Management

Provides state management for checkpoint save/restore functionality.
Supports BucketSampler, SequentialBucketSampler, and BufferedShuffleBucketSampler.

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class SamplerState(ABC):
    """Abstract base class for sampler state management."""

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Return state dictionary for checkpoint saving."""
        ...

    @abstractmethod
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        ...


class BucketSamplerState(SamplerState):
    """State management for BucketSampler."""

    def __init__(self, seed: int = 42) -> None:
        self.epoch: int = 0
        self.seed: int = seed

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch."""
        self.epoch = epoch

    def state_dict(self) -> Dict[str, Any]:
        """Return state dictionary."""
        return {
            "epoch": self.epoch,
            "seed": self.seed,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore state from dictionary."""
        self.epoch = state.get("epoch", 0)
        self.seed = state.get("seed", self.seed)

    def create_rng(self) -> random.Random:
        """Create RNG for current epoch."""
        return random.Random(self.seed + self.epoch)


class SequentialSamplerState(SamplerState):
    """State management for SequentialBucketSampler."""

    def __init__(self, order: str = "ascending") -> None:
        self.order: str = order
        self.current_bucket_idx: int = 0
        self.current_batch_in_bucket: int = 0
        self._start_bucket_idx: int = 0
        self._start_batch_idx: int = 0

    def set_state(self, bucket_idx: int, batch_idx: int) -> None:
        """Set current position."""
        self._start_bucket_idx = bucket_idx
        self._start_batch_idx = batch_idx
        self.current_bucket_idx = bucket_idx
        self.current_batch_in_bucket = batch_idx

    def get_state(self) -> Dict[str, int]:
        """Get current position state."""
        return {
            "bucket_idx": self.current_bucket_idx,
            "batch_idx": self.current_batch_in_bucket,
        }

    def state_dict(self) -> Dict[str, Any]:
        """Return state dictionary."""
        return {
            "bucket_idx": self.current_bucket_idx,
            "batch_idx": self.current_batch_in_bucket,
            "order": self.order,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore state from dictionary."""
        bucket_idx = state.get("bucket_idx", 0)
        batch_idx = state.get("batch_idx", 0)
        self.set_state(bucket_idx, batch_idx)

    def reset(self) -> None:
        """Reset state to beginning."""
        self.current_bucket_idx = 0
        self.current_batch_in_bucket = 0

    @property
    def start_bucket_idx(self) -> int:
        """Get start bucket index."""
        return self._start_bucket_idx

    @property
    def start_batch_idx(self) -> int:
        """Get start batch index."""
        return self._start_batch_idx


class BufferedShuffleSamplerState(SamplerState):
    """State management for BufferedShuffleBucketSampler."""

    def __init__(self, seed: int = 42) -> None:
        self.epoch: int = 0
        self.seed: int = seed
        self.current_chunk_idx: int = 0
        self.current_bucket_in_chunk: int = 0
        self.current_batch_in_bucket: int = 0
        self._restored_rng_state: Optional[Tuple[Any, ...]] = None

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch."""
        self.epoch = epoch

    def get_state(self) -> Dict[str, int]:
        """Get simplified state (backward compatible)."""
        return {
            "epoch": self.epoch,
            "chunk_idx": self.current_chunk_idx,
            "bucket_in_chunk": self.current_bucket_in_chunk,
            "batch_in_bucket": self.current_batch_in_bucket,
        }

    def state_dict(self) -> Dict[str, Any]:
        """Return complete state dictionary including RNG state."""
        rng = random.Random(self.seed + self.epoch)
        rng_state = rng.getstate()

        return {
            "epoch": self.epoch,
            "seed": self.seed,
            "chunk_idx": self.current_chunk_idx,
            "bucket_in_chunk": self.current_bucket_in_chunk,
            "batch_in_bucket": self.current_batch_in_bucket,
            "rng_state": rng_state,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore state from dictionary."""
        self.epoch = state.get("epoch", 0)
        self.seed = state.get("seed", self.seed)
        self.current_chunk_idx = state.get("chunk_idx", 0)
        self.current_bucket_in_chunk = state.get("bucket_in_chunk", 0)
        self.current_batch_in_bucket = state.get("batch_in_bucket", 0)
        self._restored_rng_state = state.get("rng_state", None)

    def reset(self) -> None:
        """Reset state to beginning."""
        self.current_chunk_idx = 0
        self.current_bucket_in_chunk = 0
        self.current_batch_in_bucket = 0

    def create_rng(self) -> random.Random:
        """Create RNG for current epoch."""
        return random.Random(self.seed + self.epoch)


__all__ = [
    "SamplerState",
    "BucketSamplerState",
    "SequentialSamplerState",
    "BufferedShuffleSamplerState",
]
