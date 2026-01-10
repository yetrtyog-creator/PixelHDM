"""
PixelHDM-RPEA-DinoV3 Bucket Samplers

Bucket-based samplers ensuring each batch contains samples from the same bucket.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional
from torch.utils.data import Sampler

from .batch_builder import (
    BatchCounter, BatchGenerator, BucketGrouper, BucketSorter,
    ChunkBuilder, RAMEstimator,
)
from .generator import AspectRatioBucket
from .info import BufferedShuffleSamplerInfo, SequentialSamplerInfo
from .state import (
    BucketSamplerState, BufferedShuffleSamplerState, SequentialSamplerState,
)

logger = logging.getLogger(__name__)


class BucketSampler(Sampler[int]):
    """Random bucket sampler ensuring same-bucket batches."""

    def __init__(
        self, bucket_ids: List[int], batch_size: int, drop_last: bool = True,
        shuffle: bool = True, seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.bucket_ids, self.batch_size = bucket_ids, batch_size
        self.drop_last, self.shuffle, self.seed = drop_last, shuffle, seed
        self._state = BucketSamplerState(seed=seed)
        self._grouper = BucketGrouper(bucket_ids)
        self._counter = BatchCounter(self._grouper.bucket_to_indices, batch_size, drop_last)
        self._generator = BatchGenerator(batch_size, drop_last)
        logger.info(
            f"BucketSampler: {len(bucket_ids)} samples, "
            f"{self._grouper.num_buckets} buckets, {self._counter.total_batches} batches"
        )

    @property
    def bucket_to_indices(self) -> Dict[int, List[int]]:
        return self._grouper.bucket_to_indices

    @property
    def epoch(self) -> int:
        return self._state.epoch

    @epoch.setter
    def epoch(self, value: int) -> None:
        self._state.epoch = value

    def __iter__(self) -> Iterator[List[int]]:
        rng = self._state.create_rng()
        all_batches = []
        for bucket_id, indices in self._grouper.bucket_to_indices.items():
            for batch in self._generator.generate_batches(indices, self.shuffle, rng):
                all_batches.append(batch)
        if self.shuffle:
            rng.shuffle(all_batches)
        for batch in all_batches:
            yield batch

    def __len__(self) -> int:
        return self._counter.total_batches

    def set_epoch(self, epoch: int) -> None:
        self._state.set_epoch(epoch)

    def state_dict(self) -> Dict[str, Any]:
        return self._state.state_dict()

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._state.load_state_dict(state)
        self.seed = self._state.seed


class SequentialBucketSampler(Sampler[int]):
    """Sequential bucket sampler (RAM optimized)."""

    def __init__(
        self, bucket_ids: List[int], bucket_manager: AspectRatioBucket,
        batch_size: int, drop_last: bool = True, order: str = "ascending",
        start_bucket_idx: int = 0, start_batch_idx: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if order not in ("ascending", "descending"):
            raise ValueError(f"order must be 'ascending' or 'descending'")
        self.bucket_ids, self.bucket_manager = bucket_ids, bucket_manager
        self.batch_size, self.drop_last, self.order = batch_size, drop_last, order
        self._state = SequentialSamplerState(order=order)
        self._state.set_state(start_bucket_idx, start_batch_idx)
        self._grouper = BucketGrouper(bucket_ids)
        self._sorter = BucketSorter(bucket_manager, bucket_ids, order)
        self._counter = BatchCounter(self._grouper.bucket_to_indices, batch_size, drop_last)
        self._generator = BatchGenerator(batch_size, drop_last)
        ram_est = RAMEstimator(bucket_manager, batch_size)
        self._info = SequentialSamplerInfo(
            self._grouper, self._counter, ram_est,
            self._sorter.sorted_bucket_ids, self._state
        )
        logger.info(
            f"SequentialBucketSampler: {len(bucket_ids)} samples, "
            f"{self._grouper.num_buckets} buckets, {self._counter.total_batches} batches"
        )

    @property
    def bucket_to_indices(self) -> Dict[int, List[int]]:
        return self._grouper.bucket_to_indices

    @property
    def sorted_bucket_ids(self) -> List[int]:
        return self._sorter.sorted_bucket_ids

    @property
    def current_bucket_idx(self) -> int:
        return self._state.current_bucket_idx

    @current_bucket_idx.setter
    def current_bucket_idx(self, v: int) -> None:
        self._state.current_bucket_idx = v

    @property
    def current_batch_in_bucket(self) -> int:
        return self._state.current_batch_in_bucket

    @current_batch_in_bucket.setter
    def current_batch_in_bucket(self, v: int) -> None:
        self._state.current_batch_in_bucket = v

    @property
    def start_bucket_idx(self) -> int:
        return self._state.start_bucket_idx

    @property
    def start_batch_idx(self) -> int:
        return self._state.start_batch_idx

    def __iter__(self) -> Iterator[List[int]]:
        sorted_ids = self._sorter.sorted_bucket_ids
        start_b, start_batch = self._state.start_bucket_idx, self._state.start_batch_idx
        for bucket_idx in range(start_b, len(sorted_ids)):
            bucket_id = sorted_ids[bucket_idx]
            indices = self._grouper.bucket_to_indices[bucket_id]
            skip = start_batch if bucket_idx == start_b else 0
            count = 0
            for batch in self._generator.generate_batches(indices):
                if count < skip:
                    count += 1
                    continue
                self._state.current_bucket_idx = bucket_idx
                self._state.current_batch_in_bucket = count + 1
                count += 1
                yield batch
        self._state.reset()

    def __len__(self) -> int:
        return self._counter.total_batches

    def get_state(self) -> Dict[str, int]:
        return self._state.get_state()

    def set_state(self, bucket_idx: int, batch_idx: int) -> None:
        self._state.set_state(bucket_idx, batch_idx)

    def state_dict(self) -> Dict[str, Any]:
        return self._state.state_dict()

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._state.load_state_dict(state)

    def get_current_bucket_info(self) -> Optional[Dict[str, Any]]:
        return self._info.get_current_bucket_info()

    def estimate_ram_usage(self, bytes_per_pixel: float = 12.0) -> Dict[str, float]:
        return self._info.estimate_ram_usage(bytes_per_pixel)


class BufferedShuffleBucketSampler(Sampler[int]):
    """Buffered shuffle bucket sampler (recommended)."""

    def __init__(
        self, bucket_ids: List[int], bucket_manager: AspectRatioBucket,
        batch_size: int, drop_last: bool = True, chunk_size: int = 4,
        shuffle_chunks: bool = True, shuffle_within_bucket: bool = True,
        seed: int = 42, prefetch_chunks: int = 1,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.bucket_ids, self.bucket_manager = bucket_ids, bucket_manager
        self.batch_size, self.drop_last, self.chunk_size = batch_size, drop_last, chunk_size
        self.shuffle_chunks, self.shuffle_within_bucket = shuffle_chunks, shuffle_within_bucket
        self.seed, self.prefetch_chunks = seed, prefetch_chunks
        self._state = BufferedShuffleSamplerState(seed=seed)
        self._grouper = BucketGrouper(bucket_ids)
        self._sorter = BucketSorter(bucket_manager, bucket_ids, "ascending")
        self._chunker = ChunkBuilder(self._sorter.sorted_bucket_ids, chunk_size)
        self._counter = BatchCounter(self._grouper.bucket_to_indices, batch_size, drop_last)
        self._generator = BatchGenerator(batch_size, drop_last)
        ram_est = RAMEstimator(bucket_manager, batch_size)
        self._info = BufferedShuffleSamplerInfo(
            self._grouper, self._chunker, ram_est, self._state, batch_size, prefetch_chunks
        )
        logger.info(
            f"BufferedShuffleBucketSampler: {len(bucket_ids)} samples, "
            f"{self._grouper.num_buckets} buckets, {self._chunker.num_chunks} chunks, "
            f"{self._counter.total_batches} batches"
        )

    @property
    def bucket_to_indices(self) -> Dict[int, List[int]]:
        return self._grouper.bucket_to_indices

    @property
    def sorted_bucket_ids(self) -> List[int]:
        return self._sorter.sorted_bucket_ids

    @property
    def chunks(self) -> List[List[int]]:
        return self._chunker.chunks

    @property
    def epoch(self) -> int:
        return self._state.epoch

    @property
    def current_chunk_idx(self) -> int:
        return self._state.current_chunk_idx

    @property
    def current_bucket_in_chunk(self) -> int:
        return self._state.current_bucket_in_chunk

    @property
    def current_batch_in_bucket(self) -> int:
        return self._state.current_batch_in_bucket

    def __iter__(self) -> Iterator[List[int]]:
        rng = self._state.create_rng()
        chunks = [c.copy() for c in self._chunker.chunks]
        if self.shuffle_chunks:
            rng.shuffle(chunks)
        for chunk_idx, chunk in enumerate(chunks):
            self._state.current_chunk_idx = chunk_idx
            for bucket_idx, bucket_id in enumerate(chunk):
                self._state.current_bucket_in_chunk = bucket_idx
                indices = self._grouper.bucket_to_indices[bucket_id]
                count = 0
                for batch in self._generator.generate_batches(
                    indices, self.shuffle_within_bucket, rng
                ):
                    self._state.current_batch_in_bucket = count
                    count += 1
                    yield batch
        self._state.reset()

    def __len__(self) -> int:
        return self._counter.total_batches

    def set_epoch(self, epoch: int) -> None:
        self._state.set_epoch(epoch)

    def state_dict(self) -> Dict[str, Any]:
        return self._state.state_dict()

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._state.load_state_dict(state)
        self.seed = self._state.seed

    def get_state(self) -> Dict[str, int]:
        return self._state.get_state()

    def get_current_chunk_info(self) -> Optional[Dict[str, Any]]:
        return self._info.get_current_chunk_info()

    def get_prefetch_info(self) -> List[Dict[str, Any]]:
        return self._info.get_prefetch_info()

    def estimate_buffer_size(self, bytes_per_pixel: float = 12.0) -> Dict[str, float]:
        return self._info.estimate_buffer_size(bytes_per_pixel)


__all__ = ["BucketSampler", "SequentialBucketSampler", "BufferedShuffleBucketSampler"]
