"""Thread-based prefetch buffer for dataloader iteration.

This is a test-side runtime wrapper to evaluate whether producer/consumer
buffering can smooth single-thread dataloading without touching src/.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, List


@dataclass(frozen=True)
class _ProducerError:
    exc: Exception


_END = object()


class ThreadPrefetchIterator:
    """Consumes items from a background producer thread via bounded queue."""

    def __init__(self, source: Iterable[Any], buffer_items: int, name: str) -> None:
        self._source_iter = iter(source)
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, int(buffer_items)))
        self._stop_event = threading.Event()
        self._closed = False
        self._ended = False
        self._thread = threading.Thread(
            target=self._producer_loop,
            name=f"prefetch-producer-{name}",
            daemon=True,
        )
        self._thread.start()

    def _put_with_backoff(self, item: Any) -> None:
        while not self._stop_event.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _producer_loop(self) -> None:
        try:
            for item in self._source_iter:
                if self._stop_event.is_set():
                    break
                self._put_with_backoff(item)
            self._put_with_backoff(_END)
        except Exception as exc:  # pragma: no cover - depends on runtime dataset.
            self._put_with_backoff(_ProducerError(exc=exc))

    def __iter__(self) -> "ThreadPrefetchIterator":
        return self

    def __next__(self) -> Any:
        if self._ended:
            raise StopIteration
        while True:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                if self._stop_event.is_set():
                    self._ended = True
                    raise StopIteration
                continue

            if item is _END:
                self._ended = True
                self.close()
                raise StopIteration
            if isinstance(item, _ProducerError):
                self.close()
                raise RuntimeError("Prefetch producer failed.") from item.exc
            return item

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def __del__(self) -> None:  # pragma: no cover - GC timing is non-deterministic.
        self.close()


class ThreadPrefetchDataLoader:
    """Dataloader wrapper that prefetches batches in a background thread."""

    def __init__(self, dataloader: Any, buffer_items: int = 4) -> None:
        self._dataloader = dataloader
        self._buffer_items = max(1, int(buffer_items))
        self._active_iters: List[ThreadPrefetchIterator] = []
        self._iter_count = 0

    def __len__(self) -> int:
        return len(self._dataloader)

    def __iter__(self) -> Iterator[Any]:
        self._iter_count += 1
        it = ThreadPrefetchIterator(
            source=self._dataloader,
            buffer_items=self._buffer_items,
            name=str(self._iter_count),
        )
        self._active_iters.append(it)
        return it

    def close(self) -> None:
        for it in self._active_iters:
            it.close()
        self._active_iters.clear()

    @property
    def dataset(self) -> Any:
        return self._dataloader.dataset

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dataloader, name)

