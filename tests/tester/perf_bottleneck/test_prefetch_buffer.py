from __future__ import annotations

import pytest

from tests.tester.perf_bottleneck.prefetch_buffer import ThreadPrefetchDataLoader


class _DummyLoader:
    def __init__(self, n: int):
        self.n = n
        self.dataset = list(range(n))

    def __len__(self):
        return self.n

    def __iter__(self):
        for i in range(self.n):
            yield {"value": i}


def test_prefetch_loader_preserves_order_and_length() -> None:
    base = _DummyLoader(10)
    wrapped = ThreadPrefetchDataLoader(base, buffer_items=3)
    try:
        rows = list(iter(wrapped))
    finally:
        wrapped.close()

    assert len(rows) == 10
    assert [r["value"] for r in rows] == list(range(10))
    assert len(wrapped) == 10


def test_prefetch_loader_supports_multiple_iters() -> None:
    base = _DummyLoader(5)
    wrapped = ThreadPrefetchDataLoader(base, buffer_items=2)
    try:
        first = [r["value"] for r in wrapped]
        second = [r["value"] for r in wrapped]
    finally:
        wrapped.close()

    assert first == [0, 1, 2, 3, 4]
    assert second == [0, 1, 2, 3, 4]


class _ErrorLoader:
    dataset = [0, 1]

    def __len__(self):
        return 2

    def __iter__(self):
        yield {"value": 0}
        raise ValueError("boom")


def test_prefetch_loader_propagates_producer_error() -> None:
    wrapped = ThreadPrefetchDataLoader(_ErrorLoader(), buffer_items=2)
    try:
        it = iter(wrapped)
        assert next(it)["value"] == 0
        with pytest.raises(RuntimeError):
            next(it)
    finally:
        wrapped.close()
