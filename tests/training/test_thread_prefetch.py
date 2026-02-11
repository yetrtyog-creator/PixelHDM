from __future__ import annotations

import sys

import pytest

from src.training.prefetch import (
    ThreadPrefetchDataLoader,
    resolve_thread_prefetch_config,
)
from src.training.train import parse_args


class _DummyLoader:
    def __init__(self, n: int):
        self.n = n
        self.dataset = list(range(n))

    def __len__(self):
        return self.n

    def __iter__(self):
        for i in range(self.n):
            yield {"value": i}


class _ErrorLoader:
    dataset = [0, 1]

    def __len__(self):
        return 2

    def __iter__(self):
        yield {"value": 0}
        raise ValueError("boom")


def test_resolve_prefetch_default_2x_for_single_worker() -> None:
    cfg = resolve_thread_prefetch_config(
        gradient_accumulation_steps=3,
        num_workers=0,
        cli_enabled=True,
        buffer_multiplier=2,
        buffer_items=None,
    )
    assert cfg.enabled is True
    assert cfg.resolved_items == 6


def test_resolve_prefetch_disabled_when_workers_positive() -> None:
    cfg = resolve_thread_prefetch_config(
        gradient_accumulation_steps=3,
        num_workers=2,
        cli_enabled=True,
        buffer_multiplier=2,
        buffer_items=None,
    )
    assert cfg.enabled is False
    assert cfg.resolved_items == 0


def test_resolve_prefetch_buffer_items_override_multiplier() -> None:
    cfg = resolve_thread_prefetch_config(
        gradient_accumulation_steps=3,
        num_workers=0,
        cli_enabled=True,
        buffer_multiplier=2,
        buffer_items=5,
    )
    assert cfg.enabled is True
    assert cfg.resolved_items == 5


def test_parse_args_prefetch_defaults_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py"])
    args = parse_args()
    assert args.thread_prefetch_enabled is True
    assert args.prefetch_buffer_multiplier == 2
    assert args.prefetch_buffer_items is None


def test_parse_args_disable_prefetch_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py", "--disable-thread-prefetch"])
    args = parse_args()
    assert args.thread_prefetch_enabled is False


def test_thread_prefetch_loader_preserves_order() -> None:
    wrapped = ThreadPrefetchDataLoader(_DummyLoader(6), buffer_items=2)
    try:
        rows = [row["value"] for row in wrapped]
    finally:
        wrapped.close()
    assert rows == [0, 1, 2, 3, 4, 5]


def test_thread_prefetch_loader_surfaces_producer_error() -> None:
    wrapped = ThreadPrefetchDataLoader(_ErrorLoader(), buffer_items=2)
    try:
        it = iter(wrapped)
        assert next(it)["value"] == 0
        with pytest.raises(RuntimeError):
            next(it)
    finally:
        wrapped.close()

