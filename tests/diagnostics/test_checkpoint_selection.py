"""Tests for numeric checkpoint selection in diagnostics scripts."""

from __future__ import annotations

import pytest

from tests.diagnostics.analyze_model_by_t import (
    _parse_epoch_step as parse_model,
    _pick_latest_epoch_step as pick_model,
)
from tests.diagnostics.analyze_repa_by_t import (
    _parse_epoch_step as parse_repa,
    _pick_latest_epoch_step as pick_repa,
)


HELPERS = [
    ("model", parse_model, pick_model),
    ("repa", parse_repa, pick_repa),
]


def _basename(path_obj) -> str:
    """Cross-platform basename for mixed slash inputs."""
    return str(path_obj).replace("\\", "/").rsplit("/", 1)[-1]


@pytest.mark.parametrize("_, parse_fn, __", HELPERS)
def test_parse_epoch_step_valid(_, parse_fn, __):
    assert parse_fn("checkpoint_epoch12_step345.pt") == (12, 345)
    assert parse_fn("checkpoints/checkpoint_epoch12_step345.pt") == (12, 345)
    assert parse_fn(r"checkpoints\checkpoint_epoch12_step345.pt") == (12, 345)
    assert parse_fn(r"C:\runs\checkpoint_epoch12_step345.pt") == (12, 345)


@pytest.mark.parametrize("_, parse_fn, __", HELPERS)
def test_parse_epoch_step_invalid_returns_none(_, parse_fn, __):
    assert parse_fn("checkpoint_epoch_12.pt") is None
    assert parse_fn("checkpoint_epoch12_step.pt") is None


@pytest.mark.parametrize("name, _, pick_fn", HELPERS)
def test_pick_latest_uses_numeric_epoch_then_step(name, _, pick_fn):
    latest, ignored = pick_fn(
        [
            r"checkpoints\checkpoint_epoch9_step900.pt",
            "checkpoints/checkpoint_epoch10_step1.pt",
        ]
    )
    assert ignored == 0, f"{name}: expected no ignored files"
    assert latest is not None
    assert _basename(latest) == "checkpoint_epoch10_step1.pt"


@pytest.mark.parametrize("name, _, pick_fn", HELPERS)
def test_pick_latest_with_same_epoch_uses_larger_step(name, _, pick_fn):
    latest, ignored = pick_fn(
        [
            r"checkpoints\checkpoint_epoch10_step1.pt",
            "checkpoints/checkpoint_epoch10_step99.pt",
            r"checkpoints\checkpoint_epoch10_step9.pt",
        ]
    )
    assert ignored == 0, f"{name}: expected no ignored files"
    assert latest is not None
    assert _basename(latest) == "checkpoint_epoch10_step99.pt"


@pytest.mark.parametrize("name, _, pick_fn", HELPERS)
def test_pick_latest_ignores_invalid_names_without_crashing(name, _, pick_fn):
    latest, ignored = pick_fn(
        [
            "checkpoints/checkpoint_epoch3_step7.pt",
            r"checkpoints\checkpoint_epoch_10.pt",
            r"checkpoints\checkpoint_epochX_step5.pt",
            r"checkpoints\random.pt",
        ]
    )
    assert latest is not None
    assert _basename(latest) == "checkpoint_epoch3_step7.pt"
    assert ignored == 3, f"{name}: expected 3 ignored files"


@pytest.mark.parametrize("name, _, pick_fn", HELPERS)
def test_pick_latest_returns_none_when_no_valid_files(name, _, pick_fn):
    latest, ignored = pick_fn(
        [
            r"checkpoints\checkpoint_epoch_10.pt",
            r"checkpoints\checkpoint_epochX_step5.pt",
        ]
    )
    assert latest is None, f"{name}: expected no valid checkpoint"
    assert ignored == 2, f"{name}: expected 2 ignored files"
