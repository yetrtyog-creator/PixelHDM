"""
Non-invasive smoke checks for src recovery contracts.

This script reuses existing test mocks where possible and validates a small
set of high-risk behaviors that are easy to regress.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

from unittest.mock import MagicMock
from unittest.mock import patch

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.training.trainer.test_step import (
    MockConfig,
    MockTextEncoder,
    create_batch,
    create_step_executor,
)
from tests.training.trainer.test_loop import (
    MockDataLoader,
    create_mock_metrics,
)
from src.training.trainer.loop import TrainingLoop
from src.training.trainer.metrics import TrainerState


def _check_prepare_batch_contract() -> Tuple[bool, str]:
    text_encoder = MockTextEncoder(seq_len=8, hidden_dim=64)
    config = MockConfig(cfg_dropout=0.0)
    executor, _ = create_step_executor(text_encoder=text_encoder, config=config)
    out = executor._prepare_batch(create_batch(include_text=True, seq_len=8, text_hidden_dim=64))
    ok = isinstance(out, tuple) and len(out) == 3
    return ok, f"_prepare_batch tuple_len={len(out) if isinstance(out, tuple) else 'N/A'}"


def _check_cfg_dropout_errors() -> Tuple[bool, str]:
    config = MockConfig(cfg_dropout=0.5)
    executor, _ = create_step_executor(text_encoder=None, config=config)
    try:
        executor._prepare_batch(create_batch(include_text=True))
    except ValueError as e:
        if "text_encoder" in str(e):
            return True, "cfg_dropout no-text_encoder raises ValueError"
        return False, f"unexpected ValueError: {e}"
    return False, "cfg_dropout no-text_encoder did not raise"


def _check_metrics_zero_safe() -> Tuple[bool, str]:
    executor, _ = create_step_executor()
    metrics, _ = executor.execute(create_batch(), step=0)
    finite = (
        metrics.step_time > 0.0
        and isinstance(metrics.samples_per_sec, float)
        and math.isfinite(metrics.samples_per_sec)
        and metrics.samples_per_sec > 0.0
    )
    return finite, (
        f"step_time={metrics.step_time:.8f}, samples_per_sec={metrics.samples_per_sec:.8f}"
    )


def _check_metrics_zero_safe_constant_clock() -> Tuple[bool, str]:
    executor, _ = create_step_executor()
    with patch("src.training.trainer.step.time.time", return_value=123.0):
        try:
            metrics, _ = executor.execute(create_batch(), step=0)
        except Exception as e:
            return False, f"constant-clock execute raised {type(e).__name__}: {e}"
    finite = (
        isinstance(metrics.samples_per_sec, float)
        and math.isfinite(metrics.samples_per_sec)
        and metrics.step_time > 0.0
    )
    return finite, (
        f"constant_clock step_time={metrics.step_time:.8f}, samples_per_sec={metrics.samples_per_sec:.8f}"
    )


def _check_loop_epoch_step_boundary() -> Tuple[bool, str]:
    dataloader = MockDataLoader(num_batches=3, batch_size=2)
    loop = TrainingLoop(dataloader=dataloader, training_config=None, state=TrainerState())

    def train_step(_batch):
        loop.state.step += 1
        return create_mock_metrics()

    loop.run(
        train_step_fn=train_step,
        save_checkpoint_fn=lambda *a, **k: None,
        num_steps=9,
        log_interval=0,
        save_interval=0,
        use_progress_bar=False,
    )
    return loop.state.epoch == 3, f"epoch={loop.state.epoch}, step={loop.state.step}"


def _check_loop_checkpoint_contracts() -> Tuple[bool, str]:
    training_config = MagicMock()
    training_config.checkpoint_dir = "/tmp/recovery_smoke_ckpt"
    training_config.save_every_epochs = 1
    training_config.log_every_epochs = 0
    training_config.gradient_accumulation_steps = 1
    training_config.drop_last_accumulation = True
    training_config.training_mode = "steps"
    training_config.max_steps = 6

    dataloader = MockDataLoader(num_batches=5, batch_size=2)
    loop = TrainingLoop(dataloader=dataloader, training_config=training_config, state=TrainerState())
    saves: List[Tuple[str, Any]] = []

    def train_step(_batch):
        loop.state.step += 1
        return create_mock_metrics()

    def save_ckpt(path, checkpoint_name=None, **_kwargs):
        saves.append((str(path), checkpoint_name))

    loop.run(
        train_step_fn=train_step,
        save_checkpoint_fn=save_ckpt,
        num_steps=6,
        save_interval=5,
        save_every_epochs=1,
        log_interval=0,
        save_path=None,
        use_progress_bar=False,
    )

    used_fallback = all(p == "/tmp/recovery_smoke_ckpt" for p, _ in saves) and len(saves) > 0
    periodic = [n for _, n in saves if n and n != "checkpoint_completed"]
    name_ok = any(isinstance(n, str) and "checkpoint_epoch" in n and "_step" in n for n in periodic)
    dedupe_ok = periodic.count("checkpoint_epoch1_step5") <= 1
    ok = used_fallback and name_ok and dedupe_ok
    return ok, f"saves={saves}"


def main() -> int:
    checks = [
        ("prepare_batch_contract", _check_prepare_batch_contract),
        ("cfg_dropout_errors", _check_cfg_dropout_errors),
        ("metrics_zero_safe", _check_metrics_zero_safe),
        ("metrics_zero_safe_constant_clock", _check_metrics_zero_safe_constant_clock),
        ("loop_epoch_step_boundary", _check_loop_epoch_step_boundary),
        ("loop_checkpoint_contracts", _check_loop_checkpoint_contracts),
    ]

    results: List[Dict[str, Any]] = []
    all_ok = True
    for name, fn in checks:
        try:
            ok, detail = fn()
        except Exception as e:  # pragma: no cover
            ok, detail = False, f"exception: {type(e).__name__}: {e}"
        all_ok = all_ok and ok
        results.append({"check": name, "ok": ok, "detail": detail})

    report = {
        "ok": all_ok,
        "checks": results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
