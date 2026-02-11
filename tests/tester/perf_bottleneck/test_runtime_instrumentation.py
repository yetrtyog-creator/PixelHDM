from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, Tuple

import torch

from tests.tester.perf_bottleneck.runtime_instrumentation import (
    RuntimeHookSession,
    StepMetricsCollector,
)


@dataclass
class DummyMetrics:
    loss: float = 1.0
    loss_vloss: float = 0.4
    loss_freq: float = 0.3
    loss_repa: float = 0.2
    loss_gamma_l2: float = 0.1
    grad_norm: float = 0.9
    learning_rate: float = 1e-4
    samples_per_sec: float = 128.0
    step_time: float = 0.02


class DummyLoop:
    def __init__(self) -> None:
        self.state = SimpleNamespace(step=0, epoch=0, batch_idx=0)

    def _collect_micro_batches(
        self,
        data_iter: Iterator[Dict[str, Any]],
        gradient_accumulation_steps: int,
        drop_last_accumulation: bool,
    ) -> Tuple[list[Dict[str, Any]], Iterator[Dict[str, Any]]]:
        del data_iter, drop_last_accumulation
        payload = [
            {"images": torch.zeros((2, 3, 32, 32), dtype=torch.float32)},
            {"images": torch.zeros((2, 3, 32, 32), dtype=torch.float32)},
        ]
        self.state.batch_idx += gradient_accumulation_steps
        return payload, iter(())

    def _handle_gc(self, gc_interval: int):
        if gc_interval > 0 and self.state.step % gc_interval == 0:
            return None
        return None


class DummyStepExecutor:
    def __init__(self) -> None:
        self.gradient_accumulation_steps = 2

    def _prepare_batch(self, batch: Dict[str, torch.Tensor]):
        return batch["images"], None, None

    def _forward(self, *args, **kwargs):
        del args, kwargs
        return {"total": torch.tensor(1.0)}, torch.tensor(0.0)

    def _backward_scaled(self, *args, **kwargs):
        del args, kwargs
        return None

    def forward_backward(self, batch: Dict[str, torch.Tensor], step: int, denom: int) -> int:
        del step, denom
        images, _, _ = self._prepare_batch(batch)
        loss_dict, _ = self._forward()
        self._backward_scaled(loss_dict["total"], 2)
        return int(images.shape[0])

    def optimizer_step_and_metrics(self, step: int, *args, **kwargs):
        del step, args, kwargs
        return DummyMetrics(), 4


def _run_dummy_step(loop: DummyLoop, step_exec: DummyStepExecutor, collector: StepMetricsCollector, step_zero_based: int):
    loop.state.step = step_zero_based
    payload, _ = loop._collect_micro_batches(iter(()), gradient_accumulation_steps=2, drop_last_accumulation=True)
    for batch in payload:
        step_exec.forward_backward(batch, step=step_zero_based, denom=2)
    metrics, _ = step_exec.optimizer_step_and_metrics(step=step_zero_based)
    loop.state.step = step_zero_based + 1
    loop._handle_gc(gc_interval=1)
    return collector.finalize_step(
        step=step_zero_based + 1,
        metrics=metrics,
        epoch=loop.state.epoch,
        batch_idx=loop.state.batch_idx,
    )


def test_runtime_hooks_capture_phase0_fields(tmp_path: Path) -> None:
    collector = StepMetricsCollector(
        run_id="rt_hook_ut",
        output_path=tmp_path / "step_metrics.jsonl",
        timing_mode="light",
        profile_stride=50,
    )
    try:
        with RuntimeHookSession(collector, DummyLoop, DummyStepExecutor):
            rec = _run_dummy_step(DummyLoop(), DummyStepExecutor(), collector, step_zero_based=0)
    finally:
        collector.close()

    assert rec["run_id"] == "rt_hook_ut"
    assert rec["step"] == 1
    assert rec["micro_batches_collected"] == 2
    assert rec["batch_size_total"] == 4
    assert rec["image_h"] == 32
    assert rec["image_w"] == 32
    assert rec["data_wait_ms"] >= 0.0
    assert rec["h2d_ms"] >= 0.0
    assert rec["fwd_bwd_ms"] >= 0.0
    assert rec["opt_ms"] >= 0.0
    assert rec["gc_ms"] >= 0.0
    assert rec["step_time_ms"] > 0.0
    assert rec["timing_mode"] == "light"


def test_probe_stride_marks_only_stride_steps_as_probe(tmp_path: Path) -> None:
    out_path = tmp_path / "step_metrics.jsonl"
    collector = StepMetricsCollector(
        run_id="rt_hook_probe_ut",
        output_path=out_path,
        timing_mode="probe",
        profile_stride=2,
    )
    try:
        with RuntimeHookSession(collector, DummyLoop, DummyStepExecutor):
            rec1 = _run_dummy_step(DummyLoop(), DummyStepExecutor(), collector, step_zero_based=0)
            rec2 = _run_dummy_step(DummyLoop(), DummyStepExecutor(), collector, step_zero_based=1)
    finally:
        collector.close()

    assert rec1["step"] == 1
    assert rec1["timing_mode"] == "light"
    assert rec2["step"] == 2
    assert rec2["timing_mode"] == "probe"

    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
    assert rows[0]["step"] == 1
    assert rows[1]["step"] == 2

