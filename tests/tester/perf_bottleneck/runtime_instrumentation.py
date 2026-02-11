"""Runtime-only training instrumentation for perf profiling.

This module installs monkeypatch hooks on trainer classes at runtime so we can
collect phase-0 timing slices without editing src/ code paths.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _to_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _inspect_payload(payload: Any) -> tuple[int, int, Optional[int], Optional[int]]:
    """Return (micro_batch_count, total_batch_size, image_h, image_w)."""
    if isinstance(payload, list):
        micro_batches = payload
    elif isinstance(payload, dict):
        micro_batches = [payload]
    else:
        return 0, 0, None, None

    total_batch = 0
    image_h: Optional[int] = None
    image_w: Optional[int] = None

    for batch in micro_batches:
        if not isinstance(batch, dict):
            continue
        images = batch.get("images")
        if isinstance(images, torch.Tensor) and images.ndim >= 4:
            batch_size = int(images.shape[0])
            total_batch += batch_size
            if image_h is None:
                image_h = int(images.shape[-2])
                image_w = int(images.shape[-1])

    return len(micro_batches), total_batch, image_h, image_w


@dataclass
class StepSlice:
    epoch: int = 0
    batch_idx: int = 0
    grad_accumulation_steps: int = 1
    micro_batches_collected: int = 1
    batch_size_total: int = 0
    image_h: Optional[int] = None
    image_w: Optional[int] = None
    data_wait_ms: float = 0.0
    h2d_ms: float = 0.0
    fwd_bwd_ms: float = 0.0
    opt_ms: float = 0.0
    gc_ms: float = 0.0
    oom_retry_count: int = 0


class StepMetricsCollector:
    """Collect and stream step-level perf metrics to JSONL."""

    def __init__(
        self,
        run_id: str,
        output_path: Path,
        timing_mode: str = "light",
        profile_stride: int = 50,
    ) -> None:
        self.run_id = run_id
        self.output_path = output_path
        self.timing_mode = timing_mode
        self.profile_stride = max(1, int(profile_stride))
        self._steps: Dict[int, StepSlice] = {}
        self._fp = self.output_path.open("w", encoding="utf-8")

    def close(self) -> None:
        if not self._fp.closed:
            self._fp.flush()
            self._fp.close()

    def _ensure_step(self, step: int) -> StepSlice:
        if step not in self._steps:
            self._steps[step] = StepSlice()
        return self._steps[step]

    def is_probe_step(self, step: int) -> bool:
        if self.timing_mode != "probe":
            return False
        return (step % self.profile_stride) == 0

    def effective_mode_for_step(self, step: int) -> str:
        if self.is_probe_step(step):
            return "probe"
        return "light"

    def sync_cuda_if_needed(self, step: int) -> None:
        if not self.is_probe_step(step):
            return
        if not torch.cuda.is_available():
            return
        torch.cuda.synchronize()

    def record_data_wait(
        self,
        *,
        step: int,
        epoch: int,
        batch_idx: int,
        grad_accumulation_steps: int,
        micro_batches_collected: int,
        batch_size_total: int,
        image_h: Optional[int],
        image_w: Optional[int],
        elapsed_ms: float,
    ) -> None:
        slot = self._ensure_step(step)
        slot.epoch = int(epoch)
        slot.batch_idx = int(batch_idx)
        slot.grad_accumulation_steps = max(1, int(grad_accumulation_steps))
        slot.micro_batches_collected = max(1, int(micro_batches_collected))
        if batch_size_total > 0:
            slot.batch_size_total = int(batch_size_total)
        if image_h is not None:
            slot.image_h = int(image_h)
        if image_w is not None:
            slot.image_w = int(image_w)
        slot.data_wait_ms += float(elapsed_ms)

    def record_h2d(
        self,
        *,
        step: int,
        elapsed_ms: float,
        images: Optional[torch.Tensor],
    ) -> None:
        slot = self._ensure_step(step)
        slot.h2d_ms += float(elapsed_ms)
        if isinstance(images, torch.Tensor) and images.ndim >= 4:
            if slot.batch_size_total <= 0:
                slot.batch_size_total = int(images.shape[0])
            if slot.image_h is None:
                slot.image_h = int(images.shape[-2])
                slot.image_w = int(images.shape[-1])

    def record_fwd_bwd(self, *, step: int, elapsed_ms: float) -> None:
        slot = self._ensure_step(step)
        slot.fwd_bwd_ms += float(elapsed_ms)

    def record_opt(
        self,
        *,
        step: int,
        elapsed_ms: float,
        total_batch_size: int,
        grad_accumulation_steps: int,
    ) -> None:
        slot = self._ensure_step(step)
        slot.opt_ms += float(elapsed_ms)
        slot.grad_accumulation_steps = max(1, int(grad_accumulation_steps))
        if total_batch_size > 0:
            slot.batch_size_total = int(total_batch_size)

    def record_gc(self, *, step: int, elapsed_ms: float) -> None:
        slot = self._ensure_step(step)
        slot.gc_ms += float(elapsed_ms)

    def finalize_step(
        self,
        *,
        step: int,
        metrics: Any,
        epoch: int,
        batch_idx: int,
    ) -> Dict[str, Any]:
        slot = self._steps.pop(step, StepSlice())
        slot.epoch = int(epoch)
        slot.batch_idx = int(batch_idx)

        step_time_ms = _to_float(getattr(metrics, "step_time", None), default=0.0) * 1000.0
        if step_time_ms <= 0:
            step_time_ms = (
                slot.data_wait_ms + slot.h2d_ms + slot.fwd_bwd_ms + slot.opt_ms + slot.gc_ms
            )
        if step_time_ms <= 0:
            step_time_ms = 1e-6

        if slot.batch_size_total <= 0:
            inferred = _to_float(getattr(metrics, "samples_per_sec", None), default=0.0)
            slot.batch_size_total = max(1, int(round(inferred * (step_time_ms / 1000.0))))

        row: Dict[str, Any] = {
            "run_id": self.run_id,
            "timestamp_utc": _now_utc_iso(),
            "step": int(step),
            "epoch": slot.epoch,
            "batch_idx": slot.batch_idx,
            "grad_accumulation_steps": slot.grad_accumulation_steps,
            "micro_batches_collected": slot.micro_batches_collected,
            "batch_size_total": slot.batch_size_total,
            "image_h": slot.image_h,
            "image_w": slot.image_w,
            "data_wait_ms": slot.data_wait_ms,
            "h2d_ms": slot.h2d_ms,
            "fwd_bwd_ms": slot.fwd_bwd_ms,
            "opt_ms": slot.opt_ms,
            "gc_ms": slot.gc_ms,
            "step_time_ms": step_time_ms,
            "samples_per_sec": _to_float(getattr(metrics, "samples_per_sec", None), default=0.0),
            "loss_total": _to_float(getattr(metrics, "loss", None), default=0.0),
            "loss_vloss": _to_float(getattr(metrics, "loss_vloss", None), default=0.0),
            "loss_freq": _to_float(getattr(metrics, "loss_freq", None), default=0.0),
            "loss_repa": _to_float(getattr(metrics, "loss_repa", None), default=0.0),
            "loss_gamma_l2": _to_float(getattr(metrics, "loss_gamma_l2", None), default=0.0),
            "learning_rate": _to_float(getattr(metrics, "learning_rate", None), default=0.0),
            "oom_retry_count": slot.oom_retry_count,
            "timing_mode": self.effective_mode_for_step(step),
        }
        self._fp.write(json.dumps(row) + "\n")
        self._fp.flush()
        return row


class RuntimeHookSession:
    """Install runtime monkeypatch hooks and restore them on exit."""

    def __init__(
        self,
        collector: StepMetricsCollector,
        loop_cls: Any,
        step_executor_cls: Any,
    ) -> None:
        self.collector = collector
        self.loop_cls = loop_cls
        self.step_executor_cls = step_executor_cls
        self._originals: Dict[tuple[Any, str], Any] = {}

    def __enter__(self) -> "RuntimeHookSession":
        self.install()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.uninstall()

    def _patch(self, cls: Any, method_name: str, wrapped: Any) -> None:
        key = (cls, method_name)
        if key not in self._originals:
            self._originals[key] = getattr(cls, method_name)
        setattr(cls, method_name, wrapped)

    def install(self) -> None:
        collector = self.collector

        orig_collect = getattr(self.loop_cls, "_collect_micro_batches")

        @wraps(orig_collect)
        def wrapped_collect(loop_self, data_iter, gradient_accumulation_steps, drop_last_accumulation):
            step = int(getattr(loop_self.state, "step", 0)) + 1
            t0 = time.perf_counter_ns()
            payload, next_iter = orig_collect(
                loop_self,
                data_iter,
                gradient_accumulation_steps,
                drop_last_accumulation,
            )
            elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            micro_batches, batch_size_total, image_h, image_w = _inspect_payload(payload)
            collector.record_data_wait(
                step=step,
                epoch=int(getattr(loop_self.state, "epoch", 0)),
                batch_idx=int(getattr(loop_self.state, "batch_idx", 0)),
                grad_accumulation_steps=int(gradient_accumulation_steps),
                micro_batches_collected=max(1, micro_batches),
                batch_size_total=batch_size_total,
                image_h=image_h,
                image_w=image_w,
                elapsed_ms=elapsed_ms,
            )
            return payload, next_iter

        self._patch(self.loop_cls, "_collect_micro_batches", wrapped_collect)

        orig_handle_gc = getattr(self.loop_cls, "_handle_gc")

        @wraps(orig_handle_gc)
        def wrapped_handle_gc(loop_self, gc_interval):
            step = int(getattr(loop_self.state, "step", 0))
            t0 = time.perf_counter_ns()
            out = orig_handle_gc(loop_self, gc_interval)
            elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            if gc_interval > 0 and step > 0 and (step % gc_interval) == 0:
                collector.record_gc(step=step, elapsed_ms=elapsed_ms)
            return out

        self._patch(self.loop_cls, "_handle_gc", wrapped_handle_gc)

        orig_forward_backward = getattr(self.step_executor_cls, "forward_backward")

        @wraps(orig_forward_backward)
        def wrapped_forward_backward(step_self, batch, step, denom):
            prev = getattr(step_self, "_perf_trace_active_step", None)
            step_self._perf_trace_active_step = int(step) + 1
            try:
                return orig_forward_backward(step_self, batch, step, denom)
            finally:
                step_self._perf_trace_active_step = prev

        self._patch(self.step_executor_cls, "forward_backward", wrapped_forward_backward)

        orig_prepare_batch = getattr(self.step_executor_cls, "_prepare_batch")

        @wraps(orig_prepare_batch)
        def wrapped_prepare_batch(step_self, batch):
            active_step = getattr(step_self, "_perf_trace_active_step", None)
            if isinstance(active_step, int):
                collector.sync_cuda_if_needed(active_step)
            t0 = time.perf_counter_ns()
            images, text_embeddings, text_mask = orig_prepare_batch(step_self, batch)
            elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            if isinstance(active_step, int):
                collector.sync_cuda_if_needed(active_step)
                collector.record_h2d(step=active_step, elapsed_ms=elapsed_ms, images=images)
            return images, text_embeddings, text_mask

        self._patch(self.step_executor_cls, "_prepare_batch", wrapped_prepare_batch)

        orig_forward = getattr(self.step_executor_cls, "_forward")

        @wraps(orig_forward)
        def wrapped_forward(step_self, *args, **kwargs):
            active_step = getattr(step_self, "_perf_trace_active_step", None)
            if isinstance(active_step, int):
                collector.sync_cuda_if_needed(active_step)
            t0 = time.perf_counter_ns()
            out = orig_forward(step_self, *args, **kwargs)
            elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            if isinstance(active_step, int):
                collector.sync_cuda_if_needed(active_step)
                collector.record_fwd_bwd(step=active_step, elapsed_ms=elapsed_ms)
            return out

        self._patch(self.step_executor_cls, "_forward", wrapped_forward)

        orig_backward_scaled = getattr(self.step_executor_cls, "_backward_scaled")

        @wraps(orig_backward_scaled)
        def wrapped_backward_scaled(step_self, *args, **kwargs):
            active_step = getattr(step_self, "_perf_trace_active_step", None)
            if isinstance(active_step, int):
                collector.sync_cuda_if_needed(active_step)
            t0 = time.perf_counter_ns()
            out = orig_backward_scaled(step_self, *args, **kwargs)
            elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            if isinstance(active_step, int):
                collector.sync_cuda_if_needed(active_step)
                collector.record_fwd_bwd(step=active_step, elapsed_ms=elapsed_ms)
            return out

        self._patch(self.step_executor_cls, "_backward_scaled", wrapped_backward_scaled)

        orig_opt = getattr(self.step_executor_cls, "optimizer_step_and_metrics")

        @wraps(orig_opt)
        def wrapped_opt(step_self, step, *args, **kwargs):
            one_based_step = int(step) + 1
            collector.sync_cuda_if_needed(one_based_step)
            t0 = time.perf_counter_ns()
            metrics, total_batch_size = orig_opt(step_self, step, *args, **kwargs)
            elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            collector.sync_cuda_if_needed(one_based_step)
            collector.record_opt(
                step=one_based_step,
                elapsed_ms=elapsed_ms,
                total_batch_size=int(total_batch_size),
                grad_accumulation_steps=int(getattr(step_self, "gradient_accumulation_steps", 1)),
            )
            return metrics, total_batch_size

        self._patch(self.step_executor_cls, "optimizer_step_and_metrics", wrapped_opt)

    def uninstall(self) -> None:
        for (cls, method_name), original in self._originals.items():
            setattr(cls, method_name, original)
        self._originals.clear()

