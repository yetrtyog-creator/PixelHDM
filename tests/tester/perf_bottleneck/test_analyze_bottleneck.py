import json
from pathlib import Path

import pytest

from tests.tester.perf_bottleneck.analyze_bottleneck import (
    analyze,
    filter_warmup,
    parse_gpu_telemetry,
    parse_step_metrics,
)


def _write_step_metrics(path: Path) -> None:
    records = []
    for step in range(1, 7):
        ts = f"2026-02-10T00:00:0{step}Z"
        records.append(
            {
                "run_id": "unit_test_run",
                "timestamp_utc": ts,
                "step": step,
                "epoch": 0,
                "batch_idx": step,
                "grad_accumulation_steps": 1,
                "micro_batches_collected": 1,
                "batch_size_total": 2,
                "data_wait_ms": float(step * 5),
                "h2d_ms": 10.0,
                "fwd_bwd_ms": 60.0,
                "opt_ms": 5.0,
                "gc_ms": 0.0,
                "step_time_ms": 100.0,
                "samples_per_sec": 20.0,
                "loss_total": 1.0,
                "loss_vloss": 0.4,
                "loss_freq": 0.3,
                "loss_repa": 0.2,
                "loss_gamma_l2": 0.1,
                "learning_rate": 1e-4,
            }
        )

    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _write_gpu_csv(path: Path) -> None:
    # utilization.gpu inversely correlates with data_wait_ms in the sample.
    rows = [
        "2026/02/10 00:00:01.000,90,12,1000,12000,100",
        "2026/02/10 00:00:02.000,85,12,1000,12000,100",
        "2026/02/10 00:00:03.000,80,12,1000,12000,100",
        "2026/02/10 00:00:04.000,75,12,1000,12000,100",
        "2026/02/10 00:00:05.000,70,12,1000,12000,100",
        "2026/02/10 00:00:06.000,65,12,1000,12000,100",
    ]
    path.write_text("\n".join(rows), encoding="utf-8")


def test_parse_and_warmup(tmp_path: Path) -> None:
    step_path = tmp_path / "step_metrics.jsonl"
    _write_step_metrics(step_path)

    parsed = parse_step_metrics(step_path)
    assert parsed.parse_errors == 0
    assert len(parsed.records) == 6
    assert parsed.step_gap_count == 0
    assert parsed.timestamp_coverage_pct == 100.0

    filtered = filter_warmup(parsed.records, warmup_steps=2)
    assert len(filtered) == 4
    assert filtered[0]["step"] == 3


def test_analyze_ratios_and_correlations(tmp_path: Path) -> None:
    step_path = tmp_path / "step_metrics.jsonl"
    gpu_path = tmp_path / "gpu_telemetry.csv"
    _write_step_metrics(step_path)
    _write_gpu_csv(gpu_path)

    gpu_parsed = parse_gpu_telemetry(gpu_path)
    assert gpu_parsed.parse_errors == 0
    assert len(gpu_parsed.records) == 6

    result = analyze(
        step_metrics_path=step_path,
        gpu_telemetry_path=gpu_path,
        warmup_steps=2,
        run_id=None,
        max_align_gap_sec=1.0,
    )
    summary = result["summary"]

    # Remaining steps are 3,4,5,6.
    assert summary["effective_steps"] == 4

    ratios = summary["ratios"]
    assert ratios["h2d_ratio"]["mean"] == 0.1
    assert ratios["compute_ratio"]["mean"] == 0.6
    assert ratios["opt_ratio"]["mean"] == 0.05

    # data_wait mean for steps 3..6 = (15+20+25+30)/4 = 22.5 ms.
    assert ratios["data_wait_ratio"]["mean"] == pytest.approx(0.225)

    # Strong negative relation by synthetic construction.
    corr = summary["correlations"]["gpu_util_vs_data_wait_ms"]
    assert corr is not None
    assert corr < -0.9
