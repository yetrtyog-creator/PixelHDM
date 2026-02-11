"""
Analyze single-thread training bottlenecks from step metrics and GPU telemetry.

Expected inputs:
1. step_metrics.jsonl (from trainer instrumentation)
2. gpu_telemetry.csv (from nvidia-smi query mode)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


RATIO_DEFS: Sequence[Tuple[str, str]] = (
    ("data_wait_ms", "data_wait_ratio"),
    ("h2d_ms", "h2d_ratio"),
    ("fwd_bwd_ms", "compute_ratio"),
    ("opt_ms", "opt_ratio"),
    ("gc_ms", "gc_ratio"),
)

STEP_NUMERIC_FIELDS: Sequence[str] = (
    "data_wait_ms",
    "h2d_ms",
    "fwd_bwd_ms",
    "opt_ms",
    "gc_ms",
    "step_time_ms",
    "samples_per_sec",
    "loss_total",
    "loss_vloss",
    "loss_freq",
    "loss_repa",
    "loss_gamma_l2",
    "learning_rate",
)

STEP_REQUIRED_FIELDS: Sequence[str] = (
    "run_id",
    "timestamp_utc",
    "step",
    "epoch",
    "batch_idx",
    "grad_accumulation_steps",
    "micro_batches_collected",
    "batch_size_total",
    "data_wait_ms",
    "h2d_ms",
    "fwd_bwd_ms",
    "opt_ms",
    "gc_ms",
    "step_time_ms",
)


@dataclass
class StepParseResult:
    records: List[Dict[str, Any]]
    line_count: int
    parse_errors: int
    completeness: Dict[str, float]
    step_gap_count: int
    timestamp_coverage_pct: float


@dataclass
class GpuParseResult:
    records: List[Dict[str, Any]]
    line_count: int
    parse_errors: int


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    if isinstance(value, str):
        s = value.strip().replace("%", "")
        if not s:
            return None
        try:
            f = float(s)
        except ValueError:
            return None
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    return None


def _parse_timestamp(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo is not None else raw.replace(tzinfo=timezone.utc)
    if not isinstance(raw, str):
        return None

    text = raw.strip()
    if not text:
        return None

    # ISO 8601 path
    iso_text = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        dt = datetime.fromisoformat(iso_text)
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass

    # nvidia-smi default timestamp formats
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    if q <= 0:
        return float(min(values))
    if q >= 100:
        return float(max(values))

    sorted_vals = sorted(values)
    rank = (len(sorted_vals) - 1) * (q / 100.0)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return float(sorted_vals[low])

    weight = rank - low
    return float(sorted_vals[low] * (1.0 - weight) + sorted_vals[high] * weight)


def _describe(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0.0,
            "mean": None,
            "std": None,
            "cv": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }
    count = len(values)
    mean = sum(values) / count
    if count > 1:
        variance = sum((x - mean) ** 2 for x in values) / count
        std = math.sqrt(variance)
    else:
        std = 0.0
    cv = (std / mean) if mean not in (0.0, -0.0) else None

    return {
        "count": float(count),
        "mean": mean,
        "std": std,
        "cv": cv,
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
    }


def _rank(values: Sequence[float]) -> List[float]:
    pairs = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][1] == pairs[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[pairs[k][0]] = avg_rank
        i = j + 1
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    denom = den_x * den_y
    if denom == 0:
        return None
    return num / denom


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    return _pearson(_rank(xs), _rank(ys))


def parse_step_metrics(path: Path) -> StepParseResult:
    records: List[Dict[str, Any]] = []
    line_count = 0
    parse_errors = 0
    presence_counts = {key: 0 for key in STEP_REQUIRED_FIELDS}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            text = line.strip()
            if not text:
                continue
            try:
                raw = json.loads(text)
            except json.JSONDecodeError:
                parse_errors += 1
                continue
            if not isinstance(raw, dict):
                parse_errors += 1
                continue

            step_raw = raw.get("step")
            try:
                step = int(step_raw)
            except (TypeError, ValueError):
                parse_errors += 1
                continue

            # Backward-compatible fallback if phase-0 field is not landed yet.
            step_time_ms = _to_float(raw.get("step_time_ms"))
            if step_time_ms is None:
                fallback_step_time = _to_float(raw.get("step_time"))
                if fallback_step_time is not None:
                    step_time_ms = fallback_step_time * 1000.0

            loss_total = _to_float(raw.get("loss_total"))
            if loss_total is None:
                loss_total = _to_float(raw.get("loss"))

            record: Dict[str, Any] = {
                "step": step,
                "timestamp": _parse_timestamp(raw.get("timestamp_utc") or raw.get("timestamp")),
                "run_id": raw.get("run_id"),
                "raw": raw,
                "step_time_ms": step_time_ms,
                "loss_total": loss_total,
            }
            for field in STEP_NUMERIC_FIELDS:
                if field in ("step_time_ms", "loss_total"):
                    continue
                record[field] = _to_float(raw.get(field))

            for key in STEP_REQUIRED_FIELDS:
                if raw.get(key) is not None:
                    presence_counts[key] += 1

            records.append(record)

    records.sort(key=lambda x: x["step"])
    step_gap_count = 0
    for prev, curr in zip(records, records[1:]):
        if curr["step"] - prev["step"] != 1:
            step_gap_count += 1

    completeness = {
        key: (presence_counts[key] / len(records) * 100.0 if records else 0.0)
        for key in STEP_REQUIRED_FIELDS
    }
    ts_count = sum(1 for r in records if r["timestamp"] is not None)
    ts_coverage = (ts_count / len(records) * 100.0) if records else 0.0

    return StepParseResult(
        records=records,
        line_count=line_count,
        parse_errors=parse_errors,
        completeness=completeness,
        step_gap_count=step_gap_count,
        timestamp_coverage_pct=ts_coverage,
    )


def parse_gpu_telemetry(path: Path) -> GpuParseResult:
    records: List[Dict[str, Any]] = []
    line_count = 0
    parse_errors = 0

    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            line_count += 1
            if not row:
                continue
            first = row[0].strip().lower()
            if first.startswith("timestamp"):
                continue

            if len(row) < 6:
                parse_errors += 1
                continue

            ts = _parse_timestamp(row[0])
            util_gpu = _to_float(row[1])
            util_mem = _to_float(row[2])
            mem_used = _to_float(row[3])
            mem_total = _to_float(row[4])
            power_draw = _to_float(row[5])

            if ts is None or util_gpu is None:
                parse_errors += 1
                continue

            records.append(
                {
                    "timestamp": ts,
                    "utilization_gpu": util_gpu,
                    "utilization_memory": util_mem,
                    "memory_used": mem_used,
                    "memory_total": mem_total,
                    "power_draw": power_draw,
                }
            )

    records.sort(key=lambda x: x["timestamp"])
    return GpuParseResult(records=records, line_count=line_count, parse_errors=parse_errors)


def filter_warmup(records: Sequence[Dict[str, Any]], warmup_steps: int) -> List[Dict[str, Any]]:
    if warmup_steps <= 0:
        return list(records)
    filtered = [r for r in records if r["step"] > warmup_steps]
    if filtered:
        return filtered
    if len(records) > warmup_steps:
        return list(records[warmup_steps:])
    return []


def _collect_field(records: Sequence[Dict[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for rec in records:
        val = rec.get(key)
        if isinstance(val, (int, float)) and not math.isnan(float(val)) and not math.isinf(float(val)):
            values.append(float(val))
    return values


def _compute_ratio_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for rec in records:
        step_time = rec.get("step_time_ms")
        if not isinstance(step_time, (int, float)) or step_time <= 0:
            continue

        ratio_rec: Dict[str, float] = {"step": float(rec["step"]), "step_time_ms": float(step_time)}
        for ms_key, ratio_key in RATIO_DEFS:
            val = rec.get(ms_key)
            if isinstance(val, (int, float)):
                ratio_rec[ms_key] = float(val)
                ratio_rec[ratio_key] = float(val) / float(step_time)
        out.append(ratio_rec)
    return out


def _align_gpu(
    step_records: Sequence[Dict[str, Any]],
    gpu_records: Sequence[Dict[str, Any]],
    max_align_gap_sec: float,
) -> List[Dict[str, float]]:
    if not step_records or not gpu_records:
        return []

    gpu_ts = [g["timestamp"].timestamp() for g in gpu_records]
    aligned: List[Dict[str, float]] = []

    for step in step_records:
        ts = step.get("timestamp")
        if ts is None:
            continue
        target = ts.timestamp()
        pos = bisect_left(gpu_ts, target)

        candidates: List[int] = []
        if pos < len(gpu_ts):
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
        if not candidates:
            continue

        best_idx = min(candidates, key=lambda i: abs(gpu_ts[i] - target))
        gap = abs(gpu_ts[best_idx] - target)
        if gap > max_align_gap_sec:
            continue

        row: Dict[str, float] = {
            "step": float(step["step"]),
            "gpu_util": float(gpu_records[best_idx]["utilization_gpu"]),
        }
        for field in ("data_wait_ms", "h2d_ms", "gc_ms", "step_time_ms"):
            val = step.get(field)
            if isinstance(val, (int, float)):
                row[field] = float(val)
        aligned.append(row)

    return aligned


def _correlation_summary(aligned: Sequence[Dict[str, float]]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    pairs = (
        ("gpu_util_vs_data_wait_ms", "data_wait_ms"),
        ("gpu_util_vs_h2d_ms", "h2d_ms"),
        ("step_time_ms_vs_gc_ms", "gc_ms"),
    )
    for out_key, rhs_key in pairs:
        xs: List[float] = []
        ys: List[float] = []
        for row in aligned:
            if out_key == "step_time_ms_vs_gc_ms":
                x = row.get("step_time_ms")
                y = row.get(rhs_key)
            else:
                x = row.get("gpu_util")
                y = row.get(rhs_key)
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                xs.append(float(x))
                ys.append(float(y))
        out[out_key] = _spearman(xs, ys)
    return out


def _decision_hints(
    ratio_summary: Dict[str, Dict[str, Optional[float]]],
    corr_summary: Dict[str, Optional[float]],
) -> List[str]:
    hints: List[str] = []

    data_wait = ratio_summary.get("data_wait_ratio", {}).get("mean")
    h2d = ratio_summary.get("h2d_ratio", {}).get("mean")
    compute = ratio_summary.get("compute_ratio", {}).get("mean")
    gc_ratio = ratio_summary.get("gc_ratio", {}).get("mean")

    corr_data_wait = corr_summary.get("gpu_util_vs_data_wait_ms")
    corr_h2d = corr_summary.get("gpu_util_vs_h2d_ms")
    corr_gc = corr_summary.get("step_time_ms_vs_gc_ms")

    if isinstance(data_wait, float) and data_wait > 0.25:
        if isinstance(corr_data_wait, float) and corr_data_wait < -0.2:
            hints.append(
                "Data wait appears to be the primary bottleneck; prioritize streaming/double-buffering in single-thread mode."
            )
        else:
            hints.append(
                "Data wait ratio is high; investigate dataloader cost variance and micro-batch collection behavior."
            )

    if isinstance(h2d, float) and h2d > 0.15:
        if isinstance(corr_h2d, float) and corr_h2d < -0.2:
            hints.append(
                "H2D transfer is likely limiting utilization; test pin_memory + non_blocking copies."
            )
        else:
            hints.append(
                "H2D ratio is elevated; verify host memory pinning and transfer overlap opportunities."
            )

    if isinstance(gc_ratio, float) and gc_ratio > 0.05:
        hints.append(
            "GC/empty_cache overhead is non-trivial; tune gc_interval and avoid frequent empty_cache in steady state."
        )
    elif isinstance(corr_gc, float) and corr_gc > 0.3:
        hints.append(
            "Step-time spikes correlate with gc_ms; inspect periodic GC behavior and cadence."
        )

    if isinstance(compute, float) and compute >= 0.7:
        hints.append(
            "Compute dominates step time; next-level profiling should focus on model kernels/operator efficiency."
        )

    if not hints:
        hints.append("No single dominant bottleneck detected; gather longer runs or increase probe precision.")

    return hints


def _format_num(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def write_step_ratio_csv(path: Path, ratio_records: Sequence[Dict[str, float]]) -> None:
    headers = [
        "step",
        "step_time_ms",
        "data_wait_ms",
        "h2d_ms",
        "fwd_bwd_ms",
        "opt_ms",
        "gc_ms",
        "data_wait_ratio",
        "h2d_ratio",
        "compute_ratio",
        "opt_ratio",
        "gc_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for rec in ratio_records:
            row = {h: rec.get(h) for h in headers}
            writer.writerow(row)


def write_markdown_report(path: Path, summary: Dict[str, Any]) -> None:
    quality = summary["data_quality"]
    ratio = summary["ratios"]
    gpu = summary["gpu_utilization"]
    corrs = summary["correlations"]
    hints = summary["hints"]

    lines: List[str] = []
    lines.append("# Bottleneck Analysis Summary")
    lines.append("")
    lines.append(f"- run_id: `{summary['run_id']}`")
    lines.append(f"- warmup_steps: `{summary['warmup_steps']}`")
    lines.append(f"- effective_steps: `{summary['effective_steps']}`")
    lines.append("")
    lines.append("## Data Quality")
    lines.append("")
    lines.append(
        f"- step_parse_error_rate_pct: `{_format_num(quality['step_parse_error_rate_pct'])}`"
    )
    lines.append(
        f"- gpu_parse_error_rate_pct: `{_format_num(quality['gpu_parse_error_rate_pct'])}`"
    )
    lines.append(
        f"- step_timestamp_coverage_pct: `{_format_num(quality['step_timestamp_coverage_pct'])}`"
    )
    lines.append(f"- step_gap_count: `{quality['step_gap_count']}`")
    lines.append("")
    lines.append("## Bottleneck Share (Mean Ratio)")
    lines.append("")
    for key in ("data_wait_ratio", "h2d_ratio", "compute_ratio", "opt_ratio", "gc_ratio"):
        mean = ratio[key]["mean"]
        pct = mean * 100.0 if isinstance(mean, float) else None
        lines.append(f"- {key}: `{_format_num(mean)}` ({_format_num(pct, 2)}%)")
    lines.append(
        f"- unattributed_ratio: `{_format_num(summary['unattributed_ratio'])}` "
        f"({_format_num(summary['unattributed_ratio'] * 100.0 if isinstance(summary['unattributed_ratio'], float) else None, 2)}%)"
    )
    lines.append("")
    lines.append("## Distribution")
    lines.append("")
    for key in ("step_time_ms", "data_wait_ms", "h2d_ms", "fwd_bwd_ms", "opt_ms", "gc_ms"):
        dist = summary["distributions"][key]
        lines.append(
            f"- {key}: p50={_format_num(dist['p50'], 3)} "
            f"p95={_format_num(dist['p95'], 3)} p99={_format_num(dist['p99'], 3)} "
            f"cv={_format_num(dist['cv'], 3)}"
        )
    lines.append("")
    lines.append("## GPU Utilization")
    lines.append("")
    lines.append(
        f"- p50={_format_num(gpu['p50'], 2)} p95={_format_num(gpu['p95'], 2)} "
        f"p99={_format_num(gpu['p99'], 2)} cv={_format_num(gpu['cv'], 3)}"
    )
    lines.append("")
    lines.append("## Correlation (Spearman)")
    lines.append("")
    lines.append(
        f"- gpu_util_vs_data_wait_ms: `{_format_num(corrs['gpu_util_vs_data_wait_ms'], 4)}`"
    )
    lines.append(f"- gpu_util_vs_h2d_ms: `{_format_num(corrs['gpu_util_vs_h2d_ms'], 4)}`")
    lines.append(f"- step_time_ms_vs_gc_ms: `{_format_num(corrs['step_time_ms_vs_gc_ms'], 4)}`")
    lines.append("")
    lines.append("## Hints")
    lines.append("")
    for hint in hints:
        lines.append(f"- {hint}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _error_rate(parse_errors: int, line_count: int) -> float:
    if line_count <= 0:
        return 0.0
    return parse_errors / line_count * 100.0


def analyze(
    step_metrics_path: Path,
    gpu_telemetry_path: Optional[Path],
    warmup_steps: int,
    run_id: Optional[str],
    max_align_gap_sec: float,
) -> Dict[str, Any]:
    step_result = parse_step_metrics(step_metrics_path)
    filtered_steps = filter_warmup(step_result.records, warmup_steps)
    ratio_records = _compute_ratio_records(filtered_steps)

    gpu_result = (
        parse_gpu_telemetry(gpu_telemetry_path)
        if gpu_telemetry_path is not None
        else GpuParseResult(records=[], line_count=0, parse_errors=0)
    )
    aligned = _align_gpu(filtered_steps, gpu_result.records, max_align_gap_sec)

    distributions = {
        key: _describe(_collect_field(filtered_steps, key))
        for key in ("step_time_ms", "data_wait_ms", "h2d_ms", "fwd_bwd_ms", "opt_ms", "gc_ms")
    }
    ratio_summary = {
        ratio_key: _describe(
            [r[ratio_key] for r in ratio_records if ratio_key in r]
        )
        for _, ratio_key in RATIO_DEFS
    }

    ratio_means = [
        ratio_summary[k]["mean"]
        for k in ("data_wait_ratio", "h2d_ratio", "compute_ratio", "opt_ratio", "gc_ratio")
        if isinstance(ratio_summary[k]["mean"], float)
    ]
    unattributed = 1.0 - sum(ratio_means) if ratio_means else None

    gpu_util_desc = _describe([g["utilization_gpu"] for g in gpu_result.records])
    corr_summary = _correlation_summary(aligned)
    hints = _decision_hints(ratio_summary, corr_summary)

    resolved_run_id = run_id
    if not resolved_run_id:
        resolved_run_id = "unknown_run"
        if step_result.records:
            raw_run = step_result.records[0]["run_id"]
            if isinstance(raw_run, str) and raw_run.strip():
                resolved_run_id = raw_run.strip()

    summary = {
        "run_id": resolved_run_id,
        "warmup_steps": warmup_steps,
        "effective_steps": len(filtered_steps),
        "input_files": {
            "step_metrics": str(step_metrics_path),
            "gpu_telemetry": str(gpu_telemetry_path) if gpu_telemetry_path else None,
        },
        "data_quality": {
            "step_line_count": step_result.line_count,
            "step_parse_errors": step_result.parse_errors,
            "step_parse_error_rate_pct": _error_rate(step_result.parse_errors, step_result.line_count),
            "gpu_line_count": gpu_result.line_count,
            "gpu_parse_errors": gpu_result.parse_errors,
            "gpu_parse_error_rate_pct": _error_rate(gpu_result.parse_errors, gpu_result.line_count),
            "step_gap_count": step_result.step_gap_count,
            "step_timestamp_coverage_pct": step_result.timestamp_coverage_pct,
            "step_field_completeness_pct": step_result.completeness,
            "aligned_pairs": len(aligned),
        },
        "distributions": distributions,
        "ratios": ratio_summary,
        "unattributed_ratio": unattributed,
        "gpu_utilization": gpu_util_desc,
        "correlations": corr_summary,
        "hints": hints,
    }

    return {
        "summary": summary,
        "ratio_records": ratio_records,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze single-thread training bottlenecks from instrumentation logs."
    )
    parser.add_argument(
        "--step-metrics",
        type=Path,
        required=True,
        help="Path to step_metrics.jsonl",
    )
    parser.add_argument(
        "--gpu-telemetry",
        type=Path,
        default=None,
        help="Path to gpu_telemetry.csv (optional)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=300,
        help="Discard steps <= warmup_steps from analysis.",
    )
    parser.add_argument(
        "--max-align-gap-sec",
        type=float,
        default=1.0,
        help="Max timestamp gap when aligning step and GPU telemetry.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Override run_id in output.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for summary files.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print summary JSON to stdout.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.step_metrics.exists():
        parser.error(f"step metrics file not found: {args.step_metrics}")
    if args.gpu_telemetry is not None and not args.gpu_telemetry.exists():
        parser.error(f"gpu telemetry file not found: {args.gpu_telemetry}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    result = analyze(
        step_metrics_path=args.step_metrics,
        gpu_telemetry_path=args.gpu_telemetry,
        warmup_steps=max(0, args.warmup_steps),
        run_id=args.run_id,
        max_align_gap_sec=max(0.0, args.max_align_gap_sec),
    )

    summary = result["summary"]
    ratio_records = result["ratio_records"]

    summary_json_path = args.output_dir / "bottleneck_summary.json"
    summary_md_path = args.output_dir / "bottleneck_summary.md"
    ratio_csv_path = args.output_dir / "step_ratio_breakdown.csv"

    summary_json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown_report(summary_md_path, summary)
    write_step_ratio_csv(ratio_csv_path, ratio_records)

    if args.print_summary:
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

