# Single-Thread Perf Bottleneck Toolkit

This folder is an isolated measurement workspace for Windows `num_workers=0` training runs.
It is designed to locate root causes and estimate bottleneck shares (`data_wait`, `h2d`, `compute`, `opt`, `gc`) with reproducible outputs.

## Folder Contents

- `analyze_bottleneck.py`: Parses `step_metrics.jsonl` and optional `gpu_telemetry.csv`, then writes bottleneck reports.
- `run_gpu_telemetry.ps1`: Starts `nvidia-smi` telemetry capture with a fixed sampling interval.
- `run_profile_session.ps1`: Convenience wrapper to run training + telemetry in one session.
- `test_analyze_bottleneck.py`: Unit tests for parser/ratio/correlation logic.

## Required Input Files

- `step_metrics.jsonl`:
  - One JSON object per optimizer step.
  - Prefer phase-0 fields:
    - `data_wait_ms`, `h2d_ms`, `fwd_bwd_ms`, `opt_ms`, `gc_ms`, `step_time_ms`, `timestamp_utc`.
  - Backward-compatible fallback:
    - If `step_time_ms` is missing, `step_time` (seconds) is accepted.
- `gpu_telemetry.csv`:
  - Compatible with:
    - `nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits -lms 200`

## Quick Start

1. Capture GPU telemetry:

```powershell
powershell -ExecutionPolicy Bypass -File tests/tester/perf_bottleneck/run_gpu_telemetry.ps1 `
  -OutputPath logs/profiling/run_x/gpu_telemetry.csv `
  -IntervalMs 200
```

2. Run training with your instrumentation outputting `step_metrics.jsonl`.

3. Analyze bottleneck shares:

```powershell
python tests/tester/perf_bottleneck/analyze_bottleneck.py `
  --step-metrics logs/profiling/run_x/step_metrics.jsonl `
  --gpu-telemetry logs/profiling/run_x/gpu_telemetry.csv `
  --warmup-steps 300 `
  --output-dir logs/profiling/run_x/analysis `
  --print-summary
```

## Output Files

- `bottleneck_summary.json`: Full machine-readable results.
- `bottleneck_summary.md`: Human-readable report with quality checks and hints.
- `step_ratio_breakdown.csv`: Per-step ratio table for deeper inspection.

## Accuracy Notes

- Discard warmup steps (`--warmup-steps`) before making conclusions.
- Ensure timestamp coverage in step metrics for reliable correlation results.
- If quality fields show high parse error rate or low coverage, re-run before strategy decisions.

