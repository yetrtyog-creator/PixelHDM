# Single-Thread Perf Bottleneck Toolkit

This folder is an isolated measurement workspace for Windows `num_workers=0` training runs.
It is designed to locate root causes and estimate bottleneck shares (`data_wait`, `h2d`, `compute`, `opt`, `gc`) with reproducible outputs.

## Folder Contents

- `analyze_bottleneck.py`: Parses `step_metrics.jsonl` and optional `gpu_telemetry.csv`, then writes bottleneck reports.
- `runtime_instrumentation.py`: Runtime monkeypatch hooks for `TrainingLoop`/`StepExecutor` (no src edits).
- `run_instrumented_training.py`: Test-side training entrypoint that writes phase-0 `step_metrics.jsonl`.
- `run_gpu_telemetry.ps1`: Starts `nvidia-smi` telemetry capture with a fixed sampling interval.
- `run_profile_session.ps1`: Convenience wrapper to run training + telemetry in one session (supports instrumented runner).
- `test_analyze_bottleneck.py`: Unit tests for parser/ratio/correlation logic.
- `test_runtime_instrumentation.py`: Unit tests for runtime hook and JSONL output contracts.

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

1. Run one instrumented profiling session (recommended, no src changes):

```powershell
powershell -ExecutionPolicy Bypass -File tests/tester/perf_bottleneck/run_profile_session.ps1 `
  -RunId baseline_s42_20260211 `
  -Config configs/train_config.yaml `
  -Seed 42 `
  -MaxSteps 1500 `
  -TimingMode light `
  -ProfileStride 50 `
  -IntervalMs 200
```

2. Or capture GPU telemetry separately:

```powershell
powershell -ExecutionPolicy Bypass -File tests/tester/perf_bottleneck/run_gpu_telemetry.ps1 `
  -OutputPath logs/profiling/run_x/gpu_telemetry.csv `
  -IntervalMs 200
```

3. Run training with test-side instrumentation outputting `step_metrics.jsonl`:

```powershell
python tests/tester/perf_bottleneck/run_instrumented_training.py `
  --config configs/train_config.yaml `
  --run-id baseline_s42_20260211 `
  --profile-root logs/profiling `
  --max-steps 1500 `
  --timing-mode light `
  --profile-stride 50
```

4. Analyze bottleneck shares:

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
- `timing_mode=probe` only marks every `profile_stride` step as probe; non-sampled steps stay `light`.
