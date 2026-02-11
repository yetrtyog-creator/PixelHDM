param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [string]$LogRoot = "logs/profiling",
    [string]$TrainCommand = "",
    [string]$Config = "configs/train_config.yaml",
    [int]$Seed = 42,
    [int]$MaxSteps = 0,
    [ValidateSet("light", "probe")]
    [string]$TimingMode = "light",
    [int]$ProfileStride = 50,
    [int]$GcInterval = 100,
    [switch]$UseProgressBar,
    [int]$IntervalMs = 200
)

$ErrorActionPreference = "Stop"

$runDir = Join-Path $LogRoot $RunId
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$gpuCsv = Join-Path $runDir "gpu_telemetry.csv"
if ([string]::IsNullOrWhiteSpace($TrainCommand)) {
    $TrainCommand = "python tests/tester/perf_bottleneck/run_instrumented_training.py --config `"$Config`" --run-id `"$RunId`" --profile-root `"$LogRoot`" --seed $Seed --timing-mode $TimingMode --profile-stride $ProfileStride --gc-interval $GcInterval"
    if ($MaxSteps -gt 0) {
        $TrainCommand = "$TrainCommand --max-steps $MaxSteps"
    }
    if ($UseProgressBar) {
        $TrainCommand = "$TrainCommand --use-progress-bar"
    }
}

Write-Host "Run directory: $runDir"
Write-Host "Launching GPU telemetry: $gpuCsv"
Write-Host "Training command:"
Write-Host $TrainCommand

$smiArgs = "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits -lms $IntervalMs"
$gpuProc = Start-Process -FilePath "nvidia-smi" -ArgumentList $smiArgs -RedirectStandardOutput $gpuCsv -NoNewWindow -PassThru

try {
    Invoke-Expression $TrainCommand
}
finally {
    if ($gpuProc -and -not $gpuProc.HasExited) {
        Write-Host "Stopping telemetry process..."
        Stop-Process -Id $gpuProc.Id -ErrorAction SilentlyContinue
    }
}

Write-Host "Session complete."
Write-Host "Expected step metrics path: $(Join-Path $runDir 'step_metrics.jsonl')"
Write-Host "Then run:"
Write-Host "python tests/tester/perf_bottleneck/analyze_bottleneck.py --step-metrics `"$runDir/step_metrics.jsonl`" --gpu-telemetry `"$gpuCsv`" --output-dir `"$runDir/analysis`""
