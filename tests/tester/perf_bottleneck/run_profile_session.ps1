param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [Parameter(Mandatory = $true)]
    [string]$TrainCommand,
    [string]$LogRoot = "logs/profiling",
    [int]$IntervalMs = 200
)

$ErrorActionPreference = "Stop"

$runDir = Join-Path $LogRoot $RunId
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$gpuCsv = Join-Path $runDir "gpu_telemetry.csv"
Write-Host "Run directory: $runDir"
Write-Host "Launching GPU telemetry: $gpuCsv"

$smiArgs = "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits -lms $IntervalMs"
$gpuProc = Start-Process -FilePath "nvidia-smi" -ArgumentList $smiArgs -RedirectStandardOutput $gpuCsv -NoNewWindow -PassThru

try {
    Write-Host "Executing training command:"
    Write-Host $TrainCommand
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

