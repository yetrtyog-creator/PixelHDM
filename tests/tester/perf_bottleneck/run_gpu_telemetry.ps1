param(
    [Parameter(Mandatory = $true)]
    [string]$OutputPath,
    [int]$IntervalMs = 200
)

$ErrorActionPreference = "Stop"

$outDir = Split-Path -Parent $OutputPath
if (-not [string]::IsNullOrWhiteSpace($outDir)) {
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
}

Write-Host "Starting nvidia-smi telemetry capture..."
Write-Host "Output: $OutputPath"
Write-Host "IntervalMs: $IntervalMs"
Write-Host "Press Ctrl+C to stop."

$args = "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits -lms $IntervalMs"

nvidia-smi $args | Tee-Object -FilePath $OutputPath

