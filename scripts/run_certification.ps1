param(
  [switch]$LatestPending,
  [switch]$DryRun,
  [string]$TrainingPython = "",
  [int]$TimeoutSec = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

function Resolve-RunnerPython {
  param([string]$ExplicitPython)
  if ($ExplicitPython.Trim()) {
    return $ExplicitPython.Trim()
  }
  $resolverScript = Join-Path $ProjectRoot "scripts\\resolve_training_python.ps1"
  if (Test-Path $resolverScript) {
    $resolverJson = (& powershell -ExecutionPolicy Bypass -File $resolverScript -Emit json | Out-String).Trim()
    if ($resolverJson) {
      try {
        $resolver = $resolverJson | ConvertFrom-Json
        $candidate = [string]$resolver.selected.python
        if ($candidate.Trim()) {
          return $candidate.Trim()
        }
      } catch {}
    }
  }
  return "python"
}

$py = Resolve-RunnerPython -ExplicitPython $TrainingPython
$args = @("-B", "-m", "trainer.autonomy.certification_queue")
if ($LatestPending) {
  $args += "--run-latest"
}
if ($DryRun) {
  $args += "--dry-run"
}
if ($TimeoutSec -gt 0) {
  $args += @("--timeout-sec", "$TimeoutSec")
}

$latestQueue = Join-Path $ProjectRoot "docs\\artifacts\\certification_queue\\certification_queue.json"
$latestState = Join-Path $ProjectRoot "docs\\artifacts\\certification_queue\\certification_state.json"

Write-Host ("[P61] repo_root: " + $ProjectRoot)
Write-Host ("[P61] training_python: " + $py)
Write-Host ("[P61] latest_pending: " + $LatestPending)
Write-Host ("[P61] dry_run: " + $DryRun)

$payloadJson = (& $py @args | Out-String).Trim()
if (-not $payloadJson) {
  throw "[P61] certification_queue returned empty output"
}

try {
  $payload = $payloadJson | ConvertFrom-Json
} catch {
  if (Test-Path $latestState) {
    $payload = Get-Content -Path $latestState -Raw -Encoding UTF8 | ConvertFrom-Json
  } elseif (Test-Path $latestQueue) {
    $payload = Get-Content -Path $latestQueue -Raw -Encoding UTF8 | ConvertFrom-Json
  } else {
    throw ("[P61] failed to parse certification_queue output: " + $_.Exception.Message)
  }
}

if ($payload.PSObject.Properties["status"]) {
  Write-Host ("[P61] certification_status=" + [string]$payload.status)
}
if ($payload.PSObject.Properties["queue_path"] -and $payload.queue_path) {
  Write-Host ("[P61] certification_queue=" + [string]$payload.queue_path)
}
if (Test-Path $latestState) {
  Write-Host ("[P61] certification_state=" + $latestState)
}
if (Test-Path (Join-Path $ProjectRoot "docs\\artifacts\\certification_queue\\certification_summary.md")) {
  Write-Host ("[P61] certification_summary=" + (Join-Path $ProjectRoot "docs\\artifacts\\certification_queue\\certification_summary.md"))
}

if ([string]$payload.status -eq "failed") {
  exit 1
}
exit 0
