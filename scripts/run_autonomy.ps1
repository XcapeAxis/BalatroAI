param(
  [switch]$Quick,
  [switch]$Overnight,
  [switch]$ResumeLatest,
  [switch]$DryRun,
  [string]$TrainingPython = "",
  [int]$TimeoutSec = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

$modeCount = @($Quick, $Overnight, $ResumeLatest | Where-Object { $_ }).Count
if ($modeCount -gt 1) {
  throw "[P61] choose only one of -Quick, -Overnight, or -ResumeLatest"
}

$mode = "quick"
if ($ResumeLatest) {
  $mode = "resume-latest"
} elseif ($Overnight) {
  $mode = "overnight"
}

$py = ""
if ($TrainingPython.Trim()) {
  $py = $TrainingPython.Trim()
} else {
  $resolverScript = Join-Path $ProjectRoot "scripts\\resolve_training_python.ps1"
  if (Test-Path $resolverScript) {
    $resolverJson = (& powershell -ExecutionPolicy Bypass -File $resolverScript -Emit json | Out-String).Trim()
    if ($resolverJson) {
      try {
        $resolver = $resolverJson | ConvertFrom-Json
        $py = [string]$resolver.selected.python
      } catch {
        $py = ""
      }
    }
  }
}
if (-not $py) { $py = "python" }

$args = @("-B", "-m", "trainer.autonomy.run_autonomy", "--mode", $mode)
if (-not $DryRun) { $args += "--execute" }
if ($TimeoutSec -gt 0) { $args += @("--timeout-sec", "$TimeoutSec") }
$latestAutonomyJson = Join-Path $ProjectRoot "docs\\artifacts\\p60\\latest_autonomy_entry.json"

Write-Host ("[P61] repo_root: " + $ProjectRoot)
Write-Host ("[P61] training_python: " + $py)
Write-Host ("[P61] requested_mode: " + $mode)
Write-Host ("[P61] execute: " + (-not $DryRun))

$payloadJson = (& $py @args | Out-String).Trim()
if (-not $payloadJson) {
  throw "[P61] run_autonomy returned empty output"
}

try {
  $payload = $payloadJson | ConvertFrom-Json
} catch {
  if (Test-Path $latestAutonomyJson) {
    try {
      $payload = Get-Content -Path $latestAutonomyJson -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
      throw ("[P61] failed to parse run_autonomy output and fallback JSON: " + $_.Exception.Message)
    }
  } else {
    throw ("[P61] failed to parse run_autonomy output: " + $_.Exception.Message)
  }
}

if ($payload.PSObject.Properties["latest_json"] -and $payload.latest_json) {
  Write-Host ("[P61] autonomy_entry=" + [string]$payload.latest_json)
}
if ($payload.PSObject.Properties["attention_queue_path"] -and $payload.attention_queue_path) {
  Write-Host ("[P61] attention_queue=" + [string]$payload.attention_queue_path)
}
if ($payload.PSObject.Properties["morning_summary_path"] -and $payload.morning_summary_path) {
  Write-Host ("[P61] morning_summary=" + [string]$payload.morning_summary_path)
}
if ($payload.PSObject.Properties["dashboard_path"] -and $payload.dashboard_path) {
  Write-Host ("[P61] dashboard=" + [string]$payload.dashboard_path)
}
if ($payload.PSObject.Properties["fast_check_report_path"] -and $payload.fast_check_report_path) {
  Write-Host ("[P61] fast_check_report=" + [string]$payload.fast_check_report_path)
}
if ($payload.PSObject.Properties["certification_queue_path"] -and $payload.certification_queue_path) {
  Write-Host ("[P61] certification_queue=" + [string]$payload.certification_queue_path)
}
Write-Host ("[P61] autonomy_state=" + [string]$payload.autonomy_state + " selected_plan=" + [string]$payload.selected_plan)
Write-Host ("[P61] reason=" + [string]$payload.reason)

$executionStatus = ""
if ($payload.PSObject.Properties["execution"] -and $payload.execution) {
  $executionStatus = [string]$payload.execution.status
  if ($payload.execution.PSObject.Properties["safe_run_summary_path"] -and $payload.execution.safe_run_summary_path) {
    Write-Host ("[P61] safe_run_summary=" + [string]$payload.execution.safe_run_summary_path)
  }
}

if ($executionStatus -eq "failed") {
  exit 1
}
exit 0
