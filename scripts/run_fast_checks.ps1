param(
  [string[]]$ChangedFile = @(),
  [string]$GatePlan = "",
  [string]$TrainingPython = "",
  [switch]$DryRun
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
$args = @("-B", "-m", "trainer.runtime.run_fast_checks")
if ($GatePlan.Trim()) {
  $args += @("--gate-plan", $GatePlan.Trim())
}
foreach ($path in @($ChangedFile)) {
  if ([string]::IsNullOrWhiteSpace($path)) { continue }
  $args += @("--changed-file", [string]$path)
}
if ($DryRun) {
  $args += "--dry-run"
}

$latestJson = Join-Path $ProjectRoot "docs\\artifacts\\p61\\latest_fast_check_report.json"

Write-Host ("[P61] repo_root: " + $ProjectRoot)
Write-Host ("[P61] training_python: " + $py)
Write-Host ("[P61] dry_run: " + $DryRun)
if ($GatePlan.Trim()) {
  Write-Host ("[P61] gate_plan: " + $GatePlan.Trim())
}
if (@($ChangedFile).Count -gt 0) {
  Write-Host ("[P61] changed_files: " + (@($ChangedFile) -join ", "))
}

$payloadJson = (& $py @args | Out-String).Trim()
if (-not $payloadJson) {
  throw "[P61] run_fast_checks returned empty output"
}

try {
  $payload = $payloadJson | ConvertFrom-Json
} catch {
  if (Test-Path $latestJson) {
    $payload = Get-Content -Path $latestJson -Raw -Encoding UTF8 | ConvertFrom-Json
  } else {
    throw ("[P61] failed to parse run_fast_checks output: " + $_.Exception.Message)
  }
}

Write-Host ("[P61] fast_check_status=" + [string]$payload.fast_check_status)
Write-Host ("[P61] validation_tiers_completed=" + ([string[]]@($payload.validation_tiers_completed) -join ","))
Write-Host ("[P61] pending_certification=" + [string]$payload.pending_certification)
Write-Host ("[P61] certification_status=" + [string]$payload.certification_status)
Write-Host ("[P61] required_next_step=" + [string]$payload.required_next_step)
Write-Host ("[P61] recommended_next_gate=" + [string]$payload.recommended_next_gate)
if ($payload.PSObject.Properties["validation_plan_ref"] -and $payload.validation_plan_ref) {
  Write-Host ("[P61] validation_plan=" + [string]$payload.validation_plan_ref)
}
if ($payload.PSObject.Properties["certification_queue_ref"] -and $payload.certification_queue_ref) {
  Write-Host ("[P61] certification_queue=" + [string]$payload.certification_queue_ref)
}
if ($payload.PSObject.Properties["json_path"] -and $payload.json_path) {
  Write-Host ("[P61] fast_check_report=" + [string]$payload.json_path)
}

if ([string]$payload.fast_check_status -eq "failed") {
  exit 1
}
exit 0
