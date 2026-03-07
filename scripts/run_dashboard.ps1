param(
  [string]$Input = "docs/artifacts",
  [string]$Output = "docs/artifacts/dashboard/latest",
  [switch]$Live,
  [switch]$Once,
  [string]$TrainingPython = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

$resolveTrainingPythonScript = Join-Path $ProjectRoot "scripts\\resolve_training_python.ps1"
$resolverArgs = @(
  "-ExecutionPolicy", "Bypass",
  "-File", $resolveTrainingPythonScript,
  "-Emit", "json"
)
if ($TrainingPython.Trim()) { $resolverArgs += @("-ExplicitPython", $TrainingPython) }
$resolverJson = (& powershell @resolverArgs | Out-String).Trim()
if (-not $resolverJson) {
  throw "[dashboard] training python resolver returned empty output"
}
$resolver = $resolverJson | ConvertFrom-Json
$py = [string]$resolver.selected.python
if (-not $py.Trim()) {
  throw "[dashboard] training python resolver did not return a python path"
}

$buildArgs = @("-B", "-m", "trainer.monitoring.dashboard_build", "--input", $Input, "--output", $Output)
Write-Host ("[dashboard] python: " + $py)
Write-Host ("[dashboard] python_env_type: " + [string]$resolver.selected.env_type)
Write-Host ("[dashboard] build: " + $py + " " + ($buildArgs -join " "))
& $py @buildArgs
if ($LASTEXITCODE -ne 0) {
  throw "[dashboard] static build failed"
}

if ($Live) {
  $liveArgs = @("-B", "-m", "trainer.monitoring.live_dashboard", "--watch", $Input)
  if ($Once) { $liveArgs += "--once" }
  Write-Host ("[dashboard] live: " + $py + " " + ($liveArgs -join " "))
  & $py @liveArgs
  if ($LASTEXITCODE -ne 0) {
    throw "[dashboard] live watcher failed"
  }
}
