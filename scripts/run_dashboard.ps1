param(
  [string]$Input = "docs/artifacts",
  [string]$Output = "docs/artifacts/dashboard/latest",
  [switch]$Live,
  [switch]$Once
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

$py = Join-Path $ProjectRoot ".venv_trainer\\Scripts\\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$buildArgs = @("-B", "-m", "trainer.monitoring.dashboard_build", "--input", $Input, "--output", $Output)
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
