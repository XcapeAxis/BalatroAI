param(
  [string]$BaseUrl = "http://127.0.0.1:12346",
  [string]$OutDir = "docs/artifacts/p49/readiness",
  [string]$RunId = "",
  [int]$MaxRetries = 20,
  [double]$RetryIntervalSec = 2.0,
  [double]$WarmupGraceSec = 8.0,
  [int]$ConsecutiveSuccesses = 3,
  [double]$TimeoutSec = 3.0,
  [string]$ProbeMethod = "health_gamestate",
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
  throw "[readiness] training python resolver returned empty output"
}
$resolver = $resolverJson | ConvertFrom-Json
$py = [string]$resolver.selected.python
if (-not $py.Trim()) {
  throw "[readiness] training python resolver did not return a python path"
}

$args = @(
  "-B",
  "-m", "trainer.runtime.service_readiness",
  "--base-url", $BaseUrl,
  "--out-dir", $OutDir,
  "--max-retries", "$MaxRetries",
  "--retry-interval-sec", "$RetryIntervalSec",
  "--warmup-grace-sec", "$WarmupGraceSec",
  "--consecutive-successes", "$ConsecutiveSuccesses",
  "--timeout-sec", "$TimeoutSec",
  "--probe-method", $ProbeMethod
)
if ($RunId.Trim()) { $args += @("--run-id", $RunId) }

Write-Host ("[readiness] python: " + $py)
Write-Host ("[readiness] python_env_type: " + [string]$resolver.selected.env_type)
Write-Host ("[readiness] python_cuda: " + [string]$resolver.selected.cuda_available)
if ($env:BALATRO_WINDOW_MODE) { Write-Host ("[readiness] window_mode: " + [string]$env:BALATRO_WINDOW_MODE) }
if ($env:BALATRO_BACKGROUND_VALIDATION_REF) { Write-Host ("[readiness] background_validation: " + [string]$env:BALATRO_BACKGROUND_VALIDATION_REF) }
Write-Host ("[readiness] cmd: " + $py + " " + ($args -join " "))
& $py @args
$code = $LASTEXITCODE
if ($code -ne 0) {
  throw ("[readiness] failed with exit code " + $code)
}
