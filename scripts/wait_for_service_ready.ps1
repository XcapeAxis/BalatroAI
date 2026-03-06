param(
  [string]$BaseUrl = "http://127.0.0.1:12346",
  [string]$OutDir = "docs/artifacts/p49/readiness",
  [string]$RunId = "",
  [int]$MaxRetries = 20,
  [double]$RetryIntervalSec = 2.0,
  [double]$WarmupGraceSec = 8.0,
  [int]$ConsecutiveSuccesses = 3,
  [double]$TimeoutSec = 3.0,
  [string]$ProbeMethod = "health_gamestate"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

$py = Join-Path $ProjectRoot ".venv_trainer\\Scripts\\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

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
Write-Host ("[readiness] cmd: " + $py + " " + ($args -join " "))
& $py @args
$code = $LASTEXITCODE
if ($code -ne 0) {
  throw ("[readiness] failed with exit code " + $code)
}
