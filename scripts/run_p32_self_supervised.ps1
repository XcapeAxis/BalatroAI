param(
  [string]$Config = "configs/experiments/p32_self_supervised.yaml",
  [string]$OutRoot = "docs/artifacts/p32_selfsup",
  [switch]$DryRun,
  [switch]$Quick,
  [switch]$Nightly,
  [switch]$Resume,
  [switch]$VerboseLogs,
  [int]$SeedLimit = 0,
  [string]$Seeds = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

$py = Join-Path $ProjectRoot ".venv_trainer\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$env:PYTHONUTF8 = "1"

$args = @(
  "-B",
  "-m", "trainer.experiments.orchestrator",
  "--config", $Config,
  "--out-root", $OutRoot,
  "--max-parallel", "1"
)

if ($DryRun) { $args += "--dry-run" }
if ($Nightly) { $args += "--nightly" }
if ($Resume) { $args += "--resume" }
if ($VerboseLogs) { $args += "--verbose" }
if ($SeedLimit -gt 0) { $args += @("--seed-limit", "$SeedLimit") }
if (-not [string]::IsNullOrWhiteSpace($Seeds)) { $args += @("--seeds", $Seeds) }

if ($Quick) {
  $args += @("--mode", "quick")
  if ($SeedLimit -le 0) { $args += @("--seed-limit", "3") }
}

Write-Host ("[P32-selfsup] repo_root: " + $ProjectRoot)
Write-Host ("[P32-selfsup] python: " + $py)
Write-Host ("[P32-selfsup] cmd: " + $py + " " + ($args -join " "))

& $py @args
$code = $LASTEXITCODE
if ($code -ne 0) {
  throw ("[P32-selfsup] orchestrator failed with exit code " + $code)
}
