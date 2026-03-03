param(
  [string]$Config = "configs/experiments/p22.yaml",
  [string]$OutRoot = "docs/artifacts/p22",
  [switch]$DryRun,
  [switch]$Quick,
  [switch]$Nightly,
  [switch]$Resume,
  [switch]$KeepIntermediate,
  [switch]$VerboseLogs,
  [string]$Only = "",
  [string]$Exclude = "",
  [int]$MaxParallel = 1,
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
  "--max-parallel", "$MaxParallel"
)

if ($DryRun) { $args += "--dry-run" }
if ($Nightly) { $args += "--nightly" }
if ($Resume) { $args += "--resume" }
if ($KeepIntermediate) { $args += "--keep-intermediate" }
if ($VerboseLogs) { $args += "--verbose" }
if (-not [string]::IsNullOrWhiteSpace($Only)) { $args += @("--only", $Only) }
if (-not [string]::IsNullOrWhiteSpace($Exclude)) { $args += @("--exclude", $Exclude) }
if ($SeedLimit -gt 0) { $args += @("--seed-limit", "$SeedLimit") }
if (-not [string]::IsNullOrWhiteSpace($Seeds)) { $args += @("--seeds", $Seeds) }

if ($Quick) {
  if (-not ($args -contains "--only")) { $args += @("--only", "quick_baseline,quick_candidate,quick_selfsup_pretrain,quick_selfsup_p33") }
  if ($SeedLimit -le 0) { $args += @("--seed-limit", "2") }
}

Write-Host ("[P22] repo_root: " + $ProjectRoot)
Write-Host ("[P22] python: " + $py)
Write-Host ("[P22] cmd: " + $py + " " + ($args -join " "))

& $py @args
$code = $LASTEXITCODE
if ($code -ne 0) {
  throw ("[P22] orchestrator failed with exit code " + $code)
}
