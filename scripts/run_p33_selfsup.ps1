param(
  [string]$Config = "configs/experiments/p33_selfsup.yaml",
  [string]$OutDir = "",
  [int]$MaxSamples = 0,
  [int]$Seed = 0,
  [switch]$Quiet
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

$py = Join-Path $ProjectRoot ".venv_trainer\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$args = @(
  "-B",
  "-m", "trainer.experiments.selfsupervised_p33",
  "--config", $Config
)
if (-not [string]::IsNullOrWhiteSpace($OutDir)) { $args += @("--out-dir", $OutDir) }
if ($MaxSamples -gt 0) { $args += @("--max-samples", "$MaxSamples") }
if ($Seed -gt 0) { $args += @("--seed", "$Seed") }
if ($Quiet) { $args += "--quiet" }

Write-Host ("[P33] repo_root: " + $ProjectRoot)
Write-Host ("[P33] python: " + $py)
Write-Host ("[P33] cmd: " + $py + " " + ($args -join " "))

& $py @args
$code = $LASTEXITCODE
if ($code -ne 0) {
  throw ("[P33] selfsupervised_p33 failed with exit code " + $code)
}

