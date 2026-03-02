param(
  [string]$Config = "configs/experiments/p23.yaml",
  [string]$OutRoot = "docs/artifacts/p23",
  [switch]$DryRun,
  [switch]$Quick,
  [switch]$Gate,
  [switch]$Nightly,
  [switch]$Milestone,
  [switch]$FlakeSmoke,
  [switch]$Resume,
  [switch]$KeepIntermediate,
  [string]$Only = "",
  [string]$Exclude = "",
  [int]$MaxParallel = 1,
  [int]$SeedLimit = 0,
  [int]$MaxExperiments = 0,
  [int]$MaxWallTimeMinutes = 0,
  [string]$FlakeExpId = "quick_baseline"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

function Test-HasYaml([string]$PythonExe) {
  try {
    & $PythonExe -c "import yaml" *> $null
    return ($LASTEXITCODE -eq 0)
  } catch {
    return $false
  }
}

function Get-PythonExe {
  $py = Join-Path $ProjectRoot ".venv_trainer\Scripts\python.exe"
  if (-not (Test-Path $py)) { $py = "python" }
  if (-not (Test-HasYaml -PythonExe $py)) {
    if (Test-HasYaml -PythonExe "python") {
      $py = "python"
    }
  }
  return $py
}

function Get-LatestRunRoot([string]$OutRootPath) {
  $runsRoot = Join-Path $OutRootPath "runs"
  if (-not (Test-Path $runsRoot)) { return "" }
  $latest = Get-ChildItem -Path $runsRoot -Directory -ErrorAction SilentlyContinue |
    Sort-Object Name -Descending | Select-Object -First 1
  if (-not $latest) { return "" }
  return $latest.FullName
}

function Resolve-Mode {
  if ($Nightly) { return "nightly" }
  if ($Milestone) { return "milestone" }
  if ($Quick) { return "quick" }
  return "gate"
}

function Invoke-Orchestrator(
  [string]$PythonExe,
  [string]$Mode,
  [string]$OnlyCsv,
  [int]$SeedLimitArg,
  [switch]$DoDryRun
) {
  $args = @(
    "-B",
    "-m", "trainer.experiments.orchestrator",
    "--config", $Config,
    "--out-root", $OutRoot,
    "--mode", $Mode,
    "--max-parallel", "$MaxParallel"
  )
  if ($DoDryRun) { $args += "--dry-run" }
  if ($Resume) { $args += "--resume" }
  if ($KeepIntermediate) { $args += "--keep-intermediate" }
  if ($MaxExperiments -gt 0) { $args += @("--max-experiments", "$MaxExperiments") }
  if ($MaxWallTimeMinutes -gt 0) { $args += @("--max-wall-time-minutes", "$MaxWallTimeMinutes") }
  if (-not [string]::IsNullOrWhiteSpace($OnlyCsv)) { $args += @("--only", $OnlyCsv) }
  if (-not [string]::IsNullOrWhiteSpace($Exclude)) { $args += @("--exclude", $Exclude) }
  if ($SeedLimitArg -gt 0) { $args += @("--seed-limit", "$SeedLimitArg") }

  Write-Host ("[P23] cmd: " + $PythonExe + " " + ($args -join " "))
  & $PythonExe @args
  $code = $LASTEXITCODE
  if ($code -ne 0) {
    throw ("[P23] orchestrator failed with exit code " + $code)
  }
}

$py = Get-PythonExe
$mode = Resolve-Mode
$resolvedOutRoot = (Resolve-Path (New-Item -ItemType Directory -Path $OutRoot -Force)).Path

Write-Host ("[P23] repo_root: " + $ProjectRoot)
Write-Host ("[P23] python: " + $py)
Write-Host ("[P23] mode: " + $mode)
Write-Host ("[P23] out_root: " + $resolvedOutRoot)

if ($FlakeSmoke) {
  $seedLimitForFlake = if ($SeedLimit -gt 0) { $SeedLimit } else { 6 }
  $runRoots = New-Object System.Collections.Generic.List[string]
  for ($i = 1; $i -le 3; $i++) {
    if ($i -gt 1) { Start-Sleep -Seconds 1 }
    Write-Host ("[P23-flake] repeat " + $i + "/3 exp=" + $FlakeExpId)
    Invoke-Orchestrator -PythonExe $py -Mode "quick" -OnlyCsv $FlakeExpId -SeedLimitArg $seedLimitForFlake -DoDryRun:$false
    $latestRun = Get-LatestRunRoot -OutRootPath $resolvedOutRoot
    if ([string]::IsNullOrWhiteSpace($latestRun)) {
      throw "[P23-flake] failed to locate run root after repeat"
    }
    $runRoots.Add($latestRun) | Out-Null
  }

  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $flakeOut = Join-Path $resolvedOutRoot ("flake_smoke\" + $stamp)
  New-Item -ItemType Directory -Path $flakeOut -Force | Out-Null
  $runRootsCsv = ($runRoots -join ",")
  $flakeArgs = @(
    "-B",
    "-m", "trainer.experiments.flake",
    "--run-roots", $runRootsCsv,
    "--exp-id", $FlakeExpId,
    "--out-dir", $flakeOut
  )
  Write-Host ("[P23-flake] cmd: " + $py + " " + ($flakeArgs -join " "))
  & $py @flakeArgs
  $flakeCode = $LASTEXITCODE
  if ($flakeCode -ne 0) {
    throw ("[P23-flake] flake report failed with exit code " + $flakeCode)
  }

  $flakeJson = Join-Path $flakeOut "flake_report.json"
  $flakeMd = Join-Path $flakeOut "flake_report.md"
  if (Test-Path $flakeJson) { Copy-Item -LiteralPath $flakeJson -Destination (Join-Path $resolvedOutRoot "flake_report.json") -Force }
  if (Test-Path $flakeMd) { Copy-Item -LiteralPath $flakeMd -Destination (Join-Path $resolvedOutRoot "flake_report.md") -Force }
  Write-Host ("[P23-flake] out_dir: " + $flakeOut)
  return
}

$seedLimitToUse = $SeedLimit
if ($mode -eq "quick" -and $seedLimitToUse -le 0) {
  $seedLimitToUse = 8
}
if ($mode -eq "gate" -and $seedLimitToUse -le 0) {
  $seedLimitToUse = 12
}

Invoke-Orchestrator -PythonExe $py -Mode $mode -OnlyCsv $Only -SeedLimitArg $seedLimitToUse -DoDryRun:$DryRun

$latestRunRoot = Get-LatestRunRoot -OutRootPath $resolvedOutRoot
if (-not [string]::IsNullOrWhiteSpace($latestRunRoot)) {
  $coverageArgs = @(
    "-B",
    "-m", "trainer.experiments.coverage",
    "--run-root", $latestRunRoot,
    "--out-dir", $latestRunRoot
  )
  Write-Host ("[P23] coverage cmd: " + $py + " " + ($coverageArgs -join " "))
  & $py @coverageArgs
  if ($LASTEXITCODE -ne 0) {
    throw "[P23] coverage generation failed"
  }
  Write-Host ("[P23] latest_run: " + $latestRunRoot)
}
