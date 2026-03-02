param(
  [string]$OutRoot = "docs/artifacts/p24",
  [string]$QuickCampaign = "configs/experiments/campaigns/p24_quick.yaml",
  [string]$NightlyCampaign = "configs/experiments/campaigns/p24_nightly.yaml",
  [string]$RankingConfig = "configs/experiments/ranking_p24.yaml",
  [switch]$DryRun,
  [switch]$Quick,
  [switch]$Nightly,
  [switch]$QuickNightly,
  [switch]$Resume,
  [switch]$FlakeSmoke,
  [switch]$HeadlessDashboard,
  [string]$FlakeExpId = "validation_b__quick_risk_aware"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

function Get-PythonExe {
  $py = Join-Path $ProjectRoot ".venv_trainer\Scripts\python.exe"
  if (-not (Test-Path $py)) { $py = "python" }
  return $py
}

function Get-LatestRunRoot([string]$Root) {
  $runsRoot = Join-Path $Root "runs"
  if (-not (Test-Path $runsRoot)) { return "" }
  $latest = Get-ChildItem -Path $runsRoot -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -ne "latest" } |
    Sort-Object Name -Descending | Select-Object -First 1
  if (-not $latest) { return "" }
  return $latest.FullName
}

function Sync-Latest([string]$Root, [string]$RunRoot) {
  if ([string]::IsNullOrWhiteSpace($RunRoot)) { return }
  $latestDir = Join-Path $Root "runs\latest"
  if (Test-Path $latestDir) { Remove-Item -LiteralPath $latestDir -Recurse -Force -ErrorAction SilentlyContinue }
  New-Item -ItemType Directory -Path $latestDir -Force | Out-Null
  $items = Get-ChildItem -LiteralPath $RunRoot -Force -ErrorAction SilentlyContinue
  foreach ($item in $items) {
    Copy-Item -LiteralPath $item.FullName -Destination $latestDir -Recurse -Force
  }
}

function Invoke-Campaign([string]$CampaignConfigPath) {
  $py = Get-PythonExe
  $args = @(
    "-B",
    "-m", "trainer.experiments.campaign_runner",
    "--campaign-config", $CampaignConfigPath,
    "--out-root", $OutRoot,
    "--ranking-config", $RankingConfig,
    "--python", $py
  )
  if ($DryRun) { $args += "--dry-run" }
  if ($Resume) { $args += "--resume" }
  if ($HeadlessDashboard) {
    $args += "--headless-dashboard"
    $args += @("--dashboard-out", (Join-Path $OutRoot "dashboard_headless_log.txt"))
  }
  Write-Host ("[P24] cmd: " + $py + " " + ($args -join " "))
  & $py @args
  if ($LASTEXITCODE -ne 0) { throw ("[P24] campaign runner failed: " + $LASTEXITCODE) }
}

$resolvedOutRoot = (Resolve-Path (New-Item -ItemType Directory -Path $OutRoot -Force)).Path
$campaignConfig = $QuickCampaign
if ($Nightly -or $QuickNightly) { $campaignConfig = $NightlyCampaign }

Write-Host ("[P24] repo_root: " + $ProjectRoot)
Write-Host ("[P24] out_root: " + $resolvedOutRoot)
Write-Host ("[P24] campaign_config: " + $campaignConfig)

Invoke-Campaign -CampaignConfigPath $campaignConfig
$latestRunRoot = Get-LatestRunRoot -Root $resolvedOutRoot
if ([string]::IsNullOrWhiteSpace($latestRunRoot)) {
  throw "[P24] latest run root not found"
}
Sync-Latest -Root $resolvedOutRoot -RunRoot $latestRunRoot

$py = Get-PythonExe
if ($HeadlessDashboard) {
  $dashOut = Join-Path $resolvedOutRoot "dashboard_headless_log.txt"
  $dashArgs = @(
    "-B",
    "-m", "trainer.experiments.dashboard_tui",
    "--watch", (Join-Path $resolvedOutRoot "runs\latest\telemetry.jsonl"),
    "--headless-log",
    "--out", $dashOut
  )
  Write-Host ("[P24] dashboard cmd: " + $py + " " + ($dashArgs -join " "))
  & $py @dashArgs
  if ($LASTEXITCODE -ne 0) { throw "[P24] dashboard headless failed" }
}

if ($FlakeSmoke) {
  $runRoots = Get-ChildItem -Path (Join-Path $resolvedOutRoot "runs") -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -ne "latest" } |
    Sort-Object Name -Descending | Select-Object -First 3
  if (-not $runRoots -or $runRoots.Count -lt 3) {
    throw "[P24] flake smoke requires at least 3 campaign runs"
  }
  $runCsv = (($runRoots | Select-Object -ExpandProperty FullName) -join ",")
  $flakeOut = Join-Path $resolvedOutRoot "flake_latest"
  New-Item -ItemType Directory -Path $flakeOut -Force | Out-Null
  $flakeArgs = @(
    "-B",
    "-m", "trainer.experiments.flake",
    "--run-roots", $runCsv,
    "--exp-id", $FlakeExpId,
    "--out-dir", $flakeOut
  )
  Write-Host ("[P24] flake cmd: " + $py + " " + ($flakeArgs -join " "))
  & $py @flakeArgs
  if ($LASTEXITCODE -ne 0) { throw "[P24] flake smoke failed" }
  Copy-Item -LiteralPath (Join-Path $flakeOut "flake_report.json") -Destination (Join-Path $resolvedOutRoot "flake_report.json") -Force
  Copy-Item -LiteralPath (Join-Path $flakeOut "flake_report.md") -Destination (Join-Path $resolvedOutRoot "flake_report.md") -Force
}

# convenience mirrors for external smoke commands
foreach ($name in @("triage_latest", "bisect_latest", "ranking_latest")) {
  $src = Join-Path $resolvedOutRoot ("runs\latest\" + $name)
  $dst = Join-Path $resolvedOutRoot $name
  if (Test-Path $src) {
    if (Test-Path $dst) { Remove-Item -LiteralPath $dst -Recurse -Force -ErrorAction SilentlyContinue }
    Copy-Item -LiteralPath $src -Destination $dst -Recurse -Force
  }
}

Write-Host ("[P24] latest_run: " + $latestRunRoot)
Write-Host ("[P24] latest_alias: " + (Join-Path $resolvedOutRoot "runs\latest"))
