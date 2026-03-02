param(
  [string]$OutRoot = "docs/artifacts/p26",
  [string]$TrendsRoot = "docs/artifacts/trends",
  [string]$SinceTag = "sim-p23-seed-governance-v1",
  [switch]$DryRun,
  [switch]$Quick,
  [switch]$Nightly,
  [switch]$Resume
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

function Write-JsonFile([string]$Path, $Object) {
  $dir = Split-Path -Parent $Path
  if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
  ($Object | ConvertTo-Json -Depth 32) | Out-File -LiteralPath $Path -Encoding UTF8
}

function Get-LatestRunDir([string]$RootPath) {
  if (-not (Test-Path $RootPath)) { return "" }
  $latest = Get-ChildItem -Path $RootPath -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match "^\d{8}-\d{6}$" } |
    Sort-Object Name -Descending | Select-Object -First 1
  if (-not $latest) { return "" }
  return $latest.FullName
}

function Invoke-Stage([string]$StageId, [string]$CommandText, [scriptblock]$Action) {
  $startedAt = (Get-Date).ToString("o")
  $status = "PASS"
  $details = [ordered]@{}
  try {
    & $Action
  } catch {
    $status = "FAIL"
    $details.error = [string]$_.Exception.Message
  }
  $finishedAt = (Get-Date).ToString("o")
  $entry = [ordered]@{
    stage_id = $StageId
    status = $status
    started_at = $startedAt
    finished_at = $finishedAt
    command = $CommandText
    details = $details
  }
  $script:StageEntries += ,$entry
  if ($status -ne "PASS") {
    throw ("[P26] stage failed: " + $StageId + " -> " + $details.error)
  }
}

if (-not $Quick -and -not $Nightly) {
  $Quick = $true
}
$mode = if ($Nightly) { "nightly" } else { "quick" }

$p26Root = Join-Path $ProjectRoot $OutRoot
New-Item -ItemType Directory -Path $p26Root -Force | Out-Null

$runDir = ""
if ($Resume) {
  $runDir = Get-LatestRunDir -RootPath $p26Root
}
if ([string]::IsNullOrWhiteSpace($runDir)) {
  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $runDir = Join-Path $p26Root $stamp
  New-Item -ItemType Directory -Path $runDir -Force | Out-Null
}

$StageEntries = @()
$py = Get-PythonExe
$budgetCut = $false
$cleanupExecuted = $false

Write-Host ("[P26] repo_root: " + $ProjectRoot)
Write-Host ("[P26] mode: " + $mode)
Write-Host ("[P26] dry_run: " + [string]$DryRun)
Write-Host ("[P26] run_dir: " + $runDir)

$campaignCmd = if ($DryRun) {
  "powershell -ExecutionPolicy Bypass -File scripts/run_p24.ps1 -DryRun" + $(if ($Resume) { " -Resume" } else { "" })
} elseif ($Nightly) {
  "powershell -ExecutionPolicy Bypass -File scripts/run_p24.ps1 -Nightly -HeadlessDashboard" + $(if ($Resume) { " -Resume" } else { "" })
} else {
  "powershell -ExecutionPolicy Bypass -File scripts/run_p24.ps1 -Quick -HeadlessDashboard" + $(if ($Resume) { " -Resume" } else { "" })
}
Invoke-Stage -StageId "campaign" -CommandText $campaignCmd -Action {
  $args = @("-ExecutionPolicy", "Bypass", "-File", (Join-Path $ProjectRoot "scripts/run_p24.ps1"))
  if ($DryRun) {
    $args += "-DryRun"
  } elseif ($Nightly) {
    $args += @("-Nightly", "-HeadlessDashboard")
  } else {
    $args += @("-Quick", "-HeadlessDashboard")
  }
  if ($Resume) { $args += "-Resume" }
  & powershell @args
  if ($LASTEXITCODE -ne 0) { throw ("campaign exit=" + $LASTEXITCODE) }
}

$trendCmd = "$py -m trainer.experiments.index_artifacts --scan-root docs/artifacts --out-root $TrendsRoot --append" + $(if ($Quick -or $DryRun) { " --latest-only" } else { "" })
Invoke-Stage -StageId "trend_append" -CommandText $trendCmd -Action {
  $args = @(
    "-m", "trainer.experiments.index_artifacts",
    "--scan-root", "docs/artifacts",
    "--out-root", $TrendsRoot,
    "--append"
  )
  if ($Quick -or $DryRun) { $args += "--latest-only" }
  & $py @args
  if ($LASTEXITCODE -ne 0) { throw ("trend_append exit=" + $LASTEXITCODE) }
}

$alertsDir = Join-Path $runDir "alerts_latest"
$alertCmd = "$py -m trainer.experiments.regression_alert --trends-root $TrendsRoot --config configs/experiments/regression_alert_p26.yaml --out-dir $alertsDir"
Invoke-Stage -StageId "regression_alert" -CommandText $alertCmd -Action {
  & $py -m trainer.experiments.regression_alert --trends-root $TrendsRoot --config "configs/experiments/regression_alert_p26.yaml" --out-dir $alertsDir
  if ($LASTEXITCODE -ne 0) { throw ("regression_alert exit=" + $LASTEXITCODE) }
}

$rankingOut = Join-Path $runDir "ranking_latest"
$rankingCmd = "$py -m trainer.experiments.ranking --run-root docs/artifacts/p24/runs/latest --config configs/experiments/ranking_p24.yaml --out-dir $rankingOut"
Invoke-Stage -StageId "ranking_update" -CommandText $rankingCmd -Action {
  $latestP24 = Join-Path $ProjectRoot "docs/artifacts/p24/runs/latest"
  if (-not (Test-Path $latestP24)) { throw "[P26] missing docs/artifacts/p24/runs/latest for ranking update" }
  & $py -m trainer.experiments.ranking --run-root $latestP24 --config "configs/experiments/ranking_p24.yaml" --out-dir $rankingOut
  if ($LASTEXITCODE -ne 0) { throw ("ranking_update exit=" + $LASTEXITCODE) }
}

$releaseMd = Join-Path $runDir "release_summary_p26.md"
$releaseCmd = "$py -m trainer.experiments.release_notes --since-tag $SinceTag --out $releaseMd --include-commits --include-benchmarks --include-risks"
Invoke-Stage -StageId "release_summary" -CommandText $releaseCmd -Action {
  & $py -m trainer.experiments.release_notes --since-tag $SinceTag --out $releaseMd --include-commits --include-benchmarks --include-risks --trends-root $TrendsRoot
  if ($LASTEXITCODE -ne 0) { throw ("release_summary exit=" + $LASTEXITCODE) }
}

$cleanupCmd = "powershell -ExecutionPolicy Bypass -File scripts/cleanup.ps1"
Invoke-Stage -StageId "cleanup" -CommandText $cleanupCmd -Action {
  if ($DryRun) { return }
  & powershell -ExecutionPolicy Bypass -File (Join-Path $ProjectRoot "scripts/cleanup.ps1")
  if ($LASTEXITCODE -ne 0) { throw ("cleanup exit=" + $LASTEXITCODE) }
  $script:cleanupExecuted = $true
}

$stageStatusPath = Join-Path $runDir "scheduler_stage_status.json"
$manifestPath = Join-Path $runDir "scheduler_run_manifest.json"
$summaryPath = Join-Path $runDir "scheduler_summary.md"

$stagePass = @($StageEntries | Where-Object { $_.status -eq "PASS" }).Count
$stageFail = @($StageEntries | Where-Object { $_.status -ne "PASS" }).Count
$overallStatus = if ($stageFail -eq 0) { "PASS" } else { "FAIL" }

$manifest = [ordered]@{
  schema = "p26_scheduler_run_manifest_v1"
  generated_at = (Get-Date).ToString("o")
  repo_root = $ProjectRoot
  run_dir = $runDir
  mode = $mode
  quick_mode = [bool]($mode -eq "quick")
  nightly_mode = [bool]($mode -eq "nightly")
  dry_run = [bool]$DryRun
  resume = [bool]$Resume
  since_tag = $SinceTag
  trends_root = (Resolve-Path (New-Item -ItemType Directory -Path $TrendsRoot -Force)).Path
  budget_cut = $budgetCut
  cleanup_executed = [bool]$cleanupExecuted
  pipeline = @("campaign", "trend_append", "regression_alert", "ranking_update", "release_summary", "cleanup")
}
$stageObj = [ordered]@{
  schema = "p26_scheduler_stage_status_v1"
  generated_at = (Get-Date).ToString("o")
  overall_status = $overallStatus
  stage_count = @($StageEntries).Count
  pass_count = $stagePass
  fail_count = $stageFail
  stages = @($StageEntries)
  outputs = [ordered]@{
    alerts_dir = $alertsDir
    ranking_dir = $rankingOut
    release_summary_md = $releaseMd
  }
}

Write-JsonFile -Path $manifestPath -Object $manifest
Write-JsonFile -Path $stageStatusPath -Object $stageObj

$summaryLines = @(
  "# P26 Scheduler Summary",
  "",
  "- mode: $mode",
  "- dry_run: $([string][bool]$DryRun)",
  "- resume: $([string][bool]$Resume)",
  "- since_tag: $SinceTag",
  "- stage_count: $(@($StageEntries).Count)",
  "- pass_count: $stagePass",
  "- fail_count: $stageFail",
  "- overall_status: $overallStatus",
  "- budget_cut: $([string]$budgetCut)",
  "",
  "## Stages"
)
foreach ($entry in $StageEntries) {
  $summaryLines += ("- " + $entry.stage_id + ": " + $entry.status)
}
$summaryLines -join "`n" | Out-File -LiteralPath $summaryPath -Encoding UTF8

$result = [ordered]@{
  status = $overallStatus
  run_dir = $runDir
  stage_count = @($StageEntries).Count
  pass_count = $stagePass
  fail_count = $stageFail
  budget_cut = $budgetCut
  scheduler_run_manifest = $manifestPath
  scheduler_stage_status = $stageStatusPath
  scheduler_summary = $summaryPath
}
($result | ConvertTo-Json -Depth 16)
