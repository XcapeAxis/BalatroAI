param(
  [string]$OutMd = "docs/generated/README_STATUS.md",
  [string]$OutJson = "docs/generated/README_STATUS.json",
  [string]$PublishedStatusJson = "docs/artifacts/status/latest_status.json"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

function Invoke-GitText([string[]]$GitArgs) {
  $output = & git @GitArgs 2>$null
  if ($LASTEXITCODE -ne 0) { return "" }
  return (($output | ForEach-Object { [string]$_ }) -join "`n").Trim()
}

function Read-JsonIfExists([string]$Path) {
  if (-not (Test-Path $Path)) { return $null }
  try {
    return (Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json)
  } catch {
    try {
      $raw = Get-Content -LiteralPath $Path -Raw
      if ($raw.StartsWith([char]0xFEFF)) { $raw = $raw.TrimStart([char]0xFEFF) }
      return ($raw | ConvertFrom-Json)
    } catch {
      return $null
    }
  }
}

function Get-RecentTrendSnapshot([string]$TrendRowsPath, [string]$AlertsPath) {
  $trendSignal = "unknown"
  $latestGateLine = "n/a"
  $rows = @()
  if (Test-Path $TrendRowsPath) {
    $lines = Get-Content -LiteralPath $TrendRowsPath -ErrorAction SilentlyContinue
    foreach ($ln in $lines) {
      if ([string]::IsNullOrWhiteSpace($ln)) { continue }
      try {
        $obj = $ln | ConvertFrom-Json
        if ($obj) { $rows += $obj }
      } catch {}
    }
  }
  if ($rows.Count -gt 0) {
    $gateRows = @($rows | Where-Object { $_.metric_name -in @("gate_overall_pass", "gate_pass") } | Sort-Object timestamp, run_id)
    if ($gateRows.Count -gt 0) {
      $latestGate = $gateRows[-1]
      $latestGateLine = "$($latestGate.gate_name):$($latestGate.metric_value)"
    }
  }

  $alertObj = Read-JsonIfExists -Path $AlertsPath
  if ($alertObj -and $alertObj.summary) {
    $hard = [int]($alertObj.summary.hard_regression)
    $soft = [int]($alertObj.summary.soft_regression)
    $noisy = [int]($alertObj.summary.noisy_needs_more_data)
    $improve = [int]($alertObj.summary.improvement)
    if ($hard -gt 0 -or $soft -gt 0) { $trendSignal = "regression" }
    elseif ($noisy -gt 0) { $trendSignal = "noisy" }
    elseif ($improve -gt 0) { $trendSignal = "improving" }
    else { $trendSignal = "stable" }
  } elseif ($rows.Count -gt 0) {
    $trendSignal = "stable"
  }

  return [ordered]@{
    trend_signal = $trendSignal
    latest_gate_snapshot = $latestGateLine
  }
}

$branch = Invoke-GitText -GitArgs @("rev-parse", "--abbrev-ref", "HEAD")
$originHead = Invoke-GitText -GitArgs @("symbolic-ref", "refs/remotes/origin/HEAD")
if ([string]::IsNullOrWhiteSpace($branch)) { $branch = "unknown" }
$detectedMain = "main"
if ($originHead -match "refs/remotes/origin/(.+)$") {
  $detectedMain = $Matches[1]
} elseif (-not (Test-Path ".git")) {
  $detectedMain = "unknown"
}
$statusShort = Invoke-GitText -GitArgs @("status", "--porcelain")
$isClean = [string]::IsNullOrWhiteSpace($statusShort)
$mainlineStatus = if ($branch -and $detectedMain -and $branch -eq $detectedMain) {
  if ($isClean) { "mainline-clean" } else { "mainline-dirty" }
} else {
  if ($isClean) { "non-mainline-clean" } else { "non-mainline-dirty" }
}

$runRegressions = Join-Path $ProjectRoot "scripts/run_regressions.ps1"
$highestGate = 0
if (Test-Path $runRegressions) {
  $text = Get-Content -LiteralPath $runRegressions -Raw
  $matches = [regex]::Matches($text, "RunP(\d+)")
  foreach ($m in $matches) {
    $v = [int]$m.Groups[1].Value
    if ($v -gt $highestGate) { $highestGate = $v }
  }
}

$seedPolicyPath = Join-Path $ProjectRoot "configs/experiments/seeds_p23.yaml"
$seedGovernance = Test-Path $seedPolicyPath

$platformPaths = @(
  (Join-Path $ProjectRoot "scripts/run_p22.ps1"),
  (Join-Path $ProjectRoot "scripts/run_p23.ps1"),
  (Join-Path $ProjectRoot "scripts/run_p24.ps1"),
  (Join-Path $ProjectRoot "trainer/experiments/orchestrator.py")
)
$platformReady = ($platformPaths | Where-Object { Test-Path $_ }).Count -eq $platformPaths.Count
$platformStatus = if ($platformReady) { "ready (P22+/P23+/P24+)" } else { "partial" }

$specFiles = Get-ChildItem -Path (Join-Path $ProjectRoot "docs") -Filter "P*_SPEC.md" -File -ErrorAction SilentlyContinue
$specIds = @()
foreach ($f in $specFiles) {
  if ($f.BaseName -match "^P(\d+)_SPEC$") {
    $specIds += [int]$Matches[1]
  }
}
$specIds = $specIds | Sort-Object -Unique
$specRange = "none"
if ($specIds.Count -gt 0) {
  $specRange = "P$($specIds[0])-P$($specIds[-1])"
}

$artifactGuide = @(
  "docs/artifacts/p24/runs/latest",
  "docs/artifacts/p25",
  "docs/artifacts/trends"
)

$trendsRoot = Join-Path $ProjectRoot "docs/artifacts/trends"
$trendRowsPath = Join-Path $trendsRoot "trend_rows.jsonl"
$trendSummaryPath = Join-Path $trendsRoot "trend_index_summary.json"
$alertsPath = Join-Path $ProjectRoot "docs/artifacts/p26/alerts_latest/regression_alert_report.json"
$trendWarehouseExists = (Test-Path $trendRowsPath) -and (Test-Path $trendSummaryPath)
$trendWarehouseStatus = if ($trendWarehouseExists) { "enabled (P26+)" } else { "missing" }
$trendUpdatedAt = ""
if ($trendWarehouseExists) {
  $trendUpdatedAt = (Get-Item $trendRowsPath).LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
}
$trendSnapshot = Get-RecentTrendSnapshot -TrendRowsPath $trendRowsPath -AlertsPath $alertsPath
$publishedStatusPath = Join-Path $ProjectRoot $PublishedStatusJson
$publishedStatus = Read-JsonIfExists -Path $publishedStatusPath
$publishedUsed = ($publishedStatus -ne $null)

$statusObj = [ordered]@{
  schema = "p27_readme_status_v1"
  branch = $branch
  detected_main_branch = $detectedMain
  mainline_status = $mainlineStatus
  working_tree_clean = $isClean
  highest_supported_gate = if ($highestGate -gt 0) { "RunP$highestGate" } else { "unknown" }
  latest_supported_gate = if ($highestGate -gt 0) { "RunP$highestGate" } else { "unknown" }
  seed_governance = if ($seedGovernance) { "enabled (P23+)" } else { "missing" }
  experiment_platform = $platformStatus
  orchestrator = if (Test-Path (Join-Path $ProjectRoot "trainer/experiments/orchestrator.py")) { "enabled (P22+)" } else { "missing" }
  trend_warehouse_status = $trendWarehouseStatus
  trend_warehouse_last_updated = $trendUpdatedAt
  recent_trend_signal = $trendSnapshot.trend_signal
  latest_gate_snapshot = $trendSnapshot.latest_gate_snapshot
  docs_specs_range = $specRange
  docs_specs_available = @($specIds | ForEach-Object { "P$_" })
  artifacts_guide = $artifactGuide
  published_status_source = $publishedStatusPath
  published_status_used = $publishedUsed
}

if ($publishedUsed) {
  if ($publishedStatus.latest_gate) {
    $gateName = [string]$publishedStatus.latest_gate.gate_name
    $gateStatus = [string]$publishedStatus.latest_gate.status
    if (-not [string]::IsNullOrWhiteSpace($gateName)) {
      $statusObj.latest_supported_gate = $gateName
      $statusObj.latest_gate_snapshot = ($gateName + ":" + $gateStatus)
    }
  }
  if ($publishedStatus.highest_supported_gate) {
    $statusObj.highest_supported_gate = [string]$publishedStatus.highest_supported_gate
  }
  if ($publishedStatus.seed_governance) {
    $statusObj.seed_governance = $(if ([bool]$publishedStatus.seed_governance.enabled) { "enabled (P23+)" } else { "missing" })
  }
  if ($publishedStatus.experiment_platform) {
    $platformToken = [string]$publishedStatus.experiment_platform.status
    if (-not [string]::IsNullOrWhiteSpace($platformToken)) {
      $statusObj.experiment_platform = $platformToken
    }
    if ($publishedStatus.experiment_platform.orchestrator_enabled -ne $null) {
      $statusObj.orchestrator = $(if ([bool]$publishedStatus.experiment_platform.orchestrator_enabled) { "enabled (P22+)" } else { "missing" })
    }
  }
  if ($publishedStatus.trend_warehouse) {
    $twEnabled = [bool]$publishedStatus.trend_warehouse.enabled
    $statusObj.trend_warehouse_status = $(if ($twEnabled) { "enabled (P26+)" } else { "missing" })
    if ($publishedStatus.trend_warehouse.last_updated) {
      $statusObj.trend_warehouse_last_updated = [string]$publishedStatus.trend_warehouse.last_updated
    }
  }
  if ($publishedStatus.benchmark_snapshot -and $publishedStatus.benchmark_snapshot.trend_signal) {
    $statusObj.recent_trend_signal = [string]$publishedStatus.benchmark_snapshot.trend_signal
  }
  if ($publishedStatus.docs_coverage -and $publishedStatus.docs_coverage.range) {
    $statusObj.docs_specs_range = [string]$publishedStatus.docs_coverage.range
  }
}

$mdLines = @(
  "### Repository Status (Auto-generated)",
  "",
  ("- branch: " + $statusObj.branch),
  ("- mainline_status: " + $statusObj.mainline_status + " (detected main: " + $statusObj.detected_main_branch + ")"),
  ("- working_tree_clean: " + $statusObj.working_tree_clean),
  ("- highest_supported_gate: " + $statusObj.highest_supported_gate),
  ("- latest_supported_gate: " + $statusObj.latest_supported_gate),
  ("- seed_governance: " + $statusObj.seed_governance),
  ("- experiment_platform: " + $statusObj.experiment_platform),
  ("- experiment_orchestrator: " + $statusObj.orchestrator),
  ("- trend_warehouse_status: " + $statusObj.trend_warehouse_status),
  ("- trend_warehouse_last_updated: " + $statusObj.trend_warehouse_last_updated),
  ("- recent_trend_signal: " + $statusObj.recent_trend_signal),
  ("- latest_gate_snapshot: " + $statusObj.latest_gate_snapshot),
  ("- docs_specs_range: " + $statusObj.docs_specs_range + " (available: " + (($statusObj.docs_specs_available) -join ", ") + ")"),
  "- artifacts_guide: docs/artifacts/p24/runs/latest, docs/artifacts/p25/, docs/artifacts/trends/",
  ("- published_status_used: " + $statusObj.published_status_used)
)

$mdPath = Join-Path $ProjectRoot $OutMd
$jsonPath = Join-Path $ProjectRoot $OutJson
New-Item -ItemType Directory -Path (Split-Path -Parent $mdPath) -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path -Parent $jsonPath) -Force | Out-Null
$mdBody = ($mdLines -join "`n")
$mdBody | Out-File -LiteralPath $mdPath -Encoding utf8
($statusObj | ConvertTo-Json -Depth 16) | Out-File -LiteralPath $jsonPath -Encoding utf8

$readmePath = Join-Path $ProjectRoot "README.md"
$beginMarker = "<!-- README_STATUS:BEGIN -->"
$endMarker = "<!-- README_STATUS:END -->"
if (Test-Path $readmePath) {
  $readme = Get-Content -LiteralPath $readmePath -Raw
  if ($readme.Contains($beginMarker) -and $readme.Contains($endMarker)) {
    $pattern = "(?s)<!-- README_STATUS:BEGIN -->.*?<!-- README_STATUS:END -->"
    $replacement = $beginMarker + "`n" + $mdBody + "`n" + $endMarker
    $patched = [regex]::Replace($readme, $pattern, [System.Text.RegularExpressions.MatchEvaluator]{ param($m) $replacement })
    if ($patched -ne $readme) {
      $patched | Out-File -LiteralPath $readmePath -Encoding utf8
    }
  }
}

$result = [ordered]@{
  status = "PASS"
  out_md = $mdPath
  out_json = $jsonPath
  highest_supported_gate = $statusObj.highest_supported_gate
  trend_warehouse_status = $statusObj.trend_warehouse_status
  recent_trend_signal = $statusObj.recent_trend_signal
}
($result | ConvertTo-Json -Depth 8)
