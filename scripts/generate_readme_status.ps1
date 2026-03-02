param(
  [string]$OutMd = "docs/generated/README_STATUS.md",
  [string]$OutJson = "docs/generated/README_STATUS.json"
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

$branch = Invoke-GitText -GitArgs @("rev-parse", "--abbrev-ref", "HEAD")
$originHead = Invoke-GitText -GitArgs @("symbolic-ref", "refs/remotes/origin/HEAD")
if ([string]::IsNullOrWhiteSpace($branch)) { $branch = "unknown" }
$detectedMain = "main"
if ($originHead -match "refs/remotes/origin/(.+)$") {
  $detectedMain = $Matches[1]
} elseif (-not (Test-Path ".git")) {
  $detectedMain = "unknown"
}
$mainlineStatus = if ($branch -and $detectedMain -and $branch -eq $detectedMain) { "mainline" } else { "non-mainline" }

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

$orchestratorPaths = @(
  (Join-Path $ProjectRoot "scripts/run_p22.ps1"),
  (Join-Path $ProjectRoot "trainer/experiments/orchestrator.py")
)
$orchestratorReady = ($orchestratorPaths | Where-Object { Test-Path $_ }).Count -eq $orchestratorPaths.Count

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
  "docs/artifacts/p25"
)

$statusObj = [ordered]@{
  schema = "p25_readme_status_v1"
  branch = $branch
  detected_main_branch = $detectedMain
  mainline_status = $mainlineStatus
  highest_supported_gate = if ($highestGate -gt 0) { "RunP$highestGate" } else { "unknown" }
  seed_governance = if ($seedGovernance) { "enabled (P23+)" } else { "missing" }
  orchestrator = if ($orchestratorReady) { "enabled (P22+)" } else { "missing" }
  docs_specs_range = $specRange
  docs_specs_available = @($specIds | ForEach-Object { "P$_" })
  artifacts_guide = $artifactGuide
}

$mdLines = @(
  "### Repository Status (Auto-generated)",
  "",
  ("- branch: " + $statusObj.branch),
  ("- mainline_status: " + $statusObj.mainline_status + " (detected main: " + $statusObj.detected_main_branch + ")"),
  ("- highest_supported_gate: " + $statusObj.highest_supported_gate),
  ("- seed_governance: " + $statusObj.seed_governance),
  ("- experiment_orchestrator: " + $statusObj.orchestrator),
  ("- docs_specs_range: " + $statusObj.docs_specs_range + " (available: " + (($statusObj.docs_specs_available) -join ", ") + ")"),
  "- artifacts_guide: docs/artifacts/p24/runs/latest and docs/artifacts/p25/"
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
}
($result | ConvertTo-Json -Depth 8)
