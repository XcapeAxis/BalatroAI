param(
  [string]$ReadmePath = "README.md",
  [string]$StatusJsonPath = "docs/artifacts/status/latest_status.json",
  [string]$BadgesJsonPath = "docs/artifacts/status/latest_badges.json",
  [switch]$DryRun,
  [switch]$Apply
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

if (-not $DryRun -and -not $Apply) {
  $DryRun = $true
}

function Read-Json([string]$Path) {
  if (-not (Test-Path $Path)) {
    throw "missing json file: $Path"
  }
  $raw = Get-Content -LiteralPath $Path -Raw
  try {
    return ($raw | ConvertFrom-Json)
  } catch {
    $clean = $raw.TrimStart([char]0xFEFF)
    return ($clean | ConvertFrom-Json)
  }
}

function Encode-BadgeToken([string]$Token) {
  if ([string]::IsNullOrWhiteSpace($Token)) { return "" }
  $encoded = [System.Uri]::EscapeDataString($Token)
  return $encoded.Replace("%20", "_")
}

function Get-GitHubRepoSlug() {
  $remote = ""
  try {
    $remote = (& git config --get remote.origin.url 2>$null | Select-Object -First 1)
  } catch {
    $remote = ""
  }
  if ([string]::IsNullOrWhiteSpace($remote)) { return "" }
  $token = [string]$remote
  $token = $token.Trim()
  $token = $token -replace "\.git$", ""
  if ($token -match "github\.com[:/](?<slug>[^/]+/[^/]+)$") {
    return [string]$Matches["slug"]
  }
  return ""
}

function New-BadgesBlock($badgesObj, [string]$RepoSlug) {
  $lines = New-Object System.Collections.Generic.List[string]
  $badgeRows = @($badgesObj.badges)
  foreach ($b in $badgeRows) {
    $label = [string]$b.label
    $message = [string]$b.message
    $color = [string]$b.color
    if ([string]::IsNullOrWhiteSpace($color)) { $color = "6E7781" }
    $link = [string]$b.link
    if ([string]::IsNullOrWhiteSpace($link)) { $link = "#" }
    $url = "https://img.shields.io/badge/{0}-{1}-{2}" -f (Encode-BadgeToken $label), (Encode-BadgeToken $message), (Encode-BadgeToken $color)
    $lines.Add(('[![{0}]({1})]({2})' -f $label, $url, $link)) | Out-Null
  }

  if (-not [string]::IsNullOrWhiteSpace($RepoSlug)) {
    $ciWorkflow = Join-Path $ProjectRoot ".github/workflows/ci-smoke.yml"
    if (Test-Path $ciWorkflow) {
      $ciUrl = "https://github.com/{0}/actions/workflows/ci-smoke.yml" -f $RepoSlug
      $lines.Add(('[![CI Smoke]({0}/badge.svg)]({0})' -f $ciUrl)) | Out-Null
    }
    $lines.Add(('[![GitHub Stars](https://img.shields.io/github/stars/{0}?style=social)](https://github.com/{0}/stargazers)' -f $RepoSlug)) | Out-Null
    $lines.Add(('[![GitHub Issues](https://img.shields.io/github/issues/{0})](https://github.com/{0}/issues)' -f $RepoSlug)) | Out-Null
  }

  return ($lines -join "`n")
}

function New-StatusBlock($statusObj) {
  $latestGate = [string]$statusObj.latest_gate.gate_name
  $latestGateStatus = [string]$statusObj.latest_gate.status
  $trendSignal = [string]$statusObj.benchmark_snapshot.trend_signal
  $trendUpdated = [string]$statusObj.trend_warehouse.last_updated
  $trendRows = [string]$statusObj.trend_warehouse.rows_count
  $championExp = [string]$statusObj.champion.exp_id
  $championStatus = [string]$statusObj.champion.status
  $candidateDecision = [string]$statusObj.candidate.decision
  $candidateExp = [string]$statusObj.candidate.top_candidate_exp_id
  $docsRange = [string]$statusObj.docs_coverage.range

  $lines = @(
    "<!-- README_STATUS:BEGIN -->",
    "### Repository Status (Auto-generated)",
    "",
    "- branch: $($statusObj.repo.branch)",
    "- latest_gate: $latestGate ($latestGateStatus)",
    "- recent_trend_signal: $trendSignal",
    "- trend_warehouse_last_updated: $trendUpdated",
    "- trend_rows_count: $trendRows",
    "- champion: $championExp ($championStatus)",
    "- candidate: $candidateExp (decision: $candidateDecision)",
    "- docs_coverage: $docsRange",
    "<!-- README_STATUS:END -->"
  )
  return ($lines -join "`n")
}

function Replace-Block([string]$Text, [string]$StartMarker, [string]$EndMarker, [string]$Inner) {
  if (-not $Text.Contains($StartMarker) -or -not $Text.Contains($EndMarker)) {
    throw "missing marker pair: $StartMarker ... $EndMarker"
  }
  $pattern = "(?s)" + [regex]::Escape($StartMarker) + ".*?" + [regex]::Escape($EndMarker)
  $replacement = $StartMarker + "`n" + $Inner + "`n" + $EndMarker
  return [regex]::Replace(
    $Text,
    $pattern,
    [System.Text.RegularExpressions.MatchEvaluator]{ param($m) $replacement },
    1
  )
}

$readmeFullPath = Join-Path $ProjectRoot $ReadmePath
if (-not (Test-Path $readmeFullPath)) {
  throw "missing README: $readmeFullPath"
}

$statusObj = Read-Json -Path (Join-Path $ProjectRoot $StatusJsonPath)
$badgesObj = Read-Json -Path (Join-Path $ProjectRoot $BadgesJsonPath)

$repoSlug = Get-GitHubRepoSlug
$badgesBlock = New-BadgesBlock -badgesObj $badgesObj -RepoSlug $repoSlug
$statusBlock = New-StatusBlock -statusObj $statusObj

$readmeBefore = Get-Content -LiteralPath $readmeFullPath -Raw
$patched = Replace-Block -Text $readmeBefore -StartMarker "<!-- BADGES:START -->" -EndMarker "<!-- BADGES:END -->" -Inner $badgesBlock
$patched = Replace-Block -Text $patched -StartMarker "<!-- STATUS:START -->" -EndMarker "<!-- STATUS:END -->" -Inner $statusBlock

$changed = ($patched -ne $readmeBefore)
if ($DryRun) {
  $tmp = Join-Path $env:TEMP ("readme_patch_preview_" + [guid]::NewGuid().ToString("N") + ".md")
  $patched | Out-File -LiteralPath $tmp -Encoding utf8
  Write-Host ("[dry-run] changed=" + $changed)
  & git --no-pager diff --no-index -- $readmeFullPath $tmp
  if ($LASTEXITCODE -gt 1) {
    throw "git diff preview failed with exit code $LASTEXITCODE"
  }
  Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
}

if ($Apply) {
  if ($changed) {
    $patched | Out-File -LiteralPath $readmeFullPath -Encoding utf8
    Write-Host "[apply] README updated"
  } else {
    Write-Host "[apply] README unchanged"
  }
}

$result = [ordered]@{
  status = "PASS"
  dry_run = [bool]$DryRun
  apply = [bool]$Apply
  changed = [bool]$changed
  readme = $readmeFullPath
  status_json = (Join-Path $ProjectRoot $StatusJsonPath)
  badges_json = (Join-Path $ProjectRoot $BadgesJsonPath)
}
($result | ConvertTo-Json -Depth 8)
