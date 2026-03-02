param(
  [string]$ReadmePath = "README.md",
  [string]$OutJson = "docs/generated/readme_lint_report.json"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

$fullReadmePath = Join-Path $ProjectRoot $ReadmePath
if (-not (Test-Path $fullReadmePath)) {
  throw "README not found: $fullReadmePath"
}
$readme = Get-Content -LiteralPath $fullReadmePath -Raw

$requiredChecks = [ordered]@{
  quick_start = '(?im)^##\s+Quick Start\b'
  value = '(?im)^##\s+What This Project Is\b'
  boundaries = '(?im)^##\s+Scope and Boundaries\b'
  architecture = '(?im)^##\s+Architecture Overview\b'
  reproducibility = '(?im)^##\s+Reproducibility\b'
  example_outputs = '(?im)^##\s+Example Outputs\b'
  roadmap = '(?im)^##\s+Roadmap\b'
  known_limitations = '(?im)^##\s+Known Limitations\b'
}

$checkResults = [ordered]@{}
foreach ($k in $requiredChecks.Keys) {
  $checkResults[$k] = [regex]::IsMatch($readme, [string]$requiredChecks[$k])
}

$badgeCount = [regex]::Matches($readme, 'https://img\.shields\.io/').Count
$hasReproLink = [regex]::IsMatch($readme, '\(docs/REPRODUCIBILITY_P25\.md\)')
$hasStatusMarker = ($readme.Contains('<!-- README_STATUS:BEGIN -->') -and $readme.Contains('<!-- README_STATUS:END -->'))

$linkMatches = [regex]::Matches($readme, '\[[^\]]+\]\(([^)]+)\)')
$missingLinks = New-Object System.Collections.Generic.List[string]
foreach ($m in $linkMatches) {
  $target = [string]$m.Groups[1].Value
  if ([string]::IsNullOrWhiteSpace($target)) { continue }
  if ($target.StartsWith('http://') -or $target.StartsWith('https://') -or $target.StartsWith('mailto:') -or $target.StartsWith('#')) { continue }
  $targetPath = $target.Split('#')[0]
  if ([string]::IsNullOrWhiteSpace($targetPath)) { continue }
  $normalized = Join-Path $ProjectRoot $targetPath
  if (-not (Test-Path $normalized)) {
    $missingLinks.Add($target) | Out-Null
  }
}

$allRequiredHeadings = $true
foreach ($v in $checkResults.Values) {
  if (-not $v) { $allRequiredHeadings = $false; break }
}

$pass = ($allRequiredHeadings -and $badgeCount -ge 8 -and $hasReproLink -and $missingLinks.Count -eq 0)

$report = [ordered]@{
  schema = 'p25_readme_lint_v1'
  generated_at = (Get-Date).ToString('o')
  readme_path = $fullReadmePath
  pass = $pass
  checks = $checkResults
  badge_count = $badgeCount
  has_repro_link = $hasReproLink
  has_status_marker = $hasStatusMarker
  missing_link_count = $missingLinks.Count
  missing_links = @($missingLinks)
}

$fullOutPath = Join-Path $ProjectRoot $OutJson
New-Item -ItemType Directory -Path (Split-Path -Parent $fullOutPath) -Force | Out-Null
($report | ConvertTo-Json -Depth 12) | Out-File -LiteralPath $fullOutPath -Encoding utf8
($report | ConvertTo-Json -Depth 12)

if (-not $pass) { exit 1 }
