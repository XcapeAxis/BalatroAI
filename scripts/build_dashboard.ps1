param(
  [string]$DashboardDataJson = "docs/artifacts/status/latest_dashboard_data.json",
  [string]$OutDir = "docs/dashboard"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

function Read-JsonRaw([string]$Path) {
  if (-not (Test-Path $Path)) {
    throw "missing input json: $Path"
  }
  return Get-Content -LiteralPath $Path -Raw
}

$dashboardDir = Join-Path $ProjectRoot $OutDir
$indexPath = Join-Path $dashboardDir "index.html"
$appPath = Join-Path $dashboardDir "app.js"
$stylePath = Join-Path $dashboardDir "styles.css"
if (-not (Test-Path $indexPath)) { throw "missing dashboard file: $indexPath" }
if (-not (Test-Path $appPath)) { throw "missing dashboard file: $appPath" }
if (-not (Test-Path $stylePath)) { throw "missing dashboard file: $stylePath" }

$inputPath = Join-Path $ProjectRoot $DashboardDataJson
$raw = Read-JsonRaw -Path $inputPath

$dataDir = Join-Path $dashboardDir "data"
New-Item -ItemType Directory -Path $dataDir -Force | Out-Null
$latestJsonPath = Join-Path $dataDir "latest.json"
$latestJsPath = Join-Path $dataDir "latest.js"

$raw | Out-File -LiteralPath $latestJsonPath -Encoding utf8
$jsBody = "window.DASHBOARD_DATA = " + $raw.Trim() + ";" + "`n"
$jsBody | Out-File -LiteralPath $latestJsPath -Encoding utf8

$summaryPath = Join-Path $dashboardDir "build_dashboard_summary.json"
$summary = [ordered]@{
  schema = "p27_dashboard_build_summary_v1"
  generated_at = (Get-Date).ToString("o")
  status = "PASS"
  source_json = $inputPath
  out_dir = $dashboardDir
  outputs = [ordered]@{
    latest_json = $latestJsonPath
    latest_js = $latestJsPath
    index_html = $indexPath
    app_js = $appPath
    styles_css = $stylePath
  }
}
($summary | ConvertTo-Json -Depth 12) | Out-File -LiteralPath $summaryPath -Encoding utf8

($summary | ConvertTo-Json -Depth 12)
