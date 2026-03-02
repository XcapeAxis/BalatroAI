param(
  [string]$RunRoot = "",
  [int]$Tail = 12
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

function Read-JsonSafe([string]$Path) {
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

function Get-LatestRunRoot([string]$RunsRoot) {
  if (-not (Test-Path $RunsRoot)) { return "" }
  $latest = Get-ChildItem -LiteralPath $RunsRoot -Directory -ErrorAction SilentlyContinue |
    Sort-Object Name -Descending |
    Select-Object -First 1
  if (-not $latest) { return "" }
  return $latest.FullName
}

if ([string]::IsNullOrWhiteSpace($RunRoot)) {
  $RunRoot = Get-LatestRunRoot -RunsRoot (Join-Path $ProjectRoot "docs/artifacts/p22/runs")
}

if ([string]::IsNullOrWhiteSpace($RunRoot) -or -not (Test-Path $RunRoot)) {
  throw "[P22-live] no run root found. Pass -RunRoot or run scripts/run_p22.ps1 first."
}

$runRootFull = (Resolve-Path $RunRoot).Path
$livePath = Join-Path $runRootFull "live_summary_snapshot.json"
$telemetryPath = Join-Path $runRootFull "telemetry.jsonl"
$summaryPath = Join-Path $runRootFull "summary_table.md"

$live = Read-JsonSafe -Path $livePath
if ($null -eq $live) {
  throw "[P22-live] missing or invalid live summary: $livePath"
}

$statusCounts = [ordered]@{}
if ($live.status_counts) {
  foreach ($prop in $live.status_counts.PSObject.Properties) {
    $statusCounts[$prop.Name] = [int]$prop.Value
  }
}

$rows = @()
if ($live.rows) {
  $rows = @($live.rows)
}

$tailLines = @()
if (Test-Path $telemetryPath) {
  $tailLines = Get-Content -LiteralPath $telemetryPath -Tail ([Math]::Max(1, $Tail))
}

$payload = [ordered]@{
  schema = "p30_p22_live_view_v1"
  generated_at = (Get-Date).ToString("o")
  run_root = $runRootFull
  run_id = [string]$live.run_id
  mode = [string]$live.mode
  elapsed_sec = [double]$live.elapsed_sec
  status_counts = $statusCounts
  row_count = $rows.Count
  live_summary_path = $livePath
  telemetry_path = $telemetryPath
  summary_table_path = $summaryPath
  telemetry_tail = $tailLines
}

Write-Host ("[P22-live] run_root=" + $runRootFull)
Write-Host ("[P22-live] run_id=" + $payload.run_id + " mode=" + $payload.mode + " elapsed_sec=" + [Math]::Round($payload.elapsed_sec, 1))
if ($statusCounts.Keys.Count -gt 0) {
  $parts = @()
  foreach ($k in $statusCounts.Keys) {
    $parts += ($k + "=" + $statusCounts[$k])
  }
  Write-Host ("[P22-live] status_counts: " + ($parts -join ", "))
}
Write-Host ("[P22-live] summary_table: " + $summaryPath)

if ($rows.Count -gt 0) {
  Write-Host "[P22-live] queue rows:"
  $rows |
    Select-Object exp_id, stage, status, seed, elapsed_sec, eta_sec, metric_snapshot, updated_at |
    Sort-Object exp_id |
    Format-Table -AutoSize
}

if ($tailLines.Count -gt 0) {
  Write-Host ("[P22-live] telemetry tail (" + $tailLines.Count + " lines):")
  foreach ($line in $tailLines) {
    Write-Host $line
  }
}
