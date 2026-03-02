param(
  [string]$RunDir = ""
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

$p29Root = Join-Path $ProjectRoot "docs/artifacts/p29"
if (-not (Test-Path $p29Root)) { throw "missing p29 root: $p29Root" }
if ([string]::IsNullOrWhiteSpace($RunDir)) {
  $active = Join-Path $p29Root "_active_run.txt"
  if (-not (Test-Path $active)) { throw "missing active run file: $active" }
  $stamp = (Get-Content -LiteralPath $active -Raw).Trim()
  $RunDir = Join-Path $p29Root $stamp
}

$todoPath = Join-Path $RunDir "todo_plan.md"
if (-not (Test-Path $todoPath)) { throw "missing todo plan: $todoPath" }
$text = Get-Content -LiteralPath $todoPath -Raw
$lines = $text -split "`r?`n"
$total = 0
$done = 0
$doing = 0
$blocked = 0
foreach ($line in $lines) {
  if ($line -match '^\|\s*\d+\s*\|') {
    $total += 1
    if ($line -match '\|\s*done\s*\|') { $done += 1 }
    elseif ($line -match '\|\s*doing\s*\|') { $doing += 1 }
    elseif ($line -match '\|\s*blocked\s*\|') { $blocked += 1 }
  }
}
$summary = [ordered]@{
  schema = "p29_todo_progress_v1"
  generated_at = (Get-Date).ToString("o")
  run_dir = $RunDir
  total = $total
  done = $done
  doing = $doing
  blocked = $blocked
}
$outJson = Join-Path $RunDir "todo_progress_summary.json"
$outMd = Join-Path $RunDir "todo_progress_summary.md"
($summary | ConvertTo-Json -Depth 8) | Out-File -LiteralPath $outJson -Encoding utf8
@(
  "# P29 To-do Progress Summary",
  "",
  "- total: $total",
  "- done: $done",
  "- doing: $doing",
  "- blocked: $blocked"
) -join "`n" | Out-File -LiteralPath $outMd -Encoding utf8
Write-Output $outJson
Write-Output $outMd
