param(
  [switch]$RemoveVenv
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

function Get-TreeSizeBytes([string]$Path) {
  if (-not (Test-Path $Path)) { return 0 }
  $sum = (Get-ChildItem -Force -Recurse -File -Path $Path -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
  if ($null -eq $sum) { return 0 }
  return [int64]$sum
}

function Format-GB([int64]$Bytes) {
  return ('{0:N2} GB' -f ($Bytes / 1GB))
}

function Remove-DirIfExists([string]$Path) {
  if (Test-Path $Path) {
    Write-Host "remove $Path"
    Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction SilentlyContinue
  }
}

$before = Get-TreeSizeBytes $ProjectRoot
Write-Host ("before: " + (Format-GB $before))

$safeDirs = @(
  "logs",
  "runtime",
  "sim/tests/fixtures_runtime",
  "sim/runtime"
)
foreach ($d in $safeDirs) { Remove-DirIfExists $d }

$cacheNames = @("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache")
$cacheDirs = Get-ChildItem -Force -Recurse -Directory -ErrorAction SilentlyContinue | Where-Object { $cacheNames -contains $_.Name }
foreach ($d in $cacheDirs) {
  Remove-Item -LiteralPath $d.FullName -Recurse -Force -ErrorAction SilentlyContinue
}

$venvCandidates = Get-ChildItem -Force -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -like '.venv*' -or $_.Name -like 'venv*' }
if ($venvCandidates) {
  Write-Host "venv candidates:"
  foreach ($v in $venvCandidates) {
    $sz = Get-TreeSizeBytes $v.FullName
    Write-Host ("  " + $v.Name + " -> " + (Format-GB $sz))
  }
}

if ($RemoveVenv) {
  foreach ($v in $venvCandidates) {
    Write-Host ("remove venv " + $v.FullName)
    Remove-Item -LiteralPath $v.FullName -Recurse -Force -ErrorAction SilentlyContinue
  }
} else {
  if ($venvCandidates) {
    Write-Host "skip venv removal (pass -RemoveVenv to remove)"
  }
}

$after = Get-TreeSizeBytes $ProjectRoot
$reclaimed = $before - $after
Write-Host ("after:  " + (Format-GB $after))
Write-Host ("saved:  " + (Format-GB $reclaimed))
