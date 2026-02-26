<#
.SYNOPSIS
  Run a command with a timeout and log stdout/stderr to avoid debug hangs.

.DESCRIPTION
  Wraps a command in a process with -TimeoutSec. On timeout the process is killed
  and exit code 124 is returned. Logs are written to -LogDir. Use for long-running
  regressions (RunP18, RunP19) to prevent terminal/session freezes.

.EXAMPLE
  .\scripts\safe_run.ps1 -TimeoutSec 1200 -- powershell -ExecutionPolicy Bypass -File scripts\run_regressions.ps1 -RunP18
  .\scripts\safe_run.ps1 -TimeoutSec 600 -LogDir .safe_run\logs -- python -B trainer/rollout_search_p15.py --episodes 40 ...
#>
param(
  [int]$TimeoutSec = 600,
  [string]$LogDir = ".safe_run/logs",
  [switch]$Quiet,
  [Parameter(ValueFromRemainingArguments = $true)]
  $Remaining
)

$ErrorActionPreference = "Stop"

if (-not $Remaining -or $Remaining.Count -eq 0) {
  Write-Host "Usage: .\scripts\safe_run.ps1 [-TimeoutSec SEC] [-LogDir DIR] [-Quiet] -- <exe> [args...]"
  Write-Host "Example: .\scripts\safe_run.ps1 -TimeoutSec 1200 -- powershell -ExecutionPolicy Bypass -File scripts\run_regressions.ps1 -RunP18"
  exit 2
}

$exe = $Remaining[0]
$exeArgs = @()
if ($Remaining.Count -gt 1) {
  $exeArgs = $Remaining[1..($Remaining.Count - 1)]
}

function Escape-ProcessArg([string]$s) {
  if ($s -match '\s|"') { return '"' + ($s -replace '\\','\\' -replace '"','\"') + '"' }
  return $s
}
$argStr = ($exeArgs | ForEach-Object { Escape-ProcessArg ([string]$_) }) -join ' '

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$rand = Get-Random -Minimum 1000 -Maximum 99999
$outLog = Join-Path $LogDir "${ts}_${rand}.out.log"
$errLog = Join-Path $LogDir "${ts}_${rand}.err.log"

Write-Host "[safe_run] cwd: $((Get-Location).Path)"
Write-Host "[safe_run] timeout: ${TimeoutSec}s"
Write-Host "[safe_run] cmd: $exe $argStr"
Write-Host "[safe_run] out_log: $outLog"
Write-Host "[safe_run] err_log: $errLog"

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $exe
$psi.Arguments = $argStr
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true
$psi.WorkingDirectory = (Get-Location).Path

$p = New-Object System.Diagnostics.Process
$p.StartInfo = $psi
$p.Start() | Out-Null

$timedOut = -not $p.WaitForExit($TimeoutSec * 1000)
if ($timedOut) {
  try { $p.Kill() } catch {}
  "[safe_run] TIMEOUT after ${TimeoutSec}s" | Out-File -LiteralPath $errLog -Encoding UTF8
  Write-Host "[safe_run] exit_code: 124 (timeout)"
  Write-Host "========== safe_run failure (timeout) =========="
  Write-Host "command: $exe $argStr"
  Get-Content -LiteralPath $errLog -Tail 50 -ErrorAction SilentlyContinue
  exit 124
}

$stdout = $p.StandardOutput.ReadToEnd()
$stderr = $p.StandardError.ReadToEnd()
$stdout | Out-File -LiteralPath $outLog -Encoding UTF8
$stderr | Out-File -LiteralPath $errLog -Encoding UTF8
$exitCode = $p.ExitCode

Write-Host "[safe_run] exit_code: $exitCode"

if ($exitCode -ne 0) {
  Write-Host "========== safe_run failure summary =========="
  Write-Host "command: $exe $argStr"
  Write-Host "exit_code: $exitCode"
  Write-Host "--- last 100 lines (stdout) ---"
  Get-Content -LiteralPath $outLog -Tail 100 -ErrorAction SilentlyContinue
  Write-Host "--- last 100 lines (stderr) ---"
  Get-Content -LiteralPath $errLog -Tail 100 -ErrorAction SilentlyContinue
  Write-Host "============================================="
  exit $exitCode
}

if (-not $Quiet) {
  if (Test-Path $outLog) { Get-Content -LiteralPath $outLog }
  if (Test-Path $errLog) { Get-Content -LiteralPath $errLog | Write-Host -ForegroundColor DarkYellow }
}

exit 0
