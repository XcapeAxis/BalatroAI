<#
.SYNOPSIS
  Run a command with timeout/logging/summary to avoid debug hangs.

.DESCRIPTION
  SafeRun executes a command with explicit timeout and process-tree kill on timeout.
  Stdout/stderr are redirected to log files to avoid pipe deadlocks.
  A machine-readable summary (`safe_run_result_v1`) is written for traceability.

.EXAMPLE
  .\scripts\safe_run.ps1 -TimeoutSec 1200 powershell -ExecutionPolicy Bypass -File scripts\run_regressions.ps1 -RunP18
  .\scripts\safe_run.ps1 -TimeoutSec 600 -SummaryJson .safe_run\run1.json python -B trainer/eval_long_horizon.py ...
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$TimeoutSec = 600
$LogDir = ".safe_run/logs"
$Quiet = $false
$NoEcho = $false
$HeartbeatSec = 30
$TailLines = 100
$SummaryJson = ""
$Remaining = @()

function Escape-ProcessArg([string]$s) {
  if ($s -match '\s|"') { return '"' + ($s -replace '\\','\\' -replace '"','\"') + '"' }
  return $s
}

function Stop-ProcessTree([int]$ProcessId) {
  if ($ProcessId -le 0) { return $false }
  $onWindows = $false
  if ($env:OS -eq "Windows_NT") { $onWindows = $true }
  if (-not $onWindows) {
    try {
      $isWinVar = Get-Variable -Name IsWindows -ErrorAction Stop
      if ([bool]$isWinVar.Value) { $onWindows = $true }
    } catch {}
  }
  if ($onWindows) {
    try {
      $discard = & taskkill.exe /PID $ProcessId /T /F 2>$null
      return ($LASTEXITCODE -eq 0)
    } catch {
      return $false
    }
  }
  try {
    & kill -TERM -- "-$ProcessId" 2>$null
    Start-Sleep -Milliseconds 800
    & kill -KILL -- "-$ProcessId" 2>$null
    return $true
  } catch {
    return $false
  }
}

function Write-JsonNoBom([string]$Path, $Obj) {
  $json = ($Obj | ConvertTo-Json -Depth 8)
  $enc = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText($Path, ($json + [Environment]::NewLine), $enc)
}

$argv = @($args)
$idx = 0
:parse while ($idx -lt $argv.Count) {
  $tok = [string]$argv[$idx]
  if ($tok -eq "--") {
    $idx++
    break parse
  }
  switch ($tok.ToLowerInvariant()) {
    "-timeoutsec" {
      if (($idx + 1) -ge $argv.Count) { throw "missing value for -TimeoutSec" }
      $TimeoutSec = [int]$argv[$idx + 1]
      $idx += 2
      continue parse
    }
    "-logdir" {
      if (($idx + 1) -ge $argv.Count) { throw "missing value for -LogDir" }
      $LogDir = [string]$argv[$idx + 1]
      $idx += 2
      continue parse
    }
    "-quiet" {
      $Quiet = $true
      $idx += 1
      continue parse
    }
    "-noecho" {
      $NoEcho = $true
      $idx += 1
      continue parse
    }
    "-heartbeatsec" {
      if (($idx + 1) -ge $argv.Count) { throw "missing value for -HeartbeatSec" }
      $HeartbeatSec = [int]$argv[$idx + 1]
      $idx += 2
      continue parse
    }
    "-taillines" {
      if (($idx + 1) -ge $argv.Count) { throw "missing value for -TailLines" }
      $TailLines = [int]$argv[$idx + 1]
      $idx += 2
      continue parse
    }
    "-summaryjson" {
      if (($idx + 1) -ge $argv.Count) { throw "missing value for -SummaryJson" }
      $SummaryJson = [string]$argv[$idx + 1]
      $idx += 2
      continue parse
    }
    default {
      break parse
    }
  }
}

if ($idx -lt $argv.Count) {
  $Remaining = @($argv[$idx..($argv.Count - 1)] | ForEach-Object { [string]$_ })
} else {
  $Remaining = @()
}

if ($Remaining.Count -eq 0) {
  Write-Host "[safe_run] no command provided"
  exit 2
}

$argsList = @($Remaining)

$exe = [string]$argsList[0]
$exeArgs = @()
if ($argsList.Count -gt 1) {
  $exeArgs = @($argsList[1..($argsList.Count - 1)] | ForEach-Object { [string]$_ })
}
$argStr = ($exeArgs | ForEach-Object { Escape-ProcessArg ([string]$_) }) -join " "
$displayCmd = if ([string]::IsNullOrWhiteSpace($argStr)) { $exe } else { "$exe $argStr" }

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }
$runId = "$(Get-Date -Format 'yyyyMMdd_HHmmss')_$(Get-Random -Minimum 1000 -Maximum 99999)"
$outLog = Join-Path $LogDir "${runId}.out.log"
$errLog = Join-Path $LogDir "${runId}.err.log"
$summaryPath = if ([string]::IsNullOrWhiteSpace($SummaryJson)) { Join-Path $LogDir "${runId}.summary.json" } else { $SummaryJson }
$summaryDir = Split-Path -Parent $summaryPath
if ($summaryDir -and -not (Test-Path $summaryDir)) { New-Item -ItemType Directory -Path $summaryDir -Force | Out-Null }

$echoOutput = -not ($Quiet -or $NoEcho)
$startUtc = (Get-Date).ToUniversalTime()
$endUtc = $null
$pidValue = 0
$timedOut = $false
$killedTree = $false
$exitCode = 127

Write-Host "[safe_run] run_id: $runId"
Write-Host "[safe_run] cwd: $((Get-Location).Path)"
Write-Host "[safe_run] timeout: ${TimeoutSec}s"
Write-Host "[safe_run] cmd: $displayCmd"
Write-Host "[safe_run] out_log: $outLog"
Write-Host "[safe_run] err_log: $errLog"
Write-Host "[safe_run] summary: $summaryPath"

$proc = $null
$outStream = $null
$errStream = $null
$stdoutTask = $null
$stderrTask = $null
try {
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = $exe
  $argListProp = $psi.PSObject.Properties["ArgumentList"]
  if ($argListProp -and $argListProp.Value) {
    foreach ($a in $exeArgs) { $psi.ArgumentList.Add([string]$a) }
  } else {
    $psi.Arguments = $argStr
  }
  $psi.WorkingDirectory = (Get-Location).Path
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError = $true
  $psi.UseShellExecute = $false
  $psi.CreateNoWindow = $true

  $proc = New-Object System.Diagnostics.Process
  $proc.StartInfo = $psi
  $started = $proc.Start()
  if (-not $started) { throw "failed to start process: $exe" }
  $pidValue = [int]$proc.Id

  $outStream = New-Object System.IO.FileStream($outLog, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::Read)
  $errStream = New-Object System.IO.FileStream($errLog, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::Read)
  $stdoutTask = $proc.StandardOutput.BaseStream.CopyToAsync($outStream)
  $stderrTask = $proc.StandardError.BaseStream.CopyToAsync($errStream)

  $deadline = (Get-Date).AddSeconds([double]$TimeoutSec)
  $nextHeartbeat = if ($HeartbeatSec -gt 0) { (Get-Date).AddSeconds([double]$HeartbeatSec) } else { $null }

  while (-not $proc.HasExited) {
    if ((Get-Date) -ge $deadline) {
      $timedOut = $true
      break
    }
    if ($HeartbeatSec -gt 0 -and $nextHeartbeat -and (Get-Date) -ge $nextHeartbeat) {
      $elapsed = [int](((Get-Date).ToUniversalTime() - $startUtc).TotalSeconds)
      Write-Host "[safe_run] heartbeat: alive pid=$pidValue elapsed=${elapsed}s"
      $nextHeartbeat = (Get-Date).AddSeconds([double]$HeartbeatSec)
    }
    Start-Sleep -Milliseconds 250
  }

  if ($timedOut) {
    $killedTree = Stop-ProcessTree -ProcessId $pidValue
    try { $null = $proc.WaitForExit(3000) } catch {}
    try {
      [System.Threading.Tasks.Task]::WaitAll(@($stdoutTask, $stderrTask), 5000) | Out-Null
    } catch {}
    Start-Sleep -Milliseconds 120
    try {
      if ($errStream) {
        $timeoutLine = "[safe_run] TIMEOUT after ${TimeoutSec}s`r`n"
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($timeoutLine)
        $errStream.Write($bytes, 0, $bytes.Length)
        $errStream.Flush()
      } else {
        Add-Content -LiteralPath $errLog -Encoding UTF8 -Value "[safe_run] TIMEOUT after ${TimeoutSec}s"
      }
    } catch {
      Write-Host "[safe_run] warning: failed to append timeout marker: $($_.Exception.Message)"
    }
    $exitCode = 124
  } else {
    $proc.WaitForExit()
    try {
      [System.Threading.Tasks.Task]::WaitAll(@($stdoutTask, $stderrTask), 5000) | Out-Null
    } catch {}
    $exitCode = [int]$proc.ExitCode
  }
} catch {
  if ($proc -and (-not $proc.HasExited)) {
    $null = Stop-ProcessTree -ProcessId ([int]$proc.Id)
    try { $null = $proc.WaitForExit(3000) } catch {}
  }
  $msg = "[safe_run] failed to start/run command: $($_.Exception.Message)"
  try {
    if ($errStream) {
      $line = ($msg + "`r`n")
      $bytes = [System.Text.Encoding]::UTF8.GetBytes($line)
      $errStream.Write($bytes, 0, $bytes.Length)
      $errStream.Flush()
    } else {
      Add-Content -LiteralPath $errLog -Encoding UTF8 -Value $msg
    }
  } catch {
    Write-Host $msg
  }
  $exitCode = 127
} finally {
  if ($outStream) { try { $outStream.Dispose() } catch {} }
  if ($errStream) { try { $errStream.Dispose() } catch {} }
  if ($proc) { try { $proc.Dispose() } catch {} }
}

$endUtc = (Get-Date).ToUniversalTime()
$durationSec = [math]::Round(($endUtc - $startUtc).TotalSeconds, 3)
$summary = [ordered]@{
  schema = "safe_run_result_v1"
  run_id = $runId
  generated_at = (Get-Date).ToString("o")
  cwd = (Get-Location).Path
  command = [ordered]@{
    exe = $exe
    args = $exeArgs
    display = $displayCmd
  }
  timeout_sec = $TimeoutSec
  heartbeat_sec = $HeartbeatSec
  tail_lines = $TailLines
  pid = $pidValue
  start_at_utc = $startUtc.ToString("o")
  end_at_utc = $endUtc.ToString("o")
  duration_sec = $durationSec
  exit_code = $exitCode
  timed_out = $timedOut
  killed_process_tree = $killedTree
  stdout_log = $outLog
  stderr_log = $errLog
}
Write-JsonNoBom -Path $summaryPath -Obj $summary

Write-Host "[safe_run] exit_code: $exitCode"
if ($timedOut) {
  Write-Host "[safe_run] timeout reached; killed_tree=$killedTree"
}

if ($exitCode -ne 0) {
  $tail = [Math]::Max(1, [int]$TailLines)
  Write-Host "========== safe_run failure summary =========="
  Write-Host "command: $displayCmd"
  Write-Host "exit_code: $exitCode"
  Write-Host "--- last $tail lines (stdout) ---"
  Get-Content -LiteralPath $outLog -Tail $tail -ErrorAction SilentlyContinue
  Write-Host "--- last $tail lines (stderr) ---"
  Get-Content -LiteralPath $errLog -Tail $tail -ErrorAction SilentlyContinue
  Write-Host "============================================="
  exit $exitCode
}

if ($echoOutput) {
  if (Test-Path $outLog) { Get-Content -LiteralPath $outLog }
  if (Test-Path $errLog) { Get-Content -LiteralPath $errLog | Write-Host -ForegroundColor DarkYellow }
}

exit 0
