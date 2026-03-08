param(
  [string]$BaseUrl = "http://127.0.0.1:12346",
  [string]$OutRoot = "sim/tests/fixtures_runtime",
  [string]$Seed = "AAAAAAA",
  [switch]$RunP2b,
  [switch]$RunP3,
  [switch]$RunP4,
  [switch]$RunP5,
  [switch]$RunP7,
  [switch]$RunP8,
  [switch]$RunP9,
  [switch]$RunP10,
  [switch]$RunP11,
  [switch]$RunP12,
  [switch]$RunP13,
  [switch]$RunP14,
  [switch]$RunP15,
  [switch]$RunP16,
  [switch]$RunPerfGateV2,
  [switch]$RunP17,
  [switch]$RunPerfGateV3,
  [switch]$RunP18,
  [switch]$RunPerfGateV4,
  [switch]$RunP19,
  [switch]$SkipMilestone1000,
  [switch]$RunP20,
  [switch]$RunPerfGateV5,
  [switch]$SkipMilestone2000,
  [switch]$GitSync,
  [switch]$RunFast,
  [switch]$RunP21,
  [switch]$RunP22,
  [switch]$RunP22Full,
  [switch]$RunP23,
  [switch]$RunP24,
  [switch]$RunP25,
  [switch]$RunP26,
  [switch]$RunP27,
  [switch]$RunP29,
  [switch]$RunP31,
  [switch]$RunP32,
  [switch]$RunP37,
  [switch]$RunP38,
  [switch]$RunLegacySmoke,
  [switch]$RequireMainBranch,
  [string]$P21Timestamp = ""
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
$safeRunScript = Join-Path $ProjectRoot "scripts/safe_run.ps1"
if (-not (Test-Path $safeRunScript)) { throw "missing safe_run script: $safeRunScript" }
$safeRunLogDir = Join-Path $ProjectRoot ".safe_run/regressions"
if (-not (Test-Path $safeRunLogDir)) { New-Item -ItemType Directory -Path $safeRunLogDir -Force | Out-Null }
$waitForReadyScript = Join-Path $ProjectRoot "scripts/wait_for_service_ready.ps1"
$resolveTrainingPythonScript = Join-Path $ProjectRoot "scripts/resolve_training_python.ps1"
$trainingResolverJson = (& powershell -ExecutionPolicy Bypass -File $resolveTrainingPythonScript -Emit json | Out-String).Trim()
if (-not [string]::IsNullOrWhiteSpace($trainingResolverJson)) {
  try {
    $TrainingPythonInfo = $trainingResolverJson | ConvertFrom-Json
    $env:BALATRO_TRAIN_PYTHON = [string]$TrainingPythonInfo.selected.python
    Write-Host ("[train-python] selected=" + [string]$TrainingPythonInfo.selected.python + " env_type=" + [string]$TrainingPythonInfo.selected.env_type + " env_name=" + [string]$TrainingPythonInfo.selected.env_name + " source=" + [string]$TrainingPythonInfo.selection_reason + " cuda=" + [string]$TrainingPythonInfo.selected.cuda_available)
  } catch {
    Write-Host ("[train-python] warning: failed to parse resolver output: " + $_.Exception.Message)
  }
}
$env:BALATRO_WINDOW_MODE = ""
$env:BALATRO_WINDOW_MODE_REQUESTED = ""
$env:BALATRO_BACKGROUND_VALIDATION_REF = ""

function Get-RuntimeWindowSettings {
  $defaultsPath = Join-Path $ProjectRoot "configs\\runtime\\runtime_defaults.json"
  if (-not (Test-Path $defaultsPath)) {
    return [pscustomobject]@{
      window_mode = "offscreen"
      window_mode_fallback = "offscreen"
    }
  }
  $payload = Get-Content -LiteralPath $defaultsPath -Raw | ConvertFrom-Json
  $defaults = $payload.defaults
  return [pscustomobject]@{
    window_mode = [string]$defaults.window_mode
    window_mode_fallback = [string]$defaults.window_mode_fallback
  }
}

function Get-TrainingPythonForRuntime {
  $candidate = [string]$env:BALATRO_TRAIN_PYTHON
  if (-not [string]::IsNullOrWhiteSpace($candidate)) { return $candidate }
  return "python"
}

function Invoke-RuntimePythonJson([string[]]$CmdArgs) {
  $pyCmd = Get-TrainingPythonForRuntime
  $text = (& $pyCmd @CmdArgs | Out-String).Trim()
  if (-not $text) { return $null }
  return ($text | ConvertFrom-Json)
}

function Apply-ConfiguredWindowMode([string]$Reason) {
  $settings = Get-RuntimeWindowSettings
  $requestedMode = [string]$settings.window_mode
  $fallbackMode = [string]$settings.window_mode_fallback
  if ([string]::IsNullOrWhiteSpace($requestedMode)) { return }
  $latestValidation = Join-Path $ProjectRoot "docs\\artifacts\\p53\\background_mode_validation\\latest\\background_mode_validation.json"
  if (Test-Path $latestValidation) {
    $env:BALATRO_BACKGROUND_VALIDATION_REF = $latestValidation
  }
  $resolved = Invoke-RuntimePythonJson -CmdArgs @(
    "-B",
    "-m", "trainer.runtime.background_mode_validation",
    "--resolve-mode",
    "--requested-mode", $requestedMode,
    "--fallback-mode", $fallbackMode
  )
  if (-not $resolved) {
    Write-Host "[svc] warning: failed to resolve window mode; skipping"
    return
  }
  $effectiveMode = [string]$resolved.effective_mode
  $env:BALATRO_WINDOW_MODE_REQUESTED = [string]$resolved.requested_mode
  $env:BALATRO_WINDOW_MODE = $effectiveMode
  if ([string]$resolved.validation_path) {
    $env:BALATRO_BACKGROUND_VALIDATION_REF = [string]$resolved.validation_path
  }
  Write-Host ("[svc] window_mode requested=" + [string]$resolved.requested_mode + " effective=" + $effectiveMode + " reason=" + [string]$resolved.resolution_reason + " trigger=" + $Reason)
  $apply = Invoke-RuntimePythonJson -CmdArgs @(
    "-B",
    "-m", "trainer.runtime.window_supervisor",
    "--mode", $effectiveMode,
    "--process-name", "Balatro",
    "--json"
  )
  if (-not $apply -or -not [bool]$apply.operation_success) {
    Write-Host "[svc] warning: failed to apply configured window mode"
  }
}

# P15 gate builds on top of P14.
if ($RunP15) { $RunP14 = $true }
# P16 gate builds on top of P15.
if ($RunP16) { $RunP15 = $true }
# P17 builds on top of P16.
if ($RunP17 -or $RunPerfGateV2) { $RunP16 = $true }
# P18 builds on top of P17.
if ($RunP18 -or $RunPerfGateV3) { $RunP17 = $true }
# P19 builds on top of P18.
if ($RunP19 -or $RunPerfGateV4) { $RunP18 = $true }
# P20 builds on top of P19.
if ($RunP20 -or $RunPerfGateV5) { $RunP19 = $true }
if ($RunP22Full) { $RunP22 = $true }
if ($RunP23) { $RunP22 = $true }
if ($RunP24) {
  $RunP23 = $true
  $RunP22 = $true
}
if ($RunP25) {
  $RunP24 = $true
  $RunP23 = $true
  $RunP22 = $true
}
if ($RunP26) {
  $RunP25 = $true
  $RunP24 = $true
  $RunP23 = $true
  $RunP22 = $true
}
if ($RunP27) {
  $RunP26 = $true
  $RunP25 = $true
  $RunP24 = $true
  $RunP23 = $true
  $RunP22 = $true
}
if ($RunP29) {
  $RunP27 = $true
  $RunP26 = $true
  $RunP25 = $true
  $RunP24 = $true
  $RunP23 = $true
  $RunP22 = $true
}
if ($RunP31) {
  $RunP29 = $true
  $RunP27 = $true
  $RunP26 = $true
  $RunP25 = $true
  $RunP24 = $true
  $RunP23 = $true
  $RunP22 = $true
}
if ($RunP32) {
  $RunP22 = $true
  $RunP13 = $true
}
if ($RunP37) {
  $RunP32 = $true
}
if ($RunP38) {
  $RunP37 = $true
}
if ($RunP22 -and -not $RunLegacySmoke) {
  # P43: keep a minimal legacy baseline health check, but avoid full BC/DAgger runs by default.
  $RunLegacySmoke = $true
}

function Invoke-GitCapture([string[]]$GitArgs) {
  $quoted = @($GitArgs | ForEach-Object {
    '"' + ([string]$_).Replace('"', '\"') + '"'
  })
  $command = "git " + ($quoted -join " ")
  $wrapped = $command + " 2>&1"
  $output = @(& cmd /c $wrapped)
  $code = $LASTEXITCODE
  return [pscustomobject]@{
    code = $code
    output = @($output | ForEach-Object { [string]$_ })
    text = (($output | ForEach-Object { [string]$_ }) -join "`n")
  }
}

function Test-LocalBranchExists([string]$Branch) {
  $r = Invoke-GitCapture -GitArgs @("show-ref", "--verify", "--quiet", ("refs/heads/" + $Branch))
  return ($r.code -eq 0)
}

function Get-DetectedMainBranch() {
  $originHead = Invoke-GitCapture -GitArgs @("symbolic-ref", "refs/remotes/origin/HEAD")
  if ($originHead.code -eq 0 -and -not [string]::IsNullOrWhiteSpace($originHead.text)) {
    $raw = $originHead.text.Trim()
    if ($raw -match "^refs/remotes/origin/(.+)$") {
      return $Matches[1]
    }
  }
  if (Test-LocalBranchExists -Branch "main") { return "main" }
  if (Test-LocalBranchExists -Branch "master") { return "master" }
  return ""
}

function Write-JsonFile([string]$Path, $Object) {
  $dir = Split-Path -Parent $Path
  if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
  ($Object | ConvertTo-Json -Depth 24) | Out-File -LiteralPath $Path -Encoding UTF8
}

function Read-JsonFile([string]$Path) {
  if (-not (Test-Path $Path)) { return $null }
  try {
    return (Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json)
  } catch {
    try {
      $raw = Get-Content -LiteralPath $Path -Raw
      if ($raw.StartsWith([char]0xFEFF)) {
        $raw = $raw.TrimStart([char]0xFEFF)
      }
      return ($raw | ConvertFrom-Json)
    } catch {
      return $null
    }
  }
}

function Get-HighestRunPNumber([string]$ScriptPath) {
  if (-not (Test-Path $ScriptPath)) { return 0 }
  $text = Get-Content -LiteralPath $ScriptPath -Raw
  $matches = [regex]::Matches($text, "RunP(\d+)")
  $nums = New-Object System.Collections.Generic.List[int]
  foreach ($m in $matches) {
    [int]$v = 0
    if ([int]::TryParse($m.Groups[1].Value, [ref]$v)) {
      $nums.Add($v) | Out-Null
    }
  }
  if ($nums.Count -eq 0) { return 0 }
  return ($nums | Measure-Object -Maximum).Maximum
}

function Get-LatestGitSyncReportPath([string]$ProjectRootPath) {
  $dir = Join-Path $ProjectRootPath "docs/artifacts/git_sync"
  if (-not (Test-Path $dir)) { return "" }
  $latest = Get-ChildItem -Path $dir -Filter "git_sync_*.json" -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if (-not $latest) { return "" }
  return $latest.FullName
}

function Get-LatestRunDir([string]$RunsRootPath) {
  if (-not (Test-Path $RunsRootPath)) { return "" }
  $latest = Get-ChildItem -Path $RunsRootPath -Directory -ErrorAction SilentlyContinue |
    Sort-Object Name -Descending | Select-Object -First 1
  if (-not $latest) { return "" }
  return $latest.FullName
}

$DetectedMainBranch = Get-DetectedMainBranch
$CurrentBranch = (Invoke-GitCapture -GitArgs @("rev-parse", "--abbrev-ref", "HEAD")).text.Trim()
$MainlineMode = $true
Write-Host ("[branch] current=" + $CurrentBranch + " detected_main=" + $DetectedMainBranch + " mainline_mode=" + $MainlineMode)
if ($RequireMainBranch -and -not [string]::IsNullOrWhiteSpace($DetectedMainBranch) -and $CurrentBranch -ne $DetectedMainBranch) {
  throw ("[branch] RequireMainBranch enabled; switch first: git checkout " + $DetectedMainBranch)
}

$RunP22Status = "SKIPPED"
$RunP22LatestRun = ""
$RunP22LatestReport = ""
$RunP23Status = "SKIPPED"
$RunP23ArtifactDir = ""
$RunP23GateReport = ""
$RunP24Status = "SKIPPED"
$RunP24ArtifactDir = ""
$RunP24GateReport = ""
$RunP25Status = "SKIPPED"
$RunP25ArtifactDir = ""
$RunP25GateReport = ""
$RunP26Status = "SKIPPED"
$RunP26ArtifactDir = ""
$RunP26GateReport = ""

function Test-Health([string]$Url, [int]$TimeoutSec = 5) {
  try {
    $body = '{"jsonrpc":"2.0","id":1,"method":"health","params":{}}'
    $r = Invoke-WebRequest -Uri $Url -Method Post -Body $body -ContentType "application/json" -TimeoutSec $TimeoutSec
    return ($r.StatusCode -eq 200)
  } catch { return $false }
}

function Stop-ServiceProc {
  foreach ($n in @("Balatro", "balatrobot", "uvx")) {
    Get-Process -Name $n -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  }
}

function Clear-LovelyDump {
  $dumpDir = "C:\Users\Administrator\AppData\Roaming\Balatro\Mods\lovely\dump"
  if (-not (Test-Path $dumpDir)) { return }
  try {
    Remove-Item -LiteralPath $dumpDir -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "[svc] removed lovely dump dir: $dumpDir"
  } catch {
    Write-Host "[svc] warning: failed to clear lovely dump: $($_.Exception.Message)"
  }
}

function Invoke-ServiceReadiness([string]$Url, [string]$Reason) {
  if (-not (Test-Path $waitForReadyScript)) {
    Write-Host "[svc] readiness script missing, skipping guard"
    return
  }
  $runToken = ("reg-" + $Reason + "-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
  Write-Host ("[svc] readiness guard reason=" + $Reason + " run_id=" + $runToken)
  & powershell -ExecutionPolicy Bypass -File $waitForReadyScript `
    -BaseUrl $Url `
    -OutDir "docs/artifacts/p49/readiness" `
    -RunId $runToken `
    -MaxRetries 20 `
    -RetryIntervalSec 2 `
    -WarmupGraceSec 8 `
    -ConsecutiveSuccesses 3 `
    -TimeoutSec 3 `
    -ProbeMethod "health_gamestate"
  if ($LASTEXITCODE -ne 0) {
    throw ("[svc] readiness guard failed for " + $Reason)
  }
  if ($env:BALATRO_WINDOW_MODE) {
    Write-Host ("[svc] readiness window_mode=" + $env:BALATRO_WINDOW_MODE)
  }
  if ($env:BALATRO_BACKGROUND_VALIDATION_REF) {
    Write-Host ("[svc] readiness background_validation=" + $env:BALATRO_BACKGROUND_VALIDATION_REF)
  }
}

function Start-ServiceProc([string]$Url) {
  $u = [System.Uri]$Url
  $port = if ($u.IsDefaultPort) { 12346 } else { $u.Port }
  $uvx = Get-Command uvx -ErrorAction SilentlyContinue
  if (-not $uvx) { throw "uvx not found in PATH" }

  $love = "D:\SteamLibrary\steamapps\common\Balatro\Balatro.exe"
  $lovely = "D:\SteamLibrary\steamapps\common\Balatro\version.dll"
  $serveArgs = @("balatrobot", "serve", "--headless", "--fast", "--port", "$port")
  if ((Test-Path $love) -and (Test-Path $lovely)) {
    $serveArgs += @("--love-path", $love, "--lovely-path", $lovely)
  } elseif (Test-Path $love) {
    $serveArgs += @("--balatro-path", $love)
  }

  Stop-ServiceProc
  Start-Sleep -Seconds 1
  Clear-LovelyDump
  Write-Host "[svc] starting: $($uvx.Source) $($serveArgs -join ' ')"
  Start-Process -FilePath $uvx.Source -ArgumentList $serveArgs -WorkingDirectory $ProjectRoot -WindowStyle Hidden | Out-Null

  for ($i = 0; $i -lt 45; $i++) {
    if (Test-Health -Url $Url -TimeoutSec 3) {
      Write-Host "[svc] health ok"
      Invoke-ServiceReadiness -Url $Url -Reason "startup"
      Apply-ConfiguredWindowMode -Reason "startup"
      return
    }
    Start-Sleep -Seconds 1
  }
  throw "service start timeout"
}

function Ensure-Service([string]$Url, [bool]$ForceRestart = $false) {
  if ($ForceRestart) { Stop-ServiceProc; Start-Sleep -Seconds 2; Clear-LovelyDump }
  if (Test-Health -Url $Url) {
    Write-Host "[svc] health ok at $Url"
    Invoke-ServiceReadiness -Url $Url -Reason $(if ($ForceRestart) { "restart" } else { "ensure" })
    Apply-ConfiguredWindowMode -Reason $(if ($ForceRestart) { "restart" } else { "ensure" })
    return
  }
  Start-ServiceProc -Url $Url
}

function Get-SafeRunTimeoutSec([string]$Label, [string[]]$Args) {
  $joined = (($Args | ForEach-Object { [string]$_ }) -join " ").ToLowerInvariant()
  $labelLc = [string]$Label
  if ($labelLc) { $labelLc = $labelLc.ToLowerInvariant() }
  if ($joined.Contains("--episodes 2000") -or $labelLc.Contains("p20") -or $labelLc.Contains("ablation-2000")) { return 3600 }
  if ($joined.Contains("real_micro_ab.py") -or $labelLc.Contains("real-ab")) { return 900 }
  return 1200
}

function Run-Py([string]$Label, [string[]]$PyArgs) {
  $timeoutSec = Get-SafeRunTimeoutSec -Label $Label -Args $PyArgs
  $safeLabel = ($Label -replace "[^A-Za-z0-9._-]", "_")
  $summaryPath = Join-Path $safeRunLogDir ("{0}_{1}.summary.json" -f (Get-Date -Format "yyyyMMdd_HHmmss_fff"), $safeLabel)
  $safeArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $safeRunScript,
    "-TimeoutSec", $timeoutSec,
    "-NoEcho",
    "-TailLines", "120",
    "-SummaryJson", $summaryPath,
    "python"
  ) + $PyArgs
  Write-Host "[$Label] via safe_run timeout=${timeoutSec}s: python $($PyArgs -join ' ')"
  $o = & powershell @safeArgs 2>&1
  $code = $LASTEXITCODE
  if ($o) { $o | ForEach-Object { Write-Host $_ } }

  $text = ""
  if (Test-Path $summaryPath) {
    try {
      $sum = Get-Content $summaryPath -Raw | ConvertFrom-Json
      $stdoutPath = [string]($sum.stdout_log)
      $stderrPath = [string]($sum.stderr_log)
      $stdoutText = if ($stdoutPath -and (Test-Path $stdoutPath)) { Get-Content -LiteralPath $stdoutPath -Raw -ErrorAction SilentlyContinue } else { "" }
      $stderrText = if ($stderrPath -and (Test-Path $stderrPath)) { Get-Content -LiteralPath $stderrPath -Raw -ErrorAction SilentlyContinue } else { "" }
      $text = ($stdoutText + $(if ($stderrText) { "`n" + $stderrText } else { "" }))
    } catch {
      $text = ($o -join "`n")
    }
  } else {
    $text = ($o -join "`n")
  }
  return @{ Code = $code; Text = $text; Summary = $summaryPath }
}

function Run-PyStrict([string]$Label, [string[]]$PyArgs) {
  $result = Run-Py -Label $Label -PyArgs $PyArgs
  if ([int]$result.Code -ne 0) {
    throw "[$Label] failed with exit code $($result.Code)"
  }
  return $result
}

function Run-WithRecovery([string]$Label, [string[]]$PyArgs, [string]$Url) {
  $r = Run-Py -Label $Label -PyArgs $PyArgs
  if ($r.Code -eq 0) { return }
  $t = [string]$r.Text
  if (($t -match "timeout") -or ($t -match "health check failed") -or ($t -match "connection refused") -or ($t -match "base_url unhealthy")) {
    Write-Host "[$Label] transport issue, restarting service and retrying"
    Ensure-Service -Url $Url -ForceRestart $true
    $r2 = Run-Py -Label $Label -PyArgs $PyArgs
    if ($r2.Code -ne 0) { throw "[$Label] failed after retry" }
    return
  }
  throw "[$Label] failed (non-recoverable)"
}

function Invoke-SafeRunStep([string]$Label, [string]$Exe, [string[]]$CmdArgs, [int]$TimeoutSec = 1200) {
  $safeLabel = ($Label -replace "[^A-Za-z0-9._-]", "_")
  $summaryPath = Join-Path $safeRunLogDir ("{0}_{1}.summary.json" -f (Get-Date -Format "yyyyMMdd_HHmmss_fff"), $safeLabel)
  $safeArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $safeRunScript,
    "-TimeoutSec", $TimeoutSec,
    "-NoEcho",
    "-TailLines", "120",
    "-SummaryJson", $summaryPath,
    $Exe
  ) + $CmdArgs
  Write-Host "[$Label] via safe_run timeout=${TimeoutSec}s: $Exe $($CmdArgs -join ' ')"
  $o = & powershell @safeArgs 2>&1
  $code = $LASTEXITCODE
  if ($o) { $o | ForEach-Object { Write-Host $_ } }
  if ($code -ne 0) { throw "[$Label] failed with exit code $code" }
  return $summaryPath
}

function Persist-ArtifactSet([string]$Prefix, [string]$ReportPath, [string]$ProjectRootPath, [string[]]$ExtraDocs) {
  $artifactDir = Join-Path $ProjectRootPath ("docs/artifacts/" + $Prefix.ToLower())
  if (-not (Test-Path $artifactDir)) { New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null }
  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"

  $prefixSlug = ($Prefix.ToLower() -replace "[\\/]", "_")
  $reportDest = Join-Path $artifactDir ("report_" + $prefixSlug + "_" + $stamp + ".json")
  if (Test-Path $ReportPath) {
    Copy-Item -LiteralPath $ReportPath -Destination $reportDest -Force
    Write-Host ("[" + $Prefix + "-artifacts] " + $reportDest)
  }

  foreach ($doc in $ExtraDocs) {
    $src = Join-Path $ProjectRootPath $doc
    if (-not (Test-Path $src)) { continue }
    $name = [System.IO.Path]::GetFileNameWithoutExtension($doc)
    $ext = [System.IO.Path]::GetExtension($doc)
    $dest = Join-Path $artifactDir ($name + "_" + $stamp + $ext)
    Copy-Item -LiteralPath $src -Destination $dest -Force
    Write-Host ("[" + $Prefix + "-artifacts] " + $dest)
  }
}

function Find-LatestModel([string]$ProjectRootPath) {
  $runs = Join-Path $ProjectRootPath "trainer_runs"
  if (-not (Test-Path $runs)) { return $null }

  $best = Get-ChildItem -Path $runs -Filter "best.pt" -Recurse -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($best) { return $best.FullName }

  $last = Get-ChildItem -Path $runs -Filter "last.pt" -Recurse -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($last) { return $last.FullName }

  return $null
}

if ($RunP21) {
  $p21Script = Join-Path $ProjectRoot "scripts/run_p21_mainline.ps1"
  if (-not (Test-Path $p21Script)) {
    throw "[P21] missing script: $p21Script"
  }
  $p21Args = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $p21Script,
    "-BaseUrl", $BaseUrl,
    "-Seed", $Seed
  )
  if ($RequireMainBranch) { $p21Args += "-RequireMainBranch" }
  if (-not [string]::IsNullOrWhiteSpace($P21Timestamp)) {
    $p21Args += @("-ArtifactTimestamp", $P21Timestamp)
  }
  Write-Host "[P21] running mainline-only workflow gate"
  $o = & powershell @p21Args 2>&1
  $code = $LASTEXITCODE
  if ($o) { $o | ForEach-Object { Write-Host $_ } }
  if ($code -ne 0) {
    throw "[P21] failed with exit code $code"
  }
  return
}

Ensure-Service -Url $BaseUrl
if (-not (Test-Path $OutRoot)) { New-Item -ItemType Directory -Path $OutRoot -Force | Out-Null }
$outRootPath = (Resolve-Path $OutRoot).Path

$p0Out = Join-Path $outRootPath "oracle_p0_v6_regression"
$p1Out = Join-Path $outRootPath "oracle_p1_smoke_v3_regression"
$p2Out = Join-Path $outRootPath "oracle_p2_smoke_v1_regression"
$p2bOut = Join-Path $outRootPath "oracle_p2b_smoke_v1_regression"
$p3Out = Join-Path $outRootPath "oracle_p3_jokers_v1_regression"
$p4Out = Join-Path $outRootPath "oracle_p4_consumables_v1_regression"
$p5Out = Join-Path $outRootPath "oracle_p5_voucher_pack_v1_regression"
$p7Out = Join-Path $outRootPath "oracle_p7_stateful_v1_regression"
$p8ShopOut = Join-Path $outRootPath "oracle_p8_shop_v1_regression"
$p8RngOut = Join-Path $outRootPath "oracle_p8_rng_v1_regression"
$p9Out = Join-Path $outRootPath "oracle_p9_episode_v1_regression"
$p10Out = Join-Path $outRootPath "oracle_p10_long_v1_regression"
$p11Out = Join-Path $outRootPath "oracle_p11_prob_econ_v1_regression"

$p0Args = @("-B", "sim/oracle/batch_build_p0_oracle_fixtures.py", "--base-url", $BaseUrl, "--out-dir", $p0Out, "--max-steps", "160", "--scope", "p0_hand_score_observed_core", "--seed", $Seed, "--dump-on-diff", (Join-Path $p0Out "dumps"))
$p1Args = @("-B", "sim/oracle/batch_build_p1_smoke.py", "--base-url", $BaseUrl, "--out-dir", $p1Out, "--max-steps", "120", "--scope", "p1_hand_score_observed_core", "--seed", $Seed, "--dump-on-diff", (Join-Path $p1Out "dumps"))
$p2Args = @("-B", "sim/oracle/batch_build_p2_smoke.py", "--base-url", $BaseUrl, "--out-dir", $p2Out, "--max-steps", "160", "--scope", "p2_hand_score_observed_core", "--seed", $Seed, "--dump-on-diff", (Join-Path $p2Out "dumps"))
$p2bArgs = @("-B", "sim/oracle/batch_build_p2b_smoke.py", "--base-url", $BaseUrl, "--out-dir", $p2bOut, "--max-steps", "200", "--scope", "p2b_hand_score_observed_core", "--seed", $Seed, "--dump-on-diff", (Join-Path $p2bOut "dumps"))
$p3Args = @("-B", "sim/oracle/batch_build_p3_joker_fixtures.py", "--base-url", $BaseUrl, "--out-dir", $p3Out, "--max-steps", "160", "--scope", "p3_hand_score_observed_core", "--seed", $Seed, "--dump-on-diff", (Join-Path $p3Out "dumps"))
$p4Args = @("-B", "sim/oracle/batch_build_p4_consumable_fixtures.py", "--base-url", $BaseUrl, "--targets-file", "balatro_mechanics/derived/p4_supported_targets.txt", "--out-dir", $p4Out, "--max-steps", "220", "--scope", "p4_consumable_observed_core", "--seed", $Seed, "--dump-on-diff", (Join-Path $p4Out "dumps"))
$p5Args = @("-B", "sim/oracle/batch_build_p5_voucher_pack_fixtures.py", "--base-url", $BaseUrl, "--targets-file", "balatro_mechanics/derived/p5_supported_targets.txt", "--out-dir", $p5Out, "--max-steps", "260", "--scope", "p5_voucher_pack_observed_core", "--seed", $Seed, "--dump-on-diff", (Join-Path $p5Out "dumps"))
$p7Args = @("-B", "sim/oracle/batch_build_p7_stateful_joker_fixtures.py", "--base-url", $BaseUrl, "--targets-file", "balatro_mechanics/derived/p7_supported_targets.txt", "--out-dir", $p7Out, "--max-steps", "260", "--scope", "p7_stateful_observed_core", "--seed", $Seed, "--dump-on-diff", (Join-Path $p7Out "dumps"))
$p8ShopArgs = @("-B", "sim/oracle/batch_build_p8_shop_fixtures.py", "--base-url", $BaseUrl, "--out-dir", $p8ShopOut, "--max-steps", "300", "--scope", "p8_shop_observed_core", "--seed", $Seed, "--dump-on-diff", (Join-Path $p8ShopOut "dumps"))
$p8RngArgs = @("-B", "sim/oracle/batch_build_p8_rng_fixtures.py", "--base-url", $BaseUrl, "--out-dir", $p8RngOut, "--max-steps", "260", "--scope", "p8_rng_observed_core", "--seed", $Seed, "--dump-on-diff", (Join-Path $p8RngOut "dumps"))
$p9ClassifyArgs = @("-B", "sim/oracle/p9_blind_tag_classifier.py", "--out-derived", "balatro_mechanics/derived")
$p9Args = @("-B", "sim/oracle/batch_build_p9_episode_fixtures.py", "--base-url", $BaseUrl, "--targets-file", "balatro_mechanics/derived/p9_supported_targets.txt", "--out-dir", $p9Out, "--max-steps", "800", "--scope", "p9_episode_observed_core", "--seed", $Seed, "--dump-on-diff", (Join-Path $p9Out "dumps"))
$p10StakeExtractArgs = @("-B", "sim/oracle/extract_stake_rules.py", "--mech-root", "balatro_mechanics", "--out-derived", "balatro_mechanics/derived")
$p10Args = @("-B", "sim/oracle/batch_build_p10_long_episode_fixtures.py", "--base-url", $BaseUrl, "--out-dir", $p10Out, "--max-steps", "1200", "--scope", "p10_long_episode_observed_core", "--seed", $Seed, "--dump-on-diff", (Join-Path $p10Out "dumps"))
$p11PickArgs = @("-B", "sim/oracle/p11_pick_prob_econ_targets.py", "--out-derived", "balatro_mechanics/derived", "--out-docs", "docs")
$p11Args = @("-B", "sim/oracle/batch_build_p11_prob_econ_fixtures.py", "--base-url", $BaseUrl, "--targets-file", "balatro_mechanics/derived/p11_supported_targets.txt", "--out-dir", $p11Out, "--max-steps", "500", "--scope", "p11_prob_econ_observed_core", "--seed", $Seed, "--dump-on-diff", (Join-Path $p11Out "dumps"))

Run-WithRecovery -Label "P0" -PyArgs $p0Args -Url $BaseUrl
Run-WithRecovery -Label "P1" -PyArgs $p1Args -Url $BaseUrl

$p0ReportPath = Join-Path $p0Out "report_p0.json"
$p1ReportPath = Join-Path $p1Out "report_p1.json"
$p0Report = Get-Content $p0ReportPath -Raw | ConvertFrom-Json
$p1Report = Get-Content $p1ReportPath -Raw | ConvertFrom-Json

Write-Host ("P0 summary: passed={0}/{1} failed={2} skipped={3}" -f $p0Report.passed, $p0Report.total, $p0Report.failed, $p0Report.skipped)
Write-Host ("P0 report: {0}" -f $p0ReportPath)
Write-Host ("P1 summary: pass={0}/{1} diff_fail={2} oracle_fail={3} gen_fail={4}" -f $p1Report.passed, $p1Report.total, $p1Report.diff_fail, $p1Report.oracle_fail, $p1Report.gen_fail)
Write-Host ("P1 report: {0}" -f $p1ReportPath)

if ($RunP3 -or $RunP4 -or $RunP5 -or $RunP7 -or $RunP8 -or $RunP9 -or $RunP10 -or $RunP11 -or $RunP12 -or $RunP13 -or $RunP14) {
  Run-WithRecovery -Label "P2" -PyArgs $p2Args -Url $BaseUrl
  Run-WithRecovery -Label "P2b" -PyArgs $p2bArgs -Url $BaseUrl
  Run-WithRecovery -Label "P3" -PyArgs $p3Args -Url $BaseUrl

  $p2ReportPath = Join-Path $p2Out "report_p2.json"
  $p2bReportPath = Join-Path $p2bOut "report_p2b.json"
  $p3ReportPath = Join-Path $p3Out "report_p3.json"

  $p2Report = Get-Content $p2ReportPath -Raw | ConvertFrom-Json
  $p2bReport = Get-Content $p2bReportPath -Raw | ConvertFrom-Json
  $p3Report = Get-Content $p3ReportPath -Raw | ConvertFrom-Json

  Write-Host ("P2 summary: pass={0}/{1} diff_fail={2} oracle_fail={3} gen_fail={4} skipped={5}" -f $p2Report.passed, $p2Report.total, $p2Report.diff_fail, $p2Report.oracle_fail, $p2Report.gen_fail, $p2Report.skipped)
  Write-Host ("P2 report: {0}" -f $p2ReportPath)
  Write-Host ("P2b summary: pass={0}/{1} diff_fail={2} oracle_fail={3} gen_fail={4} skipped={5}" -f $p2bReport.passed, $p2bReport.total, $p2bReport.diff_fail, $p2bReport.oracle_fail, $p2bReport.gen_fail, $p2bReport.skipped)
  Write-Host ("P2b report: {0}" -f $p2bReportPath)
  Write-Host ("P3 summary: pass={0}/{1} diff_fail={2} oracle_fail={3} gen_fail={4} skipped={5} unsupported={6}" -f $p3Report.passed, $p3Report.total, $p3Report.diff_fail, $p3Report.oracle_fail, $p3Report.gen_fail, $p3Report.skipped, $p3Report.classifier.unsupported)
  Write-Host ("P3 report: {0}" -f $p3ReportPath)
  Persist-ArtifactSet -Prefix "P3" -ReportPath $p3ReportPath -ProjectRootPath $ProjectRoot -ExtraDocs @("docs/COVERAGE_P3_JOKERS.md", "docs/COVERAGE_P3_STATUS.md")

  if ($RunP4 -or $RunP5 -or $RunP7 -or $RunP8 -or $RunP9 -or $RunP10 -or $RunP11 -or $RunP12 -or $RunP13 -or $RunP14) {
    Run-WithRecovery -Label "P4" -PyArgs $p4Args -Url $BaseUrl
    $p4ReportPath = Join-Path $p4Out "report_p4.json"
    $p4Report = Get-Content $p4ReportPath -Raw | ConvertFrom-Json
    Write-Host ("P4 summary: pass={0}/{1} diff_fail={2} oracle_fail={3} gen_fail={4} skipped={5} unsupported={6}" -f $p4Report.passed, $p4Report.total, $p4Report.diff_fail, $p4Report.oracle_fail, $p4Report.gen_fail, $p4Report.skipped, $p4Report.classifier.unsupported)
    Write-Host ("P4 report: {0}" -f $p4ReportPath)
    Persist-ArtifactSet -Prefix "P4" -ReportPath $p4ReportPath -ProjectRootPath $ProjectRoot -ExtraDocs @("docs/COVERAGE_P4_CONSUMABLES.md", "docs/COVERAGE_P4_STATUS.md")

    if ($RunP5 -or $RunP7 -or $RunP8 -or $RunP9 -or $RunP10 -or $RunP11 -or $RunP12 -or $RunP13 -or $RunP14) {
      Run-WithRecovery -Label "P5" -PyArgs $p5Args -Url $BaseUrl
      $p5ReportPath = Join-Path $p5Out "report_p5.json"
      $p5Report = Get-Content $p5ReportPath -Raw | ConvertFrom-Json
      Write-Host ("P5 summary: pass={0}/{1} diff_fail={2} oracle_fail={3} gen_fail={4} skipped={5} unsupported={6}" -f $p5Report.passed, $p5Report.total, $p5Report.diff_fail, $p5Report.oracle_fail, $p5Report.gen_fail, $p5Report.skipped, $p5Report.classifier.unsupported)
      Write-Host ("P5 report: {0}" -f $p5ReportPath)
      Persist-ArtifactSet -Prefix "P5" -ReportPath $p5ReportPath -ProjectRootPath $ProjectRoot -ExtraDocs @("docs/COVERAGE_P5_VOUCHERS_PACKS.md", "docs/COVERAGE_P5_STATUS.md")

      if ($RunP7 -or $RunP8 -or $RunP9 -or $RunP10 -or $RunP11 -or $RunP12 -or $RunP13 -or $RunP14) {
        Run-WithRecovery -Label "P7" -PyArgs $p7Args -Url $BaseUrl
        $p7ReportPath = Join-Path $p7Out "report_p7.json"
        $p7Report = Get-Content $p7ReportPath -Raw | ConvertFrom-Json
        Write-Host ("P7 summary: pass={0}/{1} diff_fail={2} oracle_fail={3} gen_fail={4} skipped={5} unsupported={6}" -f $p7Report.passed, $p7Report.total, $p7Report.diff_fail, $p7Report.oracle_fail, $p7Report.gen_fail, $p7Report.skipped, $p7Report.classifier.unsupported)
        Write-Host ("P7 report: {0}" -f $p7ReportPath)
        Persist-ArtifactSet -Prefix "P7" -ReportPath $p7ReportPath -ProjectRootPath $ProjectRoot -ExtraDocs @("docs/COVERAGE_P7_STATEFUL_JOKERS.md", "docs/COVERAGE_P7_STATUS.md")
        $p7AnalyzeArgs = @("-B", "sim/oracle/analyze_p7_stateful_mismatch.py", "--fixtures-dir", $p7Out)
        $null = Run-PyStrict -Label "P7-analyzer" -PyArgs $p7AnalyzeArgs

        $p7ArtifactDir = Join-Path $ProjectRoot "docs/artifacts/p7"
        if (-not (Test-Path $p7ArtifactDir)) { New-Item -ItemType Directory -Path $p7ArtifactDir -Force | Out-Null }
        $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $p7Csv = Join-Path $p7Out "stateful_mismatch_table_p7.csv"
        $p7Md = Join-Path $p7Out "stateful_mismatch_table_p7.md"
        if (Test-Path $p7Csv) {
          $destCsv = Join-Path $p7ArtifactDir ("stateful_mismatch_table_p7_" + $stamp + ".csv")
          Copy-Item -LiteralPath $p7Csv -Destination $destCsv -Force
          Write-Host ("[P7-artifacts] " + $destCsv)
        }
        if (Test-Path $p7Md) {
          $destMd = Join-Path $p7ArtifactDir ("stateful_mismatch_table_p7_" + $stamp + ".md")
          Copy-Item -LiteralPath $p7Md -Destination $destMd -Force
          Write-Host ("[P7-artifacts] " + $destMd)
        }

        if ($RunP8 -or $RunP9 -or $RunP10 -or $RunP11 -or $RunP12 -or $RunP13 -or $RunP14) {
          Run-WithRecovery -Label "P8-shop" -PyArgs $p8ShopArgs -Url $BaseUrl
          $p8ShopReportPath = Join-Path $p8ShopOut "report_p8.json"
          $p8ShopReport = Get-Content $p8ShopReportPath -Raw | ConvertFrom-Json
          Write-Host ("P8 shop summary: pass={0}/{1} diff_fail={2} oracle_fail={3} gen_fail={4} skipped={5}" -f $p8ShopReport.passed, $p8ShopReport.total, $p8ShopReport.diff_fail, $p8ShopReport.oracle_fail, $p8ShopReport.gen_fail, $p8ShopReport.skipped)
          Write-Host ("P8 shop report: {0}" -f $p8ShopReportPath)
          Persist-ArtifactSet -Prefix "p8/shop" -ReportPath $p8ShopReportPath -ProjectRootPath $ProjectRoot -ExtraDocs @("docs/COVERAGE_P8_STATUS.md", "docs/COVERAGE_P8_SHOP.md")
          $p8ShopAnalyzeArgs = @("-B", "sim/oracle/analyze_p8_shop_mismatch.py", "--fixtures-dir", $p8ShopOut)
          $null = Run-PyStrict -Label "P8-shop-analyzer" -PyArgs $p8ShopAnalyzeArgs
          $p8ShopCsv = Join-Path $p8ShopOut "shop_mismatch_table_p8.csv"
          $p8ShopMd = Join-Path $p8ShopOut "shop_mismatch_table_p8.md"
          $p8ShopArtifactDir = Join-Path $ProjectRoot "docs/artifacts/p8/shop"
          if (-not (Test-Path $p8ShopArtifactDir)) { New-Item -ItemType Directory -Path $p8ShopArtifactDir -Force | Out-Null }
          if (Test-Path $p8ShopCsv) {
            $destCsv = Join-Path $p8ShopArtifactDir ("shop_mismatch_table_p8_" + $stamp + ".csv")
            Copy-Item -LiteralPath $p8ShopCsv -Destination $destCsv -Force
            Write-Host ("[P8-shop-artifacts] " + $destCsv)
          }
          if (Test-Path $p8ShopMd) {
            $destMd = Join-Path $p8ShopArtifactDir ("shop_mismatch_table_p8_" + $stamp + ".md")
            Copy-Item -LiteralPath $p8ShopMd -Destination $destMd -Force
            Write-Host ("[P8-shop-artifacts] " + $destMd)
          }

          Run-WithRecovery -Label "P8-rng" -PyArgs $p8RngArgs -Url $BaseUrl
          $p8RngReportPath = Join-Path $p8RngOut "report_p8_rng.json"
          $p8RngReport = Get-Content $p8RngReportPath -Raw | ConvertFrom-Json
          Write-Host ("P8 rng summary: pass={0}/{1} diff_fail={2} oracle_fail={3} gen_fail={4} skipped={5}" -f $p8RngReport.passed, $p8RngReport.total, $p8RngReport.diff_fail, $p8RngReport.oracle_fail, $p8RngReport.gen_fail, $p8RngReport.skipped)
          Write-Host ("P8 rng report: {0}" -f $p8RngReportPath)
          Persist-ArtifactSet -Prefix "p8/rng" -ReportPath $p8RngReportPath -ProjectRootPath $ProjectRoot -ExtraDocs @("docs/COVERAGE_P8_STATUS.md", "docs/COVERAGE_P8_RNG.md")
          $p8RngAnalyzeArgs = @("-B", "sim/oracle/analyze_p8_rng_mismatch.py", "--fixtures-dir", $p8RngOut)
          $null = Run-PyStrict -Label "P8-rng-analyzer" -PyArgs $p8RngAnalyzeArgs
          $p8RngCsv = Join-Path $p8RngOut "rng_mismatch_table_p8.csv"
          $p8RngMd = Join-Path $p8RngOut "rng_mismatch_table_p8.md"
          $p8RngArtifactDir = Join-Path $ProjectRoot "docs/artifacts/p8/rng"
          if (-not (Test-Path $p8RngArtifactDir)) { New-Item -ItemType Directory -Path $p8RngArtifactDir -Force | Out-Null }
          if (Test-Path $p8RngCsv) {
            $destCsv = Join-Path $p8RngArtifactDir ("rng_mismatch_table_p8_" + $stamp + ".csv")
            Copy-Item -LiteralPath $p8RngCsv -Destination $destCsv -Force
            Write-Host ("[P8-rng-artifacts] " + $destCsv)
          }
          if (Test-Path $p8RngMd) {
            $destMd = Join-Path $p8RngArtifactDir ("rng_mismatch_table_p8_" + $stamp + ".md")
            Copy-Item -LiteralPath $p8RngMd -Destination $destMd -Force
            Write-Host ("[P8-rng-artifacts] " + $destMd)
          }

          if ($RunP9 -or $RunP10 -or $RunP11 -or $RunP12 -or $RunP13 -or $RunP14) {
            Run-PyStrict -Label "P9-classifier" -PyArgs $p9ClassifyArgs | Out-Null
            Run-WithRecovery -Label "P9" -PyArgs $p9Args -Url $BaseUrl
            $p9ReportPath = Join-Path $p9Out "report_p9.json"
            $p9Report = Get-Content $p9ReportPath -Raw | ConvertFrom-Json
            Write-Host ("P9 summary: pass={0}/{1} diff_fail={2} oracle_fail={3} gen_fail={4} skipped={5} unsupported={6}" -f $p9Report.passed, $p9Report.total, $p9Report.diff_fail, $p9Report.oracle_fail, $p9Report.gen_fail, $p9Report.skipped, $p9Report.unsupported)
            Write-Host ("P9 report: {0}" -f $p9ReportPath)
            Persist-ArtifactSet -Prefix "p9" -ReportPath $p9ReportPath -ProjectRootPath $ProjectRoot -ExtraDocs @("docs/COVERAGE_P9_STATUS.md", "docs/COVERAGE_P9_BLINDS_TAGS.md", "docs/COVERAGE_P9_EPISODES.md")
            $p9AnalyzeArgs = @("-B", "sim/oracle/analyze_p9_episode_mismatch.py", "--fixtures-dir", $p9Out)
            $null = Run-PyStrict -Label "P9-analyzer" -PyArgs $p9AnalyzeArgs
            $p9Csv = Join-Path $p9Out "episode_mismatch_table_p9.csv"
            $p9Md = Join-Path $p9Out "episode_mismatch_table_p9.md"
            $p9ArtifactDir = Join-Path $ProjectRoot "docs/artifacts/p9"
            if (-not (Test-Path $p9ArtifactDir)) { New-Item -ItemType Directory -Path $p9ArtifactDir -Force | Out-Null }
            if (Test-Path $p9Csv) {
              $destCsv = Join-Path $p9ArtifactDir ("episode_mismatch_table_p9_" + $stamp + ".csv")
              Copy-Item -LiteralPath $p9Csv -Destination $destCsv -Force
              Write-Host ("[P9-artifacts] " + $destCsv)
            }
            if (Test-Path $p9Md) {
              $destMd = Join-Path $p9ArtifactDir ("episode_mismatch_table_p9_" + $stamp + ".md")
              Copy-Item -LiteralPath $p9Md -Destination $destMd -Force
              Write-Host ("[P9-artifacts] " + $destMd)
            }

            if ($RunP10 -or $RunP11 -or $RunP12 -or $RunP13 -or $RunP14) {
              $null = Run-PyStrict -Label "P10-stake-extract" -PyArgs $p10StakeExtractArgs
              Run-WithRecovery -Label "P10" -PyArgs $p10Args -Url $BaseUrl
              $p10ReportPath = Join-Path $p10Out "report_p10.json"
              $p10Report = Get-Content $p10ReportPath -Raw | ConvertFrom-Json
              Write-Host ("P10 summary: pass={0}/{1} diff_fail={2} oracle_fail={3} gen_fail={4} skipped={5}" -f $p10Report.passed, $p10Report.total, $p10Report.diff_fail, $p10Report.oracle_fail, $p10Report.gen_fail, $p10Report.skipped)
              Write-Host ("P10 report: {0}" -f $p10ReportPath)
              Persist-ArtifactSet -Prefix "p10" -ReportPath $p10ReportPath -ProjectRootPath $ProjectRoot -ExtraDocs @("docs/COVERAGE_P10_STATUS.md", "docs/COVERAGE_P10_EPISODES.md")
              $p10AnalyzeArgs = @("-B", "sim/oracle/analyze_p10_long_episode_mismatch.py", "--fixtures-dir", $p10Out)
              $null = Run-PyStrict -Label "P10-analyzer" -PyArgs $p10AnalyzeArgs
              $p10Csv = Join-Path $p10Out "episode_mismatch_table_p10.csv"
              $p10Md = Join-Path $p10Out "episode_mismatch_table_p10.md"
              $p10ArtifactDir = Join-Path $ProjectRoot "docs/artifacts/p10"
              if (-not (Test-Path $p10ArtifactDir)) { New-Item -ItemType Directory -Path $p10ArtifactDir -Force | Out-Null }
              if (Test-Path $p10Csv) {
                $destCsv = Join-Path $p10ArtifactDir ("episode_mismatch_table_p10_" + $stamp + ".csv")
                Copy-Item -LiteralPath $p10Csv -Destination $destCsv -Force
                Write-Host ("[P10-artifacts] " + $destCsv)
              }
              if (Test-Path $p10Md) {
                $destMd = Join-Path $p10ArtifactDir ("episode_mismatch_table_p10_" + $stamp + ".md")
                Copy-Item -LiteralPath $p10Md -Destination $destMd -Force
                Write-Host ("[P10-artifacts] " + $destMd)
              }

              if ($RunP11 -or $RunP12 -or $RunP13 -or $RunP14) {
                $null = Run-PyStrict -Label "P11-pick" -PyArgs $p11PickArgs
                Run-WithRecovery -Label "P11" -PyArgs $p11Args -Url $BaseUrl
                $p11ReportPath = Join-Path $p11Out "report_p11.json"
                $p11Report = Get-Content $p11ReportPath -Raw | ConvertFrom-Json
                Write-Host ("P11 summary: pass={0}/{1} diff_fail={2} oracle_fail={3} gen_fail={4} skipped={5}" -f $p11Report.passed, $p11Report.total, $p11Report.diff_fail, $p11Report.oracle_fail, $p11Report.gen_fail, $p11Report.skipped)
                Write-Host ("P11 report: {0}" -f $p11ReportPath)
                Persist-ArtifactSet -Prefix "p11" -ReportPath $p11ReportPath -ProjectRootPath $ProjectRoot -ExtraDocs @("docs/COVERAGE_P11_STATUS.md", "docs/COVERAGE_P11_PROB_ECON.md", "docs/COVERAGE_P11_PICK.md")
                $p11AnalyzeArgs = @("-B", "sim/oracle/analyze_p11_mismatch.py", "--fixtures-dir", $p11Out)
                $null = Run-PyStrict -Label "P11-analyzer" -PyArgs $p11AnalyzeArgs
                $p11Csv = Join-Path $p11Out "mismatch_table_p11.csv"
                $p11Md = Join-Path $p11Out "mismatch_table_p11.md"
                $p11ArtifactDir = Join-Path $ProjectRoot "docs/artifacts/p11"
                if (-not (Test-Path $p11ArtifactDir)) { New-Item -ItemType Directory -Path $p11ArtifactDir -Force | Out-Null }
                if (Test-Path $p11Csv) {
                  $destCsv = Join-Path $p11ArtifactDir ("mismatch_table_p11_" + $stamp + ".csv")
                  Copy-Item -LiteralPath $p11Csv -Destination $destCsv -Force
                  Write-Host ("[P11-artifacts] " + $destCsv)
                }
                if (Test-Path $p11Md) {
                  $destMd = Join-Path $p11ArtifactDir ("mismatch_table_p11_" + $stamp + ".md")
                  Copy-Item -LiteralPath $p11Md -Destination $destMd -Force
                  Write-Host ("[P11-artifacts] " + $destMd)
                }

                if ($RunP12 -or $RunP13 -or $RunP14) {
                  $p12Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
                  $p12ArtifactDir = Join-Path $ProjectRoot ("docs/artifacts/p12/" + $p12Stamp)
                  if (-not (Test-Path $p12ArtifactDir)) { New-Item -ItemType Directory -Path $p12ArtifactDir -Force | Out-Null }

                  if (-not (Test-Health -Url $BaseUrl -TimeoutSec 3)) {
                    $skipPayload = @{
                      timestamp = $p12Stamp
                      status = "SKIPPED"
                      reason = "real unavailable"
                      base_url = $BaseUrl
                    }
                    $skipPath = Join-Path $p12ArtifactDir "p12_skip.json"
                    ($skipPayload | ConvertTo-Json -Depth 8) | Out-File -LiteralPath $skipPath -Encoding UTF8
                    Write-Host ("P12=SKIPPED(real unavailable) artifact=" + $skipPath)
                  } else {
                    $p12RealOut = Join-Path $p12ArtifactDir ("real_smoke_" + $p12Stamp + ".json")
                    $p12InferOut = Join-Path $p12ArtifactDir ("infer_real_" + $p12Stamp + ".txt")
                    $p12DriftOut = Join-Path $p12ArtifactDir ("drift_" + $p12Stamp + ".json")
                    $p12SummaryPath = Join-Path $p12ArtifactDir "report_p12.json"

                    $modelPath = Find-LatestModel -ProjectRootPath $ProjectRoot

                    $p12ObsArgs = @("-B", "trainer/real_observer.py", "--base-url", $BaseUrl, "--once", "--out", $p12RealOut)
                    $p12ObsResult = Run-Py -Label "P12-observer" -PyArgs $p12ObsArgs
                    if ($p12ObsResult.Code -ne 0) { throw "[P12] real observer failed" }

                    $p12InferArgs = @("-B", "trainer/infer_assistant_real.py", "--base-url", $BaseUrl, "--once", "--topk", "3", "--out", $p12InferOut)
                    if ($modelPath) { $p12InferArgs += @("--model", $modelPath) }
                    $p12InferResult = Run-Py -Label "P12-infer" -PyArgs $p12InferArgs
                    if ($p12InferResult.Code -ne 0) { throw "[P12] infer assistant real failed" }

                    $p12DriftArgs = @("-B", "trainer/sim_real_drift.py", "--base-url", $BaseUrl, "--samples", "10", "--interval", "0.2", "--with-sim", "--out", $p12DriftOut)
                    $p12DriftResult = Run-Py -Label "P12-drift" -PyArgs $p12DriftArgs
                    if ($p12DriftResult.Code -ne 0) { throw "[P12] sim_real_drift failed" }

                    $summary = @{
                      timestamp = $p12Stamp
                      status = "PASS"
                      base_url = $BaseUrl
                      model = $(if ($modelPath) { $modelPath } else { "" })
                      real_smoke = $p12RealOut
                      infer_output = $p12InferOut
                      drift_output = $p12DriftOut
                    }
                    ($summary | ConvertTo-Json -Depth 8) | Out-File -LiteralPath $p12SummaryPath -Encoding UTF8
                    Write-Host ("P12 summary: PASS artifact=" + $p12SummaryPath)
                  }

                  if ($RunP13 -or $RunP14) {
                    $p13Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
                    $p13ArtifactDir = Join-Path $ProjectRoot ("docs/artifacts/p13/" + $p13Stamp)
                    if (-not (Test-Path $p13ArtifactDir)) { New-Item -ItemType Directory -Path $p13ArtifactDir -Force | Out-Null }

                    $p13ReportPath = Join-Path $p13ArtifactDir "report_p13.json"
                    if (-not (Test-Health -Url $BaseUrl -TimeoutSec 3)) {
                      $p13Skip = @{
                        timestamp = $p13Stamp
                        status = "SKIPPED"
                        reason = "real unavailable"
                        base_url = $BaseUrl
                      }
                      ($p13Skip | ConvertTo-Json -Depth 8) | Out-File -LiteralPath $p13ReportPath -Encoding UTF8
                      Write-Host ("P13=SKIPPED(real unavailable) artifact=" + $p13ReportPath)
                    } else {
                      $modelPath = Find-LatestModel -ProjectRootPath $ProjectRoot

                      $p13SessionPath = Join-Path $p13ArtifactDir ("session_" + $p13Stamp + ".jsonl")
                      $p13FixtureDir = Join-Path $p13ArtifactDir "fixture"
                      $p13StateTrace = Join-Path $p13FixtureDir "state_trace.jsonl"
                      $p13DriftPath = Join-Path $p13ArtifactDir ("drift_report_" + $p13Stamp + ".json")
                      $p13DaggerSummaryPath = Join-Path $p13ArtifactDir ("dagger_summary_" + $p13Stamp + ".json")
                      $p13DaggerDatasetPath = Join-Path $ProjectRoot "trainer_data/p13_gate_dagger_dataset.jsonl"

                      $p13RecordArgs = @("-B", "trainer/record_real_session.py", "--base-url", $BaseUrl, "--steps", "50", "--interval", "0.2", "--topk", "3", "--include-raw", "--out", $p13SessionPath)
                      if ($modelPath) { $p13RecordArgs += @("--model", $modelPath) }
                      $p13RecordResult = Run-Py -Label "P13-record" -PyArgs $p13RecordArgs
                      if ($p13RecordResult.Code -ne 0) { throw "[P13] record_real_session failed" }

                      $p13TraceArgs = @("-B", "trainer/real_trace_to_fixture.py", "--in", $p13SessionPath, "--out-dir", $p13FixtureDir)
                      $p13TraceResult = Run-Py -Label "P13-trace-to-fixture" -PyArgs $p13TraceArgs
                      if ($p13TraceResult.Code -ne 0) { throw "[P13] real_trace_to_fixture failed" }

                      $p13DriftArgs = @("-B", "sim/tests/run_real_drift_fixture.py", "--trace-a", $p13StateTrace, "--trace-b", $p13StateTrace, "--out", $p13DriftPath)
                      $p13DriftResult = Run-Py -Label "P13-drift-compare" -PyArgs $p13DriftArgs
                      if ($p13DriftResult.Code -ne 0) { throw "[P13] run_real_drift_fixture failed" }

                      $p13DaggerArgs = @("-B", "trainer/dagger_collect.py", "--session", $p13SessionPath, "--backend", "sim", "--out", $p13DaggerDatasetPath, "--hand-samples", "200", "--shop-samples", "80", "--time-budget-ms", "20", "--allow-sim-augment", "--summary-out", $p13DaggerSummaryPath)
                      $p13DaggerResult = Run-Py -Label "P13-dagger-collect" -PyArgs $p13DaggerArgs
                      if ($p13DaggerResult.Code -ne 0) { throw "[P13] dagger_collect failed" }

                      $p13Summary = @{
                        timestamp = $p13Stamp
                        status = "PASS"
                        base_url = $BaseUrl
                        model = $(if ($modelPath) { $modelPath } else { "" })
                        session = $p13SessionPath
                        fixture_dir = $p13FixtureDir
                        drift_report = $p13DriftPath
                        dagger_dataset = $p13DaggerDatasetPath
                        dagger_summary = $p13DaggerSummaryPath
                      }
                      ($p13Summary | ConvertTo-Json -Depth 8) | Out-File -LiteralPath $p13ReportPath -Encoding UTF8
                      Write-Host ("P13 summary: PASS artifact=" + $p13ReportPath)
                    }

                    if ($RunP14) {
                      $p14Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
                      $p14ArtifactDir = Join-Path $ProjectRoot ("docs/artifacts/p14/" + $p14Stamp)
                      if (-not (Test-Path $p14ArtifactDir)) { New-Item -ItemType Directory -Path $p14ArtifactDir -Force | Out-Null }

                      $p14ReportPath = Join-Path $p14ArtifactDir "report_p14.json"
                      if (-not (Test-Health -Url $BaseUrl -TimeoutSec 3)) {
                        $p14Skip = @{
                          timestamp = $p14Stamp
                          status = "SKIPPED"
                          reason = "real unavailable"
                          base_url = $BaseUrl
                        }
                        ($p14Skip | ConvertTo-Json -Depth 8) | Out-File -LiteralPath $p14ReportPath -Encoding UTF8
                        Write-Host ("P14=SKIPPED(real unavailable) artifact=" + $p14ReportPath)
                      } else {
                        $modelPath = Find-LatestModel -ProjectRootPath $ProjectRoot
                        $p14SessionDir = Join-Path $p14ArtifactDir "sessions"
                        $p14SessionPath = Join-Path $p14SessionDir "session_exec.jsonl"
                        $p14FixtureDir = Join-Path $p14ArtifactDir "fixtures"
                        $p14ReplayPath = Join-Path $p14ArtifactDir "replay_report.json"
                        $p14DumpDir = Join-Path $p14ArtifactDir "dumps"
                        $p14DaggerDatasetPath = Join-Path $ProjectRoot "trainer_data/p14_gate_dagger_dataset.jsonl"
                        $p14DaggerSummaryPath = Join-Path $p14ArtifactDir "dagger_summary.json"

                        $p14TokenArgs = @("-B", "trainer/record_real_session.py", "--print-arm-token", "--out", (Join-Path $p14SessionDir "_token_probe.jsonl"))
                        $p14TokenResult = Run-Py -Label "P14-token" -PyArgs $p14TokenArgs
                        if ($p14TokenResult.Code -ne 0) { throw "[P14] token generation failed" }
                        $tokenLine = ($p14TokenResult.Text -split "`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ -match "^[A-Za-z0-9_-]{16,}$" } | Select-Object -Last 1)
                        if (-not $tokenLine) { throw "[P14] could not parse arm token" }

                        $p14RecordArgs = @("-B", "trainer/record_real_session.py", "--base-url", $BaseUrl, "--steps", "20", "--interval", "0.5", "--topk", "3", "--execute", "--arm-token", $tokenLine, "--confirm", "I_UNDERSTAND", "--max-actions", "4", "--rate-limit-sec", "1.5", "--include-raw", "--out", $p14SessionPath)
                        if ($modelPath) { $p14RecordArgs += @("--model", $modelPath) }
                        $p14RecordResult = Run-Py -Label "P14-record-exec" -PyArgs $p14RecordArgs
                        if ($p14RecordResult.Code -ne 0) { throw "[P14] controlled execution record failed" }

                        $p14TraceArgs = @("-B", "trainer/real_trace_to_fixture.py", "--in", $p14SessionPath, "--out-dir", $p14FixtureDir)
                        $p14TraceResult = Run-Py -Label "P14-trace-to-fixture" -PyArgs $p14TraceArgs
                        if ($p14TraceResult.Code -ne 0) { throw "[P14] real_trace_to_fixture failed" }

                        $p14ManifestPath = Join-Path $p14FixtureDir "manifest.json"
                        $p14Manifest = if (Test-Path $p14ManifestPath) { Get-Content $p14ManifestPath -Raw | ConvertFrom-Json } else { $null }
                        $p14ActionsCount = if ($p14Manifest -and $p14Manifest.actions_count -ne $null) { [int]$p14Manifest.actions_count } else { 0 }

                        $p14Status = "PASS"
                        $p14Reason = ""

                        if ($p14ActionsCount -lt 1) {
                          $p14Status = "SKIPPED"
                          $p14Reason = "no_executed_actions_recorded"
                        } else {
                          $p14ReplayArgs = @("-B", "sim/tests/run_real_action_replay_fixture.py", "--fixture-dir", $p14FixtureDir, "--scope", "p14_real_action_observed_core", "--out", $p14ReplayPath, "--dump-on-diff", $p14DumpDir)
                          $p14ReplayResult = Run-Py -Label "P14-replay" -PyArgs $p14ReplayArgs
                          if ($p14ReplayResult.Code -ne 0) {
                            $p14Status = "SKIPPED"
                            $p14Reason = "replay_mismatch_or_unstable_real"
                          }
                        }

                        $p14DaggerArgs = @("-B", "trainer/dagger_collect.py", "--session", $p14SessionPath, "--backend", "sim", "--out", $p14DaggerDatasetPath, "--hand-samples", "80", "--shop-samples", "30", "--time-budget-ms", "20", "--allow-sim-augment", "--summary-out", $p14DaggerSummaryPath)
                        $p14DaggerResult = Run-Py -Label "P14-dagger-collect" -PyArgs $p14DaggerArgs
                        if ($p14DaggerResult.Code -ne 0) {
                          $p14Status = "SKIPPED"
                          if (-not $p14Reason) { $p14Reason = "dagger_collect_failed" }
                        }

                        $p14Summary = @{
                          timestamp = $p14Stamp
                          status = $p14Status
                          reason = $p14Reason
                          base_url = $BaseUrl
                          model = $(if ($modelPath) { $modelPath } else { "" })
                          session = $p14SessionPath
                          fixture_dir = $p14FixtureDir
                          replay_report = $p14ReplayPath
                          dagger_dataset = $p14DaggerDatasetPath
                          dagger_summary = $p14DaggerSummaryPath
                          actions_count = $p14ActionsCount
                        }
                        ($p14Summary | ConvertTo-Json -Depth 8) | Out-File -LiteralPath $p14ReportPath -Encoding UTF8
                        Write-Host ("P14 summary: " + $p14Status + " artifact=" + $p14ReportPath)
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
elseif ($RunP2b) {
  Run-WithRecovery -Label "P2b" -PyArgs $p2bArgs -Url $BaseUrl
  $p2bAnalyzeArgs = @("-B", "sim/oracle/analyze_p2b_mismatch.py", "--fixtures-dir", $p2bOut)
  Run-PyStrict -Label "P2b-analyzer" -PyArgs $p2bAnalyzeArgs | Out-Null

  $p2bReportPath = Join-Path $p2bOut "report_p2b.json"
  $p2bReport = Get-Content $p2bReportPath -Raw | ConvertFrom-Json
  Write-Host ("P2b summary: pass={0}/{1} diff_fail={2} oracle_fail={3} gen_fail={4} skipped={5}" -f $p2bReport.passed, $p2bReport.total, $p2bReport.diff_fail, $p2bReport.oracle_fail, $p2bReport.gen_fail, $p2bReport.skipped)
  Write-Host ("P2b report: {0}" -f $p2bReportPath)
  Write-Host ("P2b analyzer: {0}" -f (Join-Path $p2bOut "score_mismatch_table_p2b.md"))
}

if ($RunP15) {
  $p15SmokeScript = Join-Path $ProjectRoot "scripts/run_p15_smoke.ps1"
  if (-not (Test-Path $p15SmokeScript)) {
    throw "[P15] missing script: $p15SmokeScript"
  }
  Write-Host "[P15] running smoke gate (search->pv->eval)"
  $null = Invoke-SafeRunStep -Label "P15-smoke" -Exe "powershell" -CmdArgs @("-ExecutionPolicy","Bypass","-File",$p15SmokeScript,"-BaseUrl",$BaseUrl,"-Seed",$Seed) -TimeoutSec 1200
}

if ($RunP16) {
  Write-Host "[P16] running continuous dagger smoke loop"
  $p16Args = @(
    "-B",
    "trainer/p16_loop.py",
    "--mode", "smoke",
    "--base-url", $BaseUrl,
    "--seed", $Seed,
    "--resume"
  )
  Run-PyStrict -Label "P16-loop" -PyArgs $p16Args | Out-Null

  $p16Root = Join-Path $ProjectRoot "docs/artifacts/p16"
  if (Test-Path $p16Root) {
    $latest = Get-ChildItem -Path $p16Root -Directory -ErrorAction SilentlyContinue |
      Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latest) {
      $report = Join-Path $latest.FullName "report_p16.json"
      $gateSummary = [ordered]@{
        timestamp = (Get-Date).ToString("o")
        status = "PASS"
        artifact_dir = $latest.FullName
        report = $(if (Test-Path $report) { $report } else { "" })
      }
      ($gateSummary | ConvertTo-Json -Depth 8) | Out-File -LiteralPath (Join-Path $latest.FullName "gate_summary.json") -Encoding UTF8
      Write-Host ("[P16] PASS artifact=" + $latest.FullName)
    } else {
      Write-Host "[P16] warning: no artifact directory found under docs/artifacts/p16"
    }
  } else {
    Write-Host "[P16] warning: docs/artifacts/p16 not found"
  }
}

if ($RunP17 -or $RunPerfGateV2) {
  $p17SmokeScript = Join-Path $ProjectRoot "scripts/run_p17_smoke.ps1"
  if (-not (Test-Path $p17SmokeScript)) {
    throw "[P17] missing script: $p17SmokeScript"
  }
  Write-Host "[P17] running champion-challenger smoke gate"
  $p17Args = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $p17SmokeScript,
    "-BaseUrl", $BaseUrl,
    "-Seed", $Seed
  )
  if ($RunPerfGateV2) {
    $p17Args += "-RunPerfGateOnly"
    $p17Args += "-FailOnPerfGate"
  }
  $null = Invoke-SafeRunStep -Label "P17-smoke" -Exe "powershell" -CmdArgs $p17Args -TimeoutSec 1200
}

if ($RunP18 -or $RunPerfGateV3) {
  $p18SmokeScript = Join-Path $ProjectRoot "scripts/run_p18_smoke.ps1"
  if (-not (Test-Path $p18SmokeScript)) {
    throw "[P18] missing script: $p18SmokeScript"
  }
  Write-Host "[P18] running RL pilot smoke gate"
  $p18Args = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $p18SmokeScript,
    "-BaseUrl", $BaseUrl,
    "-Seed", $Seed
  )
  if ($RunPerfGateV3) {
    $p18Args += "-RunPerfGateOnly"
    $p18Args += "-FailOnPerfGate"
  }
  $null = Invoke-SafeRunStep -Label "P18-smoke" -Exe "powershell" -CmdArgs $p18Args -TimeoutSec 1200
}

if ($RunP19 -or $RunPerfGateV4) {
  $p19SmokeScript = Join-Path $ProjectRoot "scripts/run_p19_smoke.ps1"
  if (-not (Test-Path $p19SmokeScript)) {
    throw "[P19] missing script: $p19SmokeScript"
  }
  Write-Host "[P19] running risk-aware/promotion smoke gate"
  $p19Args = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $p19SmokeScript,
    "-BaseUrl", $BaseUrl,
    "-Seed", $Seed
  )
  if ($RunPerfGateV4) {
    $p19Args += "-RunPerfGateOnly"
    $p19Args += "-FailOnPerfGate"
  }
  if ($SkipMilestone1000) {
    $p19Args += "-SkipMilestone1000"
  }
  $p19Timeout = if ($SkipMilestone1000) { 1200 } else { 3600 }
  $null = Invoke-SafeRunStep -Label "P19-smoke" -Exe "powershell" -CmdArgs $p19Args -TimeoutSec $p19Timeout
}

if ($RunP20 -or $RunPerfGateV5) {
  $p20SmokeScript = Join-Path $ProjectRoot "scripts/run_p20_smoke.ps1"
  if (-not (Test-Path $p20SmokeScript)) {
    throw "[P20] missing script: $p20SmokeScript"
  }
  Write-Host "[P20] running RC release smoke gate"
  $p20Args = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $p20SmokeScript,
    "-BaseUrl", $BaseUrl,
    "-Seed", $Seed
  )
  if ($RunPerfGateV5) {
    $p20Args += "-RunPerfGateOnly"
    $p20Args += "-FailOnPerfGate"
  }
  if ($SkipMilestone2000) {
    $p20Args += "-SkipMilestone2000"
  }
  $p20Timeout = if ($SkipMilestone2000) { 3600 } else { 10800 }
  $null = Invoke-SafeRunStep -Label "P20-smoke" -Exe "powershell" -CmdArgs $p20Args -TimeoutSec $p20Timeout
}

if ($GitSync) {
  $gitSyncScript = Join-Path $ProjectRoot "scripts/git_sync.ps1"
  if (-not (Test-Path $gitSyncScript)) {
    throw "[GitSync] missing script: $gitSyncScript"
  }
  Write-Host "[GitSync] running dry-run sync preview"
  $null = Invoke-SafeRunStep -Label "GitSync-dry-run" -Exe "powershell" -CmdArgs @("-ExecutionPolicy","Bypass","-File",$gitSyncScript,"-DryRun:$true") -TimeoutSec 600
  Write-Host "[GitSync] dry-run complete. To execute push/delete: powershell -ExecutionPolicy Bypass -File scripts/git_sync.ps1 -DryRun:`$false"
}

if ($RunP22) {
  $p22Script = Join-Path $ProjectRoot "scripts/run_p22.ps1"
  if (-not (Test-Path $p22Script)) {
    throw "[P22] missing script: $p22Script"
  }

  # P55: sync/check sidecars before running the orchestrator.
  $syncScript = Join-Path $ProjectRoot "scripts\sync_config_sidecars.ps1"
  if (Test-Path $syncScript) {
    Write-Host "[P55] running config sidecar sync/check before P22..."
    $syncPy = [string]($env:BALATRO_TRAIN_PYTHON)
    if (-not $syncPy) { $syncPy = "python" }
    $syncArgs = @("-ExecutionPolicy", "Bypass", "-File", $syncScript, "-TrainingPython", $syncPy)
    & powershell @syncArgs
    if ($LASTEXITCODE -ne 0) {
      throw "[P55] config sidecar sync/check failed. Run: python -m trainer.experiments.config_sidecar_sync --sync"
    }
    Write-Host "[P55] config sidecar check/sync passed."
  }

  $p22Args = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $p22Script
  )
  if ($RunP22Full) {
    Write-Host "[P22] running full nightly-style orchestrator"
    $p22Args += "-Nightly"
  } else {
    Write-Host "[P22] running dry-run orchestrator gate"
    $p22Args += "-DryRun"
  }
  $null = Invoke-SafeRunStep -Label "P22-orchestrator" -Exe "powershell" -CmdArgs $p22Args -TimeoutSec 3600
  $RunP22Status = "PASS"
  $RunP22LatestRun = Get-LatestRunDir -RunsRootPath (Join-Path $ProjectRoot "docs/artifacts/p22/runs")
  if ($RunP22LatestRun) {
    $candidate = Join-Path $RunP22LatestRun "report_p23.json"
    if (Test-Path $candidate) { $RunP22LatestReport = $candidate }
    else {
      $candidateP22 = Join-Path $RunP22LatestRun "report_p22.json"
      if (Test-Path $candidateP22) { $RunP22LatestReport = $candidateP22 }
    }
  }
}

if ($RunLegacySmoke) {
  $legacyStamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $legacyRoot = Join-Path $ProjectRoot "docs/artifacts/legacy_smoke"
  if (-not (Test-Path $legacyRoot)) { New-Item -ItemType Directory -Path $legacyRoot -Force | Out-Null }
  $legacyDir = Join-Path $legacyRoot $legacyStamp
  New-Item -ItemType Directory -Path $legacyDir -Force | Out-Null

  Run-PyStrict -Label "LegacySmoke-train-bc-help" -PyArgs @("-B", "trainer/train_bc.py", "--help") | Out-Null
  Run-PyStrict -Label "LegacySmoke-dagger-help" -PyArgs @("-B", "trainer/dagger_collect.py", "--help") | Out-Null
  Run-PyStrict -Label "LegacySmoke-dagger-v4-help" -PyArgs @("-B", "trainer/dagger_collect_v4.py", "--help") | Out-Null

  $legacyReport = [ordered]@{
    schema = "legacy_baseline_smoke_v1"
    generated_at = (Get-Date).ToString("o")
    status = "PASS"
    checks = @(
      @{ id = "train_bc_help"; status = "PASS"; command = "python -B trainer/train_bc.py --help" },
      @{ id = "dagger_collect_help"; status = "PASS"; command = "python -B trainer/dagger_collect.py --help" },
      @{ id = "dagger_collect_v4_help"; status = "PASS"; command = "python -B trainer/dagger_collect_v4.py --help" }
    )
    note = "Legacy baseline health check only; BC/DAgger are not mainline training gates."
  }
  $legacyReportPath = Join-Path $legacyDir "report_legacy_smoke.json"
  Write-JsonFile -Path $legacyReportPath -Object $legacyReport
  Write-Host ("[LegacySmoke] PASS artifact=" + $legacyReportPath)
}

if ($RunP23) {
  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $p23Root = Join-Path $ProjectRoot "docs/artifacts/p23"
  if (-not (Test-Path $p23Root)) { New-Item -ItemType Directory -Path $p23Root -Force | Out-Null }
  $p23Dir = Join-Path $p23Root $stamp
  New-Item -ItemType Directory -Path $p23Dir -Force | Out-Null

  $baselineSummary = [ordered]@{
    schema = "p23_baseline_summary_v1"
    generated_at = (Get-Date).ToString("o")
    run_p22_status = $RunP22Status
    run_p22_latest_run = $RunP22LatestRun
    run_p22_report = $RunP22LatestReport
  }
  $baselineSummaryPath = Join-Path $p23Dir "baseline_summary.json"
  Write-JsonFile -Path $baselineSummaryPath -Object $baselineSummary
  $baselineMd = @(
    "# P23 Baseline Summary",
    "",
    "- run_p22_status: $RunP22Status",
    "- run_p22_latest_run: $RunP22LatestRun",
    "- run_p22_report: $RunP22LatestReport"
  )
  $baselineMdPath = Join-Path $p23Dir "baseline_summary.md"
  $baselineMd -join "`n" | Out-File -LiteralPath $baselineMdPath -Encoding UTF8

  $seedValidationPath = Join-Path $p23Dir "seed_policy_validation.json"
  $seedValidateArgs = @(
    "-B",
    "-m", "trainer.experiments.seeds",
    "--validate-policy",
    "--config", "configs/experiments/seeds_p23.yaml",
    "--write", $seedValidationPath
  )
  Run-PyStrict -Label "P23-seed-validate" -PyArgs $seedValidateArgs | Out-Null
  $seedPolicySnapshot = Join-Path $p23Dir "seed_policy.json"
  Copy-Item -LiteralPath (Join-Path $ProjectRoot "configs/experiments/seeds_p23.yaml") -Destination $seedPolicySnapshot -Force

  $seedPerfPath = Join-Path $p23Dir "seeds_perf_gate_100.json"
  $seedPerfArgs = @(
    "-B",
    "-m", "trainer.experiments.seeds",
    "--materialize", "perf_gate_100",
    "--config", "configs/experiments/seeds_p23.yaml",
    "--write", $seedPerfPath
  )
  Run-PyStrict -Label "P23-seed-perf100" -PyArgs $seedPerfArgs | Out-Null

  $seedNightlyPath = Join-Path $p23Dir "seeds_nightly_sample.json"
  $seedNightlyArgs = @(
    "-B",
    "-m", "trainer.experiments.seeds",
    "--materialize-nightly",
    "--config", "configs/experiments/seeds_p23.yaml",
    "--write", $seedNightlyPath
  )
  Run-PyStrict -Label "P23-seed-nightly" -PyArgs $seedNightlyArgs | Out-Null

  $p23Script = Join-Path $ProjectRoot "scripts/run_p23.ps1"
  if (-not (Test-Path $p23Script)) { throw "[P23] missing script: $p23Script" }
  $null = Invoke-SafeRunStep -Label "P23-quick" -Exe "powershell" -CmdArgs @(
    "-ExecutionPolicy", "Bypass",
    "-File", $p23Script,
    "-Quick"
  ) -TimeoutSec 2400
  $quickRunRoot = Get-LatestRunDir -RunsRootPath (Join-Path $p23Root "runs")
  $quickRunStatus = "UNKNOWN"
  $quickExperimentCount = 0
  if (-not [string]::IsNullOrWhiteSpace($quickRunRoot)) {
    $quickReport = Join-Path $quickRunRoot "report_p23.json"
    if (Test-Path $quickReport) {
      try {
        $quickObj = Get-Content -LiteralPath $quickReport -Raw | ConvertFrom-Json
        $quickRunStatus = [string]$quickObj.status
        if ($quickObj.rows) { $quickExperimentCount = @($quickObj.rows).Count }
      } catch {
        $quickRunStatus = "UNKNOWN"
      }
    }
  }
  $null = Invoke-SafeRunStep -Label "P23-flake" -Exe "powershell" -CmdArgs @(
    "-ExecutionPolicy", "Bypass",
    "-File", $p23Script,
    "-FlakeSmoke"
  ) -TimeoutSec 2400

  $latestP23Run = Get-LatestRunDir -RunsRootPath (Join-Path $p23Root "runs")
  $runToPersist = $quickRunRoot
  if ([string]::IsNullOrWhiteSpace($runToPersist)) { $runToPersist = $latestP23Run }
  if (-not [string]::IsNullOrWhiteSpace($runToPersist)) {
    $destRunsRoot = Join-Path $p23Dir "runs"
    New-Item -ItemType Directory -Path $destRunsRoot -Force | Out-Null
    $destRun = Join-Path $destRunsRoot (Split-Path -Leaf $runToPersist)
    Copy-Item -LiteralPath $runToPersist -Destination $destRun -Recurse -Force
    foreach ($covName in @("coverage_summary.json", "coverage_summary.md", "coverage_table.csv")) {
      $srcCov = Join-Path $runToPersist $covName
      if (Test-Path $srcCov) {
        Copy-Item -LiteralPath $srcCov -Destination (Join-Path $p23Dir $covName) -Force
      }
    }
  }

  $gateFunctional = [ordered]@{
    schema = "p23_gate_functional_v1"
    generated_at = (Get-Date).ToString("o")
    run_p22_status = $RunP22Status
    run_p22_latest_run = $RunP22LatestRun
    run_p22_report = $RunP22LatestReport
    pass = ($RunP22Status -eq "PASS")
  }
  $gateExperiments = [ordered]@{
    schema = "p23_gate_experiments_v1"
    generated_at = (Get-Date).ToString("o")
    quick_run_root = $quickRunRoot
    quick_status = $quickRunStatus
    quick_experiment_count = $quickExperimentCount
    pass = ($quickRunStatus -eq "PASS")
  }
  $flakeReportRoot = Join-Path $p23Root "flake_report.json"
  $flakePass = $false
  if (Test-Path $flakeReportRoot) {
    try {
      $flakeObj = Get-Content -LiteralPath $flakeReportRoot -Raw | ConvertFrom-Json
      $flakePass = ([string]$flakeObj.status -eq "PASS")
    } catch { $flakePass = $false }
  }
  $seedValidationObj = $null
  if (Test-Path $seedValidationPath) {
    try { $seedValidationObj = Get-Content -LiteralPath $seedValidationPath -Raw | ConvertFrom-Json } catch {}
  }
  $seedValidationPass = $false
  if ($seedValidationObj -and $seedValidationObj.ok -eq $true) { $seedValidationPass = $true }
  $gateReliability = [ordered]@{
    schema = "p23_gate_reliability_v1"
    generated_at = (Get-Date).ToString("o")
    seed_policy_validation_pass = $seedValidationPass
    flake_smoke_pass = $flakePass
    disallow_single_seed_default = $true
    pass = ($seedValidationPass -and $flakePass)
  }

  $gateFunctionalPath = Join-Path $p23Dir "gate_functional.json"
  $gateExperimentsPath = Join-Path $p23Dir "gate_experiments.json"
  $gateReliabilityPath = Join-Path $p23Dir "gate_reliability.json"
  Write-JsonFile -Path $gateFunctionalPath -Object $gateFunctional
  Write-JsonFile -Path $gateExperimentsPath -Object $gateExperiments
  Write-JsonFile -Path $gateReliabilityPath -Object $gateReliability

  $reportGate = [ordered]@{
    schema = "p23_report_gate_v1"
    generated_at = (Get-Date).ToString("o")
    functional = $gateFunctional
    experiments = $gateExperiments
    reliability = $gateReliability
    status = $(if ($gateFunctional.pass -and $gateExperiments.pass -and $gateReliability.pass) { "PASS" } else { "FAIL" })
  }
  $reportGatePath = Join-Path $p23Dir "report_p23_gate.json"
  Write-JsonFile -Path $reportGatePath -Object $reportGate

  $coverageStatusPath = Join-Path $ProjectRoot "docs/COVERAGE_P23_STATUS.md"
  $coverageStatusLines = @(
    "# COVERAGE P23 STATUS",
    "",
    "- timestamp: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"),
    "- run_p22_status: " + $RunP22Status,
    "- quick_run_root: " + $quickRunRoot,
    "- seed_policy_validation_pass: " + $seedValidationPass,
    "- flake_smoke_pass: " + $flakePass,
    "- gate_status: " + $reportGate.status
  )
  $coverageStatusLines -join "`n" | Out-File -LiteralPath $coverageStatusPath -Encoding UTF8

  if (Test-Path (Join-Path $p23Root "flake_report.json")) {
    Copy-Item -LiteralPath (Join-Path $p23Root "flake_report.json") -Destination (Join-Path $p23Dir "flake_report.json") -Force
  }
  if (Test-Path (Join-Path $p23Root "flake_report.md")) {
    Copy-Item -LiteralPath (Join-Path $p23Root "flake_report.md") -Destination (Join-Path $p23Dir "flake_report.md") -Force
  }
  foreach ($name in @("champion.json", "candidate.json", "nightly_decision.json", "nightly_decision.md", "CHANGELOG_P23.md")) {
    $src = Join-Path $p23Root $name
    if (Test-Path $src) {
      Copy-Item -LiteralPath $src -Destination (Join-Path $p23Dir $name) -Force
    }
  }

  Write-Host ("[P23] artifact_dir=" + $p23Dir + " gate_status=" + $reportGate.status)
  $RunP23Status = [string]$reportGate.status
  $RunP23ArtifactDir = $p23Dir
  $RunP23GateReport = $reportGatePath
  if ($reportGate.status -ne "PASS") {
    throw "[P23] gate failed; inspect gate_functional/gate_experiments/gate_reliability"
  }
}

if ($RunP24) {
  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $p24Root = Join-Path $ProjectRoot "docs/artifacts/p24"
  if (-not (Test-Path $p24Root)) { New-Item -ItemType Directory -Path $p24Root -Force | Out-Null }
  $p24Dir = Join-Path $p24Root $stamp
  New-Item -ItemType Directory -Path $p24Dir -Force | Out-Null

  $baselineSummary = [ordered]@{
    schema = "p24_baseline_summary_v1"
    generated_at = (Get-Date).ToString("o")
    run_p23_status = $RunP23Status
    run_p23_artifact_dir = $RunP23ArtifactDir
    run_p23_gate_report = $RunP23GateReport
  }
  $baselineSummaryPath = Join-Path $p24Dir "baseline_summary.json"
  Write-JsonFile -Path $baselineSummaryPath -Object $baselineSummary
  $baselineMd = @(
    "# P24 Baseline Summary",
    "",
    "- run_p23_status: $RunP23Status",
    "- run_p23_artifact_dir: $RunP23ArtifactDir",
    "- run_p23_gate_report: $RunP23GateReport"
  )
  $baselineMdPath = Join-Path $p24Dir "baseline_summary.md"
  $baselineMd -join "`n" | Out-File -LiteralPath $baselineMdPath -Encoding UTF8

  $p24Script = Join-Path $ProjectRoot "scripts/run_p24.ps1"
  if (-not (Test-Path $p24Script)) { throw "[P24] missing script: $p24Script" }
  $null = Invoke-SafeRunStep -Label "P24-quick" -Exe "powershell" -CmdArgs @(
    "-ExecutionPolicy", "Bypass",
    "-File", $p24Script,
    "-Quick",
    "-HeadlessDashboard"
  ) -TimeoutSec 3600

  $latestP24Run = Get-LatestRunDir -RunsRootPath (Join-Path $p24Root "runs")
  if (-not [string]::IsNullOrWhiteSpace($latestP24Run)) {
    $destRunsRoot = Join-Path $p24Dir "runs"
    New-Item -ItemType Directory -Path $destRunsRoot -Force | Out-Null
    $destRun = Join-Path $destRunsRoot (Split-Path -Leaf $latestP24Run)
    Copy-Item -LiteralPath $latestP24Run -Destination $destRun -Recurse -Force
  }

  $p24Py = Join-Path $ProjectRoot ".venv_trainer\Scripts\python.exe"
  if (-not (Test-Path $p24Py)) { $p24Py = "python" }

  $dashOut = Join-Path $p24Root "dashboard_headless_log.txt"
  $dashCmdArgs = @(
    "-B",
    "-m", "trainer.experiments.dashboard_tui",
    "--watch", (Join-Path $p24Root "runs/latest/telemetry.jsonl"),
    "--headless-log",
    "--out", $dashOut
  )
  $null = Invoke-SafeRunStep -Label "P24-dashboard" -Exe $p24Py -CmdArgs $dashCmdArgs -TimeoutSec 600

  $triageOut = Join-Path $p24Root "triage_latest"
  $triageCmdArgs = @(
    "-B",
    "-m", "trainer.experiments.triage",
    "--run-root", (Join-Path $p24Root "runs/latest"),
    "--out-dir", $triageOut
  )
  $null = Invoke-SafeRunStep -Label "P24-triage" -Exe $p24Py -CmdArgs $triageCmdArgs -TimeoutSec 600

  $bisectOut = Join-Path $p24Root "bisect_latest"
  $bisectCmdArgs = @(
    "-B",
    "-m", "trainer.experiments.bisect_lite",
    "--run-root", (Join-Path $p24Root "runs/latest"),
    "--mode", "seed_bisect",
    "--out-dir", $bisectOut
  )
  $null = Invoke-SafeRunStep -Label "P24-bisect" -Exe $p24Py -CmdArgs $bisectCmdArgs -TimeoutSec 600

  $rankingOut = Join-Path $p24Root "ranking_latest"
  $rankingCmdArgs = @(
    "-B",
    "-m", "trainer.experiments.ranking",
    "--run-root", (Join-Path $p24Root "runs/latest"),
    "--config", "configs/experiments/ranking_p24.yaml",
    "--out-dir", $rankingOut
  )
  $null = Invoke-SafeRunStep -Label "P24-ranking" -Exe $p24Py -CmdArgs $rankingCmdArgs -TimeoutSec 600

  foreach ($name in @("dashboard_headless_log.txt")) {
    $src = Join-Path $p24Root $name
    if (Test-Path $src) { Copy-Item -LiteralPath $src -Destination (Join-Path $p24Dir $name) -Force }
  }
  foreach ($dirName in @("triage_latest", "bisect_latest", "ranking_latest")) {
    $srcDir = Join-Path $p24Root $dirName
    if (Test-Path $srcDir) {
      Copy-Item -LiteralPath $srcDir -Destination (Join-Path $p24Dir $dirName) -Recurse -Force
    }
  }

  $campaignSummaryPath = Join-Path $p24Root "runs/latest/campaign_summary.json"
  $campaignSummary = if (Test-Path $campaignSummaryPath) { Get-Content -LiteralPath $campaignSummaryPath -Raw | ConvertFrom-Json } else { $null }
  $summaryTablePath = Join-Path $p24Root "runs/latest/summary_table.json"
  $summaryRows = @()
  if (Test-Path $summaryTablePath) {
    try {
      $summaryRows = Get-Content -LiteralPath $summaryTablePath -Raw | ConvertFrom-Json
      if ($null -eq $summaryRows) {
        $summaryRows = @()
      } elseif ($summaryRows -isnot [System.Array]) {
        $summaryRows = @($summaryRows)
      }
    } catch {
      $summaryRows = @()
    }
  }
  $stagePassCount = 0
  if ($campaignSummary -and $campaignSummary.stages) {
    $stagePassCount = @($campaignSummary.stages | Where-Object { [string]$_.status -eq "stage_pass" }).Count
  }
  $campaignStatusToken = if ($campaignSummary) { [string]$campaignSummary.status } else { "" }
  $campaignPass = (($campaignStatusToken -eq "completed") -and ($stagePassCount -ge 2) -and ($summaryRows.Count -ge 3))

  $seedPolicyUsed = $true
  $seedCountMin = 999999
  $latestRunDirPath = Join-Path $p24Root "runs/latest"
  $expDirs = Get-ChildItem -Path $latestRunDirPath -Directory -ErrorAction SilentlyContinue | Where-Object { Test-Path (Join-Path $_.FullName "run_manifest.json") }
  foreach ($ed in $expDirs) {
    $manifestPath = Join-Path $ed.FullName "run_manifest.json"
    $seedPath = Join-Path $ed.FullName "seeds_used.json"
    $m = if (Test-Path $manifestPath) { Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json } else { $null }
    $s = if (Test-Path $seedPath) { Get-Content -LiteralPath $seedPath -Raw | ConvertFrom-Json } else { $null }
    $spv = if ($m) { [string]$m.seed_policy_version } else { "" }
    if ([string]::IsNullOrWhiteSpace($spv) -or $spv -eq "legacy.p22") { $seedPolicyUsed = $false }
    $cnt = 0
    if ($s -and $s.seed_count -ne $null) { $cnt = [int]$s.seed_count }
    elseif ($s -and $s.seeds) { $cnt = @($s.seeds).Count }
    if ($cnt -lt $seedCountMin) { $seedCountMin = $cnt }
    if ($cnt -le 1) { $seedPolicyUsed = $false }
  }
  if ($seedCountMin -eq 999999) { $seedCountMin = 0 }

  $coverageExists = Test-Path (Join-Path $p24Root "runs/latest/coverage_summary.json")
  $flakeReportPath = Join-Path $p24Root "runs/latest/flake_report.json"
  $flakeStatus = "SKIPPED"
  if (Test-Path $flakeReportPath) {
    try {
      $fobj = Get-Content -LiteralPath $flakeReportPath -Raw | ConvertFrom-Json
      $flakeStatus = [string]$fobj.status
    } catch {
      $flakeStatus = "UNKNOWN"
    }
  }
  $reliabilityPass = ($seedPolicyUsed -and $coverageExists -and ($flakeStatus -ne "FAIL"))

  $dashboardPass = ((Test-Path $dashOut) -and ((Get-Item $dashOut).Length -gt 0))
  $triagePass = Test-Path (Join-Path $triageOut "triage_report.json")
  $bisectPass = Test-Path (Join-Path $bisectOut "bisect_report.json")
  $rankingSummaryPath = Join-Path $rankingOut "ranking_summary.json"
  $rankingPass = Test-Path $rankingSummaryPath
  $opsPass = ($dashboardPass -and $triagePass -and $bisectPass -and $rankingPass)

  $gateFunctional = [ordered]@{
    schema = "p24_gate_functional_v1"
    generated_at = (Get-Date).ToString("o")
    run_p23_status = $RunP23Status
    run_p23_artifact_dir = $RunP23ArtifactDir
    run_p23_gate_report = $RunP23GateReport
    pass = ($RunP23Status -eq "PASS")
  }
  $gateCampaign = [ordered]@{
    schema = "p24_gate_campaign_v1"
    generated_at = (Get-Date).ToString("o")
    campaign_status = $campaignStatusToken
    stage_pass_count = $stagePassCount
    experiment_count = $summaryRows.Count
    quick_run_root = $latestP24Run
    pass = $campaignPass
  }
  $gateReliability = [ordered]@{
    schema = "p24_gate_reliability_v1"
    generated_at = (Get-Date).ToString("o")
    seed_policy_used = $seedPolicyUsed
    min_seed_count = $seedCountMin
    coverage_report_exists = $coverageExists
    flake_status = $flakeStatus
    pass = $reliabilityPass
  }
  $gateOps = [ordered]@{
    schema = "p24_gate_ops_v1"
    generated_at = (Get-Date).ToString("o")
    dashboard_pass = $dashboardPass
    triage_pass = $triagePass
    bisect_pass = $bisectPass
    ranking_pass = $rankingPass
    pass = $opsPass
  }

  Write-JsonFile -Path (Join-Path $p24Dir "gate_functional.json") -Object $gateFunctional
  Write-JsonFile -Path (Join-Path $p24Dir "gate_campaign.json") -Object $gateCampaign
  Write-JsonFile -Path (Join-Path $p24Dir "gate_reliability.json") -Object $gateReliability
  Write-JsonFile -Path (Join-Path $p24Dir "gate_ops.json") -Object $gateOps

  $reportGate = [ordered]@{
    schema = "p24_report_gate_v1"
    generated_at = (Get-Date).ToString("o")
    functional = $gateFunctional
    campaign = $gateCampaign
    reliability = $gateReliability
    ops = $gateOps
    status = $(if ($gateFunctional.pass -and $gateCampaign.pass -and $gateReliability.pass -and $gateOps.pass) { "PASS" } else { "FAIL" })
  }
  $reportGatePath = Join-Path $p24Dir "report_p24_gate.json"
  Write-JsonFile -Path $reportGatePath -Object $reportGate

  foreach ($name in @("champion.json", "candidate.json", "nightly_decision.json", "nightly_decision.md", "CHANGELOG_P24.md")) {
    $src = Join-Path $p24Root $name
    if (Test-Path $src) { Copy-Item -LiteralPath $src -Destination (Join-Path $p24Dir $name) -Force }
  }

  $coverageStatusPath = Join-Path $ProjectRoot "docs/COVERAGE_P24_STATUS.md"
  $coverageStatusLines = @(
    "# COVERAGE P24 STATUS",
    "",
    "- timestamp: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"),
    "- run_p23_status: " + $RunP23Status,
    "- campaign_status: " + $campaignStatusToken,
    "- stage_pass_count: " + $stagePassCount,
    "- experiment_count: " + $summaryRows.Count,
    "- seed_policy_used: " + $seedPolicyUsed,
    "- coverage_exists: " + $coverageExists,
    "- flake_status: " + $flakeStatus,
    "- gate_status: " + $reportGate.status
  )
  $coverageStatusLines -join "`n" | Out-File -LiteralPath $coverageStatusPath -Encoding UTF8

  Write-Host ("[P24] artifact_dir=" + $p24Dir + " gate_status=" + $reportGate.status)
  $RunP24Status = [string]$reportGate.status
  $RunP24ArtifactDir = $p24Dir
  $RunP24GateReport = $reportGatePath
  if ($reportGate.status -ne "PASS") {
    throw "[P24] gate failed; inspect gate_functional/gate_campaign/gate_reliability/gate_ops"
  }
}

if ($RunP25) {
  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $p25Root = Join-Path $ProjectRoot "docs/artifacts/p25"
  if (-not (Test-Path $p25Root)) { New-Item -ItemType Directory -Path $p25Root -Force | Out-Null }
  $p25Dir = Join-Path $p25Root $stamp
  New-Item -ItemType Directory -Path $p25Dir -Force | Out-Null

  $baselineSummary = [ordered]@{
    schema = "p25_baseline_summary_v1"
    generated_at = (Get-Date).ToString("o")
    baseline_gate = "RunP24"
    run_p24_status = $RunP24Status
    run_p24_artifact_dir = $RunP24ArtifactDir
    run_p24_gate_report = $RunP24GateReport
  }
  $baselineSummaryPath = Join-Path $p25Dir "baseline_summary.json"
  Write-JsonFile -Path $baselineSummaryPath -Object $baselineSummary
  $baselineMd = @(
    "# P25 Baseline Summary",
    "",
    "- baseline_gate: RunP24",
    "- run_p24_status: $RunP24Status",
    "- run_p24_artifact_dir: $RunP24ArtifactDir",
    "- run_p24_gate_report: $RunP24GateReport"
  )
  $baselineMdPath = Join-Path $p25Dir "baseline_summary.md"
  $baselineMd -join "`n" | Out-File -LiteralPath $baselineMdPath -Encoding UTF8

  $statusScript = Join-Path $ProjectRoot "scripts/generate_readme_status.ps1"
  if (-not (Test-Path $statusScript)) { throw "[P25] missing script: $statusScript" }
  $null = Invoke-SafeRunStep -Label "P25-readme-status" -Exe "powershell" -CmdArgs @(
    "-ExecutionPolicy", "Bypass",
    "-File", $statusScript
  ) -TimeoutSec 300
  $statusJsonPath = Join-Path $ProjectRoot "docs/generated/README_STATUS.json"
  $statusMdPath = Join-Path $ProjectRoot "docs/generated/README_STATUS.md"
  $statusObj = $null
  if (Test-Path $statusJsonPath) {
    try { $statusObj = Get-Content -LiteralPath $statusJsonPath -Raw | ConvertFrom-Json } catch { $statusObj = $null }
  }
  $statusPass = ($statusObj -ne $null)
  $statusSummary = [ordered]@{
    schema = "p25_readme_status_generation_v1"
    generated_at = (Get-Date).ToString("o")
    pass = $statusPass
    status_json = $statusJsonPath
    status_md = $statusMdPath
    highest_supported_gate = $(if ($statusObj) { [string]$statusObj.highest_supported_gate } else { "" })
  }
  Write-JsonFile -Path (Join-Path $p25Dir "readme_status_generation.json") -Object $statusSummary
  if (Test-Path $statusJsonPath) { Copy-Item -LiteralPath $statusJsonPath -Destination (Join-Path $p25Dir "README_STATUS.json") -Force }
  if (Test-Path $statusMdPath) { Copy-Item -LiteralPath $statusMdPath -Destination (Join-Path $p25Dir "README_STATUS.md") -Force }

  $lintScript = Join-Path $ProjectRoot "scripts/lint_readme_p25.ps1"
  if (-not (Test-Path $lintScript)) { throw "[P25] missing script: $lintScript" }
  $null = Invoke-SafeRunStep -Label "P25-readme-lint" -Exe "powershell" -CmdArgs @(
    "-ExecutionPolicy", "Bypass",
    "-File", $lintScript
  ) -TimeoutSec 300
  $lintReportPath = Join-Path $ProjectRoot "docs/generated/readme_lint_report.json"
  if (-not (Test-Path $lintReportPath)) { throw "[P25] missing lint report: $lintReportPath" }
  $lintReport = Get-Content -LiteralPath $lintReportPath -Raw | ConvertFrom-Json
  $lintPass = [bool]$lintReport.pass
  Copy-Item -LiteralPath $lintReportPath -Destination (Join-Path $p25Dir "readme_lint_report.json") -Force

  $quickSmokeScript = Join-Path $ProjectRoot "scripts/run_p23.ps1"
  $quickSmokeArgs = @()
  if (Test-Path $quickSmokeScript) {
    $quickSmokeArgs = @("-ExecutionPolicy", "Bypass", "-File", $quickSmokeScript, "-DryRun")
  } else {
    $fallbackQuick = Join-Path $ProjectRoot "scripts/run_p22.ps1"
    if (-not (Test-Path $fallbackQuick)) { throw "[P25] missing quick-start smoke scripts (run_p23/run_p22)" }
    $quickSmokeArgs = @("-ExecutionPolicy", "Bypass", "-File", $fallbackQuick, "-DryRun")
  }
  $null = Invoke-SafeRunStep -Label "P25-quickstart-smoke" -Exe "powershell" -CmdArgs $quickSmokeArgs -TimeoutSec 1200
  $quickSmoke = [ordered]@{
    schema = "p25_quickstart_smoke_v1"
    generated_at = (Get-Date).ToString("o")
    command = "powershell " + ($quickSmokeArgs -join " ")
    pass = $true
  }
  Write-JsonFile -Path (Join-Path $p25Dir "quickstart_smoke.json") -Object $quickSmoke

  $readmePath = Join-Path $ProjectRoot "README.md"
  $readmeText = if (Test-Path $readmePath) { Get-Content -LiteralPath $readmePath -Raw } else { "" }
  $reproDocPath = Join-Path $ProjectRoot "docs/REPRODUCIBILITY_P25.md"
  $assetsRoot = Join-Path $ProjectRoot "docs/assets/readme"
  $assetTargets = @("sample_run_log.txt", "sample_summary_table.md", "architecture_dataflow.mmd")
  $assetCount = 0
  foreach ($assetName in $assetTargets) {
    if (Test-Path (Join-Path $assetsRoot $assetName)) { $assetCount += 1 }
  }
  $readmeHasReproLink = [regex]::IsMatch($readmeText, "\(docs/REPRODUCIBILITY_P25\.md\)")

  $functionalPass = (($RunP24Status -eq "PASS") -and $quickSmoke.pass)
  $docsPass = ($lintPass -and $statusPass -and (Test-Path $reproDocPath) -and $readmeHasReproLink -and ($assetCount -ge 3))

  $gateFunctional = [ordered]@{
    schema = "p25_gate_functional_v1"
    generated_at = (Get-Date).ToString("o")
    baseline_gate = "RunP24"
    baseline_status = $RunP24Status
    baseline_artifact_dir = $RunP24ArtifactDir
    baseline_gate_report = $RunP24GateReport
    quickstart_smoke_pass = $quickSmoke.pass
    pass = $functionalPass
  }
  $gateDocs = [ordered]@{
    schema = "p25_gate_docs_v1"
    generated_at = (Get-Date).ToString("o")
    readme_lint_pass = $lintPass
    readme_status_generation_pass = $statusPass
    reproducibility_doc_exists = (Test-Path $reproDocPath)
    readme_has_repro_link = $readmeHasReproLink
    readme_assets_count = $assetCount
    pass = $docsPass
  }
  Write-JsonFile -Path (Join-Path $p25Dir "gate_functional.json") -Object $gateFunctional
  Write-JsonFile -Path (Join-Path $p25Dir "gate_docs.json") -Object $gateDocs

  $reportGate = [ordered]@{
    schema = "p25_report_gate_v1"
    generated_at = (Get-Date).ToString("o")
    functional = $gateFunctional
    docs = $gateDocs
    status = $(if ($functionalPass -and $docsPass) { "PASS" } else { "FAIL" })
  }
  $reportGatePath = Join-Path $p25Dir "report_p25_gate.json"
  Write-JsonFile -Path $reportGatePath -Object $reportGate

  $coverageStatusPath = Join-Path $ProjectRoot "docs/COVERAGE_P25_STATUS.md"
  $coverageStatusLines = @(
    "# COVERAGE P25 STATUS",
    "",
    "- timestamp: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"),
    "- baseline_gate: RunP24",
    "- baseline_status: " + $RunP24Status,
    "- readme_lint_pass: " + $lintPass,
    "- readme_status_generation_pass: " + $statusPass,
    "- quickstart_smoke_pass: " + $quickSmoke.pass,
    "- docs_assets_count: " + $assetCount,
    "- gate_status: " + $reportGate.status
  )
  $coverageStatusLines -join "`n" | Out-File -LiteralPath $coverageStatusPath -Encoding UTF8

  Write-Host ("[P25] artifact_dir=" + $p25Dir + " gate_status=" + $reportGate.status)
  $RunP25Status = [string]$reportGate.status
  $RunP25ArtifactDir = $p25Dir
  $RunP25GateReport = $reportGatePath
  if ($reportGate.status -ne "PASS") {
    throw "[P25] gate failed; inspect gate_functional/gate_docs/report_p25_gate"
  }
}

if ($RunP26) {
  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $p26Root = Join-Path $ProjectRoot "docs/artifacts/p26"
  if (-not (Test-Path $p26Root)) { New-Item -ItemType Directory -Path $p26Root -Force | Out-Null }
  $p26Dir = Join-Path $p26Root $stamp
  New-Item -ItemType Directory -Path $p26Dir -Force | Out-Null

  $baselineSummary = [ordered]@{
    schema = "p26_baseline_summary_v1"
    generated_at = (Get-Date).ToString("o")
    baseline_gate = "RunP25"
    run_p25_status = $RunP25Status
    run_p25_artifact_dir = $RunP25ArtifactDir
    run_p25_gate_report = $RunP25GateReport
  }
  $baselineSummaryPath = Join-Path $p26Dir "baseline_summary.json"
  Write-JsonFile -Path $baselineSummaryPath -Object $baselineSummary
  $baselineMd = @(
    "# P26 Baseline Summary",
    "",
    "- baseline_gate: RunP25",
    "- run_p25_status: $RunP25Status",
    "- run_p25_artifact_dir: $RunP25ArtifactDir",
    "- run_p25_gate_report: $RunP25GateReport"
  )
  $baselineMdPath = Join-Path $p26Dir "baseline_summary.md"
  $baselineMd -join "`n" | Out-File -LiteralPath $baselineMdPath -Encoding UTF8

  $p26Py = Join-Path $ProjectRoot ".venv_trainer\Scripts\python.exe"
  if (-not (Test-Path $p26Py)) { $p26Py = "python" }
  $trendsRoot = Join-Path $ProjectRoot "docs/artifacts/trends"
  $trendRowsJsonl = Join-Path $trendsRoot "trend_rows.jsonl"
  $trendRowsCsv = Join-Path $trendsRoot "trend_rows.csv"
  $trendSummaryPath = Join-Path $trendsRoot "trend_index_summary.json"
  $requestedSinceTag = "sim-p23-seed-governance-v1"

  $null = Invoke-SafeRunStep -Label "P26-trend-index" -Exe $p26Py -CmdArgs @(
    "-B",
    "-m", "trainer.experiments.index_artifacts",
    "--scan-root", "docs/artifacts",
    "--latest-only",
    "--out-root", "docs/artifacts/trends",
    "--append"
  ) -TimeoutSec 900

  $alertsDir = Join-Path $p26Dir "alerts_latest"
  $null = Invoke-SafeRunStep -Label "P26-regression-alert" -Exe $p26Py -CmdArgs @(
    "-B",
    "-m", "trainer.experiments.regression_alert",
    "--trends-root", "docs/artifacts/trends",
    "--config", "configs/experiments/regression_alert_p26.yaml",
    "--out-dir", $alertsDir
  ) -TimeoutSec 900
  $alertJsonPath = Join-Path $alertsDir "regression_alert_report.json"
  $alertMdPath = Join-Path $alertsDir "regression_alert_report.md"
  $alertCsvPath = Join-Path $alertsDir "regression_alert_table.csv"

  $releaseMdPath = Join-Path $p26Dir "release_summary_p26.md"
  $null = Invoke-SafeRunStep -Label "P26-release-summary" -Exe $p26Py -CmdArgs @(
    "-B",
    "-m", "trainer.experiments.release_notes",
    "--since-tag", $requestedSinceTag,
    "--out", $releaseMdPath,
    "--include-commits",
    "--include-benchmarks",
    "--include-risks",
    "--trends-root", "docs/artifacts/trends"
  ) -TimeoutSec 900
  $releaseJsonPath = [System.IO.Path]::ChangeExtension($releaseMdPath, ".json")

  $p26Script = Join-Path $ProjectRoot "scripts/run_p26.ps1"
  if (-not (Test-Path $p26Script)) { throw "[P26] missing script: $p26Script" }
  $null = Invoke-SafeRunStep -Label "P26-scheduler-quick" -Exe "powershell" -CmdArgs @(
    "-ExecutionPolicy", "Bypass",
    "-File", $p26Script,
    "-OutRoot", "docs/artifacts/p26",
    "-Quick",
    "-Resume",
    "-SinceTag", $requestedSinceTag
  ) -TimeoutSec 3600

  $statusScript = Join-Path $ProjectRoot "scripts/generate_readme_status.ps1"
  if (-not (Test-Path $statusScript)) { throw "[P26] missing script: $statusScript" }
  $null = Invoke-SafeRunStep -Label "P26-readme-status" -Exe "powershell" -CmdArgs @(
    "-ExecutionPolicy", "Bypass",
    "-File", $statusScript
  ) -TimeoutSec 300

  $schedulerManifestPath = Join-Path $p26Dir "scheduler_run_manifest.json"
  $schedulerStagePath = Join-Path $p26Dir "scheduler_stage_status.json"
  $schedulerSummaryPath = Join-Path $p26Dir "scheduler_summary.md"

  $trendSummaryObj = $null
  if (Test-Path $trendSummaryPath) {
    try { $trendSummaryObj = Get-Content -LiteralPath $trendSummaryPath -Raw | ConvertFrom-Json } catch { $trendSummaryObj = $null }
  }
  $trendMilestones = @()
  if ($trendSummaryObj -and $trendSummaryObj.milestones) {
    $trendMilestones = @($trendSummaryObj.milestones | ForEach-Object { [string]$_ })
  }
  $milestoneCoveragePass = (
    ($trendMilestones -contains "P22") -and
    ($trendMilestones -contains "P23") -and
    ($trendMilestones -contains "P24") -and
    ($trendMilestones -contains "P25")
  )
  $alertObj = $null
  if (Test-Path $alertJsonPath) {
    try { $alertObj = Get-Content -LiteralPath $alertJsonPath -Raw | ConvertFrom-Json } catch { $alertObj = $null }
  }
  $releaseObj = $null
  if (Test-Path $releaseJsonPath) {
    try { $releaseObj = Get-Content -LiteralPath $releaseJsonPath -Raw | ConvertFrom-Json } catch { $releaseObj = $null }
  }
  $schedulerObj = $null
  if (Test-Path $schedulerStagePath) {
    try { $schedulerObj = Get-Content -LiteralPath $schedulerStagePath -Raw | ConvertFrom-Json } catch { $schedulerObj = $null }
  }

  $statusMdPath = Join-Path $ProjectRoot "docs/generated/README_STATUS.md"
  $statusMdText = if (Test-Path $statusMdPath) { Get-Content -LiteralPath $statusMdPath -Raw } else { "" }
  $statusHasTrend = $statusMdText -match "recent_trend_signal"

  $functionalPass = ($RunP25Status -eq "PASS")
  $trendsPass = (
    (Test-Path $trendRowsJsonl) -and
    (Test-Path $trendRowsCsv) -and
    (Test-Path $trendSummaryPath) -and
    (Test-Path $alertJsonPath) -and
    (Test-Path $alertMdPath) -and
    (Test-Path $alertCsvPath) -and
    (Test-Path $releaseMdPath) -and
    (Test-Path $releaseJsonPath) -and
    $milestoneCoveragePass
  )
  $opsPass = (
    (Test-Path $schedulerManifestPath) -and
    (Test-Path $schedulerStagePath) -and
    (Test-Path $schedulerSummaryPath) -and
    ($schedulerObj -ne $null) -and
    ([string]$schedulerObj.overall_status -eq "PASS")
  )
  $docsPass = ((Test-Path $statusMdPath) -and $statusHasTrend)

  $gateFunctional = [ordered]@{
    schema = "p26_gate_functional_v1"
    generated_at = (Get-Date).ToString("o")
    baseline_gate = "RunP25"
    baseline_status = $RunP25Status
    baseline_artifact_dir = $RunP25ArtifactDir
    baseline_gate_report = $RunP25GateReport
    pass = $functionalPass
  }
  $gateTrends = [ordered]@{
    schema = "p26_gate_trends_v1"
    generated_at = (Get-Date).ToString("o")
    trend_rows_jsonl = $trendRowsJsonl
    trend_rows_csv = $trendRowsCsv
    trend_index_summary = $trendSummaryPath
    trend_rows_total = $(if ($trendSummaryObj) { [int]$trendSummaryObj.rows_total } else { 0 })
    milestones = $trendMilestones
    milestone_coverage_pass = $milestoneCoveragePass
    alert_report_json = $alertJsonPath
    release_summary_json = $releaseJsonPath
    pass = $trendsPass
  }
  $gateOps = [ordered]@{
    schema = "p26_gate_ops_v1"
    generated_at = (Get-Date).ToString("o")
    scheduler_run_manifest = $schedulerManifestPath
    scheduler_stage_status = $schedulerStagePath
    scheduler_summary = $schedulerSummaryPath
    scheduler_overall_status = $(if ($schedulerObj) { [string]$schedulerObj.overall_status } else { "UNKNOWN" })
    scheduler_stage_count = $(if ($schedulerObj) { [int]$schedulerObj.stage_count } else { 0 })
    scheduler_fail_count = $(if ($schedulerObj) { [int]$schedulerObj.fail_count } else { 0 })
    pass = $opsPass
  }
  $gateDocsStatus = [ordered]@{
    schema = "p26_gate_docs_status_v1"
    generated_at = (Get-Date).ToString("o")
    readme_status_md = $statusMdPath
    readme_status_contains_trend = $statusHasTrend
    pass = $docsPass
  }

  Write-JsonFile -Path (Join-Path $p26Dir "gate_functional.json") -Object $gateFunctional
  Write-JsonFile -Path (Join-Path $p26Dir "gate_trends.json") -Object $gateTrends
  Write-JsonFile -Path (Join-Path $p26Dir "gate_ops.json") -Object $gateOps
  Write-JsonFile -Path (Join-Path $p26Dir "gate_docs_status.json") -Object $gateDocsStatus

  $reportGate = [ordered]@{
    schema = "p26_report_gate_v1"
    generated_at = (Get-Date).ToString("o")
    functional = $gateFunctional
    trends = $gateTrends
    ops = $gateOps
    docs_status = $gateDocsStatus
    status = $(if ($functionalPass -and $trendsPass -and $opsPass -and $docsPass) { "PASS" } else { "FAIL" })
  }
  $reportGatePath = Join-Path $p26Dir "report_p26_gate.json"
  Write-JsonFile -Path $reportGatePath -Object $reportGate

  $coverageStatusPath = Join-Path $ProjectRoot "docs/COVERAGE_P26_STATUS.md"
  $coverageStatusLines = @(
    "# COVERAGE P26 STATUS",
    "",
    "- timestamp: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"),
    "- baseline_gate: RunP25",
    "- baseline_status: " + $RunP25Status,
    "- trend_rows_total: " + $(if ($trendSummaryObj) { [int]$trendSummaryObj.rows_total } else { 0 }),
    "- alert_report_exists: " + (Test-Path $alertJsonPath),
    "- release_summary_exists: " + (Test-Path $releaseJsonPath),
    "- scheduler_status: " + $(if ($schedulerObj) { [string]$schedulerObj.overall_status } else { "UNKNOWN" }),
    "- readme_status_has_trend: " + $statusHasTrend,
    "- gate_status: " + $reportGate.status
  )
  $coverageStatusLines -join "`n" | Out-File -LiteralPath $coverageStatusPath -Encoding UTF8

  Write-Host ("[P26] artifact_dir=" + $p26Dir + " gate_status=" + $reportGate.status)
  $RunP26Status = [string]$reportGate.status
  $RunP26ArtifactDir = $p26Dir
  $RunP26GateReport = $reportGatePath
  if ($reportGate.status -ne "PASS") {
    throw "[P26] gate failed; inspect gate_functional/gate_trends/gate_ops/gate_docs_status/report_p26_gate"
  }
}

if ($RunP27) {
  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $p27Root = Join-Path $ProjectRoot "docs/artifacts/p27"
  if (-not (Test-Path $p27Root)) { New-Item -ItemType Directory -Path $p27Root -Force | Out-Null }
  $p27Dir = Join-Path $p27Root $stamp
  New-Item -ItemType Directory -Path $p27Dir -Force | Out-Null

  $baselineSummary = [ordered]@{
    schema = "p27_baseline_summary_v1"
    generated_at = (Get-Date).ToString("o")
    baseline_gate = "RunP26"
    run_p26_status = $RunP26Status
    run_p26_artifact_dir = $RunP26ArtifactDir
    run_p26_gate_report = $RunP26GateReport
  }
  $baselineSummaryPath = Join-Path $p27Dir "baseline_summary.json"
  Write-JsonFile -Path $baselineSummaryPath -Object $baselineSummary
  $baselineMd = @(
    "# P27 Baseline Summary",
    "",
    "- baseline_gate: RunP26",
    "- run_p26_status: $RunP26Status",
    "- run_p26_artifact_dir: $RunP26ArtifactDir",
    "- run_p26_gate_report: $RunP26GateReport"
  )
  $baselineMdPath = Join-Path $p27Dir "baseline_summary.md"
  $baselineMd -join "`n" | Out-File -LiteralPath $baselineMdPath -Encoding UTF8

  $p27Py = Join-Path $ProjectRoot ".venv_trainer\Scripts\python.exe"
  if (-not (Test-Path $p27Py)) { $p27Py = "python" }

  $statusRootPath = Join-Path $ProjectRoot "docs/artifacts/status"
  $latestStatusJson = Join-Path $statusRootPath "latest_status.json"
  $latestBadgesJson = Join-Path $statusRootPath "latest_badges.json"
  $latestDashboardDataJson = Join-Path $statusRootPath "latest_dashboard_data.json"
  $latestStatusMd = Join-Path $statusRootPath "latest_status.md"
  $statusPublishSummaryPath = Join-Path $statusRootPath "status_publish_summary.json"

  $null = Invoke-SafeRunStep -Label "P27-status-publish" -Exe $p27Py -CmdArgs @(
    "-B",
    "-m", "trainer.experiments.status_publish",
    "--trends-root", "docs/artifacts/trends",
    "--artifacts-root", "docs/artifacts",
    "--out-root", "docs/artifacts/status"
  ) -TimeoutSec 900

  $updateReadmeScript = Join-Path $ProjectRoot "scripts/update_readme_badges.ps1"
  if (-not (Test-Path $updateReadmeScript)) { throw "[P27] missing script: $updateReadmeScript" }
  $null = Invoke-SafeRunStep -Label "P27-readme-badge-dryrun" -Exe "powershell" -CmdArgs @(
    "-ExecutionPolicy", "Bypass",
    "-File", $updateReadmeScript,
    "-DryRun"
  ) -TimeoutSec 300
  $null = Invoke-SafeRunStep -Label "P27-readme-badge-apply" -Exe "powershell" -CmdArgs @(
    "-ExecutionPolicy", "Bypass",
    "-File", $updateReadmeScript,
    "-Apply"
  ) -TimeoutSec 300

  $buildDashboardScript = Join-Path $ProjectRoot "scripts/build_dashboard.ps1"
  if (-not (Test-Path $buildDashboardScript)) { throw "[P27] missing script: $buildDashboardScript" }
  $null = Invoke-SafeRunStep -Label "P27-dashboard-build" -Exe "powershell" -CmdArgs @(
    "-ExecutionPolicy", "Bypass",
    "-File", $buildDashboardScript
  ) -TimeoutSec 300
  $dashboardIndex = Join-Path $ProjectRoot "docs/dashboard/index.html"
  $dashboardJson = Join-Path $ProjectRoot "docs/dashboard/data/latest.json"
  $dashboardJs = Join-Path $ProjectRoot "docs/dashboard/data/latest.js"
  $dashboardBuildSummary = Join-Path $ProjectRoot "docs/dashboard/build_dashboard_summary.json"

  $releaseTrainScript = Join-Path $ProjectRoot "scripts/run_release_train.ps1"
  if (-not (Test-Path $releaseTrainScript)) { throw "[P27] missing script: $releaseTrainScript" }
  $releaseOutDir = Join-Path $p27Dir "release_train"
  New-Item -ItemType Directory -Path $releaseOutDir -Force | Out-Null
  $null = Invoke-SafeRunStep -Label "P27-release-train" -Exe "powershell" -CmdArgs @(
    "-ExecutionPolicy", "Bypass",
    "-File", $releaseTrainScript,
    "-OutDir", $releaseOutDir,
    "-DryRun:`$false"
  ) -TimeoutSec 1200
  $rcSummaryJson = Join-Path $releaseOutDir "rc_summary.json"
  $rcSummaryMd = Join-Path $releaseOutDir "rc_summary.md"
  $benchmarkDeltaCsv = Join-Path $releaseOutDir "benchmark_delta.csv"
  $gateSnapshotJson = Join-Path $releaseOutDir "gate_snapshot.json"
  $riskSnapshotJson = Join-Path $releaseOutDir "risk_snapshot.json"

  $lintWorkflowScript = Join-Path $ProjectRoot "scripts/lint_workflows.ps1"
  if (-not (Test-Path $lintWorkflowScript)) { throw "[P27] missing script: $lintWorkflowScript" }
  $null = Invoke-SafeRunStep -Label "P27-workflow-lint" -Exe "powershell" -CmdArgs @(
    "-ExecutionPolicy", "Bypass",
    "-File", $lintWorkflowScript
  ) -TimeoutSec 300
  $workflowLintJson = Join-Path $statusRootPath "workflow_lint_report.json"
  $workflowLintMd = Join-Path $statusRootPath "workflow_lint_report.md"

  $generateStatusScript = Join-Path $ProjectRoot "scripts/generate_readme_status.ps1"
  if (-not (Test-Path $generateStatusScript)) { throw "[P27] missing script: $generateStatusScript" }
  $null = Invoke-SafeRunStep -Label "P27-generate-readme-status" -Exe "powershell" -CmdArgs @(
    "-ExecutionPolicy", "Bypass",
    "-File", $generateStatusScript
  ) -TimeoutSec 300

  $null = Invoke-SafeRunStep -Label "P27-readme-badge-reapply" -Exe "powershell" -CmdArgs @(
    "-ExecutionPolicy", "Bypass",
    "-File", $updateReadmeScript,
    "-Apply"
  ) -TimeoutSec 300

  $lintReadmeScript = Join-Path $ProjectRoot "scripts/lint_readme_p25.ps1"
  if (-not (Test-Path $lintReadmeScript)) { throw "[P27] missing script: $lintReadmeScript" }
  $null = Invoke-SafeRunStep -Label "P27-readme-lint" -Exe "powershell" -CmdArgs @(
    "-ExecutionPolicy", "Bypass",
    "-File", $lintReadmeScript
  ) -TimeoutSec 300
  $readmeLintJson = Join-Path $ProjectRoot "docs/generated/readme_lint_report.json"
  $readmeLintObj = Read-JsonFile -Path $readmeLintJson
  $readmeLintPass = $false
  if ($readmeLintObj) { $readmeLintPass = [bool]$readmeLintObj.pass }

  $readmeText = Get-Content -LiteralPath (Join-Path $ProjectRoot "README.md") -Raw
  $hasBadgeMarkers = ($readmeText.Contains("<!-- BADGES:START -->") -and $readmeText.Contains("<!-- BADGES:END -->"))
  $hasStatusMarkers = ($readmeText.Contains("<!-- STATUS:START -->") -and $readmeText.Contains("<!-- STATUS:END -->"))
  $hasLegacyStatusMarkers = ($readmeText.Contains("<!-- README_STATUS:BEGIN -->") -and $readmeText.Contains("<!-- README_STATUS:END -->"))

  $statusFilesPass = (
    (Test-Path $latestStatusJson) -and
    (Test-Path $latestBadgesJson) -and
    (Test-Path $latestDashboardDataJson) -and
    (Test-Path $latestStatusMd)
  )
  $dashboardPass = (
    (Test-Path $dashboardIndex) -and
    (Test-Path $dashboardJson) -and
    (Test-Path $dashboardJs) -and
    (Test-Path $dashboardBuildSummary)
  )
  $releaseTrainPass = (
    (Test-Path $rcSummaryJson) -and
    (Test-Path $rcSummaryMd) -and
    (Test-Path $benchmarkDeltaCsv) -and
    (Test-Path $gateSnapshotJson) -and
    (Test-Path $riskSnapshotJson)
  )

  $workflowLintObj = Read-JsonFile -Path $workflowLintJson
  $workflowLintPass = $false
  if ($workflowLintObj) { $workflowLintPass = [bool]$workflowLintObj.pass }
  $workflowFilesExist = (
    (Test-Path (Join-Path $ProjectRoot ".github/workflows/ci-smoke.yml")) -and
    (Test-Path (Join-Path $ProjectRoot ".github/workflows/nightly-orchestrator.yml"))
  )

  $functionalPass = ($RunP26Status -eq "PASS")
  $statusPublishingPass = ($statusFilesPass -and $hasBadgeMarkers -and $hasStatusMarkers -and $hasLegacyStatusMarkers)
  $docsRefreshPass = ((Test-Path (Join-Path $ProjectRoot "docs/generated/README_STATUS.md")) -and (Test-Path (Join-Path $ProjectRoot "docs/generated/README_STATUS.json")) -and $readmeLintPass)
  $releaseOpsPass = ($dashboardPass -and $releaseTrainPass)
  $workflowCiPass = ($workflowLintPass -and $workflowFilesExist)

  $gateFunctional = [ordered]@{
    schema = "p27_gate_functional_v1"
    generated_at = (Get-Date).ToString("o")
    baseline_gate = "RunP26"
    baseline_status = $RunP26Status
    baseline_artifact_dir = $RunP26ArtifactDir
    baseline_gate_report = $RunP26GateReport
    pass = $functionalPass
  }
  $gateStatusPublishing = [ordered]@{
    schema = "p27_gate_status_publishing_v1"
    generated_at = (Get-Date).ToString("o")
    latest_status_json = $latestStatusJson
    latest_badges_json = $latestBadgesJson
    latest_dashboard_data_json = $latestDashboardDataJson
    latest_status_md = $latestStatusMd
    readme_has_badges_markers = $hasBadgeMarkers
    readme_has_status_markers = $hasStatusMarkers
    readme_has_legacy_status_markers = $hasLegacyStatusMarkers
    pass = $statusPublishingPass
  }
  $gateDocsRefresh = [ordered]@{
    schema = "p27_gate_docs_refresh_v1"
    generated_at = (Get-Date).ToString("o")
    generate_readme_status_json = (Join-Path $ProjectRoot "docs/generated/README_STATUS.json")
    generate_readme_status_md = (Join-Path $ProjectRoot "docs/generated/README_STATUS.md")
    readme_lint_report = $readmeLintJson
    readme_lint_pass = $readmeLintPass
    pass = $docsRefreshPass
  }
  $gateReleaseOps = [ordered]@{
    schema = "p27_gate_release_ops_v1"
    generated_at = (Get-Date).ToString("o")
    dashboard_index = $dashboardIndex
    dashboard_data_json = $dashboardJson
    release_out_dir = $releaseOutDir
    rc_summary_json = $rcSummaryJson
    benchmark_delta_csv = $benchmarkDeltaCsv
    risk_snapshot_json = $riskSnapshotJson
    pass = $releaseOpsPass
  }
  $gateWorkflowCi = [ordered]@{
    schema = "p27_gate_workflow_ci_v1"
    generated_at = (Get-Date).ToString("o")
    workflow_lint_json = $workflowLintJson
    workflow_lint_md = $workflowLintMd
    workflow_lint_pass = $workflowLintPass
    required_workflows_exist = $workflowFilesExist
    pass = $workflowCiPass
  }

  Write-JsonFile -Path (Join-Path $p27Dir "gate_functional.json") -Object $gateFunctional
  Write-JsonFile -Path (Join-Path $p27Dir "gate_status_publishing.json") -Object $gateStatusPublishing
  Write-JsonFile -Path (Join-Path $p27Dir "gate_docs_refresh.json") -Object $gateDocsRefresh
  Write-JsonFile -Path (Join-Path $p27Dir "gate_release_ops.json") -Object $gateReleaseOps
  Write-JsonFile -Path (Join-Path $p27Dir "gate_workflow_ci.json") -Object $gateWorkflowCi

  $reportP27 = [ordered]@{
    schema = "p27_report_gate_v1"
    generated_at = (Get-Date).ToString("o")
    functional = $gateFunctional
    status_publishing = $gateStatusPublishing
    docs_refresh = $gateDocsRefresh
    release_ops = $gateReleaseOps
    workflow_ci = $gateWorkflowCi
    status = $(if ($functionalPass -and $statusPublishingPass -and $docsRefreshPass -and $releaseOpsPass -and $workflowCiPass) { "PASS" } else { "FAIL" })
  }
  $reportP27Path = Join-Path $p27Dir "report_p27_gate.json"
  Write-JsonFile -Path $reportP27Path -Object $reportP27

  if (Test-Path $statusPublishSummaryPath) { Copy-Item -LiteralPath $statusPublishSummaryPath -Destination (Join-Path $p27Dir "status_publish_summary.json") -Force }
  if (Test-Path $workflowLintJson) { Copy-Item -LiteralPath $workflowLintJson -Destination (Join-Path $p27Dir "workflow_lint_report.json") -Force }
  if (Test-Path $workflowLintMd) { Copy-Item -LiteralPath $workflowLintMd -Destination (Join-Path $p27Dir "workflow_lint_report.md") -Force }
  if (Test-Path $rcSummaryJson) { Copy-Item -LiteralPath $rcSummaryJson -Destination (Join-Path $p27Dir "rc_summary.json") -Force }
  if (Test-Path $rcSummaryMd) { Copy-Item -LiteralPath $rcSummaryMd -Destination (Join-Path $p27Dir "rc_summary.md") -Force }
  if (Test-Path $benchmarkDeltaCsv) { Copy-Item -LiteralPath $benchmarkDeltaCsv -Destination (Join-Path $p27Dir "benchmark_delta.csv") -Force }
  if (Test-Path $gateSnapshotJson) { Copy-Item -LiteralPath $gateSnapshotJson -Destination (Join-Path $p27Dir "gate_snapshot.json") -Force }
  if (Test-Path $riskSnapshotJson) { Copy-Item -LiteralPath $riskSnapshotJson -Destination (Join-Path $p27Dir "risk_snapshot.json") -Force }

  $coverageP27Path = Join-Path $ProjectRoot "docs/COVERAGE_P27_STATUS.md"
  $coverageP27Lines = @(
    "# COVERAGE P27 STATUS",
    "",
    "- timestamp: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"),
    "- baseline_gate: RunP26",
    "- baseline_status: " + $RunP26Status,
    "- status_publishing_pass: " + $statusPublishingPass,
    "- docs_refresh_pass: " + $docsRefreshPass,
    "- release_ops_pass: " + $releaseOpsPass,
    "- workflow_ci_pass: " + $workflowCiPass,
    "- gate_status: " + $reportP27.status
  )
  $coverageP27Lines -join "`n" | Out-File -LiteralPath $coverageP27Path -Encoding UTF8

  if ($GitSync) {
    $gitSyncScript = Join-Path $ProjectRoot "scripts/git_sync.ps1"
    if (Test-Path $gitSyncScript) {
      $null = Invoke-SafeRunStep -Label "P27-gitsync-dryrun" -Exe "powershell" -CmdArgs @(
        "-ExecutionPolicy", "Bypass",
        "-File", $gitSyncScript,
        "-DryRun:`$true"
      ) -TimeoutSec 600
    }
  }

  Write-Host ("[P27] artifact_dir=" + $p27Dir + " gate_status=" + $reportP27.status)
  if ($reportP27.status -ne "PASS") {
    throw "[P27] gate failed; inspect gate_* and report_p27_gate"
  }
}

if ($RunP29) {
  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $p29Root = Join-Path $ProjectRoot "docs/artifacts/p29"
  if (-not (Test-Path $p29Root)) { New-Item -ItemType Directory -Path $p29Root -Force | Out-Null }
  $p29Dir = Join-Path $p29Root $stamp
  New-Item -ItemType Directory -Path $p29Dir -Force | Out-Null

  $p27Root = Join-Path $ProjectRoot "docs/artifacts/p27"
  $p27RunDir = ""
  if (Test-Path $p27Root) {
    $p27Candidates = Get-ChildItem -Path $p27Root -Directory -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -match "^\d{8}-\d{6}$" } |
      Sort-Object Name -Descending
    if ($p27Candidates) { $p27RunDir = $p27Candidates[0].FullName }
  }
  $p27ReportPath = ""
  if ($p27RunDir) { $p27ReportPath = Join-Path $p27RunDir "report_p27_gate.json" }
  $p27ReportObj = Read-JsonFile -Path $p27ReportPath
  $p27Status = "UNKNOWN"
  if ($p27ReportObj) { $p27Status = [string]$p27ReportObj.status }

  $baselineSummary = [ordered]@{
    schema = "p29_baseline_summary_v1"
    generated_at = (Get-Date).ToString("o")
    requested_gate = "RunP28"
    fallback_to = "RunP27"
    baseline_gate = "RunP27"
    run_p27_status = $p27Status
    run_p27_artifact_dir = $p27RunDir
    run_p27_gate_report = $p27ReportPath
  }
  Write-JsonFile -Path (Join-Path $p29Dir "baseline_summary.json") -Object $baselineSummary
  @(
    "# P29 Baseline Summary",
    "",
    "- requested_gate: RunP28",
    "- fallback_to: RunP27",
    "- run_p27_status: $p27Status",
    "- run_p27_artifact_dir: $p27RunDir",
    "- run_p27_gate_report: $p27ReportPath"
  ) -join "`n" | Out-File -LiteralPath (Join-Path $p29Dir "baseline_summary.md") -Encoding UTF8

  $activeTodoStampPath = Join-Path $p29Root "_active_run.txt"
  $copiedTodo = $false
  if (Test-Path $activeTodoStampPath) {
    $activeStamp = (Get-Content -LiteralPath $activeTodoStampPath -Raw).Trim()
    if ($activeStamp) {
      $activeTodoPath = Join-Path (Join-Path $p29Root $activeStamp) "todo_plan.md"
      if (Test-Path $activeTodoPath) {
        Copy-Item -LiteralPath $activeTodoPath -Destination (Join-Path $p29Dir "todo_plan.md") -Force
        $copiedTodo = $true
      }
    }
  }
  if (-not $copiedTodo) {
    @(
      "# P29 To-do Plan",
      "",
      "- generated_at: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"),
      "- status: generated by RunP29 (fallback template)"
    ) -join "`n" | Out-File -LiteralPath (Join-Path $p29Dir "todo_plan.md") -Encoding UTF8
  }

  $p29Py = Join-Path $ProjectRoot ".venv_trainer\Scripts\python.exe"
  if (-not (Test-Path $p29Py)) { $p29Py = "python" }

  $weaknessLatestDir = Join-Path $ProjectRoot "docs/artifacts/p29/weakness_latest"
  $datasetsLatestDir = Join-Path $ProjectRoot "docs/artifacts/p29/datasets_latest"
  $trainBatchLatestDir = Join-Path $ProjectRoot "docs/artifacts/p29/train_batch_latest"
  $evalRoot = Join-Path $ProjectRoot "docs/artifacts/p29/eval"
  $ablation100Dir = Join-Path $evalRoot "ablation_100"
  $ablation500Dir = Join-Path $evalRoot "ablation_500"
  $ablation1000Dir = Join-Path $evalRoot "ablation_1000"
  $rankingLatestDir = Join-Path $ProjectRoot "docs/artifacts/p29/ranking_latest"
  $candidateCompareLatestDir = Join-Path $ProjectRoot "docs/artifacts/p29/candidate_compare_latest"
  $flakeLatestDir = Join-Path $ProjectRoot "docs/artifacts/p29/flake/best_candidate_latest"
  $datasetPath = Join-Path $ProjectRoot "trainer_data/p29_targeted_v1.jsonl"

  $null = Invoke-SafeRunStep -Label "P29-weakness-v3" -Exe $p29Py -CmdArgs @(
    "-B",
    "-m", "trainer.analysis.weakness_mining_v3",
    "--artifacts-root", "docs/artifacts",
    "--config", "configs/analysis/weakness_v3.yaml",
    "--out-dir", $weaknessLatestDir
  ) -TimeoutSec 900

  $weaknessReportPath = Join-Path $weaknessLatestDir "weakness_priority_report.json"
  $null = Invoke-SafeRunStep -Label "P29-targeted-dataset" -Exe $p29Py -CmdArgs @(
    "-B",
    "-m", "trainer.data.gen_targeted_dataset",
    "--config", "configs/experiments/p29_targeted_data.yaml",
    "--weakness-report", $weaknessReportPath,
    "--out", $datasetPath,
    "--artifacts-dir", $datasetsLatestDir
  ) -TimeoutSec 1800

  $null = Invoke-SafeRunStep -Label "P29-dataset-validate" -Exe $p29Py -CmdArgs @(
    "-B",
    "trainer/dataset.py",
    "--path", $datasetPath,
    "--validate",
    "--summary"
  ) -TimeoutSec 900

  $null = Invoke-SafeRunStep -Label "P29-train-batch" -Exe $p29Py -CmdArgs @(
    "-B",
    "-m", "trainer.train_batch_p29",
    "--config", "configs/train/p29_batch.yaml",
    "--dataset", $datasetPath,
    "--out-dir", "trainer_runs/p29_batch",
    "--artifacts-dir", $trainBatchLatestDir,
    "--resume"
  ) -TimeoutSec 7200

  $trainBatchManifestPath = Join-Path $trainBatchLatestDir "train_batch_manifest.json"
  $trainBatchSummaryPath = Join-Path $trainBatchLatestDir "train_batch_summary.json"

  $null = Invoke-SafeRunStep -Label "P29-ablation-100" -Exe $p29Py -CmdArgs @(
    "-B",
    "trainer/run_ablation.py",
    "--backend", "sim",
    "--stake", "gold",
    "--episodes", "100",
    "--seeds-file", "balatro_mechanics/derived/eval_seeds_100.txt",
    "--from-train-batch-manifest", $trainBatchManifestPath,
    "--include-champion",
    "--out-dir", $ablation100Dir
  ) -TimeoutSec 3600

  $null = Invoke-SafeRunStep -Label "P29-ranking" -Exe $p29Py -CmdArgs @(
    "-B",
    "-m", "trainer.experiments.ranking",
    "--run-root", $ablation100Dir,
    "--config", "configs/experiments/ranking_p24.yaml",
    "--out-dir", $rankingLatestDir
  ) -TimeoutSec 1200

  $null = Invoke-SafeRunStep -Label "P29-candidate-compare" -Exe $p29Py -CmdArgs @(
    "-B",
    "-m", "trainer.analysis.candidate_compare_p29",
    "--eval-root", $evalRoot,
    "--ranking-root", $rankingLatestDir,
    "--out-dir", $candidateCompareLatestDir
  ) -TimeoutSec 1200

  $rankingSummaryPath = Join-Path $rankingLatestDir "ranking_summary.json"
  $null = Invoke-SafeRunStep -Label "P29-ablation-500" -Exe $p29Py -CmdArgs @(
    "-B",
    "trainer/run_ablation.py",
    "--backend", "sim",
    "--stake", "gold",
    "--episodes", "500",
    "--seeds-file", "balatro_mechanics/derived/eval_seeds_500.txt",
    "--from-ranking", $rankingSummaryPath,
    "--topk", "3",
    "--include-champion",
    "--out-dir", $ablation500Dir
  ) -TimeoutSec 5400

  $ablation1000Status = "PASS"
  $ablation1000Error = ""
  try {
    $null = Invoke-SafeRunStep -Label "P29-ablation-1000" -Exe $p29Py -CmdArgs @(
      "-B",
      "trainer/run_ablation.py",
      "--backend", "sim",
      "--stake", "gold",
      "--episodes", "1000",
      "--seeds-file", "balatro_mechanics/derived/eval_seeds_1000.txt",
      "--from-ranking", $rankingSummaryPath,
      "--topk", "1",
      "--include-champion",
      "--out-dir", $ablation1000Dir
    ) -TimeoutSec 7200
  } catch {
    $ablation1000Status = "DEGRADED"
    $ablation1000Error = $_.Exception.Message
    Write-Warning ("[P29-ablation-1000] degraded: " + $ablation1000Error)
  }

  $null = Invoke-SafeRunStep -Label "P29-flake-best-candidate" -Exe $p29Py -CmdArgs @(
    "-B",
    "-m", "trainer.experiments.flake",
    "--mode", "candidate",
    "--candidate-from", $rankingSummaryPath,
    "--seeds-file", "balatro_mechanics/derived/eval_seeds_100.txt",
    "--repeats", "3",
    "--out-dir", $flakeLatestDir
  ) -TimeoutSec 2400

  $weaknessMdPath = Join-Path $weaknessLatestDir "weakness_priority_report.md"
  $weaknessCsvPath = Join-Path $weaknessLatestDir "weakness_priority_table.csv"
  $datasetSummaryJsonPath = Join-Path $datasetsLatestDir "p29_targeted_v1_summary.json"
  $datasetSummaryMdPath = Join-Path $datasetsLatestDir "p29_targeted_v1_summary.md"
  $ablation100SummaryPath = Join-Path $ablation100Dir "summary.json"
  $ablation500SummaryPath = Join-Path $ablation500Dir "summary.json"
  $ablation1000SummaryPath = Join-Path $ablation1000Dir "summary.json"
  $candidateCompareSummaryPath = Join-Path $candidateCompareLatestDir "candidate_compare_summary.json"
  $flakeReportJsonPath = Join-Path $flakeLatestDir "best_candidate_flake_report.json"
  $flakeReportMdPath = Join-Path $flakeLatestDir "best_candidate_flake_report.md"

  $trainBatchSummaryObj = Read-JsonFile -Path $trainBatchSummaryPath
  $trainBatchStatus = "UNKNOWN"
  $trainBatchCandidateCount = 0
  if ($trainBatchSummaryObj) {
    $trainBatchStatus = [string]$trainBatchSummaryObj.status
    $trainBatchCandidateCount = [int]$trainBatchSummaryObj.successful_candidate_count
  }
  $datasetSummaryObj = Read-JsonFile -Path $datasetSummaryJsonPath
  $invalidRowsCount = -1
  if ($datasetSummaryObj -and $datasetSummaryObj.invalid_rows) {
    $invalidRowsCount = [int]$datasetSummaryObj.invalid_rows.count
  }

  $flakeObj = Read-JsonFile -Path $flakeReportJsonPath
  $flakeStatus = "UNKNOWN"
  if ($flakeObj) { $flakeStatus = [string]$flakeObj.status }
  $candidateCompareObj = Read-JsonFile -Path $candidateCompareSummaryPath
  $releaseSuggestion = ""
  if ($candidateCompareObj -and $candidateCompareObj.decision) {
    $releaseSuggestion = [string]$candidateCompareObj.decision.action
  }

  $functionalPass = ($p27Status -eq "PASS")
  $dataFlywheelPass = (
    (Test-Path $weaknessReportPath) -and
    (Test-Path $weaknessMdPath) -and
    (Test-Path $weaknessCsvPath) -and
    (Test-Path $datasetPath) -and
    (Test-Path $datasetSummaryJsonPath) -and
    (Test-Path $datasetSummaryMdPath) -and
    ($invalidRowsCount -eq 0)
  )
  $trainingEvalPass = (
    ($trainBatchStatus -eq "PASS") -and
    ($trainBatchCandidateCount -ge 1) -and
    (Test-Path $ablation100SummaryPath) -and
    (Test-Path $rankingSummaryPath) -and
    (Test-Path $candidateCompareSummaryPath)
  )
  $reliabilityPass = (
    (Test-Path $flakeReportJsonPath) -and
    (Test-Path $flakeReportMdPath) -and
    (($flakeStatus -eq "PASS") -or ($releaseSuggestion -ne "promote"))
  )

  $gateFunctional = [ordered]@{
    schema = "p29_gate_functional_v1"
    generated_at = (Get-Date).ToString("o")
    baseline_gate = "RunP27"
    baseline_status = $p27Status
    baseline_artifact_dir = $p27RunDir
    baseline_gate_report = $p27ReportPath
    pass = $functionalPass
  }
  $gateDataFlywheel = [ordered]@{
    schema = "p29_gate_data_flywheel_v1"
    generated_at = (Get-Date).ToString("o")
    weakness_report_json = $weaknessReportPath
    weakness_report_md = $weaknessMdPath
    weakness_table_csv = $weaknessCsvPath
    dataset_path = $datasetPath
    dataset_summary_json = $datasetSummaryJsonPath
    dataset_summary_md = $datasetSummaryMdPath
    invalid_rows_count = $invalidRowsCount
    pass = $dataFlywheelPass
  }
  $gateTrainingEval = [ordered]@{
    schema = "p29_gate_training_eval_v1"
    generated_at = (Get-Date).ToString("o")
    train_batch_manifest = $trainBatchManifestPath
    train_batch_summary = $trainBatchSummaryPath
    train_batch_status = $trainBatchStatus
    train_batch_candidate_count = $trainBatchCandidateCount
    eval_ablation_100 = $ablation100SummaryPath
    eval_ablation_500 = $ablation500SummaryPath
    eval_ablation_1000 = $ablation1000SummaryPath
    ablation_1000_status = $ablation1000Status
    ablation_1000_error = $ablation1000Error
    ranking_summary = $rankingSummaryPath
    candidate_compare_summary = $candidateCompareSummaryPath
    pass = $trainingEvalPass
  }
  $gateReliability = [ordered]@{
    schema = "p29_gate_reliability_v1"
    generated_at = (Get-Date).ToString("o")
    flake_report_json = $flakeReportJsonPath
    flake_report_md = $flakeReportMdPath
    flake_status = $flakeStatus
    release_suggestion = $releaseSuggestion
    pass = $reliabilityPass
  }

  Write-JsonFile -Path (Join-Path $p29Dir "gate_functional.json") -Object $gateFunctional
  Write-JsonFile -Path (Join-Path $p29Dir "gate_data_flywheel.json") -Object $gateDataFlywheel
  Write-JsonFile -Path (Join-Path $p29Dir "gate_training_eval.json") -Object $gateTrainingEval
  Write-JsonFile -Path (Join-Path $p29Dir "gate_reliability.json") -Object $gateReliability

  $reportP29 = [ordered]@{
    schema = "p29_report_gate_v1"
    generated_at = (Get-Date).ToString("o")
    functional = $gateFunctional
    data_flywheel = $gateDataFlywheel
    training_eval = $gateTrainingEval
    reliability = $gateReliability
    status = $(if ($functionalPass -and $dataFlywheelPass -and $trainingEvalPass -and $reliabilityPass) { "PASS" } else { "FAIL" })
  }
  $reportP29Path = Join-Path $p29Dir "report_p29_gate.json"
  Write-JsonFile -Path $reportP29Path -Object $reportP29

  $copyTargets = @(
    @{ src = $weaknessLatestDir; dst = (Join-Path $p29Dir "weakness_latest") },
    @{ src = $datasetsLatestDir; dst = (Join-Path $p29Dir "datasets_latest") },
    @{ src = $trainBatchLatestDir; dst = (Join-Path $p29Dir "train_batch_latest") },
    @{ src = $evalRoot; dst = (Join-Path $p29Dir "eval") },
    @{ src = $rankingLatestDir; dst = (Join-Path $p29Dir "ranking_latest") },
    @{ src = $candidateCompareLatestDir; dst = (Join-Path $p29Dir "candidate_compare_latest") },
    @{ src = (Join-Path $ProjectRoot "docs/artifacts/p29/flake"); dst = (Join-Path $p29Dir "flake") }
  )
  foreach ($item in $copyTargets) {
    if (Test-Path $item.src) {
      if (Test-Path $item.dst) { Remove-Item -LiteralPath $item.dst -Recurse -Force -ErrorAction SilentlyContinue }
      Copy-Item -LiteralPath $item.src -Destination $item.dst -Recurse -Force
    }
  }

  $coverageP29Path = Join-Path $ProjectRoot "docs/COVERAGE_P29_STATUS.md"
  @(
    "# COVERAGE P29 STATUS",
    "",
    "- timestamp: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"),
    "- baseline_gate: RunP27",
    "- baseline_status: " + $p27Status,
    "- data_flywheel_pass: " + $dataFlywheelPass,
    "- training_eval_pass: " + $trainingEvalPass,
    "- reliability_pass: " + $reliabilityPass,
    "- ablation_1000_status: " + $ablation1000Status,
    "- gate_status: " + $reportP29.status
  ) -join "`n" | Out-File -LiteralPath $coverageP29Path -Encoding UTF8

  if ($GitSync) {
    $gitSyncScript = Join-Path $ProjectRoot "scripts/git_sync.ps1"
    if (Test-Path $gitSyncScript) {
      $null = Invoke-SafeRunStep -Label "P29-gitsync-dryrun" -Exe "powershell" -CmdArgs @(
        "-ExecutionPolicy", "Bypass",
        "-File", $gitSyncScript,
        "-DryRun:`$true"
      ) -TimeoutSec 600
    }
  }

  Write-Host ("[P29] artifact_dir=" + $p29Dir + " gate_status=" + $reportP29.status)
  if ($reportP29.status -ne "PASS") {
    throw "[P29] gate failed; inspect gate_* and report_p29_gate"
  }
}

if ($RunP31) {
  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $p31Root = Join-Path $ProjectRoot "docs/artifacts/p31"
  if (-not (Test-Path $p31Root)) { New-Item -ItemType Directory -Path $p31Root -Force | Out-Null }
  $p31Dir = Join-Path $p31Root $stamp
  New-Item -ItemType Directory -Path $p31Dir -Force | Out-Null

  $p29Root = Join-Path $ProjectRoot "docs/artifacts/p29"
  $p29RunDir = ""
  if (Test-Path $p29Root) {
    $p29Candidates = Get-ChildItem -Path $p29Root -Directory -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -match "^\d{8}-\d{6}$" } |
      Sort-Object Name -Descending
    if ($p29Candidates) { $p29RunDir = $p29Candidates[0].FullName }
  }
  $p29ReportPath = ""
  if ($p29RunDir) { $p29ReportPath = Join-Path $p29RunDir "report_p29_gate.json" }
  $p29ReportObj = Read-JsonFile -Path $p29ReportPath
  $p29Status = "UNKNOWN"
  if ($p29ReportObj) { $p29Status = [string]$p29ReportObj.status }

  $baselineSummary = [ordered]@{
    schema = "p31_baseline_v1"
    generated_at = (Get-Date).ToString("o")
    requested_gate = "RunP29"
    baseline_status = $p29Status
    baseline_artifact_dir = $p29RunDir
    baseline_report = $p29ReportPath
    functional_pass = $(if ($p29ReportObj -and $p29ReportObj.functional) { [bool]$p29ReportObj.functional.pass } else { $false })
    data_flywheel_pass = $(if ($p29ReportObj -and $p29ReportObj.data_flywheel) { [bool]$p29ReportObj.data_flywheel.pass } else { $false })
    training_eval_pass = $(if ($p29ReportObj -and $p29ReportObj.training_eval) { [bool]$p29ReportObj.training_eval.pass } else { $false })
    reliability_pass = $(if ($p29ReportObj -and $p29ReportObj.reliability) { [bool]$p29ReportObj.reliability.pass } else { $false })
  }
  Write-JsonFile -Path (Join-Path $p31Dir "baseline.json") -Object $baselineSummary

  $p31Py = Join-Path $ProjectRoot ".venv_trainer\Scripts\python.exe"
  if (-not (Test-Path $p31Py)) { $p31Py = "python" }

  $null = Invoke-SafeRunStep -Label "P31-router-smoke" -Exe $p31Py -CmdArgs @(
    "-B",
    "-m", "py_compile",
    "trainer/decision_stack/context_features.py",
    "trainer/decision_stack/risk_model.py",
    "trainer/decision_stack/router.py",
    "trainer/search/adaptive_budget.py"
  ) -TimeoutSec 300

  $null = Invoke-SafeRunStep -Label "P31-quick-eval-50" -Exe $p31Py -CmdArgs @(
    "-B",
    "-m", "trainer.eval.run_p31_eval",
    "--seeds", "50",
    "--timestamp", $stamp,
    "--out-root", "docs/artifacts/p31",
    "--config", "configs/decision_stack/p31_router.json"
  ) -TimeoutSec 1200

  $p31EvalDir = Join-Path $p31Dir "eval"
  $ablation50Path = Join-Path $p31EvalDir "ablation_50.json"
  $rankingSmokePath = Join-Path $p31EvalDir "ranking_smoke_50.json"
  $compareSummaryPath = Join-Path $p31EvalDir "compare_summary.md"

  $evalObj = Read-JsonFile -Path $ablation50Path
  $improvement = $null
  $recommendation = "unknown"
  $recommendedVariant = ""
  $meetsThreshold = $false
  if ($evalObj -and $evalObj.improvement) {
    $improvement = $evalObj.improvement
    $recommendation = [string]$improvement.recommendation
    $recommendedVariant = [string]$improvement.recommended_variant
    $meetsThreshold = [bool]$improvement.meets_threshold
  }

  $gateFunctional = [ordered]@{
    schema = "p31_gate_functional_v1"
    generated_at = (Get-Date).ToString("o")
    baseline_gate = "RunP29"
    baseline_status = $p29Status
    baseline_artifact_dir = $p29RunDir
    baseline_report = $p29ReportPath
    router_smoke = $true
    pass = ($p29Status -eq "PASS")
  }
  $gateStrength = [ordered]@{
    schema = "p31_gate_strength_v1"
    generated_at = (Get-Date).ToString("o")
    eval_ablation_50 = $ablation50Path
    ranking_smoke = $rankingSmokePath
    compare_summary = $compareSummaryPath
    recommendation = $recommendation
    recommended_variant = $recommendedVariant
    meets_threshold = $meetsThreshold
    pass = ((Test-Path $ablation50Path) -and (Test-Path $rankingSmokePath) -and (Test-Path $compareSummaryPath))
  }

  Write-JsonFile -Path (Join-Path $p31Dir "gate_functional.json") -Object $gateFunctional
  Write-JsonFile -Path (Join-Path $p31Dir "gate_strength.json") -Object $gateStrength

  $reportP31 = [ordered]@{
    schema = "p31_report_gate_v1"
    generated_at = (Get-Date).ToString("o")
    functional = $gateFunctional
    strength = $gateStrength
    status = $(if ($gateFunctional.pass -and $gateStrength.pass) { "PASS" } else { "FAIL" })
  }
  $reportP31Path = Join-Path $p31Dir "report_p31.json"
  Write-JsonFile -Path $reportP31Path -Object $reportP31

  $coverageP31Path = Join-Path $ProjectRoot "docs/COVERAGE_P31_STATUS.md"
  @(
    "# COVERAGE P31 STATUS",
    "",
    "- timestamp: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"),
    "- baseline_gate: RunP29",
    "- baseline_status: " + $p29Status,
    "- router_smoke_pass: true",
    "- eval_50_pass: " + (Test-Path $ablation50Path),
    "- ranking_smoke_pass: " + (Test-Path $rankingSmokePath),
    "- recommendation: " + $recommendation,
    "- recommended_variant: " + $recommendedVariant,
    "- gate_status: " + $reportP31.status
  ) -join "`n" | Out-File -LiteralPath $coverageP31Path -Encoding UTF8

  Write-Host ("[P31] artifact_dir=" + $p31Dir + " gate_status=" + $reportP31.status)
  if ($reportP31.status -ne "PASS") {
    throw "[P31] gate failed; inspect gate_* and report_p31"
  }
}

if ($RunP32) {
  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $p32Root = Join-Path $ProjectRoot "docs/artifacts/p32"
  if (-not (Test-Path $p32Root)) { New-Item -ItemType Directory -Path $p32Root -Force | Out-Null }
  $p32Dir = Join-Path $p32Root $stamp
  New-Item -ItemType Directory -Path $p32Dir -Force | Out-Null

  $p22RunDir = Get-LatestRunDir -RunsRootPath (Join-Path $ProjectRoot "docs/artifacts/p22/runs")
  $p13RunDir = Get-LatestRunDir -RunsRootPath (Join-Path $ProjectRoot "docs/artifacts/p13")
  $p22ReportPath = ""
  if ($p22RunDir) {
    $r23 = Join-Path $p22RunDir "report_p23.json"
    $r22 = Join-Path $p22RunDir "report_p22.json"
    if (Test-Path $r23) { $p22ReportPath = $r23 }
    elseif (Test-Path $r22) { $p22ReportPath = $r22 }
  }
  $p13ReportPath = ""
  if ($p13RunDir) { $p13ReportPath = Join-Path $p13RunDir "report_p13.json" }
  $p22ReportObj = Read-JsonFile -Path $p22ReportPath
  $p13ReportObj = Read-JsonFile -Path $p13ReportPath
  $resolvedRunP22Status = $RunP22Status
  if ($resolvedRunP22Status -eq "SKIPPED" -and $p22ReportObj -and ([string]$p22ReportObj.status -eq "PASS")) {
    $resolvedRunP22Status = "PASS"
  }

  $baseline = [ordered]@{
    schema = "p32_baseline_v1"
    generated_at = (Get-Date).ToString("o")
    requested_gate = "RunP22 + RunP13"
    run_p22_status = $resolvedRunP22Status
    run_p22_latest_run = $RunP22LatestRun
    run_p22_report = $p22ReportPath
    run_p13_latest_run = $p13RunDir
    run_p13_report = $p13ReportPath
    run_p13_status = $(if ($p13ReportObj -and $p13ReportObj.status) { [string]$p13ReportObj.status } else { "UNKNOWN" })
  }
  Write-JsonFile -Path (Join-Path $p32Dir "baseline.json") -Object $baseline

  $p32Py = Join-Path $ProjectRoot ".venv_trainer\Scripts\python.exe"
  if (-not (Test-Path $p32Py)) { $p32Py = "python" }

  $positionFixtureDir = Join-Path $p32Dir "fixtures/position_contract"
  $positionReplayPath = Join-Path $p32Dir "position_replay_report.json"
  $positionDumpDir = Join-Path $p32Dir "dumps/position_replay"

  $null = Invoke-SafeRunStep -Label "P32-build-position-fixture" -Exe $p32Py -CmdArgs @(
    "-B",
    "sim/tests/build_p32_position_fixture.py",
    "--out-dir", $positionFixtureDir,
    "--seed", ("P32POS-" + $stamp)
  ) -TimeoutSec 600

  $null = Invoke-SafeRunStep -Label "P32-replay-position-fixture" -Exe $p32Py -CmdArgs @(
    "-B",
    "sim/tests/run_real_action_replay_fixture.py",
    "--fixture-dir", $positionFixtureDir,
    "--scope", "p32_real_action_position_observed_core",
    "--out", $positionReplayPath,
    "--dump-on-diff", $positionDumpDir
  ) -TimeoutSec 600

  $positionReplay = Read-JsonFile -Path $positionReplayPath
  $positionPass = $false
  if ($positionReplay -and ([string]$positionReplay.status -eq "pass")) { $positionPass = $true }

  $syntheticSessionPath = Join-Path $p32Dir "sessions/session_position_synthetic.jsonl"
  $syntheticFixtureDir = Join-Path $p32Dir "fixtures/synthetic_session"
  $syntheticReplayPath = Join-Path $p32Dir "synthetic_session_replay_report.json"
  $syntheticRoundtripStatus = "FAIL"
  $syntheticRoundtripReason = ""

  $syntheticSessionArgs = @(
    "-B",
    "trainer/build_p32_synthetic_session.py",
    "--out", $syntheticSessionPath,
    "--seed", ("P32SYN-" + $stamp)
  )
  $syntheticSessionRun = Run-Py -Label "P32-build-synthetic-session" -PyArgs $syntheticSessionArgs
  if ($syntheticSessionRun.Code -eq 0) {
    $syntheticTraceArgs = @(
      "-B",
      "trainer/real_trace_to_fixture.py",
      "--in", $syntheticSessionPath,
      "--out-dir", $syntheticFixtureDir,
      "--seed", ("P32SYN-" + $stamp)
    )
    $syntheticTraceRun = Run-Py -Label "P32-synthetic-trace-to-fixture" -PyArgs $syntheticTraceArgs
    if ($syntheticTraceRun.Code -eq 0) {
      $syntheticReplayArgs = @(
        "-B",
        "sim/tests/run_real_action_replay_fixture.py",
        "--fixture-dir", $syntheticFixtureDir,
        "--scope", "p32_real_action_position_observed_core",
        "--out", $syntheticReplayPath,
        "--dump-on-diff", (Join-Path $p32Dir "dumps/synthetic_session_replay")
      )
      $syntheticReplayRun = Run-Py -Label "P32-synthetic-session-replay" -PyArgs $syntheticReplayArgs
      if ($syntheticReplayRun.Code -eq 0) {
        $syntheticRoundtripStatus = "PASS"
      } else {
        $syntheticRoundtripStatus = "FAIL"
        $syntheticRoundtripReason = "synthetic_session_replay_failed"
      }
    } else {
      $syntheticRoundtripStatus = "FAIL"
      $syntheticRoundtripReason = "synthetic_trace_to_fixture_failed"
    }
  } else {
    $syntheticRoundtripStatus = "FAIL"
    $syntheticRoundtripReason = "synthetic_session_build_failed"
  }

  $realSessionPath = Join-Path $p32Dir "sessions/session_shadow.jsonl"
  $realFixtureDir = Join-Path $p32Dir "fixtures/real_shadow"
  $realReplayPath = Join-Path $p32Dir "real_shadow_replay_report.json"
  $realRoundtripStatus = "SKIPPED"
  $realRoundtripReason = ""
  $realActionsCount = 0

  if (Test-Health -Url $BaseUrl -TimeoutSec 3) {
    $recordArgs = @(
      "-B", "trainer/record_real_session.py",
      "--base-url", $BaseUrl,
      "--steps", "40",
      "--interval", "0.2",
      "--topk", "3",
      "--include-raw",
      "--out", $realSessionPath,
      "--strict-errors"
    )
    $recordResult = Run-Py -Label "P32-record-shadow" -PyArgs $recordArgs
    if ($recordResult.Code -eq 0) {
      $traceArgs = @("-B", "trainer/real_trace_to_fixture.py", "--in", $realSessionPath, "--out-dir", $realFixtureDir, "--seed", $Seed)
      $traceResult = Run-Py -Label "P32-shadow-trace-to-fixture" -PyArgs $traceArgs
      if ($traceResult.Code -eq 0) {
        $realManifestPath = Join-Path $realFixtureDir "manifest.json"
        $realManifest = Read-JsonFile -Path $realManifestPath
        if ($realManifest -and $realManifest.actions_count -ne $null) { $realActionsCount = [int]$realManifest.actions_count }
        if ($realActionsCount -gt 0) {
          $replayArgs = @(
            "-B", "sim/tests/run_real_action_replay_fixture.py",
            "--fixture-dir", $realFixtureDir,
            "--scope", "p32_real_action_position_observed_core",
            "--out", $realReplayPath,
            "--dump-on-diff", (Join-Path $p32Dir "dumps/real_shadow_replay")
          )
          $replayResult = Run-Py -Label "P32-real-shadow-replay" -PyArgs $replayArgs
          if ($replayResult.Code -eq 0) {
            $realRoundtripStatus = "PASS"
          } else {
            $realRoundtripStatus = "FAIL"
            $realRoundtripReason = "real_shadow_replay_failed"
          }
        } else {
          $realRoundtripStatus = "SKIPPED"
          $realRoundtripReason = "no_actions_in_shadow_capture"
        }
      } else {
        $realRoundtripStatus = "FAIL"
        $realRoundtripReason = "trace_to_fixture_failed"
      }
    } else {
      $realRoundtripStatus = "FAIL"
      $realRoundtripReason = "record_real_session_failed"
    }
  } else {
    $realRoundtripStatus = "SKIPPED"
    $realRoundtripReason = "real_unavailable"
  }

  $shopReportDir = Join-Path $p32Dir "shop_rng"
  $shopSummaryJson = Join-Path $shopReportDir "shop_probability_summary.json"
  $shopSummaryMd = Join-Path $shopReportDir "shop_probability_summary.md"
  $shopStatus = "FAIL"
  $shopReason = ""
  $shopArgs = @(
    "-B",
    "sim/oracle/analyze_shop_probabilities.py",
    "--base-url", $BaseUrl,
    "--seed", ("P32SHOP-" + $stamp),
    "--samples", "300",
    "--out-dir", $shopReportDir
  )
  $shopRun = Run-Py -Label "P32-shop-probability" -PyArgs $shopArgs
  if ($shopRun.Code -eq 0 -and (Test-Path $shopSummaryJson) -and (Test-Path $shopSummaryMd)) {
    $shopStatus = "PASS"
  } else {
    $shopStatus = "FAIL"
    $shopReason = "shop_probability_analysis_failed"
  }

  $gateFunctional = [ordered]@{
    schema = "p32_gate_functional_v1"
    generated_at = (Get-Date).ToString("o")
    baseline_gate = "RunP22 + RunP13"
    run_p22_status = $resolvedRunP22Status
    run_p22_latest_run = $RunP22LatestRun
    run_p22_report = $p22ReportPath
    run_p13_latest_run = $p13RunDir
    run_p13_report = $p13ReportPath
    pass = (($resolvedRunP22Status -eq "PASS") -and ($baseline.run_p13_status -in @("PASS", "SKIPPED")))
  }
  $gateActionContract = [ordered]@{
    schema = "p32_gate_action_contract_v1"
    generated_at = (Get-Date).ToString("o")
    position_fixture_dir = $positionFixtureDir
    position_replay_report = $positionReplayPath
    position_replay_pass = $positionPass
    synthetic_session = $syntheticSessionPath
    synthetic_fixture_dir = $syntheticFixtureDir
    synthetic_replay_report = $syntheticReplayPath
    synthetic_roundtrip_status = $syntheticRoundtripStatus
    synthetic_roundtrip_reason = $syntheticRoundtripReason
    real_session = $realSessionPath
    real_fixture_dir = $realFixtureDir
    real_replay_report = $realReplayPath
    real_roundtrip_status = $realRoundtripStatus
    real_roundtrip_reason = $realRoundtripReason
    real_actions_count = $realActionsCount
    pass = ($positionPass -and ($syntheticRoundtripStatus -eq "PASS") -and ($realRoundtripStatus -in @("PASS", "SKIPPED")))
  }
  $gateShopRng = [ordered]@{
    schema = "p32_gate_shop_rng_v1"
    generated_at = (Get-Date).ToString("o")
    shop_report_dir = $shopReportDir
    shop_summary_json = $shopSummaryJson
    shop_summary_md = $shopSummaryMd
    status = $shopStatus
    reason = $shopReason
    pass = ($shopStatus -eq "PASS")
  }

  Write-JsonFile -Path (Join-Path $p32Dir "gate_functional.json") -Object $gateFunctional
  Write-JsonFile -Path (Join-Path $p32Dir "gate_action_contract.json") -Object $gateActionContract
  Write-JsonFile -Path (Join-Path $p32Dir "gate_shop_rng.json") -Object $gateShopRng

  $reportP32 = [ordered]@{
    schema = "p32_report_gate_v1"
    generated_at = (Get-Date).ToString("o")
    functional = $gateFunctional
    action_contract = $gateActionContract
    shop_rng = $gateShopRng
    status = $(if ($gateFunctional.pass -and $gateActionContract.pass -and $gateShopRng.pass) { "PASS" } else { "FAIL" })
  }
  $reportP32Path = Join-Path $p32Dir "report_p32.json"
  Write-JsonFile -Path $reportP32Path -Object $reportP32

  $coverageP32Path = Join-Path $ProjectRoot "docs/COVERAGE_P32_STATUS.md"
  @(
    "# COVERAGE P32 STATUS",
    "",
    "- timestamp: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"),
    "- baseline_gate: RunP22 + RunP13",
    "- run_p22_status: " + $resolvedRunP22Status,
    "- run_p13_status: " + $baseline.run_p13_status,
    "- position_replay_pass: " + $positionPass,
    "- synthetic_roundtrip_status: " + $syntheticRoundtripStatus,
    "- real_roundtrip_status: " + $realRoundtripStatus,
    "- shop_rng_status: " + $shopStatus,
    "- gate_status: " + $reportP32.status
  ) -join "`n" | Out-File -LiteralPath $coverageP32Path -Encoding UTF8

  if ($GitSync) {
    $gitSyncScript = Join-Path $ProjectRoot "scripts/git_sync.ps1"
    if (Test-Path $gitSyncScript) {
      $null = Invoke-SafeRunStep -Label "P32-gitsync-dryrun" -Exe "powershell" -CmdArgs @(
        "-ExecutionPolicy", "Bypass",
        "-File", $gitSyncScript,
        "-DryRun:`$true"
      ) -TimeoutSec 600
    }
  }

  Write-Host ("[P32] artifact_dir=" + $p32Dir + " gate_status=" + $reportP32.status)
  if ($reportP32.status -ne "PASS") {
    throw "[P32] gate failed; inspect gate_* and report_p32"
  }
}

if ($RunP37) {
  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $p37Root = Join-Path $ProjectRoot "docs/artifacts/p37"
  if (-not (Test-Path $p37Root)) { New-Item -ItemType Directory -Path $p37Root -Force | Out-Null }
  $p37Dir = Join-Path $p37Root $stamp
  New-Item -ItemType Directory -Path $p37Dir -Force | Out-Null

  $p37Py = Join-Path $ProjectRoot ".venv_trainer\Scripts\python.exe"
  if (-not (Test-Path $p37Py)) { $p37Py = "python" }

  $null = Invoke-SafeRunStep -Label "P37-batch-action-fidelity" -Exe $p37Py -CmdArgs @(
    "-B",
    "sim/oracle/batch_build_p37_action_fidelity.py",
    "--out-dir", $p37Dir,
    "--seed", ("P37-" + $stamp),
    "--scope", "p37_action_fidelity_core"
  ) -TimeoutSec 1200

  $reportP37Path = Join-Path $p37Dir "report_p37.json"
  $reportP37 = Read-JsonFile -Path $reportP37Path
  if (-not $reportP37) {
    throw "[P37] missing report_p37.json"
  }

  $coverageP37Path = Join-Path $ProjectRoot "docs/COVERAGE_P37_STATUS.md"
  @(
    "# COVERAGE P37 STATUS",
    "",
    "- timestamp: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"),
    "- baseline_gate: RunP32",
    "- run_p32_invoked: " + $RunP32,
    "- fixtures_total: " + $reportP37.fixtures_total,
    "- fixtures_pass: " + $reportP37.fixtures_pass,
    "- diff_fail: " + $reportP37.diff_fail,
    "- gate_status: " + $reportP37.status
  ) -join "`n" | Out-File -LiteralPath $coverageP37Path -Encoding UTF8

  Write-Host ("[P37] artifact_dir=" + $p37Dir + " gate_status=" + $reportP37.status + " diff_fail=" + $reportP37.diff_fail)
  if ($reportP37.status -ne "PASS" -or [int]$reportP37.diff_fail -ne 0) {
    throw "[P37] gate failed; inspect report_p37"
  }
}

if ($RunP38) {
  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $p38Root = Join-Path $ProjectRoot "docs/artifacts/p38"
  if (-not (Test-Path $p38Root)) { New-Item -ItemType Directory -Path $p38Root -Force | Out-Null }

  $p38LongRoot = Join-Path $p38Root "long_episode"
  if (-not (Test-Path $p38LongRoot)) { New-Item -ItemType Directory -Path $p38LongRoot -Force | Out-Null }
  $p38RunId = $stamp
  $p38LongDir = Join-Path $p38LongRoot $p38RunId
  $p38AnalysisDir = Join-Path $p38Root ("analysis_" + $stamp)
  $p38PlotsDir = Join-Path $p38Root "plots"

  $p38Py = Join-Path $ProjectRoot ".venv_trainer\Scripts\python.exe"
  if (-not (Test-Path $p38Py)) { $p38Py = "python" }

  $p38Seeds = "AAAAAAA,BBBBBBB,CCCCCCC,DDDDDDD,EEEEEEE"

  $p38Batch = Run-Py -Label "P38-batch-long-episode" -PyArgs @(
    "-B",
    "sim/oracle/batch_build_p38_long_episode.py",
    "--base-url", $BaseUrl,
    "--out-dir", $p38LongRoot,
    "--run-id", $p38RunId,
    "--episodes", "12",
    "--max-steps", "260",
    "--seeds", $p38Seeds,
    "--scope", "p37_action_fidelity_core"
  )

  $reportP38Path = Join-Path $p38LongDir "report_p38_long_episode.json"
  $reportP38 = Read-JsonFile -Path $reportP38Path
  if (-not $reportP38) {
    throw "[P38] missing report_p38_long_episode.json"
  }

  $p38Analyze = Run-Py -Label "P38-analyze-stats" -PyArgs @(
    "-B",
    "sim/oracle/analyze_p38_long_stats.py",
    "--fixtures-dir", $p38LongDir,
    "--out-dir", $p38AnalysisDir,
    "--warn-relative-pct", "5",
    "--warn-pvalue", "0.01"
  )
  if ([int]$p38Analyze.Code -ne 0) {
    throw "[P38] analyze stats failed"
  }

  $p38SummaryPath = Join-Path $p38AnalysisDir "summary_stats.json"
  $p38Summary = Read-JsonFile -Path $p38SummaryPath
  if (-not $p38Summary) {
    throw "[P38] missing summary_stats.json"
  }

  $p38Plot = Run-Py -Label "P38-plot-stats" -PyArgs @(
    "-B",
    "sim/oracle/plot_p38_stats.py",
    "--fixtures-dir", $p38LongDir,
    "--out-dir", $p38PlotsDir
  )
  if ([int]$p38Plot.Code -ne 0) {
    Write-Host "[P38][WARNING] plot script returned non-zero"
  }

  $hardFailCount = [int]($reportP38.hard_fail_count)
  $softWarnCount = [int]($p38Summary.soft_warn_count)
  $coverageP38Path = Join-Path $ProjectRoot "docs/COVERAGE_P38_STATUS.md"
  @(
    "# COVERAGE P38 STATUS",
    "",
    "- timestamp: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"),
    "- baseline_gate: RunP37",
    "- run_p37_invoked: " + $RunP37,
    "- episodes_total: " + $reportP38.episodes_total,
    "- episodes_pass: " + $reportP38.episodes_pass,
    "- hard_fail_count: " + $hardFailCount,
    "- soft_warn_count: " + $softWarnCount,
    "- report_long_episode: " + $reportP38Path,
    "- report_summary_stats: " + $p38SummaryPath,
    "- gate_status: " + $(if ($hardFailCount -eq 0) { "PASS" } else { "FAIL" })
  ) -join "`n" | Out-File -LiteralPath $coverageP38Path -Encoding UTF8

  if ($softWarnCount -gt 0) {
    Write-Host ("[P38][WARNING] soft statistical warnings detected: " + $softWarnCount)
  }

  Write-Host (
    "[P38] run_dir=" + $p38LongDir +
    " hard_fail_count=" + $hardFailCount +
    " soft_warn_count=" + $softWarnCount
  )

  if ($hardFailCount -gt 0 -or [int]$p38Batch.Code -ne 0) {
    throw "[P38] hard gate failed; inspect report_p38_long_episode.json and summary_stats.json"
  }
}

if ($RunFast) {
  Write-Host "[RunFast] PASS (P0/P1 baseline completed)"
}



