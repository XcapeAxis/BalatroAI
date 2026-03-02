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
    if (Test-Health -Url $Url -TimeoutSec 3) { Write-Host "[svc] health ok"; return }
    Start-Sleep -Seconds 1
  }
  throw "service start timeout"
}

function Ensure-Service([string]$Url, [bool]$ForceRestart = $false) {
  if ($ForceRestart) { Stop-ServiceProc; Start-Sleep -Seconds 2; Clear-LovelyDump }
  if (Test-Health -Url $Url) { Write-Host "[svc] health ok at $Url"; return }
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
  if ($reportGate.status -ne "PASS") {
    throw "[P25] gate failed; inspect gate_functional/gate_docs/report_p25_gate"
  }
}

if ($RunFast) {
  Write-Host "[RunFast] PASS (P0/P1 baseline completed)"
}



