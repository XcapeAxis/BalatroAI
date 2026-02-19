param(
  [string]$BaseUrl = "http://127.0.0.1:12346",
  [string]$OutRoot = "sim/tests/fixtures_runtime",
  [string]$Seed = "AAAAAAA",
  [switch]$RunP2b,
  [switch]$RunP3
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

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

  Write-Host "[svc] starting: $($uvx.Source) $($serveArgs -join ' ')"
  Start-Process -FilePath $uvx.Source -ArgumentList $serveArgs -WorkingDirectory $ProjectRoot -WindowStyle Hidden | Out-Null

  for ($i = 0; $i -lt 45; $i++) {
    if (Test-Health -Url $Url -TimeoutSec 3) { Write-Host "[svc] health ok"; return }
    Start-Sleep -Seconds 1
  }
  throw "service start timeout"
}

function Ensure-Service([string]$Url, [bool]$ForceRestart = $false) {
  if ($ForceRestart) { Stop-ServiceProc; Start-Sleep -Seconds 2 }
  if (Test-Health -Url $Url) { Write-Host "[svc] health ok at $Url"; return }
  Start-ServiceProc -Url $Url
}

function Run-Py([string]$Label, [string[]]$PyArgs) {
  Write-Host "[$Label] running: python $($PyArgs -join ' ')"
  $o = & python @PyArgs 2>&1
  $code = $LASTEXITCODE
  if ($o) { $o | ForEach-Object { Write-Host $_ } }
  return @{ Code = $code; Text = ($o -join "`n") }
}

function Run-WithRecovery([string]$Label, [string[]]$PyArgs, [string]$Url) {
  $r = Run-Py -Label $Label -PyArgs $PyArgs
  if ($r.Code -eq 0) { return }
  $t = [string]$r.Text
  if (($t -match "timeout") -or ($t -match "health check failed") -or ($t -match "connection refused")) {
    Write-Host "[$Label] transport issue, restarting service and retrying"
    Ensure-Service -Url $Url -ForceRestart $true
    $r2 = Run-Py -Label $Label -PyArgs $PyArgs
    if ($r2.Code -ne 0) { throw "[$Label] failed after retry" }
    return
  }
  throw "[$Label] failed (non-recoverable)"
}

Ensure-Service -Url $BaseUrl
if (-not (Test-Path $OutRoot)) { New-Item -ItemType Directory -Path $OutRoot -Force | Out-Null }
$outRootPath = (Resolve-Path $OutRoot).Path

$p0Out = Join-Path $outRootPath "oracle_p0_v6_regression"
$p1Out = Join-Path $outRootPath "oracle_p1_smoke_v3_regression"
$p2Out = Join-Path $outRootPath "oracle_p2_smoke_v1_regression"
$p2bOut = Join-Path $outRootPath "oracle_p2b_smoke_v1_regression"
$p3Out = Join-Path $outRootPath "oracle_p3_jokers_v1_regression"

$p0Args = @("-B", "sim/oracle/batch_build_p0_oracle_fixtures.py", "--base-url", $BaseUrl, "--out-dir", $p0Out, "--max-steps", "160", "--scope", "p0_hand_score_observed_core", "--seed", $Seed, "--dump-on-diff", (Join-Path $p0Out "dumps"))
$p1Args = @("-B", "sim/oracle/batch_build_p1_smoke.py", "--base-url", $BaseUrl, "--out-dir", $p1Out, "--max-steps", "120", "--scope", "p1_hand_score_observed_core", "--seed", $Seed, "--dump-on-diff", (Join-Path $p1Out "dumps"))
$p2Args = @("-B", "sim/oracle/batch_build_p2_smoke.py", "--base-url", $BaseUrl, "--out-dir", $p2Out, "--max-steps", "160", "--scope", "p2_hand_score_observed_core", "--seed", $Seed, "--dump-on-diff", (Join-Path $p2Out "dumps"))
$p2bArgs = @("-B", "sim/oracle/batch_build_p2b_smoke.py", "--base-url", $BaseUrl, "--out-dir", $p2bOut, "--max-steps", "200", "--scope", "p2b_hand_score_observed_core", "--seed", $Seed, "--dump-on-diff", (Join-Path $p2bOut "dumps"))
$p3Args = @("-B", "sim/oracle/batch_build_p3_joker_fixtures.py", "--base-url", $BaseUrl, "--out-dir", $p3Out, "--max-steps", "160", "--scope", "p3_hand_score_observed_core", "--seed", $Seed, "--dump-on-diff", (Join-Path $p3Out "dumps"))

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

if ($RunP3) {
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
}
elseif ($RunP2b) {
  Run-WithRecovery -Label "P2b" -PyArgs $p2bArgs -Url $BaseUrl
  $p2bAnalyzeArgs = @("-B", "sim/oracle/analyze_p2b_mismatch.py", "--fixtures-dir", $p2bOut)
  Run-Py -Label "P2b-analyzer" -PyArgs $p2bAnalyzeArgs | Out-Null

  $p2bReportPath = Join-Path $p2bOut "report_p2b.json"
  $p2bReport = Get-Content $p2bReportPath -Raw | ConvertFrom-Json
  Write-Host ("P2b summary: pass={0}/{1} diff_fail={2} oracle_fail={3} gen_fail={4} skipped={5}" -f $p2bReport.passed, $p2bReport.total, $p2bReport.diff_fail, $p2bReport.oracle_fail, $p2bReport.gen_fail, $p2bReport.skipped)
  Write-Host ("P2b report: {0}" -f $p2bReportPath)
  Write-Host ("P2b analyzer: {0}" -f (Join-Path $p2bOut "score_mismatch_table_p2b.md"))
}

