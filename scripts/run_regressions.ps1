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
  [switch]$GitSync
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

if ($RunP3 -or $RunP4 -or $RunP5 -or $RunP7 -or $RunP8 -or $RunP9 -or $RunP10) {
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

  if ($RunP4 -or $RunP5 -or $RunP7 -or $RunP8 -or $RunP9 -or $RunP10) {
    Run-WithRecovery -Label "P4" -PyArgs $p4Args -Url $BaseUrl
    $p4ReportPath = Join-Path $p4Out "report_p4.json"
    $p4Report = Get-Content $p4ReportPath -Raw | ConvertFrom-Json
    Write-Host ("P4 summary: pass={0}/{1} diff_fail={2} oracle_fail={3} gen_fail={4} skipped={5} unsupported={6}" -f $p4Report.passed, $p4Report.total, $p4Report.diff_fail, $p4Report.oracle_fail, $p4Report.gen_fail, $p4Report.skipped, $p4Report.classifier.unsupported)
    Write-Host ("P4 report: {0}" -f $p4ReportPath)
    Persist-ArtifactSet -Prefix "P4" -ReportPath $p4ReportPath -ProjectRootPath $ProjectRoot -ExtraDocs @("docs/COVERAGE_P4_CONSUMABLES.md", "docs/COVERAGE_P4_STATUS.md")

    if ($RunP5 -or $RunP7 -or $RunP8 -or $RunP9 -or $RunP10) {
      Run-WithRecovery -Label "P5" -PyArgs $p5Args -Url $BaseUrl
      $p5ReportPath = Join-Path $p5Out "report_p5.json"
      $p5Report = Get-Content $p5ReportPath -Raw | ConvertFrom-Json
      Write-Host ("P5 summary: pass={0}/{1} diff_fail={2} oracle_fail={3} gen_fail={4} skipped={5} unsupported={6}" -f $p5Report.passed, $p5Report.total, $p5Report.diff_fail, $p5Report.oracle_fail, $p5Report.gen_fail, $p5Report.skipped, $p5Report.classifier.unsupported)
      Write-Host ("P5 report: {0}" -f $p5ReportPath)
      Persist-ArtifactSet -Prefix "P5" -ReportPath $p5ReportPath -ProjectRootPath $ProjectRoot -ExtraDocs @("docs/COVERAGE_P5_VOUCHERS_PACKS.md", "docs/COVERAGE_P5_STATUS.md")

      if ($RunP7 -or $RunP8 -or $RunP9 -or $RunP10) {
        Run-WithRecovery -Label "P7" -PyArgs $p7Args -Url $BaseUrl
        $p7ReportPath = Join-Path $p7Out "report_p7.json"
        $p7Report = Get-Content $p7ReportPath -Raw | ConvertFrom-Json
        Write-Host ("P7 summary: pass={0}/{1} diff_fail={2} oracle_fail={3} gen_fail={4} skipped={5} unsupported={6}" -f $p7Report.passed, $p7Report.total, $p7Report.diff_fail, $p7Report.oracle_fail, $p7Report.gen_fail, $p7Report.skipped, $p7Report.classifier.unsupported)
        Write-Host ("P7 report: {0}" -f $p7ReportPath)
        Persist-ArtifactSet -Prefix "P7" -ReportPath $p7ReportPath -ProjectRootPath $ProjectRoot -ExtraDocs @("docs/COVERAGE_P7_STATEFUL_JOKERS.md", "docs/COVERAGE_P7_STATUS.md")
        $p7AnalyzeArgs = @("-B", "sim/oracle/analyze_p7_stateful_mismatch.py", "--fixtures-dir", $p7Out)
        $null = Run-Py -Label "P7-analyzer" -PyArgs $p7AnalyzeArgs

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

        if ($RunP8 -or $RunP9 -or $RunP10) {
          Run-WithRecovery -Label "P8-shop" -PyArgs $p8ShopArgs -Url $BaseUrl
          $p8ShopReportPath = Join-Path $p8ShopOut "report_p8.json"
          $p8ShopReport = Get-Content $p8ShopReportPath -Raw | ConvertFrom-Json
          Write-Host ("P8 shop summary: pass={0}/{1} diff_fail={2} oracle_fail={3} gen_fail={4} skipped={5}" -f $p8ShopReport.passed, $p8ShopReport.total, $p8ShopReport.diff_fail, $p8ShopReport.oracle_fail, $p8ShopReport.gen_fail, $p8ShopReport.skipped)
          Write-Host ("P8 shop report: {0}" -f $p8ShopReportPath)
          Persist-ArtifactSet -Prefix "p8/shop" -ReportPath $p8ShopReportPath -ProjectRootPath $ProjectRoot -ExtraDocs @("docs/COVERAGE_P8_STATUS.md", "docs/COVERAGE_P8_SHOP.md")
          $p8ShopAnalyzeArgs = @("-B", "sim/oracle/analyze_p8_shop_mismatch.py", "--fixtures-dir", $p8ShopOut)
          $null = Run-Py -Label "P8-shop-analyzer" -PyArgs $p8ShopAnalyzeArgs
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
          $null = Run-Py -Label "P8-rng-analyzer" -PyArgs $p8RngAnalyzeArgs
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

          if ($RunP9 -or $RunP10) {
            Run-Py -Label "P9-classifier" -PyArgs $p9ClassifyArgs | Out-Null
            Run-WithRecovery -Label "P9" -PyArgs $p9Args -Url $BaseUrl
            $p9ReportPath = Join-Path $p9Out "report_p9.json"
            $p9Report = Get-Content $p9ReportPath -Raw | ConvertFrom-Json
            Write-Host ("P9 summary: pass={0}/{1} diff_fail={2} oracle_fail={3} gen_fail={4} skipped={5} unsupported={6}" -f $p9Report.passed, $p9Report.total, $p9Report.diff_fail, $p9Report.oracle_fail, $p9Report.gen_fail, $p9Report.skipped, $p9Report.unsupported)
            Write-Host ("P9 report: {0}" -f $p9ReportPath)
            Persist-ArtifactSet -Prefix "p9" -ReportPath $p9ReportPath -ProjectRootPath $ProjectRoot -ExtraDocs @("docs/COVERAGE_P9_STATUS.md", "docs/COVERAGE_P9_BLINDS_TAGS.md", "docs/COVERAGE_P9_EPISODES.md")
            $p9AnalyzeArgs = @("-B", "sim/oracle/analyze_p9_episode_mismatch.py", "--fixtures-dir", $p9Out)
            $null = Run-Py -Label "P9-analyzer" -PyArgs $p9AnalyzeArgs
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

            if ($RunP10) {
              $null = Run-Py -Label "P10-stake-extract" -PyArgs $p10StakeExtractArgs
              Run-WithRecovery -Label "P10" -PyArgs $p10Args -Url $BaseUrl
              $p10ReportPath = Join-Path $p10Out "report_p10.json"
              $p10Report = Get-Content $p10ReportPath -Raw | ConvertFrom-Json
              Write-Host ("P10 summary: pass={0}/{1} diff_fail={2} oracle_fail={3} gen_fail={4} skipped={5}" -f $p10Report.passed, $p10Report.total, $p10Report.diff_fail, $p10Report.oracle_fail, $p10Report.gen_fail, $p10Report.skipped)
              Write-Host ("P10 report: {0}" -f $p10ReportPath)
              Persist-ArtifactSet -Prefix "p10" -ReportPath $p10ReportPath -ProjectRootPath $ProjectRoot -ExtraDocs @("docs/COVERAGE_P10_STATUS.md", "docs/COVERAGE_P10_EPISODES.md")
              $p10AnalyzeArgs = @("-B", "sim/oracle/analyze_p10_long_episode_mismatch.py", "--fixtures-dir", $p10Out)
              $null = Run-Py -Label "P10-analyzer" -PyArgs $p10AnalyzeArgs
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
  Run-Py -Label "P2b-analyzer" -PyArgs $p2bAnalyzeArgs | Out-Null

  $p2bReportPath = Join-Path $p2bOut "report_p2b.json"
  $p2bReport = Get-Content $p2bReportPath -Raw | ConvertFrom-Json
  Write-Host ("P2b summary: pass={0}/{1} diff_fail={2} oracle_fail={3} gen_fail={4} skipped={5}" -f $p2bReport.passed, $p2bReport.total, $p2bReport.diff_fail, $p2bReport.oracle_fail, $p2bReport.gen_fail, $p2bReport.skipped)
  Write-Host ("P2b report: {0}" -f $p2bReportPath)
  Write-Host ("P2b analyzer: {0}" -f (Join-Path $p2bOut "score_mismatch_table_p2b.md"))
}

if ($GitSync) {
  $gitSyncScript = Join-Path $ProjectRoot "scripts/git_sync.ps1"
  if (-not (Test-Path $gitSyncScript)) {
    throw "[GitSync] missing script: $gitSyncScript"
  }
  Write-Host "[GitSync] running dry-run sync preview"
  & powershell -ExecutionPolicy Bypass -File $gitSyncScript -DryRun:$true
  if ($LASTEXITCODE -ne 0) {
    throw "[GitSync] dry-run failed"
  }
  Write-Host "[GitSync] dry-run complete. To execute push/delete: powershell -ExecutionPolicy Bypass -File scripts/git_sync.ps1 -DryRun:`$false"
}

