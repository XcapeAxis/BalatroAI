param(
  [string]$BaseUrl = "http://127.0.0.1:12346",
  [string]$Seed = "AAAAAAA",
  [switch]$RunPerfGateOnly,
  [switch]$IncludeMilestone1000,
  [switch]$SkipMilestone1000,
  [switch]$FailOnPerfGate
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

function Write-Json([string]$Path, $Obj) {
  $dir = Split-Path -Parent $Path
  if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
  ($Obj | ConvertTo-Json -Depth 16) | Out-File -LiteralPath $Path -Encoding UTF8
}

function Run-Step([string]$Label, [string]$Exe, [string[]]$CmdArgs) {
  $filtered = @($CmdArgs | Where-Object { $_ -ne $null -and $_ -ne "" })
  Write-Host "[$Label] $Exe $($filtered -join ' ')"
  $out = & $Exe @filtered 2>&1
  $code = $LASTEXITCODE
  if ($out) { $out | ForEach-Object { Write-Host $_ } }
  if ($code -ne 0) { throw "[$Label] failed with exit code $code" }
}

function Ensure-Seeds([string]$DerivedDir) {
  if (-not (Test-Path $DerivedDir)) { New-Item -ItemType Directory -Path $DerivedDir -Force | Out-Null }
  $specs = @(
    @{N=20; S=20260224},
    @{N=100; S=20260225},
    @{N=500; S=20260226},
    @{N=1000; S=20260227}
  )
  foreach ($x in $specs) {
    $p = Join-Path $DerivedDir ("eval_seeds_" + $x.N + ".txt")
    if (-not (Test-Path $p)) {
      $rng = [System.Random]::new([int]$x.S)
      $lines = New-Object System.Collections.Generic.List[string]
      for ($i = 0; $i -lt [int]$x.N; $i++) {
        $lines.Add([string]$rng.Next(1, [int]::MaxValue))
      }
      $lines | Out-File -LiteralPath $p -Encoding UTF8
      Write-Host "[seeds] created $p"
    }
  }
}

function Find-LatestModel([string]$Root, [string]$Hint = "") {
  $runs = Join-Path $Root "trainer_runs"
  if (-not (Test-Path $runs)) { return $null }
  $all = Get-ChildItem -Path $runs -Recurse -Filter "best.pt" -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending
  if ($Hint) {
    $hit = $all | Where-Object { $_.FullName.ToLowerInvariant().Contains($Hint.ToLowerInvariant()) } | Select-Object -First 1
    if ($hit) { return $hit.FullName }
  }
  $first = $all | Select-Object -First 1
  if ($first) { return $first.FullName }
  return $null
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$artifactDir = Join-Path $ProjectRoot ("docs/artifacts/p19/" + $stamp)
New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
$registryDir = Join-Path $artifactDir "registry"
New-Item -ItemType Directory -Path $registryDir -Force | Out-Null

function Write-BaselineSummary([string]$ArtifactDir, [string]$ProjectRoot) {
  $p18Root = Join-Path $ProjectRoot "docs/artifacts/p18"
  $baselineJson = Join-Path $ArtifactDir "baseline_summary.json"
  $baselineMd = Join-Path $ArtifactDir "baseline_summary.md"
  if (-not (Test-Path $p18Root)) {
    $obj = @{ schema = "p19_baseline_summary_v1"; status = "no_p18_baseline"; reason = "docs/artifacts/p18 not found"; generated_at = (Get-Date).ToString("o") }
    Write-Json -Path $baselineJson -Obj $obj
    ("# P19 Baseline Summary", "", "- status: no_p18_baseline", "- reason: docs/artifacts/p18 not found") | Out-File -LiteralPath $baselineMd -Encoding UTF8
    return
  }
  $latestP18 = Get-ChildItem -Path $p18Root -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if (-not $latestP18) {
    $obj = @{ schema = "p19_baseline_summary_v1"; status = "no_p18_baseline"; reason = "no p18 artifact dir"; generated_at = (Get-Date).ToString("o") }
    Write-Json -Path $baselineJson -Obj $obj
    ("# P19 Baseline Summary", "", "- status: no_p18_baseline", "- reason: no p18 artifact dir") | Out-File -LiteralPath $baselineMd -Encoding UTF8
    return
  }
  $reportPath = Join-Path $latestP18.FullName "report_p18.json"
  $gatePath = Join-Path $latestP18.FullName "gate_perf.json"
  $payload = $null
  if (Test-Path $gatePath) { $payload = Get-Content $gatePath -Raw | ConvertFrom-Json }
  elseif (Test-Path $reportPath) { $payload = Get-Content $reportPath -Raw | ConvertFrom-Json }
  if (-not $payload) {
    $obj = @{ schema = "p19_baseline_summary_v1"; status = "no_p18_baseline"; reason = "no report_p18 or gate_perf in latest p18"; p18_dir = $latestP18.FullName; generated_at = (Get-Date).ToString("o") }
    Write-Json -Path $baselineJson -Obj $obj
    ("# P19 Baseline Summary", "", "- status: no_p18_baseline", "- p18_dir: " + $latestP18.FullName) | Out-File -LiteralPath $baselineMd -Encoding UTF8
    return
  }
  $obj = @{
    schema = "p19_baseline_summary_v1"
    status = "from_p18"
    p18_artifact_dir = $latestP18.FullName
    generated_at = (Get-Date).ToString("o")
    perf_gate_pass = $payload.perf_gate_pass
    risk_guard_pass = $payload.risk_guard_pass
    final_decision = $payload.final_decision
    reason = $payload.reason
  }
  Write-Json -Path $baselineJson -Obj $obj
  $md = @(
    "# P19 Baseline Summary",
    "",
    "- status: from_p18",
    "- p18_artifact_dir: " + $latestP18.FullName,
    "- perf_gate_pass: " + $payload.perf_gate_pass,
    "- risk_guard_pass: " + $payload.risk_guard_pass,
    "- final_decision: " + $payload.final_decision,
    "- reason: " + $payload.reason
  )
  $md | Out-File -LiteralPath $baselineMd -Encoding UTF8
}
Write-BaselineSummary -ArtifactDir $artifactDir -ProjectRoot $ProjectRoot

$py = Join-Path $ProjectRoot ".venv_trainer\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
Ensure-Seeds -DerivedDir (Join-Path $ProjectRoot "balatro_mechanics/derived")

$champModel = $null
$p18Root = Join-Path $ProjectRoot "docs/artifacts/p18"
if (Test-Path $p18Root) {
  $latestP18 = Get-ChildItem -Path $p18Root -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($latestP18) {
    $cc = Join-Path $latestP18.FullName "registry/current_champion.json"
    if (Test-Path $cc) {
      try {
        $cur = Get-Content $cc -Raw | ConvertFrom-Json
        if ($cur.model_path -and (Test-Path $cur.model_path)) { $champModel = $cur.model_path }
      } catch {}
    }
  }
}
if (-not $champModel) { $champModel = Find-LatestModel -Root $ProjectRoot -Hint "_pv_" }
if (-not $champModel) { $champModel = Find-LatestModel -Root $ProjectRoot }
if (-not $champModel) { throw "[P19] no available champion/pv model" }

$rlOutDir = Join-Path $ProjectRoot "trainer_runs/p19_rl_smoke"
$rlArtDir = Join-Path $artifactDir "rl_smoke"
$rlModel = Join-Path $rlOutDir "best.pt"
$riskCfg = Join-Path $ProjectRoot "trainer/config/p19_risk_controller.yaml"
$rulesCfg = Join-Path $ProjectRoot "trainer/config/p19_champion_rules.yaml"
$calibDir = Join-Path $artifactDir "calibration_smoke"
$ablation100 = Join-Path $artifactDir "ablation_100"
$ablation1000 = Join-Path $artifactDir "ablation_1000"
$failureDir = Join-Path $artifactDir "failure_mining_rl"
$decision100 = Join-Path $artifactDir "promotion_decision_100.json"
$daggerOut = Join-Path $ProjectRoot "trainer_data/p19_dagger_v4.jsonl"
$daggerSummary = Join-Path $artifactDir ("dagger_v4_summary_" + $stamp + ".json")
$canaryDir = Join-Path $artifactDir "real_canary_latest"

$gateFunctional = @{
  status = "PASS"
  generated_at = (Get-Date).ToString("o")
  artifact_dir = $artifactDir
}
$riskSmokePass = $false
$calibrationPass = $false
$rollbackSmokePass = $false
$canaryStatus = "SKIP"

try {
  if (-not $RunPerfGateOnly) {
    $currPlan = Join-Path $artifactDir "curriculum_plan_smoke.json"
    $failureInput = Join-Path $ProjectRoot "docs/artifacts/p18/failure_mining_rl/failure_buckets_latest.json"
    if (-not (Test-Path $failureInput)) { $failureInput = "" }
    $currArgs = @("-B","trainer/curriculum_sampler.py","--config","trainer/config/p18_curriculum.yaml","--out",$currPlan,"--mode","smoke")
    if ($failureInput) { $currArgs += @("--failure-input",$failureInput) }
    Run-Step -Label "P19-curriculum" -Exe $py -CmdArgs $currArgs

    Run-Step -Label "P19-rl-train-smoke" -Exe $py -CmdArgs @(
      "-B","trainer/train_rl.py",
      "--config","trainer/config/p18_rl.yaml",
      "--mode","smoke",
      "--curriculum-plan",$currPlan,
      "--out-dir",$rlOutDir,
      "--artifacts-dir",$rlArtDir,
      "--warm-start-model",$champModel
    )
  }

  if (-not (Test-Path $rlModel)) { $rlModel = $champModel }

  Run-Step -Label "P19-risk-infer" -Exe $py -CmdArgs @(
    "-B","trainer/infer_assistant.py",
    "--backend","sim",
    "--policy","risk_aware",
    "--model",$champModel,
    "--rl-model",$rlModel,
    "--risk-config",$riskCfg,
    "--search-budget-ms","10",
    "--topk","5",
    "--once"
  )
  $riskSmokePass = $true

  Run-Step -Label "P19-calibration" -Exe $py -CmdArgs @(
    "-B","trainer/eval_calibration.py",
    "--backend","sim",
    "--stake","gold",
    "--episodes","20",
    "--seeds-file","balatro_mechanics/derived/eval_seeds_100.txt",
    "--pv-model",$champModel,
    "--rl-model",$rlModel,
    "--out-dir",$calibDir
  )
  $calibrationPass = $true

  Run-Step -Label "P19-ablation-100" -Exe $py -CmdArgs @(
    "-B","trainer/run_ablation.py",
    "--backend","sim",
    "--stake","gold",
    "--episodes","100",
    "--max-steps-per-episode","90",
    "--seeds-file","balatro_mechanics/derived/eval_seeds_100.txt",
    "--heuristic",
    "--pv-model",$champModel,
    "--hybrid-model",$champModel,
    "--rl-model",$rlModel,
    "--risk-aware-config",$riskCfg,
    "--champion-model",$champModel,
    "--out-dir",$ablation100
  )

  Run-Step -Label "P19-cc-v3" -Exe $py -CmdArgs @(
    "-B","trainer/champion_manager.py",
    "--rules",$rulesCfg,
    "--registry-root",$registryDir,
    "--baseline",(Join-Path $ablation100 "eval_gold_champion.json"),
    "--candidate",(Join-Path $ablation100 "eval_gold_rl.json"),
    "--compare-summary",(Join-Path $ablation100 "summary.json"),
    "--candidate-model",$rlModel,
    "--decision-out",$decision100,
    "--auto-rollback"
  )
  $rollbackSmokePass = $true

  Run-Step -Label "P19-failure-mine-rl" -Exe $py -CmdArgs @(
    "-B","trainer/failure_mining.py",
    "--episode-logs",(Join-Path $ablation100 "episodes_rl.jsonl"),
    "--baseline-episode-logs",(Join-Path $ablation100 "episodes_champion.jsonl"),
    "--out-dir",$failureDir
  )
  Run-Step -Label "P19-dagger-v4" -Exe $py -CmdArgs @(
    "-B","trainer/dagger_collect_v4.py",
    "--from-failure-buckets",(Join-Path $failureDir "failure_buckets_latest.json"),
    "--backend","sim",
    "--out",$daggerOut,
    "--hand-samples","1000",
    "--shop-samples","400",
    "--failure-weight","0.7",
    "--uniform-weight","0.3",
    "--source-policies","rl,risk_aware,pv",
    "--time-budget-ms","20",
    "--summary-out",$daggerSummary
  )
  Run-Step -Label "P19-dagger-dataset" -Exe $py -CmdArgs @("-B","trainer/dataset.py","--path",$daggerOut,"--validate","--summary")

  if (-not $SkipMilestone1000) {
    Run-Step -Label "P19-ablation-1000" -Exe $py -CmdArgs @(
      "-B","trainer/run_ablation.py",
      "--backend","sim",
      "--stake","gold",
      "--episodes","1000",
      "--max-steps-per-episode","120",
      "--seeds-file","balatro_mechanics/derived/eval_seeds_1000.txt",
      "--champion-model",$champModel,
      "--rl-model",$rlModel,
      "--risk-aware-config",$riskCfg,
      "--strategies","champion,rl,risk_aware",
      "--out-dir",$ablation1000
    )
  }

  try {
    Run-Step -Label "P19-canary" -Exe $py -CmdArgs @(
      "-B","trainer/real_shadow_canary.py",
      "--base-url",$BaseUrl,
      "--model",$champModel,
      "--models",("pv=" + $champModel + ",hybrid=" + $champModel + ",rl=" + $rlModel),
      "--risk-aware-config",$riskCfg,
      "--steps","120",
      "--interval","1.0",
      "--topk","3",
      "--out-dir",$canaryDir
    )
    $canaryStatus = "PASS"
  } catch {
    Write-Json -Path (Join-Path $artifactDir "canary_skip.json") -Obj @{
      status = "SKIP"
      reason = $_.Exception.Message
      generated_at = (Get-Date).ToString("o")
      base_url = $BaseUrl
    }
    $canaryStatus = "SKIP"
  }

  $decisionPayload = Get-Content $decision100 -Raw | ConvertFrom-Json
  $perfPass = [bool]$decisionPayload.perf_gate_pass
  $riskPass = [bool]$decisionPayload.risk_guard_pass
  $finalDecision = [string]$decisionPayload.final_decision

  $gatePerf = @{
    status = $(if ($perfPass -and $riskPass) { "PASS" } else { "FAIL" })
    perf_gate_pass = $perfPass
    risk_guard_pass = $riskPass
    final_decision = $finalDecision
    reason = [string]$decisionPayload.reason
    generated_at = (Get-Date).ToString("o")
  }
  Write-Json -Path (Join-Path $artifactDir "gate_perf.json") -Obj $gatePerf

  $gateRisk = @{
    status = $(if ($riskSmokePass -and $calibrationPass -and $rollbackSmokePass) { "PASS" } else { "FAIL" })
    risk_controller_smoke_pass = $riskSmokePass
    calibration_smoke_pass = $calibrationPass
    rollback_smoke_pass = $rollbackSmokePass
    canary_status = $canaryStatus
    generated_at = (Get-Date).ToString("o")
  }
  Write-Json -Path (Join-Path $artifactDir "gate_risk.json") -Obj $gateRisk

  @(
    "# P19 Perf Gate Summary",
    "",
    "- perf_gate_pass: $($gatePerf.perf_gate_pass)",
    "- risk_guard_pass: $($gatePerf.risk_guard_pass)",
    "- final_decision: $($gatePerf.final_decision)",
    "- reason: $($gatePerf.reason)"
  ) | Out-File -LiteralPath (Join-Path $artifactDir "PERF_GATE_SUMMARY.md") -Encoding UTF8

  Write-Json -Path (Join-Path $artifactDir "gate_functional.json") -Obj $gateFunctional
  Write-Json -Path (Join-Path $artifactDir "report_p19.json") -Obj @{
    schema = "p19_report_v1"
    generated_at = (Get-Date).ToString("o")
    artifact_dir = $artifactDir
    gate_functional = (Join-Path $artifactDir "gate_functional.json")
    gate_perf = (Join-Path $artifactDir "gate_perf.json")
    gate_risk = (Join-Path $artifactDir "gate_risk.json")
    decision = $decision100
  }

  $statusPath = Join-Path $ProjectRoot "docs/COVERAGE_P19_STATUS.md"
  @(
    "# P19 Status",
    "",
    "- status: $(if ($gatePerf.status -eq 'PASS' -and $gateRisk.status -eq 'PASS') { 'PASS' } else { 'FAIL' })",
    "- updated_at_utc: $((Get-Date).ToUniversalTime().ToString('o'))",
    "- latest_artifact_dir: $artifactDir",
    "- functional_gate: PASS",
    "- perf_gate: $($gatePerf.status)",
    "- risk_gate: $($gateRisk.status)",
    "- decision: $($gatePerf.final_decision)"
  ) | Out-File -LiteralPath $statusPath -Encoding UTF8

  if (($gatePerf.status -ne "PASS") -and $FailOnPerfGate) {
    throw "[P19] perf gate failed"
  }
}
catch {
  $gateFunctional.status = "FAIL"
  $gateFunctional.reason = $_.Exception.Message
  Write-Json -Path (Join-Path $artifactDir "gate_functional.json") -Obj $gateFunctional
  Write-Host $_.Exception.Message
  exit 1
}

exit 0
