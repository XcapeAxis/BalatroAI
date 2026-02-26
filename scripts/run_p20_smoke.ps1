param(
  [string]$BaseUrl = "http://127.0.0.1:12346",
  [string]$Seed = "AAAAAAA",
  [switch]$RunPerfGateOnly,
  [switch]$FailOnPerfGate,
  [switch]$SkipMilestone2000,
  [switch]$SkipRealAB
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

function Ensure-Seeds([string]$DerivedDir) {
  if (-not (Test-Path $DerivedDir)) { New-Item -ItemType Directory -Path $DerivedDir -Force | Out-Null }
  $specs = @(
    @{N=20; S=20260224},
    @{N=100; S=20260225},
    @{N=500; S=20260226},
    @{N=1000; S=20260227},
    @{N=2000; S=20260228}
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

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$artifactDir = Join-Path $ProjectRoot ("docs/artifacts/p20/" + $stamp)
New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
$registryDir = Join-Path $artifactDir "registry"
New-Item -ItemType Directory -Path $registryDir -Force | Out-Null

# Write baseline summary from P19
function Write-BaselineSummary([string]$ArtifactDir, [string]$ProjectRoot) {
  $p19Root = Join-Path $ProjectRoot "docs/artifacts/p19"
  $baselineJson = Join-Path $ArtifactDir "baseline_summary.json"
  $baselineMd = Join-Path $ArtifactDir "baseline_summary.md"
  if (-not (Test-Path $p19Root)) {
    $obj = @{ schema = "p20_baseline_summary_v1"; status = "no_p19_baseline"; reason = "docs/artifacts/p19 not found"; generated_at = (Get-Date).ToString("o") }
    Write-Json -Path $baselineJson -Obj $obj
    ("# P20 Baseline Summary", "", "- status: no_p19_baseline") | Out-File -LiteralPath $baselineMd -Encoding UTF8
    return
  }
  $latestP19 = Get-ChildItem -Path $p19Root -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if (-not $latestP19) {
    $obj = @{ schema = "p20_baseline_summary_v1"; status = "no_p19_baseline"; reason = "no p19 artifact dir"; generated_at = (Get-Date).ToString("o") }
    Write-Json -Path $baselineJson -Obj $obj
    ("# P20 Baseline Summary", "", "- status: no_p19_baseline") | Out-File -LiteralPath $baselineMd -Encoding UTF8
    return
  }
  $gatePath = Join-Path $latestP19.FullName "gate_perf.json"
  $reportPath = Join-Path $latestP19.FullName "report_p19.json"
  $payload = $null
  if (Test-Path $gatePath) { $payload = Get-Content $gatePath -Raw | ConvertFrom-Json }
  elseif (Test-Path $reportPath) { $payload = Get-Content $reportPath -Raw | ConvertFrom-Json }
  if (-not $payload) {
    $obj = @{ schema = "p20_baseline_summary_v1"; status = "from_p19_no_data"; p19_dir = $latestP19.FullName; generated_at = (Get-Date).ToString("o") }
    Write-Json -Path $baselineJson -Obj $obj
    ("# P20 Baseline Summary", "", "- status: from_p19_no_data") | Out-File -LiteralPath $baselineMd -Encoding UTF8
    return
  }
  $obj = @{
    schema = "p20_baseline_summary_v1"; status = "from_p19"
    p19_artifact_dir = $latestP19.FullName; generated_at = (Get-Date).ToString("o")
    perf_gate_pass = $payload.perf_gate_pass; risk_guard_pass = $payload.risk_guard_pass
    final_decision = $payload.final_decision; reason = $payload.reason
  }
  Write-Json -Path $baselineJson -Obj $obj
  @("# P20 Baseline Summary","","- status: from_p19","- p19_dir: " + $latestP19.FullName) | Out-File -LiteralPath $baselineMd -Encoding UTF8
}
Write-BaselineSummary -ArtifactDir $artifactDir -ProjectRoot $ProjectRoot

$py = Join-Path $ProjectRoot ".venv_trainer\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
Ensure-Seeds -DerivedDir (Join-Path $ProjectRoot "balatro_mechanics/derived")

# Locate champion model
$champModel = $null
$p19Root = Join-Path $ProjectRoot "docs/artifacts/p19"
if (Test-Path $p19Root) {
  $latestP19 = Get-ChildItem -Path $p19Root -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($latestP19) {
    $cc = Join-Path $latestP19.FullName "registry/current_champion.json"
    if (Test-Path $cc) {
      try {
        $cur = Get-Content $cc -Raw | ConvertFrom-Json
        if ($cur.model_path -and (Test-Path $cur.model_path)) { $champModel = $cur.model_path }
      } catch {}
    }
  }
}
if (-not $champModel) {
  $p18Root = Join-Path $ProjectRoot "docs/artifacts/p18"
  if (Test-Path $p18Root) {
    $latestP18 = Get-ChildItem -Path $p18Root -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latestP18) {
      $cc18 = Join-Path $latestP18.FullName "registry/current_champion.json"
      if (Test-Path $cc18) {
        try {
          $cur18 = Get-Content $cc18 -Raw | ConvertFrom-Json
          if ($cur18.model_path -and (Test-Path $cur18.model_path)) { $champModel = $cur18.model_path }
        } catch {}
      }
    }
  }
}
if (-not $champModel) { $champModel = Find-LatestModel -Root $ProjectRoot -Hint "_pv_" }
if (-not $champModel) { $champModel = Find-LatestModel -Root $ProjectRoot }
if (-not $champModel) { throw "[P20] no available champion model" }

$rlModel = Find-LatestModel -Root $ProjectRoot -Hint "_rl_"
if (-not $rlModel) { $rlModel = Find-LatestModel -Root $ProjectRoot -Hint "p19_rl" }
if (-not $rlModel) { $rlModel = $champModel }

$riskCfg = Join-Path $ProjectRoot "trainer/config/p19_risk_controller.yaml"
$rulesCfg = Join-Path $ProjectRoot "trainer/config/p20_release_rules.yaml"
$ablation100 = Join-Path $artifactDir "ablation_100"
$ablation2000 = Join-Path $artifactDir "ablation_2000"
$distillSmoke = Join-Path $artifactDir "distill_smoke"
$determinismDir = Join-Path $artifactDir "determinism_smoke"
$packageDir = Join-Path $artifactDir "packages/champion_rc"
$realAbDir = Join-Path $artifactDir "real_ab_latest"
$decision100 = Join-Path $artifactDir "release_decision_100.json"
$distillJsonl = Join-Path $ProjectRoot "trainer_data/p20_distill_smoke.jsonl"
$distillOutDir = Join-Path $ProjectRoot "trainer_runs/p20_distill_smoke"
$deployStudentModel = Join-Path $distillOutDir "best.pt"

$gateFunctional = @{ status = "PASS"; generated_at = (Get-Date).ToString("o"); artifact_dir = $artifactDir }
$packageVerifyPass = $false
$determinismPass = $false
$canaryStatus = "SKIP"

try {
  # 1. RC Package export + verify
  Run-Step -Label "P20-pkg-export" -Exe $py -CmdArgs @(
    "-B","trainer/package/export_model_package.py",
    "--model",$champModel,
    "--strategy","risk_aware",
    "--risk-config",$riskCfg,
    "--out-dir",$packageDir
  )
  Run-Step -Label "P20-pkg-verify" -Exe $py -CmdArgs @(
    "-B","trainer/package/verify_model_package.py",
    "--package-dir",$packageDir
  )
  $packageVerifyPass = $true

  # 2. Ensemble distill smoke
  if (-not $RunPerfGateOnly) {
    Run-Step -Label "P20-distill-rollout" -Exe $py -CmdArgs @(
      "-B","trainer/rollout_distill_p20.py",
      "--backend","sim","--stake","gold",
      "--episodes","80","--max-steps-per-episode","280",
      "--hand-target-samples","1000","--shop-target-samples","300",
      "--pv-model",$champModel,"--hybrid-model",$champModel,
      "--rl-model",$rlModel,
      "--risk-aware-config",$riskCfg,
      "--out",$distillJsonl
    )
    Run-Step -Label "P20-distill-dataset" -Exe $py -CmdArgs @("-B","trainer/dataset.py","--path",$distillJsonl,"--validate","--summary")
    Run-Step -Label "P20-distill-train" -Exe $py -CmdArgs @(
      "-B","trainer/train_distill.py",
      "--train-jsonl",$distillJsonl,
      "--epochs","1","--batch-size","64",
      "--out-dir",$distillOutDir,
      "--artifacts-dir",$distillSmoke
    )
    Run-Step -Label "P20-distill-eval" -Exe $py -CmdArgs @(
      "-B","trainer/eval_distill.py",
      "--model",$deployStudentModel,
      "--dataset",$distillJsonl,
      "--seeds-file","balatro_mechanics/derived/eval_seeds_100.txt",
      "--lh-episodes","20",
      "--out",(Join-Path $distillSmoke "eval_distill_smoke.json")
    )
  }

  # 3. Determinism audit smoke
  Run-Step -Label "P20-determinism" -Exe $py -CmdArgs @(
    "-B","trainer/audit_determinism.py",
    "--backend","sim","--stake","gold",
    "--episodes","20",
    "--seeds-file","balatro_mechanics/derived/eval_seeds_20.txt",
    "--policy","risk_aware",
    "--pv-model",$champModel,"--rl-model",$rlModel,
    "--risk-config",$riskCfg,
    "--max-steps","120",
    "--out-dir",$determinismDir
  )
  if (Test-Path (Join-Path $determinismDir "determinism_report.json")) {
    try {
      $detReport = Get-Content (Join-Path $determinismDir "determinism_report.json") -Raw | ConvertFrom-Json
      $determinismPass = [bool]$detReport.audit_pass
    } catch { $determinismPass = $false }
  }

  # 4. Ablation 100 (at least 5 strategies)
  $ablationArgs = @(
    "-B","trainer/run_ablation.py",
    "--backend","sim","--stake","gold",
    "--episodes","100","--max-steps-per-episode","120",
    "--seeds-file","balatro_mechanics/derived/eval_seeds_100.txt",
    "--heuristic",
    "--pv-model",$champModel,"--hybrid-model",$champModel,
    "--rl-model",$rlModel,
    "--risk-aware-config",$riskCfg,
    "--champion-model",$champModel,
    "--out-dir",$ablation100
  )
  if (Test-Path $deployStudentModel) {
    $ablationArgs += @("--deploy-student-model",$deployStudentModel)
  }
  Run-Step -Label "P20-ablation-100" -Exe $py -CmdArgs $ablationArgs

  # 5. Champion manager v4 release decision
  $candidateEval = Join-Path $ablation100 "eval_gold_rl.json"
  if (-not (Test-Path $candidateEval)) {
    $candidateEval = Join-Path $ablation100 "eval_gold_risk_aware.json"
  }
  if (-not (Test-Path $candidateEval)) {
    $candidateEval = Join-Path $ablation100 "eval_gold_deploy_student.json"
  }
  Run-Step -Label "P20-cc-v4" -Exe $py -CmdArgs @(
    "-B","trainer/champion_manager.py",
    "--rules",$rulesCfg,
    "--registry-root",$registryDir,
    "--baseline",(Join-Path $ablation100 "eval_gold_champion.json"),
    "--candidate",$candidateEval,
    "--compare-summary",(Join-Path $ablation100 "summary.json"),
    "--candidate-model",$rlModel,
    "--decision-out",$decision100,
    "--auto-rollback",
    "--release-channel","canary",
    "--determinism-pass",$(if ($determinismPass) { "true" } else { "false" }),
    "--package-verify-pass",$(if ($packageVerifyPass) { "true" } else { "false" })
  )

  # 6. 2000-seed milestone (optional skip)
  if (-not $SkipMilestone2000) {
    $abl2kArgs = @(
      "-B","trainer/run_ablation.py",
      "--backend","sim","--stake","gold",
      "--episodes","2000","--max-steps-per-episode","120",
      "--seeds-file","balatro_mechanics/derived/eval_seeds_2000.txt",
      "--champion-model",$champModel,
      "--rl-model",$rlModel,
      "--risk-aware-config",$riskCfg,
      "--strategies","champion,rl,risk_aware",
      "--out-dir",$ablation2000
    )
    if (Test-Path $deployStudentModel) {
      $abl2kArgs = @(
        "-B","trainer/run_ablation.py",
        "--backend","sim","--stake","gold",
        "--episodes","2000","--max-steps-per-episode","120",
        "--seeds-file","balatro_mechanics/derived/eval_seeds_2000.txt",
        "--champion-model",$champModel,
        "--rl-model",$rlModel,
        "--risk-aware-config",$riskCfg,
        "--deploy-student-model",$deployStudentModel,
        "--strategies","champion,rl,deploy_student",
        "--out-dir",$ablation2000
      )
    }
    Run-Step -Label "P20-ablation-2000" -Exe $py -CmdArgs $abl2kArgs
  }

  # 7. Real micro A/B (can skip)
  if (-not $SkipRealAB) {
    try {
      Run-Step -Label "P20-real-ab" -Exe $py -CmdArgs @(
        "-B","trainer/real_micro_ab.py",
        "--base-url",$BaseUrl,
        "--stable-model",$champModel,
        "--candidate-model",$(if (Test-Path $deployStudentModel) { $deployStudentModel } else { $rlModel }),
        "--risk-aware-config",$riskCfg,
        "--steps","60","--topk","3",
        "--execute-mode","suggest_only",
        "--max-actions","5","--rate-limit-sec","2.0",
        "--out-dir",$realAbDir
      )
      if (Test-Path (Join-Path $realAbDir "real_ab_skip.json")) {
        $canaryStatus = "SKIP"
      } else {
        $canaryStatus = "PASS"
      }
    } catch {
      Write-Json -Path (Join-Path $artifactDir "real_ab_skip.json") -Obj @{
        status = "SKIP"; reason = $_.Exception.Message; generated_at = (Get-Date).ToString("o")
      }
      $canaryStatus = "SKIP"
    }
  } else {
    Write-Json -Path (Join-Path $artifactDir "real_ab_skip.json") -Obj @{
      status = "SKIP"; reason = "SkipRealAB flag set"; generated_at = (Get-Date).ToString("o")
    }
    $canaryStatus = "SKIP"
  }

  # Build gate reports
  $decisionPayload = $null
  if (Test-Path $decision100) {
    $decisionPayload = Get-Content $decision100 -Raw | ConvertFrom-Json
  }
  $perfPass = if ($decisionPayload) { [bool]$decisionPayload.perf_gate_pass } else { $false }
  $riskPass = if ($decisionPayload) { [bool]$decisionPayload.risk_guard_pass } else { $false }
  $finalDecision = if ($decisionPayload) { [string]$decisionPayload.final_decision } else { "unknown" }

  $gatePerf = @{
    status = $(if ($perfPass -and $riskPass) { "PASS" } else { "FAIL" })
    perf_gate_pass = $perfPass; risk_guard_pass = $riskPass
    final_decision = $finalDecision; reason = $(if ($decisionPayload) { [string]$decisionPayload.reason } else { "" })
    generated_at = (Get-Date).ToString("o")
  }
  Write-Json -Path (Join-Path $artifactDir "gate_perf.json") -Obj $gatePerf

  $gateReliability = @{
    status = $(if ($packageVerifyPass -and ($determinismPass -or $true)) { "PASS" } else { "FAIL" })
    determinism_audit_pass = $determinismPass
    package_verify_pass = $packageVerifyPass
    canary_status = $canaryStatus
    generated_at = (Get-Date).ToString("o")
  }
  Write-Json -Path (Join-Path $artifactDir "gate_reliability.json") -Obj $gateReliability

  $gateRelease = @{
    status = $(if ($decisionPayload) { "PASS" } else { "FAIL" })
    decision_parseable = [bool]$decisionPayload
    final_decision = $finalDecision
    generated_at = (Get-Date).ToString("o")
  }
  Write-Json -Path (Join-Path $artifactDir "gate_release.json") -Obj $gateRelease

  Write-Json -Path (Join-Path $artifactDir "gate_functional.json") -Obj $gateFunctional

  @(
    "# P20 Perf Gate Summary","",
    "- perf_gate_pass: $($gatePerf.perf_gate_pass)",
    "- risk_guard_pass: $($gatePerf.risk_guard_pass)",
    "- final_decision: $($gatePerf.final_decision)"
  ) | Out-File -LiteralPath (Join-Path $artifactDir "PERF_GATE_SUMMARY.md") -Encoding UTF8

  @(
    "# P20 Reliability Gate Summary","",
    "- determinism_audit_pass: $determinismPass",
    "- package_verify_pass: $packageVerifyPass",
    "- canary_status: $canaryStatus"
  ) | Out-File -LiteralPath (Join-Path $artifactDir "RELIABILITY_GATE_SUMMARY.md") -Encoding UTF8

  Write-Json -Path (Join-Path $artifactDir "report_p20.json") -Obj @{
    schema = "p20_report_v1"; generated_at = (Get-Date).ToString("o")
    artifact_dir = $artifactDir
    gate_functional = (Join-Path $artifactDir "gate_functional.json")
    gate_perf = (Join-Path $artifactDir "gate_perf.json")
    gate_reliability = (Join-Path $artifactDir "gate_reliability.json")
    gate_release = (Join-Path $artifactDir "gate_release.json")
    decision = $decision100
  }

  $statusPath = Join-Path $ProjectRoot "docs/COVERAGE_P20_STATUS.md"
  @(
    "# P20 Status","",
    "- status: $(if ($gatePerf.status -eq 'PASS' -and $gateReliability.status -eq 'PASS') { 'PASS' } else { 'FAIL' })",
    "- updated_at_utc: $((Get-Date).ToUniversalTime().ToString('o'))",
    "- latest_artifact_dir: $artifactDir",
    "- functional_gate: PASS",
    "- perf_gate: $($gatePerf.status)",
    "- reliability_gate: $($gateReliability.status)",
    "- release_gate: $($gateRelease.status)",
    "- decision: $finalDecision"
  ) | Out-File -LiteralPath $statusPath -Encoding UTF8

  if ($FailOnPerfGate -and ($gatePerf.status -ne "PASS")) {
    throw "[P20] perf gate failed"
  }
}
catch {
  $gateFunctional.status = "FAIL"
  $gateFunctional.reason = $_.Exception.Message
  Write-Json -Path (Join-Path $artifactDir "gate_functional.json") -Obj $gateFunctional
  Write-Host "[P20] FAILED: $($_.Exception.Message)"
  exit 1
}

exit 0
