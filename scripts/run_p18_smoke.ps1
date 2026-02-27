param(
  [string]$BaseUrl = "http://127.0.0.1:12346",
  [string]$Seed = "AAAAAAA",
  [switch]$RunPerfGateOnly,
  [switch]$IncludeMilestone500,
  [switch]$FailOnPerfGate
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
$safeRunScript = Join-Path $ProjectRoot "scripts/safe_run.ps1"
if (-not (Test-Path $safeRunScript)) { throw "[P18] missing safe_run script: $safeRunScript" }
$safeRunRuns = New-Object System.Collections.Generic.List[object]

function Write-Json([string]$Path, $Obj) {
  $dir = Split-Path -Parent $Path
  if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
  ($Obj | ConvertTo-Json -Depth 16) | Out-File -LiteralPath $Path -Encoding UTF8
}

function Get-StepTimeoutSec([string]$Label, [string[]]$CmdArgs) {
  $joined = (($CmdArgs | ForEach-Object { [string]$_ }) -join " ").ToLowerInvariant()
  if ($joined.Contains("--episodes 500")) { return 1800 }
  return 1200
}

function Run-Step([string]$Label, [string]$Exe, [string[]]$CmdArgs) {
  $filtered = @($CmdArgs | Where-Object { $_ -ne $null -and $_ -ne "" })
  $timeoutSec = Get-StepTimeoutSec -Label $Label -CmdArgs $filtered
  $safeDir = Join-Path $artifactDir "safe_run"
  if (-not (Test-Path $safeDir)) { New-Item -ItemType Directory -Path $safeDir -Force | Out-Null }
  $safeLabel = ($Label -replace "[^A-Za-z0-9._-]", "_")
  $summaryPath = Join-Path $safeDir ("{0}_{1}.summary.json" -f (Get-Date -Format "yyyyMMdd_HHmmss_fff"), $safeLabel)
  $safeArgs = @(
    "-ExecutionPolicy","Bypass",
    "-File",$safeRunScript,
    "-TimeoutSec",$timeoutSec,
    "-NoEcho",
    "-TailLines","120",
    "-SummaryJson",$summaryPath,
    $Exe
  ) + $filtered
  Write-Host "[$Label] via safe_run timeout=${timeoutSec}s :: $Exe $($filtered -join ' ')"
  $out = & powershell @safeArgs 2>&1
  $code = $LASTEXITCODE
  if ($out) { $out | ForEach-Object { Write-Host $_ } }
  $safeRunRuns.Add([ordered]@{ label = $Label; timeout_sec = $timeoutSec; summary = $summaryPath; exit_code = $code })
  if ($code -ne 0) {
    throw "[$Label] failed with exit code $code"
  }
}

function Ensure-Seeds([string]$DerivedDir) {
  if (-not (Test-Path $DerivedDir)) { New-Item -ItemType Directory -Path $DerivedDir -Force | Out-Null }
  $specs = @(
    @{N=20; S=20260224},
    @{N=100; S=20260225},
    @{N=500; S=20260226}
  )
  foreach ($x in $specs) {
    $p = Join-Path $DerivedDir ("eval_seeds_" + $x.N + ".txt")
    if (-not (Test-Path $p)) {
      $rng = [System.Random]::new([int]$x.S)
      $lines = New-Object System.Collections.Generic.List[string]
      for ($i = 0; $i -lt [int]$x.N; $i++) {
        $v = $rng.Next(1, [int]::MaxValue)
        $lines.Add([string]$v)
      }
      $lines | Out-File -LiteralPath $p -Encoding UTF8
      Write-Host "[seeds] created $p"
    }
  }
}

function Find-AnyBestPvModel([string]$Root, [string]$ExcludePath = "") {
  $all = Get-ChildItem -Path (Join-Path $Root "trainer_runs") -Recurse -Filter "best.pt" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending
  $pv = @($all | Where-Object { $_.FullName.ToLowerInvariant().Contains("_pv_") })
  foreach ($c in $pv) {
    if ($ExcludePath -and ((Resolve-Path $c.FullName).Path -eq (Resolve-Path $ExcludePath).Path)) { continue }
    return $c.FullName
  }
  foreach ($c in $all) {
    if ($ExcludePath -and ((Resolve-Path $c.FullName).Path -eq (Resolve-Path $ExcludePath).Path)) { continue }
    return $c.FullName
  }
  return $null
}

function Find-LatestP17Artifact([string]$Root) {
  $p = Join-Path $Root "docs/artifacts/p17"
  if (-not (Test-Path $p)) { return $null }
  $latest = Get-ChildItem -Path $p -Directory -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($latest) { return $latest.FullName }
  return $null
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$artifactDir = Join-Path $ProjectRoot ("docs/artifacts/p18/" + $stamp)
New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
$registryDir = Join-Path $artifactDir "registry"
New-Item -ItemType Directory -Path $registryDir -Force | Out-Null

$py = Join-Path $ProjectRoot ".venv_trainer\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
Ensure-Seeds -DerivedDir (Join-Path $ProjectRoot "balatro_mechanics/derived")

$gateFunctional = @{
  status = "PASS"
  generated_at = (Get-Date).ToString("o")
  artifact_dir = $artifactDir
}

try {
  $daggerHandSamples = 800
  $daggerShopSamples = 240
  $daggerEpochs = 1
  $ablationMaxSteps100 = 120
  $ablationMaxSteps500 = 120

  $p17Latest = Find-LatestP17Artifact -Root $ProjectRoot
  $failureInput = ""
  if ($p17Latest) {
    $cand = Join-Path $p17Latest "failure_mining/failure_buckets_latest.json"
    if (Test-Path $cand) { $failureInput = $cand }
  }

  $champModel = $null
  if ($p17Latest) {
    $cc = Join-Path $p17Latest "registry/current_champion.json"
    if (Test-Path $cc) {
      try {
        $cur = Get-Content $cc -Raw | ConvertFrom-Json
        if ($cur.model_path -and (Test-Path $cur.model_path)) { $champModel = $cur.model_path }
      } catch {}
    }
  }
  if (-not $champModel) {
    $champModel = Find-AnyBestPvModel -Root $ProjectRoot
  }
  if (-not $champModel) {
    throw "[P18] no champion/pv model found"
  }

  $currPlan = Join-Path $artifactDir "curriculum_plan_smoke.json"
  $rlOutDir = Join-Path $ProjectRoot "trainer_runs/p18_rl_smoke"
  $rlArtDir = Join-Path $artifactDir "rl_smoke"
  $rlModel = Join-Path $rlOutDir "best.pt"

  if (-not $RunPerfGateOnly) {
    $currArgs = @("-B","trainer/curriculum_sampler.py","--config","trainer/config/p18_curriculum.yaml","--out",$currPlan,"--mode","smoke")
    if ($failureInput) { $currArgs += @("--failure-input",$failureInput) }
    Run-Step -Label "P18-curriculum" -Exe $py -CmdArgs $currArgs

    Run-Step -Label "P18-rl-train-smoke" -Exe $py -CmdArgs @("-B","trainer/train_rl.py","--config","trainer/config/p18_rl.yaml","--mode","smoke","--curriculum-plan",$currPlan,"--out-dir",$rlOutDir,"--artifacts-dir",$rlArtDir,"--warm-start-model",$champModel)
  }

  if (-not (Test-Path $rlModel)) { $rlModel = $champModel }

  $ablation100 = Join-Path $artifactDir "ablation_100"
  Run-Step -Label "P18-ablation-100" -Exe $py -CmdArgs @(
    "-B","trainer/run_ablation.py",
    "--backend","sim",
    "--stake","gold",
    "--episodes","100",
    "--max-steps-per-episode",$ablationMaxSteps100,
    "--seeds-file","balatro_mechanics/derived/eval_seeds_100.txt",
    "--heuristic",
    "--pv-model",$champModel,
    "--hybrid-model",$champModel,
    "--rl-model",$rlModel,
    "--champion-model",$champModel,
    "--out-dir",$ablation100
  )

  $summaryPath = Join-Path $ablation100 "summary.json"
  if (Test-Path $summaryPath) {
    try {
      $sum = Get-Content $summaryPath -Raw | ConvertFrom-Json
      foreach ($row in $sum.rows) {
        $strat = $row.strategy
        $ep = [int]$row.episodes
        $fb = $row.failure_breakdown
        if ($ep -gt 0 -and $fb -and $null -ne $fb.PSObject.Properties["max_steps_cutoff"]) {
          $cut = [int]$fb.max_steps_cutoff
          if ($cut -ge $ep) {
            Write-Warning "[P18] ablation strategy '$strat' had all $ep episodes end with max_steps_cutoff; consider increasing --max-steps-per-episode."
          }
        }
      }
    } catch {}
  }

  $decision100 = Join-Path $artifactDir "promotion_decision_100.json"
  Run-Step -Label "P18-cc-v2" -Exe $py -CmdArgs @(
    "-B","trainer/champion_manager.py",
    "--rules","trainer/config/p18_champion_rules.yaml",
    "--registry-root",$registryDir,
    "--baseline",(Join-Path $ablation100 "eval_gold_champion.json"),
    "--candidate",(Join-Path $ablation100 "eval_gold_rl.json"),
    "--compare-summary",(Join-Path $ablation100 "summary.json"),
    "--candidate-model",$rlModel,
    "--decision-out",$decision100
  )

  $failureRlDir = Join-Path $artifactDir "failure_mining_rl"
  Run-Step -Label "P18-failure-mine-rl" -Exe $py -CmdArgs @(
    "-B","trainer/failure_mining.py",
    "--episode-logs",(Join-Path $ablation100 "episodes_rl.jsonl"),
    "--baseline-episode-logs",(Join-Path $ablation100 "episodes_champion.jsonl"),
    "--out-dir",$failureRlDir
  )

  $daggerOut = Join-Path $ProjectRoot "trainer_data/p18_dagger_v3.jsonl"
  $daggerSummary = Join-Path $artifactDir ("dagger_v3_summary_" + $stamp + ".json")
  Run-Step -Label "P18-dagger-v3" -Exe $py -CmdArgs @(
    "-B","trainer/dagger_collect.py",
    "--from-failure-buckets",(Join-Path $failureRlDir "failure_buckets_latest.json"),
    "--backend","sim",
    "--out",$daggerOut,
    "--hand-samples",$daggerHandSamples,
    "--shop-samples",$daggerShopSamples,
    "--failure-weight","0.7",
    "--uniform-weight","0.3",
    "--time-budget-ms","15",
    "--summary-out",$daggerSummary
  )
  Run-Step -Label "P18-dagger-dataset" -Exe $py -CmdArgs @("-B","trainer/dataset.py","--path",$daggerOut,"--validate","--summary")
  Run-Step -Label "P18-dagger-train" -Exe $py -CmdArgs @("-B","trainer/train_bc.py","--train-jsonl",$daggerOut,"--epochs",$daggerEpochs,"--batch-size","64","--out-dir",(Join-Path $ProjectRoot "trainer_runs/p18_bc_dagger_v3"))
  Run-Step -Label "P18-dagger-eval" -Exe $py -CmdArgs @("-B","trainer/eval.py","--offline","--model",(Join-Path $ProjectRoot "trainer_runs/p18_bc_dagger_v3/best.pt"),"--dataset",$daggerOut)

  if ($IncludeMilestone500) {
    $ablation500 = Join-Path $artifactDir "ablation_500"
    Run-Step -Label "P18-ablation-500" -Exe $py -CmdArgs @(
      "-B","trainer/run_ablation.py",
      "--backend","sim",
      "--stake","gold",
      "--episodes","500",
      "--max-steps-per-episode",$ablationMaxSteps500,
      "--seeds-file","balatro_mechanics/derived/eval_seeds_500.txt",
      "--champion-model",$champModel,
      "--rl-model",$rlModel,
      "--strategies","champion,rl",
      "--out-dir",$ablation500
    )
  }

  $canaryDir = Join-Path $artifactDir "real_canary_latest"
  try {
    Run-Step -Label "P18-canary" -Exe $py -CmdArgs @("-B","trainer/real_shadow_canary.py","--base-url",$BaseUrl,"--model",$champModel,"--steps","60","--interval","1.0","--topk","3","--out-dir",$canaryDir)
  } catch {
    Write-Json -Path (Join-Path $artifactDir "canary_skip.json") -Obj @{
      status = "SKIP"
      reason = $_.Exception.Message
      generated_at = (Get-Date).ToString("o")
      base_url = $BaseUrl
    }
  }

  Run-Step -Label "P18-reg-dataset-smoke" -Exe $py -CmdArgs @("-B","trainer/registry/datasets.py","--registry-root",$registryDir,"--dataset-id",("p18_smoke_" + $stamp),"--source-type","mixed","--file-path",$daggerOut,"--hand-records",$daggerHandSamples,"--shop-records",$daggerShopSamples,"--invalid-rows","0")
  Run-Step -Label "P18-reg-model-rl" -Exe $py -CmdArgs @("-B","trainer/registry/models.py","--registry-root",$registryDir,"--model-id",("p18_rl_" + $stamp),"--dataset-id",("p18_smoke_" + $stamp),"--model-path",$rlModel,"--decision","candidate","--offline-metrics-json",(Join-Path $rlArtDir "rl_train_summary.json"),"--eval100-json",(Join-Path $ablation100 "eval_gold_rl.json"))

  $decisionPayload = Get-Content $decision100 -Raw | ConvertFrom-Json
  $perfPass = [bool]$decisionPayload.perf_gate_pass
  $riskPass = $true
  if ($decisionPayload.PSObject.Properties.Name -contains "risk_guard_pass") {
    $riskPass = [bool]$decisionPayload.risk_guard_pass
  }
  $finalDecision = [string]$decisionPayload.decision
  if ($decisionPayload.PSObject.Properties.Name -contains "final_decision") {
    $finalDecision = [string]$decisionPayload.final_decision
  }

  $gatePerf = @{
    status = $(if ($perfPass -and $riskPass) { "PASS" } else { "FAIL" })
    perf_gate_pass = $perfPass
    risk_guard_pass = $riskPass
    final_decision = $finalDecision
    reason = [string]$decisionPayload.reason
    decision = $decision100
    generated_at = (Get-Date).ToString("o")
  }
  Write-Json -Path (Join-Path $artifactDir "gate_perf.json") -Obj $gatePerf

  $gateRl = @{
    status = $(if (Test-Path (Join-Path $rlArtDir "rl_train_summary.json")) { "PASS" } else { "FAIL" })
    generated_at = (Get-Date).ToString("o")
    rl_summary = (Join-Path $rlArtDir "rl_train_summary.json")
    rl_model = $rlModel
  }
  Write-Json -Path (Join-Path $artifactDir "gate_rl_smoke.json") -Obj $gateRl

  @(
    "# P18 Perf Gate Summary",
    "",
    "- perf_gate_pass: $($gatePerf.perf_gate_pass)",
    "- risk_guard_pass: $($gatePerf.risk_guard_pass)",
    "- final_decision: $($gatePerf.final_decision)",
    "- reason: $($gatePerf.reason)"
  ) | Out-File -LiteralPath (Join-Path $artifactDir "PERF_GATE_SUMMARY.md") -Encoding UTF8

  $statusPath = Join-Path $ProjectRoot "docs/COVERAGE_P18_STATUS.md"
  @(
    "# P18 Status",
    "",
    "- status: $(if ($gatePerf.status -eq 'PASS') { 'PASS' } else { 'FAIL' })",
    "- updated_at_utc: $((Get-Date).ToUniversalTime().ToString('o'))",
    "- latest_artifact_dir: $artifactDir",
    "- functional_gate: PASS",
    "- perf_gate: $($gatePerf.status)",
    "- rl_smoke_gate: $($gateRl.status)",
    "- decision: $($gatePerf.final_decision)"
  ) | Out-File -LiteralPath $statusPath -Encoding UTF8

  Write-Json -Path (Join-Path $artifactDir "gate_functional.json") -Obj $gateFunctional
  Write-Json -Path (Join-Path $artifactDir "report_p18.json") -Obj @{
    schema = "p18_report_v1"
    generated_at = (Get-Date).ToString("o")
    artifact_dir = $artifactDir
    gate_functional = (Join-Path $artifactDir "gate_functional.json")
    gate_perf = (Join-Path $artifactDir "gate_perf.json")
    gate_rl_smoke = (Join-Path $artifactDir "gate_rl_smoke.json")
    safe_run_runs = @($safeRunRuns.ToArray())
  }

  if (($gatePerf.status -ne "PASS") -and $FailOnPerfGate) {
    throw "[P18] perf/risk gate failed"
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
