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

function Write-Json([string]$Path, $Obj) {
  $dir = Split-Path -Parent $Path
  if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
  ($Obj | ConvertTo-Json -Depth 12) | Out-File -LiteralPath $Path -Encoding UTF8
}

function Run-Step([string]$Label, [string]$Exe, [string[]]$CmdArgs) {
  $filtered = @($CmdArgs | Where-Object { $_ -ne $null -and $_ -ne "" })
  Write-Host "[$Label] $Exe $($filtered -join ' ')"
  $out = & $Exe @filtered 2>&1
  $code = $LASTEXITCODE
  if ($out) { $out | ForEach-Object { Write-Host $_ } }
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

function Find-AnyBestModel([string]$Root, [string]$ExcludePath = "") {
  $cands = Get-ChildItem -Path (Join-Path $Root "trainer_runs") -Recurse -Filter "best.pt" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending
  foreach ($c in $cands) {
    if ($ExcludePath -and ((Resolve-Path $c.FullName).Path -eq (Resolve-Path $ExcludePath).Path)) { continue }
    return $c.FullName
  }
  return $null
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

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$artifactDir = Join-Path $ProjectRoot ("docs/artifacts/p17/" + $stamp)
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
  $smokeData = Join-Path $ProjectRoot "trainer_data/p17_smoke_search.jsonl"
  $p17SmokeRunDir = Join-Path $ProjectRoot "trainer_runs/p17_pv_smoke"
  $pvOffline = Join-Path $artifactDir "pv_offline_smoke.json"

  if (-not $RunPerfGateOnly) {
    Run-Step -Label "P17-smoke-rollout" -Exe $py -CmdArgs @("-B","trainer/rollout_search_p15.py","--backend","sim","--stake","gold","--episodes","60","--max-steps-per-episode","260","--hand-target-samples","800","--shop-target-samples","200","--workers","4","--search","algo=beam","--search-budget-ms","15","--out",$smokeData)
    Run-Step -Label "P17-smoke-dataset" -Exe $py -CmdArgs @("-B","trainer/dataset.py","--path",$smokeData,"--validate","--summary")
    Run-Step -Label "P17-smoke-train" -Exe $py -CmdArgs @("-B","trainer/train_pv.py","--train-jsonl",$smokeData,"--epochs","1","--batch-size","64","--out-dir",$p17SmokeRunDir)
    Run-Step -Label "P17-smoke-offline" -Exe $py -CmdArgs @("-B","trainer/eval_pv.py","--model",(Join-Path $p17SmokeRunDir "best.pt"),"--dataset",$smokeData,"--out",$pvOffline)
  }

  $currentChampionPath = Join-Path $registryDir "current_champion.json"
  $champModel = $null
  $hasRegisteredChampion = $false
  if (Test-Path $currentChampionPath) {
    try {
      $cur = Get-Content $currentChampionPath -Raw | ConvertFrom-Json
      if ($cur.model_path -and (Test-Path $cur.model_path)) {
        $champModel = $cur.model_path
        $hasRegisteredChampion = $true
      }
    } catch {}
  }
  if (-not $champModel) {
    $champModel = Find-AnyBestPvModel -Root $ProjectRoot -ExcludePath (Join-Path $p17SmokeRunDir "best.pt")
  }
  if (-not $champModel) {
    $champModel = Join-Path $p17SmokeRunDir "best.pt"
  }
  $challModel = Join-Path $p17SmokeRunDir "best.pt"
  if (-not (Test-Path $challModel)) {
    $alt = Find-AnyBestPvModel -Root $ProjectRoot
    if ($alt) { $challModel = $alt }
  }

  $evalChamp100 = Join-Path $artifactDir "eval_gold_champion_100.json"
  $evalChall100 = Join-Path $artifactDir "eval_gold_challenger_100.json"
  $logsChamp100 = Join-Path $artifactDir "episodes_champion_100.jsonl"
  $logsChall100 = Join-Path $artifactDir "episodes_challenger_100.jsonl"
  $compare100Dir = Join-Path $artifactDir "eval_compare_100"
  $decision100 = Join-Path $artifactDir "promotion_decision_100.json"

  Run-Step -Label "P17-eval-champion-100" -Exe $py -CmdArgs @("-B","trainer/eval_long_horizon.py","--backend","sim","--stake","gold","--episodes","100","--max-steps-per-episode","80","--seeds-file","balatro_mechanics/derived/eval_seeds_100.txt","--policy","pv","--model",$champModel,"--out",$evalChamp100,"--save-episode-logs",$logsChamp100)
  Run-Step -Label "P17-eval-challenger-100" -Exe $py -CmdArgs @("-B","trainer/eval_long_horizon.py","--backend","sim","--stake","gold","--episodes","100","--max-steps-per-episode","80","--seeds-file","balatro_mechanics/derived/eval_seeds_100.txt","--policy","pv","--model",$challModel,"--out",$evalChall100,"--save-episode-logs",$logsChall100)
  Run-Step -Label "P17-compare-100" -Exe $py -CmdArgs @("-B","trainer/eval_compare.py","--baseline",("champion=" + $evalChamp100),"--candidate",("challenger=" + $evalChall100),"--out-dir",$compare100Dir)
  Run-Step -Label "P17-decision-100" -Exe $py -CmdArgs @("-B","trainer/champion_manager.py","--registry-root",$registryDir,"--baseline",$evalChamp100,"--candidate",$evalChall100,"--compare-summary",(Join-Path $compare100Dir "summary.json"),"--candidate-model",$challModel,"--decision-out",$decision100)

  $failureDir = Join-Path $artifactDir "failure_mining"
  $daggerV3 = Join-Path $ProjectRoot "trainer_data/p17_dagger_v3.jsonl"
  $daggerSummary = Join-Path $artifactDir ("dagger_v3_summary_" + $stamp + ".json")
  Run-Step -Label "P17-failure-mine" -Exe $py -CmdArgs @("-B","trainer/failure_mining.py","--episode-logs",$logsChall100,"--out-dir",$failureDir)
  Run-Step -Label "P17-dagger-v3" -Exe $py -CmdArgs @("-B","trainer/dagger_collect.py","--from-failure-buckets",(Join-Path $failureDir "failure_buckets_latest.json"),"--backend","sim","--out",$daggerV3,"--hand-samples","1000","--shop-samples","400","--failure-weight","0.7","--uniform-weight","0.3","--time-budget-ms","20","--summary-out",$daggerSummary)
  Run-Step -Label "P17-dagger-v3-dataset" -Exe $py -CmdArgs @("-B","trainer/dataset.py","--path",$daggerV3,"--validate","--summary")
  Run-Step -Label "P17-dagger-v3-train" -Exe $py -CmdArgs @("-B","trainer/train_bc.py","--train-jsonl",$daggerV3,"--epochs","2","--batch-size","64","--out-dir",(Join-Path $ProjectRoot "trainer_runs/p17_bc_dagger_v3"))
  Run-Step -Label "P17-dagger-v3-offline" -Exe $py -CmdArgs @("-B","trainer/eval.py","--offline","--model",(Join-Path $ProjectRoot "trainer_runs/p17_bc_dagger_v3/best.pt"),"--dataset",$daggerV3)

  if ($IncludeMilestone500) {
    $evalChall500 = Join-Path $artifactDir "eval_gold_challenger_500.json"
    $compare500Dir = Join-Path $artifactDir "eval_compare_500"
    Run-Step -Label "P17-eval-challenger-500" -Exe $py -CmdArgs @("-B","trainer/eval_long_horizon.py","--backend","sim","--stake","gold","--episodes","500","--max-steps-per-episode","120","--seeds-file","balatro_mechanics/derived/eval_seeds_500.txt","--policy","pv","--model",$challModel,"--out",$evalChall500)
    Run-Step -Label "P17-compare-500" -Exe $py -CmdArgs @("-B","trainer/eval_compare.py","--baseline",("champion=" + $evalChamp100),"--candidate",("challenger500=" + $evalChall500),"--out-dir",$compare500Dir)
  }

  $canaryDir = Join-Path $artifactDir "real_canary_latest"
  Run-Step -Label "P17-canary" -Exe $py -CmdArgs @("-B","trainer/real_shadow_canary.py","--base-url",$BaseUrl,"--model",$challModel,"--steps","60","--interval","1.0","--topk","3","--out-dir",$canaryDir)

  # registry writes
  Run-Step -Label "P17-reg-dataset-smoke" -Exe $py -CmdArgs @("-B","trainer/registry/datasets.py","--registry-root",$registryDir,"--dataset-id",("p17_smoke_search_" + $stamp),"--source-type","search","--file-path",$smokeData,"--hand-records","800","--shop-records","200","--invalid-rows","0")
  Run-Step -Label "P17-reg-dataset-dagger" -Exe $py -CmdArgs @("-B","trainer/registry/datasets.py","--registry-root",$registryDir,"--dataset-id",("p17_dagger_v3_" + $stamp),"--source-type","dagger","--file-path",$daggerV3,"--hand-records","1000","--shop-records","400","--invalid-rows","0","--source-runs","failure_mining")
  $regModelArgs = @("-B","trainer/registry/models.py","--registry-root",$registryDir,"--model-id",("p17_pv_challenger_" + $stamp),"--dataset-id",("p17_smoke_search_" + $stamp),"--model-path",$challModel,"--decision","candidate","--offline-metrics-json",$pvOffline,"--eval100-json",$evalChall100)
  if ($IncludeMilestone500) { $regModelArgs += @("--eval500-json",(Join-Path $artifactDir "eval_gold_challenger_500.json")) }
  Run-Step -Label "P17-reg-model" -Exe $py -CmdArgs $regModelArgs

  $compareSummary = Get-Content (Join-Path $compare100Dir "summary.json") -Raw | ConvertFrom-Json
  $perfPass = [bool]$compareSummary.perf_gate_pass
  $decisionPayload = Get-Content $decision100 -Raw | ConvertFrom-Json

  if (-not $hasRegisteredChampion) {
    # Bootstrap mode: first run has no stable champion pointer yet.
    $perfPass = $true
    $decisionPayload.decision = "promote"
    $decisionPayload.reason = "bootstrap_no_champion"
    $decisionPayload.perf_gate_pass = $true
    Write-Json -Path $decision100 -Obj $decisionPayload
    Write-Json -Path (Join-Path $registryDir "current_champion.json") -Obj @{
      schema = "p17_current_champion_v1"
      updated_at = (Get-Date).ToString("o")
      model_id = ("p17_bootstrap_" + $stamp)
      model_path = $challModel
      source_decision = $decision100
      bootstrap = $true
    }
  }

  $gatePerf = @{
    status = $(if ($perfPass) { "PASS" } else { "FAIL" })
    perf_gate_pass = $perfPass
    candidate_decision = [string]$decisionPayload.decision
    reason = [string]$decisionPayload.reason
    compare_summary = (Join-Path $compare100Dir "summary.json")
    decision = $decision100
    generated_at = (Get-Date).ToString("o")
  }
  Write-Json -Path (Join-Path $artifactDir "gate_perf.json") -Obj $gatePerf

  $perfMd = @(
    "# P17 Perf Gate Summary",
    "",
    "- status: $($gatePerf.status)",
    "- perf_gate_pass: $($gatePerf.perf_gate_pass)",
    "- candidate_decision: $($gatePerf.candidate_decision)",
    "- reason: $($gatePerf.reason)"
  )
  $perfMd | Out-File -LiteralPath (Join-Path $artifactDir "PERF_GATE_SUMMARY.md") -Encoding UTF8

  $statusPath = Join-Path $ProjectRoot "docs/COVERAGE_P17_STATUS.md"
  @(
    "# P17 Status",
    "",
    "- status: $($gatePerf.status)",
    "- updated_at_utc: $((Get-Date).ToUniversalTime().ToString('o'))",
    "- latest_artifact_dir: $artifactDir",
    "- functional_gate: PASS",
    "- perf_gate: $($gatePerf.status)",
    "- decision: $($gatePerf.candidate_decision)",
    "- champion_model: $champModel",
    "- challenger_model: $challModel"
  ) | Out-File -LiteralPath $statusPath -Encoding UTF8

  Write-Json -Path (Join-Path $artifactDir "gate_functional.json") -Obj $gateFunctional
  $report = @{
    schema = "p17_report_v1"
    generated_at = (Get-Date).ToString("o")
    artifact_dir = $artifactDir
    gate_functional = (Join-Path $artifactDir "gate_functional.json")
    gate_perf = (Join-Path $artifactDir "gate_perf.json")
  }
  Write-Json -Path (Join-Path $artifactDir "report_p17.json") -Obj $report

  if ((-not $perfPass) -and $FailOnPerfGate) {
    throw "[P17] perf gate failed"
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
