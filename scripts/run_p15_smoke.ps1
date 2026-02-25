param(
  [string]$BaseUrl = "http://127.0.0.1:12346",
  [string]$Seed = "AAAAAAA"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
$env:SIM_FAIL_FAST = "1"

function Resolve-Python {
  $venvPy = Join-Path $ProjectRoot ".venv_trainer\\Scripts\\python.exe"
  if (Test-Path $venvPy) { return $venvPy }
  $py = Get-Command python -ErrorAction Stop
  return $py.Source
}

function Ensure-SeedsFile([string]$Path, [int]$Count, [int]$Start = 123) {
  if (Test-Path $Path) { return }
  $dir = Split-Path -Parent $Path
  if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
  $lines = @()
  for ($i = 0; $i -lt $Count; $i++) {
    $lines += ("{0}" -f ($Start + $i))
  }
  [System.IO.File]::WriteAllLines($Path, $lines, [System.Text.Encoding]::UTF8)
}

function Run-Step([string]$Label, [string]$Py, [string[]]$Args) {
  Write-Host "[$Label] $Py $($Args -join ' ')"
  & $Py @Args
  if ($LASTEXITCODE -ne 0) {
    throw "[$Label] failed with exit code $LASTEXITCODE"
  }
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$artifactDir = Join-Path $ProjectRoot ("docs/artifacts/p15/" + $stamp)
New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null

$py = Resolve-Python
$seed20 = Join-Path $ProjectRoot "balatro_mechanics/derived/eval_seeds_20.txt"
Ensure-SeedsFile -Path $seed20 -Count 20 -Start 1201

$smokeOut = Join-Path $ProjectRoot "trainer_data/p15_smoke_search.jsonl"
$smokeModelDir = Join-Path $ProjectRoot "trainer_runs/p15_pv_smoke"
$pvOffline = Join-Path $artifactDir "pv_offline_smoke_latest.json"
$evalH = Join-Path $artifactDir "eval_gold_heuristic_smoke.json"
$evalPV = Join-Path $artifactDir "eval_gold_pv_smoke.json"
$summaryTxt = Join-Path $artifactDir "dataset_summary.txt"
$reportPath = Join-Path $artifactDir "report_gate.json"

Run-Step -Label "p15-rollout" -Py $py -Args @(
  "trainer/rollout_search_p15.py",
  "--backend", "sim",
  "--stake", "gold",
  "--episodes", "40",
  "--max-steps-per-episode", "240",
  "--hand-target-samples", "500",
  "--shop-target-samples", "120",
  "--workers", "4",
  "--search", "algo=beam",
  "--search-budget-ms", "15",
  "--fail-fast",
  "--seed", "7",
  "--seed-prefix", $Seed,
  "--out", $smokeOut
)

$summaryOutput = & $py "trainer/dataset.py" --path $smokeOut --validate --summary 2>&1
if ($LASTEXITCODE -ne 0) { throw "[p15-dataset] validate/summary failed" }
[System.IO.File]::WriteAllText($summaryTxt, ($summaryOutput -join "`r`n"), [System.Text.Encoding]::UTF8)

Run-Step -Label "p15-train" -Py $py -Args @(
  "trainer/train_pv.py",
  "--train-jsonl", $smokeOut,
  "--epochs", "1",
  "--batch-size", "64",
  "--out-dir", $smokeModelDir
)

Run-Step -Label "p15-eval-pv-offline" -Py $py -Args @(
  "trainer/eval_pv.py",
  "--model", (Join-Path $smokeModelDir "best.pt"),
  "--dataset", $smokeOut,
  "--out", $pvOffline
)

Run-Step -Label "p15-eval-heuristic" -Py $py -Args @(
  "trainer/eval_long_horizon.py",
  "--backend", "sim",
  "--stake", "gold",
  "--episodes", "20",
  "--seeds-file", $seed20,
  "--policy", "heuristic",
  "--out", $evalH
)

Run-Step -Label "p15-eval-pv" -Py $py -Args @(
  "trainer/eval_long_horizon.py",
  "--backend", "sim",
  "--stake", "gold",
  "--episodes", "20",
  "--seeds-file", $seed20,
  "--policy", "pv",
  "--model", (Join-Path $smokeModelDir "best.pt"),
  "--out", $evalPV
)

$report = @{
  timestamp = $stamp
  status = "PASS"
  base_url = $BaseUrl
  smoke_dataset = $smokeOut
  model = (Join-Path $smokeModelDir "best.pt")
  offline_eval = $pvOffline
  eval_gold_heuristic = $evalH
  eval_gold_pv = $evalPV
  dataset_summary = $summaryTxt
}
($report | ConvertTo-Json -Depth 8) | Out-File -LiteralPath $reportPath -Encoding UTF8
Write-Host ("[P15] PASS artifact=" + $reportPath)
