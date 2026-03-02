param(
  [string]$CandidateFrom = "docs/artifacts/p29/ranking_latest/ranking_summary.json",
  [string]$SeedsFile = "balatro_mechanics/derived/eval_seeds_100.txt",
  [int]$Repeats = 3,
  [string]$OutDir = "docs/artifacts/p29/flake/best_candidate_latest"
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

$py = Join-Path $ProjectRoot ".venv_trainer\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

& $py -B -m trainer.experiments.flake --mode candidate --candidate-from $CandidateFrom --seeds-file $SeedsFile --repeats $Repeats --out-dir $OutDir
if ($LASTEXITCODE -ne 0) { throw "run_flake_candidate failed with exit code $LASTEXITCODE" }
