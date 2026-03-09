param(
  [Alias("Host")]
  [string]$ListenHost = "127.0.0.1",
  [int]$Port = 8050,
  [switch]$OpenBrowser,
  [switch]$Detach,
  [string]$TrainingPython = "",
  [int]$Episodes = 220,
  [int]$MaxSteps = 32,
  [int]$ScenarioCopies = 64,
  [int]$Epochs = 4,
  [switch]$ForceRetrain
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

$resolveTrainingPythonScript = Join-Path $ProjectRoot "scripts\\resolve_training_python.ps1"
$resolverArgs = @(
  "-ExecutionPolicy", "Bypass",
  "-File", $resolveTrainingPythonScript,
  "-Emit", "json"
)
if ($TrainingPython.Trim()) { $resolverArgs += @("-ExplicitPython", $TrainingPython) }
$resolverJson = (& powershell @resolverArgs | Out-String).Trim()
if (-not $resolverJson) {
  throw "[bootstrap-mvp] training python resolver returned empty output"
}
$resolver = $resolverJson | ConvertFrom-Json
$py = [string]$resolver.selected.python
if (-not $py.Trim()) {
  throw "[bootstrap-mvp] training python resolver did not return a python path"
}

$modelRoot = Join-Path $ProjectRoot "docs\\artifacts\\mvp\\model_train"
$latestRunPath = Join-Path $modelRoot "latest_run.txt"
$latestRun = ""
$checkpointPath = ""
if (Test-Path $latestRunPath) {
  $latestRun = (Get-Content $latestRunPath -Raw).Trim()
  if ($latestRun) {
    $checkpointPath = Join-Path $modelRoot ($latestRun + "\\mvp_policy.pt")
  }
}

if (-not $ForceRetrain -and $checkpointPath -and (Test-Path $checkpointPath)) {
  Write-Host ("[bootstrap-mvp] reusing checkpoint: " + $checkpointPath)
} else {
  if (-not (Test-Path $modelRoot)) { New-Item -ItemType Directory -Path $modelRoot -Force | Out-Null }
  $runId = Get-Date -Format "yyyyMMdd_HHmmss"
  $runDir = Join-Path $modelRoot $runId
  $summaryRoot = Join-Path $ProjectRoot "docs\\artifacts\\mvp\\runtime"
  if (-not (Test-Path $summaryRoot)) { New-Item -ItemType Directory -Path $summaryRoot -Force | Out-Null }

  $datasetSummary = Join-Path $summaryRoot ("bootstrap_dataset_" + $runId + ".summary.json")
  $trainSummary = Join-Path $summaryRoot ("bootstrap_train_" + $runId + ".summary.json")

  Write-Host ("[bootstrap-mvp] building dataset into " + $runDir)
  & powershell -ExecutionPolicy Bypass -File scripts\safe_run.ps1 `
    -TimeoutSec 3600 `
    -SummaryJson $datasetSummary `
    -- $py -B demo\build_mvp_dataset.py --episodes $Episodes --max-steps $MaxSteps --scenario-copies $ScenarioCopies --run-dir $runDir
  if ($LASTEXITCODE -ne 0) {
    throw "[bootstrap-mvp] dataset build failed"
  }

  Write-Host ("[bootstrap-mvp] training model into " + $runDir)
  & powershell -ExecutionPolicy Bypass -File scripts\safe_run.ps1 `
    -TimeoutSec 3600 `
    -SummaryJson $trainSummary `
    -- $py -B demo\train_mvp_model.py --run-dir $runDir --epochs $Epochs --batch-size 256 --device cpu
  if ($LASTEXITCODE -ne 0) {
    throw "[bootstrap-mvp] model training failed"
  }

  $checkpointPath = Join-Path $runDir "mvp_policy.pt"
  if (-not (Test-Path $checkpointPath)) {
    throw "[bootstrap-mvp] training finished without mvp_policy.pt"
  }
  Write-Host ("[bootstrap-mvp] new checkpoint: " + $checkpointPath)
}

$launchArgs = @(
  "-ExecutionPolicy", "Bypass",
  "-File", (Join-Path $ProjectRoot "scripts\\run_mvp_demo.ps1"),
  "-ListenHost", $ListenHost,
  "-Port", "$Port"
)
if ($OpenBrowser) { $launchArgs += "-OpenBrowser" }
if ($Detach) { $launchArgs += "-Detach" }
if ($TrainingPython.Trim()) { $launchArgs += @("-TrainingPython", $TrainingPython) }

& powershell @launchArgs
exit $LASTEXITCODE

