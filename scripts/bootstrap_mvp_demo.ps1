param(
  [Alias("Host")]
  [string]$ListenHost = "127.0.0.1",
  [int]$Port = 8050,
  [switch]$OpenBrowser,
  [switch]$Detach,
  [string]$TrainingPython = "",
  [ValidateSet("smoke", "standard")]
  [string]$TrainProfile = "smoke",
  [switch]$ForceRetrain
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

$resolveTrainingPythonScript = Join-Path $ProjectRoot "scripts\resolve_training_python.ps1"
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

$modelRoot = Join-Path $ProjectRoot "docs\artifacts\mvp\model_train"
$runtimeRoot = Join-Path $ProjectRoot "docs\artifacts\mvp\runtime"
$statusPath = Join-Path $ProjectRoot "docs\artifacts\mvp\training_status\latest.json"
$latestRunPath = Join-Path $modelRoot "latest_run.txt"
if (-not (Test-Path $modelRoot)) { New-Item -ItemType Directory -Path $modelRoot -Force | Out-Null }
if (-not (Test-Path $runtimeRoot)) { New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null }

function Get-LatestCheckpoint {
  if (-not (Test-Path $latestRunPath)) { return "" }
  $latestRun = (Get-Content $latestRunPath -Raw).Trim()
  if (-not $latestRun) { return "" }
  $candidate = Join-Path $modelRoot ($latestRun + "\mvp_policy.pt")
  if (Test-Path $candidate) { return $candidate }
  return ""
}

$checkpointPath = Get-LatestCheckpoint
if (-not $ForceRetrain -and $checkpointPath) {
  Write-Host ("[bootstrap-mvp] reusing checkpoint: " + $checkpointPath)
} else {
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $summaryPath = Join-Path $runtimeRoot ("bootstrap_" + $TrainProfile + "_" + $stamp + ".summary.json")

  if ($TrainProfile -eq "standard") {
    $budgetMinutes = 120
    $episodes = 1800
    $maxSteps = 44
    $scenarioCopies = 256
    $batchSize = 768
    $finalEpochs = 18
    $sweepEpochs = 8
    $timeoutSec = 12600
  } else {
    $budgetMinutes = 8
    $episodes = 180
    $maxSteps = 28
    $scenarioCopies = 48
    $batchSize = 256
    $finalEpochs = 4
    $sweepEpochs = 2
    $timeoutSec = 1800
  }

  Write-Host ("[bootstrap-mvp] no reusable checkpoint found, launching " + $TrainProfile + " training")
  Write-Host ("[bootstrap-mvp] summary=" + $summaryPath)

  & powershell -ExecutionPolicy Bypass -File scripts\safe_run.ps1 `
    -TimeoutSec $timeoutSec `
    -SummaryJson $summaryPath `
    -- $py -B -m demo.train_mvp_pipeline `
      --status-path $statusPath `
      --budget-minutes $budgetMinutes `
      --episodes $episodes `
      --max-steps $maxSteps `
      --scenario-copies $scenarioCopies `
      --device auto `
      --batch-size $batchSize `
      --final-epochs $finalEpochs `
      --sweep-epochs $sweepEpochs
  if ($LASTEXITCODE -ne 0) {
    throw "[bootstrap-mvp] MVP training pipeline failed"
  }

  $checkpointPath = Get-LatestCheckpoint
  if (-not $checkpointPath) {
    throw "[bootstrap-mvp] training finished without mvp_policy.pt"
  }
  Write-Host ("[bootstrap-mvp] new checkpoint: " + $checkpointPath)
}

$launchArgs = @(
  "-ExecutionPolicy", "Bypass",
  "-File", (Join-Path $ProjectRoot "scripts\run_mvp_demo.ps1"),
  "-ListenHost", $ListenHost,
  "-Port", "$Port"
)
if ($OpenBrowser) { $launchArgs += "-OpenBrowser" }
if ($Detach) { $launchArgs += "-Detach" }
if ($TrainingPython.Trim()) { $launchArgs += @("-TrainingPython", $TrainingPython) }

Write-Host "[bootstrap-mvp] launching demo"
& powershell @launchArgs
exit $LASTEXITCODE
