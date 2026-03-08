param(
  [ValidateSet("auto", "cpu", "cuda")]
  [string]$Mode = "auto",
  [switch]$ForceRecreate,
  [switch]$SkipSmoke,
  [string]$PythonPath = "",
  [switch]$NoGitCheck,
  [ValidateSet("human", "json", "path")]
  [string]$Emit = "human"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

function Resolve-HostPython {
  param([string]$PreferredPath)
  if ($PreferredPath.Trim()) {
    return @{ exe = $PreferredPath; prefix = @() }
  }
  $cpuVenv = Join-Path $ProjectRoot ".venv_trainer\Scripts\python.exe"
  if (Test-Path $cpuVenv) {
    return @{ exe = $cpuVenv; prefix = @() }
  }
  $cudaVenv = Join-Path $ProjectRoot ".venv_trainer_cuda\Scripts\python.exe"
  if (Test-Path $cudaVenv) {
    return @{ exe = $cudaVenv; prefix = @() }
  }
  $pyCmd = Get-Command py -ErrorAction SilentlyContinue
  if ($pyCmd) {
    try {
      $pyList = (& $pyCmd.Source -0p | Out-String)
      if ($pyList -match '-V:3\.14') {
        return @{ exe = $pyCmd.Source; prefix = @("-3.14") }
      }
    } catch {}
    return @{ exe = $pyCmd.Source; prefix = @() }
  }
  $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
  if ($pythonCmd) {
    return @{ exe = $pythonCmd.Source; prefix = @() }
  }
  throw "[setup-windows] no host python found"
}

function Invoke-PythonJson {
  param(
    [string]$PythonExe,
    [string[]]$PythonArgs
  )
  $jsonText = (& $PythonExe @PythonArgs | Out-String).Trim()
  if (-not $jsonText) {
    throw "[setup-windows] expected JSON output from python helper but received empty output"
  }
  try {
    return ($jsonText | ConvertFrom-Json)
  } catch {
    throw ("[setup-windows] failed to parse python JSON output: " + $_.Exception.Message)
  }
}

function Get-LatestConfigSyncReport {
  $root = Join-Path $ProjectRoot "docs\artifacts\p55\config_sidecar_sync"
  if (-not (Test-Path $root)) { return "" }
  $latest = Get-ChildItem -Path $root -Recurse -Filter "sidecar_sync_report.json" -ErrorAction SilentlyContinue |
    Sort-Object -Property @{ Expression = "LastWriteTime"; Descending = $true }, @{ Expression = "FullName"; Descending = $true } |
    Select-Object -First 1
  if (-not $latest) { return "" }
  return $latest.FullName
}

function Get-LatestP22Summary {
  $runsRoot = Join-Path $ProjectRoot "docs\artifacts\p22\runs"
  if (-not (Test-Path $runsRoot)) { return "" }
  $latest = Get-ChildItem -Path $runsRoot -Directory -ErrorAction SilentlyContinue |
    Sort-Object -Property @{ Expression = "LastWriteTime"; Descending = $true }, @{ Expression = "Name"; Descending = $true } |
    Select-Object -First 1
  if (-not $latest) { return "" }
  $summaryPath = Join-Path $latest.FullName "summary_table.json"
  if (-not (Test-Path $summaryPath)) { return "" }
  return $summaryPath
}

$hostPython = Resolve-HostPython -PreferredPath $PythonPath
$runId = "bootstrap-" + (Get-Date -Format "yyyyMMdd-HHmmss")
$bootstrapRunRoot = Join-Path $ProjectRoot ("docs\artifacts\p58\bootstrap\" + $runId)
if (-not (Test-Path $bootstrapRunRoot)) { New-Item -ItemType Directory -Path $bootstrapRunRoot -Force | Out-Null }

if (-not $NoGitCheck) {
  $branch = (& git rev-parse --abbrev-ref HEAD 2>$null | Out-String).Trim()
  if ($LASTEXITCODE -ne 0) {
    throw "[setup-windows] git is not available; pass -NoGitCheck only when bootstrapping outside git checks"
  }
  if ($branch -ne "main") {
    throw ("[setup-windows] expected branch 'main' but found '" + $branch + "'")
  }
}

$cudaVisible = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
$modeResolved = if ($Mode -eq "cpu") {
  "cpu"
} elseif ($Mode -eq "cuda") {
  if (-not $cudaVisible) {
    throw "[setup-windows] Mode=cuda requested but nvidia-smi is unavailable"
  }
  "cuda"
} else {
  if ($cudaVisible) { "cuda" } else { "cpu" }
}

$notes = @()
if ($cudaVisible) {
  $notes += "nvidia_smi_visible"
} else {
  $notes += "nvidia_smi_missing"
}
if ($Mode -eq "auto" -and $modeResolved -eq "cpu") {
  $notes += "auto_mode_fell_back_to_cpu"
}

$cpuSetupPath = Join-Path $bootstrapRunRoot "setup_cpu_env.json"
$cpuArgs = @("-ExecutionPolicy", "Bypass", "-File", (Join-Path $ProjectRoot "scripts\setup_cpu_env.ps1"), "-OutPath", $cpuSetupPath)
if ($PythonPath.Trim()) { $cpuArgs += @("-PythonPath", $PythonPath) }
if ($ForceRecreate) { $cpuArgs += "-ForceRecreate" }
Write-Host ("[setup-windows] setup cpu env: powershell " + ($cpuArgs -join " "))
& powershell @cpuArgs
if ($LASTEXITCODE -ne 0) {
  throw ("[setup-windows] cpu env setup failed with exit code " + $LASTEXITCODE)
}
if (-not (Test-Path $cpuSetupPath)) {
  throw "[setup-windows] cpu env setup did not write its JSON summary"
}
$cpuSetup = Get-Content -LiteralPath $cpuSetupPath -Raw | ConvertFrom-Json

$cudaSetup = $null
$cudaSetupPath = ""
if ($modeResolved -eq "cuda") {
  $cudaSetupPath = Join-Path $bootstrapRunRoot "setup_cuda_env.json"
  $cudaArgs = @("-ExecutionPolicy", "Bypass", "-File", (Join-Path $ProjectRoot "scripts\setup_cuda_env.ps1"), "-OutPath", $cudaSetupPath)
  if ($PythonPath.Trim()) { $cudaArgs += @("-PythonPath", $PythonPath) }
  if ($ForceRecreate) { $cudaArgs += "-ForceRecreate" }
  Write-Host ("[setup-windows] setup cuda env: powershell " + ($cudaArgs -join " "))
  & powershell @cudaArgs
  if ($LASTEXITCODE -ne 0) {
    throw ("[setup-windows] cuda env setup failed with exit code " + $LASTEXITCODE)
  }
  if (-not (Test-Path $cudaSetupPath)) {
    throw "[setup-windows] cuda env setup did not write its JSON summary"
  }
  $cudaSetup = Get-Content -LiteralPath $cudaSetupPath -Raw | ConvertFrom-Json
}

$selectedPython = if ($modeResolved -eq "cuda" -and $cudaSetup) { [string]$cudaSetup.python } else { [string]$cpuSetup.python }
if (-not $selectedPython.Trim()) {
  throw "[setup-windows] selected training python is empty after environment setup"
}

$syncArgs = @("-ExecutionPolicy", "Bypass", "-File", (Join-Path $ProjectRoot "scripts\sync_config_sidecars.ps1"), "-TrainingPython", $selectedPython)
Write-Host ("[setup-windows] config sidecar sync: powershell " + ($syncArgs -join " "))
& powershell @syncArgs
if ($LASTEXITCODE -ne 0) {
  throw ("[setup-windows] config sidecar sync failed with exit code " + $LASTEXITCODE)
}
$configSyncReport = Get-LatestConfigSyncReport

$smokeSummary = ""
if (-not $SkipSmoke) {
  $smokeArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $ProjectRoot "scripts\run_p22.ps1"),
    "-DryRun",
    "-SetupMode", $modeResolved,
    "-TrainingPython", $selectedPython
  )
  Write-Host ("[setup-windows] smoke: powershell " + ($smokeArgs -join " "))
  & powershell @smokeArgs
  if ($LASTEXITCODE -ne 0) {
    throw ("[setup-windows] dry-run smoke failed with exit code " + $LASTEXITCODE)
  }
  $smokeSummary = Get-LatestP22Summary
}

$bootstrapArgs = @()
$bootstrapArgs += $hostPython.prefix
$bootstrapArgs += @(
  "-B",
  "-m", "trainer.runtime.bootstrap_env",
  "--write-state",
  "--run-id", $runId,
  "--mode-requested", $Mode,
  "--mode-resolved", $modeResolved,
  "--cpu-env", (Join-Path $ProjectRoot ".venv_trainer"),
  "--cuda-env", (Join-Path $ProjectRoot ".venv_trainer_cuda"),
  "--selected-python", $selectedPython,
  "--config-sync-report", $configSyncReport
)
if ($smokeSummary) { $bootstrapArgs += @("--smoke-summary", $smokeSummary) }
foreach ($item in $notes) {
  if ([string]$item) { $bootstrapArgs += @("--note", [string]$item) }
}
if ($ForceRecreate) { $bootstrapArgs += "--force-recreate" }
if ($SkipSmoke) { $bootstrapArgs += "--skip-smoke" }

Write-Host ("[setup-windows] bootstrap state: " + $hostPython.exe + " " + ($bootstrapArgs -join " "))
$bootstrapState = Invoke-PythonJson -PythonExe $hostPython.exe -PythonArgs $bootstrapArgs

if ($Emit -eq "json") {
  $bootstrapState | ConvertTo-Json -Depth 12
} elseif ($Emit -eq "path") {
  Write-Output ([string]$bootstrapState.json_path)
} else {
  Write-Host ("[setup-windows] mode_requested=" + $Mode + " mode_resolved=" + $modeResolved)
  Write-Host ("[setup-windows] selected_python=" + [string]$bootstrapState.selected_training_python)
  Write-Host ("[setup-windows] bootstrap_complete=" + [string]$bootstrapState.bootstrap_complete + " recommended_mode=" + [string]$bootstrapState.recommended_mode)
  Write-Host ("[setup-windows] bootstrap_state=" + [string]$bootstrapState.json_path)
  Write-Host ("[setup-windows] bootstrap_md=" + [string]$bootstrapState.md_path)
  if ($configSyncReport) { Write-Host ("[setup-windows] config_sync_report=" + $configSyncReport) }
  if ($smokeSummary) { Write-Host ("[setup-windows] smoke_summary=" + $smokeSummary) }
  foreach ($command in @($bootstrapState.next_commands)) {
    Write-Host ("[setup-windows] next=" + [string]$command)
  }
}

if (-not [bool]$bootstrapState.bootstrap_complete) {
  exit 1
}
exit 0
