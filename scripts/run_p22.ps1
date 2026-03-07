param(
  [string]$Config = "configs/experiments/p22.yaml",
  [string]$OutRoot = "docs/artifacts/p22",
  [switch]$DryRun,
  [switch]$Quick,
  [switch]$Nightly,
  [switch]$Dashboard,
  [switch]$SkipReadinessGuard,
  [switch]$IncludeLegacy,
  [switch]$LegacyOnly,
  [switch]$Resume,
  [switch]$KeepIntermediate,
  [switch]$VerboseLogs,
  [switch]$RunP44,
  [switch]$RunP45,
  [switch]$RunP46,
  [switch]$RunP47,
  [switch]$RunP48,
  [switch]$RunP49,
  [switch]$RunP50,
  [switch]$RunP51,
  [switch]$RunP52,
  [switch]$RunP53,
  [string]$Only = "",
  [string]$Exclude = "",
  [string]$TrainingPython = "",
  [ValidateSet("visible", "minimized", "hidden", "offscreen", "restore")]
  [string]$WindowMode = "",
  [ValidateSet("visible", "minimized", "hidden", "offscreen", "restore")]
  [string]$WindowModeFallback = "",
  [switch]$StartOpsUI,
  [int]$OpsUiPort = 8765,
  [int]$MaxParallel = 1,
  [int]$SeedLimit = 0,
  [string]$Seeds = ""
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
if ($RunP50 -or $RunP52) { $resolverArgs += "-RequireCuda" }
$resolverJson = (& powershell @resolverArgs | Out-String).Trim()
if (-not $resolverJson) {
  throw "[P22] training python resolver returned empty output"
}
$resolver = $resolverJson | ConvertFrom-Json
$py = [string]$resolver.selected.python
if (-not $py.Trim()) {
  throw "[P22] training python resolver did not return a python path"
}
$env:BALATRO_TRAIN_PYTHON = $py
$env:BALATRO_READINESS_REPORT = ""
$env:BALATRO_WINDOW_MODE = ""
$env:BALATRO_WINDOW_MODE_REQUESTED = ""
$env:BALATRO_BACKGROUND_VALIDATION_REF = ""
$env:BALATRO_OPS_UI_PATH = ("http://127.0.0.1:" + $OpsUiPort + "/")

$env:PYTHONUTF8 = "1"

function Get-RuntimeWindowSettings {
  $defaultsPath = Join-Path $ProjectRoot "configs\\runtime\\runtime_defaults.json"
  if (-not (Test-Path $defaultsPath)) {
    return [pscustomobject]@{
      window_mode = "offscreen"
      window_mode_fallback = "offscreen"
      window_restore_on_failure = $true
      window_restore_on_exit = $true
      validate_background_mode_before_run = $true
    }
  }
  $payload = Get-Content -LiteralPath $defaultsPath -Raw | ConvertFrom-Json
  $defaults = $payload.defaults
  return [pscustomobject]@{
    window_mode = [string]$defaults.window_mode
    window_mode_fallback = [string]$defaults.window_mode_fallback
    window_restore_on_failure = [bool]$defaults.window_restore_on_failure
    window_restore_on_exit = [bool]$defaults.window_restore_on_exit
    validate_background_mode_before_run = [bool]$defaults.validate_background_mode_before_run
  }
}

function Invoke-PythonJson([string[]]$CmdArgs) {
  $text = (& $py @CmdArgs | Out-String).Trim()
  if (-not $text) { return $null }
  return ($text | ConvertFrom-Json)
}

function Ensure-BackgroundValidation([string]$RequestedMode) {
  $latestPath = Join-Path $ProjectRoot "docs\\artifacts\\p53\\background_mode_validation\\latest\\background_mode_validation.json"
  if (Test-Path $latestPath) {
    return $latestPath
  }
  if ([string]::IsNullOrWhiteSpace($RequestedMode) -or $RequestedMode -eq "visible") {
    return ""
  }
  $validationRunId = "p22-window-validation-" + (Get-Date -Format "yyyyMMdd-HHmmss")
  $validationArgs = @(
    "-B",
    "-m", "trainer.runtime.background_mode_validation",
    "--base-url", "http://127.0.0.1:12346",
    "--run-id", $validationRunId,
    "--modes", "visible,offscreen,minimized,hidden",
    "--seed", "AAAAAAA",
    "--scope", "p1_hand_score_observed_core",
    "--max-steps", "120",
    "--timeout-sec", "1200"
  )
  Write-Host ("[P22] background validation cmd: " + $py + " " + ($validationArgs -join " "))
  & $py @validationArgs
  if ($LASTEXITCODE -ne 0) {
    throw ("[P22] background mode validation failed with exit code " + $LASTEXITCODE)
  }
  return $latestPath
}

function Resolve-WindowMode([string]$RequestedMode, [string]$FallbackMode) {
  $resolveArgs = @(
    "-B",
    "-m", "trainer.runtime.background_mode_validation",
    "--resolve-mode",
    "--requested-mode", $RequestedMode,
    "--fallback-mode", $FallbackMode
  )
  return Invoke-PythonJson -CmdArgs $resolveArgs
}

function Set-ManagedWindowMode([string]$Mode) {
  $windowArgs = @(
    "-B",
    "-m", "trainer.runtime.window_supervisor",
    "--mode", $Mode,
    "--process-name", "Balatro",
    "--json"
  )
  return Invoke-PythonJson -CmdArgs $windowArgs
}

function Start-OpsUiServer([int]$Port) {
  $opsUiScript = Join-Path $ProjectRoot "scripts\\run_ops_ui.ps1"
  if (-not (Test-Path $opsUiScript)) {
    throw "[P22] missing ops ui script: $opsUiScript"
  }
  $opsArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $opsUiScript,
    "-Port", "$Port",
    "-TrainingPython", $py,
    "-Detach"
  )
  Write-Host ("[P22] ops ui cmd: powershell " + ($opsArgs -join " "))
  & powershell @opsArgs
  if ($LASTEXITCODE -ne 0) {
    throw ("[P22] ops ui start failed with exit code " + $LASTEXITCODE)
  }
}

$args = @(
  "-B",
  "-m", "trainer.experiments.orchestrator",
  "--config", $Config,
  "--out-root", $OutRoot,
  "--max-parallel", "$MaxParallel"
)

if ($DryRun) { $args += "--dry-run" }
if ($Nightly) { $args += "--nightly" }
if ($IncludeLegacy) { $args += "--include-legacy" }
if ($LegacyOnly) { $args += "--legacy-only" }
if ($Resume) { $args += "--resume" }
if ($KeepIntermediate) { $args += "--keep-intermediate" }
if ($VerboseLogs) { $args += "--verbose" }
if (($RunP44 -or $RunP45 -or $RunP46 -or $RunP47 -or $RunP48 -or $RunP49 -or $RunP50 -or $RunP51 -or $RunP52 -or $RunP53) -and [string]::IsNullOrWhiteSpace($Only)) {
  $selected = @()
  if ($RunP44) { $selected += if ($Nightly) { "p44_rl_nightly" } else { "p44_rl_smoke" } }
  if ($RunP45) { $selected += if ($Nightly) { "p45_world_model_nightly" } else { "p45_world_model_smoke" } }
  if ($RunP46) { $selected += if ($Nightly) { "p46_imagination_nightly" } else { "p46_imagination_smoke" } }
  if ($RunP47) { $selected += if ($Nightly) { "p47_wm_search_nightly" } else { "p47_wm_search_smoke" } }
  if ($RunP48) { $selected += if ($Nightly) { "p48_hybrid_controller_nightly" } else { "p48_hybrid_controller_smoke" } }
  if ($RunP49) { $selected += if ($Nightly) { "p49_gpu_mainline_nightly" } else { "p49_gpu_mainline_smoke" } }
  if ($RunP50) { $selected += if ($Nightly) { "p50_gpu_validation_nightly" } else { "p50_gpu_validation_smoke" } }
  if ($RunP51) { $selected += if ($Nightly) { "p51_resumeable_nightly" } else { "p51_registry_smoke" } }
  if ($RunP52) { $selected += if ($Nightly) { "p52_learned_router_nightly" } else { "p52_learned_router_smoke" } }
  if ($RunP53) { $selected += if ($Nightly) { "p53_background_ops_nightly" } else { "p53_background_ops_smoke" } }
  $Only = ($selected -join ",")
}
if (-not [string]::IsNullOrWhiteSpace($Only)) { $args += @("--only", $Only) }
if (-not [string]::IsNullOrWhiteSpace($Exclude)) { $args += @("--exclude", $Exclude) }
if ($SeedLimit -gt 0) { $args += @("--seed-limit", "$SeedLimit") }
if (-not [string]::IsNullOrWhiteSpace($Seeds)) { $args += @("--seeds", $Seeds) }

if ($Quick) {
  if (-not ($args -contains "--only")) {
    $quickIds = @(
      "quick_baseline",
      "quick_candidate",
      "quick_selfsup_pretrain",
      "quick_selfsup_p33",
      "quick_selfsup_future_value",
      "quick_selfsup_action_type",
      "quick_ssl_pretrain_v1",
      "quick_ssl_probe_v1",
      "rl_ppo_smoke",
      "p39_policy_arena_smoke",
      "p40_closed_loop_smoke",
      "p41_closed_loop_v2_smoke",
      "p42_rl_candidate_smoke",
      "p45_world_model_smoke",
      "p46_imagination_smoke",
      "p47_wm_search_smoke",
      "p48_hybrid_controller_smoke",
      "p49_gpu_mainline_smoke",
      "p50_gpu_validation_smoke",
      "p51_registry_smoke",
      "p52_learned_router_smoke",
      "p53_background_ops_smoke"
    )
    if ($IncludeLegacy -or $LegacyOnly) {
      $quickIds += @("legacy_bc_dagger_probe")
    }
    if ($LegacyOnly) {
      $quickIds = @("legacy_bc_dagger_probe")
    }
    $args += @("--only", ($quickIds -join ","))
  }
  if ($SeedLimit -le 0) { $args += @("--seed-limit", "2") }
}

if ($LegacyOnly) {
  Write-Host "[P22] Selected categories: legacy_baseline"
} elseif ($IncludeLegacy) {
  Write-Host "[P22] Selected categories: mainline + legacy_baseline"
} else {
  Write-Host "[P22] Selected categories: mainline"
}

Write-Host ("[P22] repo_root: " + $ProjectRoot)
Write-Host ("[P22] python: " + $py)
Write-Host ("[P22] python_env_type: " + [string]$resolver.selected.env_type)
Write-Host ("[P22] python_torch: " + [string]$resolver.selected.torch_version)
Write-Host ("[P22] python_cuda: " + [string]$resolver.selected.cuda_available)
Write-Host ("[P22] cmd: " + $py + " " + ($args -join " "))

$readinessReport = ""
$dashboardPath = ""
$windowSettings = Get-RuntimeWindowSettings
$requestedWindowMode = if ($WindowMode.Trim()) { $WindowMode } else { [string]$windowSettings.window_mode }
$fallbackWindowMode = if ($WindowModeFallback.Trim()) { $WindowModeFallback } else { [string]$windowSettings.window_mode_fallback }
$restoreWindowOnFailure = [bool]$windowSettings.window_restore_on_failure
$restoreWindowOnExit = [bool]$windowSettings.window_restore_on_exit
$validateBackgroundModeBeforeRun = [bool]$windowSettings.validate_background_mode_before_run
$windowModeApplied = $false
$runSucceeded = $false

try {
  if (-not $DryRun -and -not $SkipReadinessGuard) {
    $readinessRunId = "p22-" + (Get-Date -Format "yyyyMMdd-HHmmss")
    $waitArgs = @(
      "-ExecutionPolicy", "Bypass",
      "-File", (Join-Path $ProjectRoot "scripts\\wait_for_service_ready.ps1"),
      "-BaseUrl", "http://127.0.0.1:12346",
      "-OutDir", "docs/artifacts/p49/readiness",
      "-RunId", $readinessRunId,
      "-TrainingPython", $py
    )
    Write-Host ("[P22] readiness cmd: powershell " + ($waitArgs -join " "))
    & powershell @waitArgs
    if ($LASTEXITCODE -ne 0) {
      throw ("[P22] readiness guard failed with exit code " + $LASTEXITCODE)
    }
    $readinessReport = (Join-Path $ProjectRoot ("docs\\artifacts\\p49\\readiness\\" + $readinessRunId + "\\service_readiness_report.json"))
    $env:BALATRO_READINESS_REPORT = $readinessReport
  }

  if ($StartOpsUI) {
    Start-OpsUiServer -Port $OpsUiPort
  }
  $env:BALATRO_OPS_UI_PATH = ("http://127.0.0.1:" + $OpsUiPort + "/")

  if (-not $DryRun -and -not [string]::IsNullOrWhiteSpace($requestedWindowMode)) {
    if ($validateBackgroundModeBeforeRun -and $requestedWindowMode -ne "visible") {
      $validationRef = Ensure-BackgroundValidation -RequestedMode $requestedWindowMode
      if ($validationRef) { $env:BALATRO_BACKGROUND_VALIDATION_REF = $validationRef }
    }
    $resolvedWindowMode = Resolve-WindowMode -RequestedMode $requestedWindowMode -FallbackMode $fallbackWindowMode
    if ($resolvedWindowMode) {
      $effectiveWindowMode = [string]$resolvedWindowMode.effective_mode
      $env:BALATRO_WINDOW_MODE_REQUESTED = [string]$resolvedWindowMode.requested_mode
      $env:BALATRO_WINDOW_MODE = $effectiveWindowMode
      if ([string]$resolvedWindowMode.validation_path) {
        $env:BALATRO_BACKGROUND_VALIDATION_REF = [string]$resolvedWindowMode.validation_path
      }
      Write-Host ("[P22] window_mode requested=" + [string]$resolvedWindowMode.requested_mode + " effective=" + $effectiveWindowMode + " reason=" + [string]$resolvedWindowMode.resolution_reason)
      if ($effectiveWindowMode -and $effectiveWindowMode -ne "restore") {
        $windowApply = Set-ManagedWindowMode -Mode $effectiveWindowMode
        if (-not $windowApply -or -not [bool]$windowApply.operation_success) {
          throw "[P22] failed to apply requested background window mode"
        }
        $windowModeApplied = $true
      }
    }
  }

  & $py @args
  $code = $LASTEXITCODE
  if ($code -ne 0) {
    throw ("[P22] orchestrator failed with exit code " + $code)
  }

  if ($Dashboard -or $Quick -or $Nightly -or $RunP49 -or $RunP50 -or $RunP51 -or $RunP52 -or $RunP53) {
    $dashArgs = @(
      "-B",
      "-m", "trainer.monitoring.dashboard_build",
      "--input", "docs/artifacts",
      "--output", "docs/artifacts/dashboard/latest"
    )
    Write-Host ("[P22] dashboard cmd: " + $py + " " + ($dashArgs -join " "))
    & $py @dashArgs
    if ($LASTEXITCODE -ne 0) {
      throw ("[P22] dashboard build failed with exit code " + $LASTEXITCODE)
    }
    $dashboardPath = (Join-Path $ProjectRoot "docs\\artifacts\\dashboard\\latest\\index.html")
  }
  $runSucceeded = $true
} finally {
  if ($windowModeApplied -and (($runSucceeded -and $restoreWindowOnExit) -or ((-not $runSucceeded) -and ($restoreWindowOnFailure -or $restoreWindowOnExit)))) {
    try {
      $null = Set-ManagedWindowMode -Mode "restore"
      Write-Host "[P22] restored managed window to visible mode"
    } catch {
      Write-Host ("[P22] warning: failed to restore managed window: " + $_.Exception.Message)
    }
  }
}

if ($readinessReport) { Write-Host ("[P22] readiness_report=" + $readinessReport) }
if ($env:BALATRO_WINDOW_MODE) { Write-Host ("[P22] window_mode=" + $env:BALATRO_WINDOW_MODE) }
if ($env:BALATRO_BACKGROUND_VALIDATION_REF) { Write-Host ("[P22] background_validation=" + $env:BALATRO_BACKGROUND_VALIDATION_REF) }
if ($env:BALATRO_OPS_UI_PATH) { Write-Host ("[P22] ops_ui=" + $env:BALATRO_OPS_UI_PATH) }
if ($dashboardPath) { Write-Host ("[P22] dashboard=" + $dashboardPath) }

$runsRoot = Join-Path $ProjectRoot (Join-Path $OutRoot "runs")
if (Test-Path $runsRoot) {
  $latestRun = Get-ChildItem -Path $runsRoot -Directory -ErrorAction SilentlyContinue |
    Sort-Object -Property @(
      @{ Expression = "LastWriteTime"; Descending = $true },
      @{ Expression = "Name"; Descending = $true }
    ) |
    Select-Object -First 1
  if ($latestRun) {
    Write-Host ("[P22] run_id=" + [string]$latestRun.Name)
    $summaryPath = Join-Path $latestRun.FullName "summary_table.json"
    if (Test-Path $summaryPath) { Write-Host ("[P22] summary=" + $summaryPath) }
    if (Test-Path $summaryPath) {
      try {
        $rows = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
        if ($rows) {
          foreach ($row in @($rows)) {
            if (-not $row) { continue }
            $campaignStateProp = $row.PSObject.Properties["campaign_state_path"]
            $registrySnapshotProp = $row.PSObject.Properties["registry_snapshot_path"]
            $promotionQueueProp = $row.PSObject.Properties["promotion_queue_path"]
            if ($campaignStateProp -and $campaignStateProp.Value) { Write-Host ("[P22] campaign_state=" + [string]$campaignStateProp.Value) }
            if ($registrySnapshotProp -and $registrySnapshotProp.Value) { Write-Host ("[P22] registry_snapshot=" + [string]$registrySnapshotProp.Value) }
            if ($promotionQueueProp -and $promotionQueueProp.Value) { Write-Host ("[P22] promotion_queue=" + [string]$promotionQueueProp.Value) }
          }
        }
      } catch {
        Write-Host ("[P22] warning: failed to read summary_table.json: " + $_.Exception.Message)
      }
    }
  }
}
