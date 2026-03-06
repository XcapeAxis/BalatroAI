param(
  [string]$Config = "configs/experiments/p22.yaml",
  [string]$OutRoot = "docs/artifacts/p22",
  [switch]$DryRun,
  [switch]$Quick,
  [switch]$Nightly,
  [switch]$IncludeLegacy,
  [switch]$LegacyOnly,
  [switch]$Resume,
  [switch]$KeepIntermediate,
  [switch]$VerboseLogs,
  [switch]$RunP44,
  [switch]$RunP45,
  [string]$Only = "",
  [string]$Exclude = "",
  [int]$MaxParallel = 1,
  [int]$SeedLimit = 0,
  [string]$Seeds = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

$py = Join-Path $ProjectRoot ".venv_trainer\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$env:PYTHONUTF8 = "1"

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
if (($RunP44 -or $RunP45) -and [string]::IsNullOrWhiteSpace($Only)) {
  $selected = @()
  if ($RunP44) { $selected += if ($Nightly) { "p44_rl_nightly" } else { "p44_rl_smoke" } }
  if ($RunP45) { $selected += if ($Nightly) { "p45_world_model_nightly" } else { "p45_world_model_smoke" } }
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
      "p45_world_model_smoke"
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
Write-Host ("[P22] cmd: " + $py + " " + ($args -join " "))

& $py @args
$code = $LASTEXITCODE
if ($code -ne 0) {
  throw ("[P22] orchestrator failed with exit code " + $code)
}
