<#
.SYNOPSIS
    P55 YAML/JSON config sidecar sync and consistency checker.

.DESCRIPTION
    Wraps `python -m trainer.experiments.config_sidecar_sync`.
    In --Check mode: reports drift, exits 1 if any sidecar is stale/missing.
    In --Sync mode (default): regenerates all JSON sidecars from YAML sources.

.PARAMETER Check
    Only check parity; do not write files. Exit 1 on drift.

.PARAMETER Sync
    Generate/refresh JSON sidecars from YAML (default if neither --Check nor --Sync given).

.PARAMETER Path
    Specific YAML file to check/sync. If omitted, all configs/experiments/**/*.yaml are scanned.

.PARAMETER Quiet
    Suppress per-file output from the sync tool.

.PARAMETER TrainingPython
    Override the Python interpreter to use.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\sync_config_sidecars.ps1
    powershell -ExecutionPolicy Bypass -File scripts\sync_config_sidecars.ps1 -Check
    powershell -ExecutionPolicy Bypass -File scripts\sync_config_sidecars.ps1 -Sync
#>
param(
    [switch]$Check,
    [switch]$Sync,
    [string]$Path = "",
    [switch]$Quiet,
    [string]$TrainingPython = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

# Resolve Python interpreter: prefer the one from env, then resolve script, then 'python'
$py = $TrainingPython.Trim()
if (-not $py) {
    $py = [string]($env:BALATRO_TRAIN_PYTHON)
}
if (-not $py) {
    $resolveScript = Join-Path $ProjectRoot "scripts\resolve_training_python.ps1"
    if (Test-Path $resolveScript) {
        $resolverJson = (& powershell -ExecutionPolicy Bypass -File $resolveScript -Emit json | Out-String).Trim()
        if ($resolverJson) {
            try {
                $resolver = $resolverJson | ConvertFrom-Json
                $py = [string]$resolver.selected.python
            } catch {}
        }
    }
}
if (-not $py) { $py = "python" }

# Determine mode
$mode = "sync"
if ($Check) { $mode = "check" }
elseif ($Sync) { $mode = "sync" }

Write-Host ("[sync-sidecars] mode=$mode python=$py")

$cmdArgs = @("-B", "-m", "trainer.experiments.config_sidecar_sync", "--$mode")
if ($Path.Trim()) { $cmdArgs += @("--path", $Path) }
if ($Quiet) { $cmdArgs += "--quiet" }

& $py @cmdArgs
$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    if ($mode -eq "check") {
        Write-Host "[sync-sidecars] DRIFT DETECTED. Run with --Sync to fix." -ForegroundColor Red
    } else {
        Write-Host "[sync-sidecars] sync failed with exit code $exitCode" -ForegroundColor Red
    }
    exit $exitCode
}

Write-Host "[sync-sidecars] $mode completed successfully."
