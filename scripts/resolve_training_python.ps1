param(
  [string]$ExplicitPython = "",
  [string]$ExplicitEnv = "",
  [switch]$RequireCuda,
  [switch]$NoPreferCuda,
  [switch]$NoCpuFallback,
  [string]$Emit = "json",
  [string]$Out = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

function Resolve-BootstrapPython {
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
    return @{ exe = $pyCmd.Source; prefix = @("-3.14") }
  }
  $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
  if ($pythonCmd) {
    return @{ exe = $pythonCmd.Source; prefix = @() }
  }
  throw "[resolve_training_python] no bootstrap python found"
}

$bootstrap = Resolve-BootstrapPython
$args = @()
$args += $bootstrap.prefix
$args += @("-B", "-m", "trainer.runtime.python_resolver", "--emit", $Emit)
$args += @("--timeout-sec", "180")
if ($ExplicitPython.Trim()) { $args += @("--explicit-python", $ExplicitPython) }
if ($ExplicitEnv.Trim()) { $args += @("--explicit-env", $ExplicitEnv) }
if ($RequireCuda) { $args += "--require-cuda" }
if ($NoPreferCuda) { $args += "--no-prefer-cuda" }
if ($NoCpuFallback) { $args += "--no-cpu-fallback" }
if ($Out.Trim()) { $args += @("--out", $Out) }

& $bootstrap.exe @args
$code = $LASTEXITCODE
if ($code -ne 0) {
  throw ("[resolve_training_python] failed with exit code " + $code)
}
