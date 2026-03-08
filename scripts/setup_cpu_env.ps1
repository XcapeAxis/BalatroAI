param(
  [string]$PythonPath = "",
  [switch]$ForceRecreate,
  [string]$OutPath = ""
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
  throw "[setup-cpu] no Python interpreter found"
}

function Invoke-Pip {
  param(
    [string]$PythonExe,
    [string[]]$PipArgs
  )
  & $PythonExe -m pip @PipArgs
  if ($LASTEXITCODE -ne 0) {
    throw ("[setup-cpu] pip failed: " + ($PipArgs -join " "))
  }
}

function Get-EnvProbe {
  param([string]$PythonExe)
  if (-not (Test-Path $PythonExe)) { return $null }
  $json = (& $PythonExe -B -m trainer.runtime.bootstrap_env --probe-python $PythonExe | Out-String).Trim()
  if (-not $json) { return $null }
  try {
    return ($json | ConvertFrom-Json)
  } catch {
    return $null
  }
}

$hostPython = Resolve-HostPython -PreferredPath $PythonPath
$envDir = Join-Path $ProjectRoot ".venv_trainer"
$venvPython = Join-Path $envDir "Scripts\python.exe"

if ($ForceRecreate -and (Test-Path $envDir)) {
  Write-Host ("[setup-cpu] removing existing env: " + $envDir)
  Remove-Item -LiteralPath $envDir -Recurse -Force
}

if (-not (Test-Path $venvPython)) {
  Write-Host ("[setup-cpu] creating venv with " + $hostPython.exe)
  & $hostPython.exe @($hostPython.prefix + @("-m", "venv", $envDir))
  if ($LASTEXITCODE -ne 0) {
    throw "[setup-cpu] failed to create .venv_trainer"
  }
}

Invoke-Pip -PythonExe $venvPython -PipArgs @("install", "--upgrade", "pip", "setuptools", "wheel")
Invoke-Pip -PythonExe $venvPython -PipArgs @("install", "-r", "trainer/requirements.txt")

$probeBeforeTorch = Get-EnvProbe -PythonExe $venvPython
$needsCpuTorch = $true
if ($probeBeforeTorch -and $probeBeforeTorch.torch_available -and (-not $probeBeforeTorch.cuda_available)) {
  $needsCpuTorch = $false
}
if ($needsCpuTorch) {
  Write-Host "[setup-cpu] installing CPU torch stack"
  Invoke-Pip -PythonExe $venvPython -PipArgs @("install", "--index-url", "https://download.pytorch.org/whl/cpu", "torch", "torchvision", "torchaudio")
}

$probe = Get-EnvProbe -PythonExe $venvPython
if (-not $probe) {
  throw "[setup-cpu] failed to probe .venv_trainer"
}
if (-not $probe.yaml_available) {
  throw "[setup-cpu] PyYAML is still unavailable after setup"
}
if (-not $probe.torch_available) {
  throw "[setup-cpu] torch is still unavailable after setup"
}

$payload = [ordered]@{
  schema = "p58_env_setup_v1"
  generated_at = (Get-Date).ToString("o")
  mode = "cpu"
  env_dir = $envDir
  python = $probe.python
  python_version = $probe.python_version
  torch_version = $probe.torch_version
  cuda_available = [bool]$probe.cuda_available
  pyyaml_available = [bool]$probe.yaml_available
  pyyaml_version = [string]$probe.yaml_version
  installed_profile_suggestion = "cpu_safe_fallback"
  host_python = $hostPython.exe
}

if ($OutPath.Trim()) {
  $target = $OutPath
  if (-not [System.IO.Path]::IsPathRooted($target)) {
    $target = Join-Path $ProjectRoot $target
  }
  $dir = Split-Path -Parent $target
  if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
  ($payload | ConvertTo-Json -Depth 8) | Out-File -LiteralPath $target -Encoding UTF8
}

$payload | ConvertTo-Json -Depth 8
