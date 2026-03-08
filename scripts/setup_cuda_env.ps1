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
  throw "[setup-cuda] no Python interpreter found"
}

function Invoke-Pip {
  param(
    [string]$PythonExe,
    [string[]]$PipArgs
  )
  & $PythonExe -m pip @PipArgs
  if ($LASTEXITCODE -ne 0) {
    throw ("[setup-cuda] pip failed: " + ($PipArgs -join " "))
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

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
  throw "[setup-cuda] nvidia-smi not found; CUDA mode requires NVIDIA driver visibility"
}

$hostPython = Resolve-HostPython -PreferredPath $PythonPath
$envDir = Join-Path $ProjectRoot ".venv_trainer_cuda"
$venvPython = Join-Path $envDir "Scripts\python.exe"

if ($ForceRecreate -and (Test-Path $envDir)) {
  Write-Host ("[setup-cuda] removing existing env: " + $envDir)
  Remove-Item -LiteralPath $envDir -Recurse -Force
}

if (-not (Test-Path $venvPython)) {
  Write-Host ("[setup-cuda] creating venv with " + $hostPython.exe)
  & $hostPython.exe @($hostPython.prefix + @("-m", "venv", $envDir))
  if ($LASTEXITCODE -ne 0) {
    throw "[setup-cuda] failed to create .venv_trainer_cuda"
  }
}

Invoke-Pip -PythonExe $venvPython -PipArgs @("install", "--upgrade", "pip", "setuptools", "wheel")
Invoke-Pip -PythonExe $venvPython -PipArgs @("install", "-r", "trainer/requirements.txt")

$probeBeforeTorch = Get-EnvProbe -PythonExe $venvPython
$needsCudaTorch = $true
if ($probeBeforeTorch -and $probeBeforeTorch.torch_available -and $probeBeforeTorch.cuda_available) {
  $needsCudaTorch = $false
}
if ($needsCudaTorch) {
  Write-Host "[setup-cuda] installing CUDA torch stack"
  Invoke-Pip -PythonExe $venvPython -PipArgs @("install", "--index-url", "https://download.pytorch.org/whl/cu128", "torch", "torchvision", "torchaudio")
}

$probe = Get-EnvProbe -PythonExe $venvPython
if (-not $probe) {
  throw "[setup-cuda] failed to probe .venv_trainer_cuda"
}
if (-not $probe.yaml_available) {
  throw "[setup-cuda] PyYAML is still unavailable after setup"
}
if (-not $probe.torch_available) {
  throw "[setup-cuda] torch is still unavailable after setup"
}
if (-not $probe.cuda_available) {
  throw "[setup-cuda] CUDA torch was not activated successfully"
}

$payload = [ordered]@{
  schema = "p58_env_setup_v1"
  generated_at = (Get-Date).ToString("o")
  mode = "cuda"
  env_dir = $envDir
  python = $probe.python
  python_version = $probe.python_version
  torch_version = $probe.torch_version
  cuda_available = [bool]$probe.cuda_available
  pyyaml_available = [bool]$probe.yaml_available
  pyyaml_version = [string]$probe.yaml_version
  installed_profile_suggestion = "single_gpu_mainline"
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
