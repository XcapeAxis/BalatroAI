param(
  [Alias("Host")]
  [string]$ListenHost = "127.0.0.1",
  [int]$Port = 8050,
  [switch]$Detach,
  [switch]$OpenBrowser,
  [string]$TrainingPython = ""
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
  throw "[mvp-demo] training python resolver returned empty output"
}
$resolver = $resolverJson | ConvertFrom-Json
$py = [string]$resolver.selected.python
if (-not $py.Trim()) {
  throw "[mvp-demo] training python resolver did not return a python path"
}

$modelRoot = Join-Path $ProjectRoot "docs\artifacts\mvp\model_train"
$latestRunPath = Join-Path $modelRoot "latest_run.txt"
$latestRun = ""
$checkpointPath = ""
if (Test-Path $latestRunPath) {
  $latestRun = (Get-Content $latestRunPath -Raw).Trim()
  if ($latestRun) {
    $checkpointPath = Join-Path $modelRoot ($latestRun + "\mvp_policy.pt")
  }
}

if (-not $checkpointPath -or -not (Test-Path $checkpointPath)) {
  Write-Warning "[mvp-demo] no trained MVP checkpoint found. Run scripts\bootstrap_mvp_demo.ps1 first if needed."
} else {
  Write-Host ("[mvp-demo] current checkpoint: " + $checkpointPath)
}

$args = @("-B", "-m", "demo.app", "--host", $ListenHost, "--port", "$Port")
if ($OpenBrowser) { $args += "--open-browser" }
$url = "http://" + $ListenHost + ":" + $Port + "/"

if ($Detach) {
  $runtimeRoot = Join-Path $ProjectRoot "docs\artifacts\mvp\runtime"
  if (-not (Test-Path $runtimeRoot)) { New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null }
  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $stdoutPath = Join-Path $runtimeRoot ("mvp_demo_" + $stamp + ".stdout.log")
  $stderrPath = Join-Path $runtimeRoot ("mvp_demo_" + $stamp + ".stderr.log")
  $process = Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory $ProjectRoot -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru -WindowStyle Hidden
  Write-Host ("[mvp-demo] detached pid=" + [string]$process.Id)
  Write-Host ("[mvp-demo] url=" + $url)
  Write-Host ("[mvp-demo] stdout=" + $stdoutPath)
  Write-Host ("[mvp-demo] stderr=" + $stderrPath)
  exit 0
}

Write-Host ("[mvp-demo] python=" + $py)
Write-Host ("[mvp-demo] env_name=" + [string]$resolver.selected.env_name)
Write-Host ("[mvp-demo] env_source=" + [string]$resolver.selection_reason)
Write-Host ("[mvp-demo] url=" + $url)
Write-Host ("[mvp-demo] cmd=" + $py + " " + ($args -join " "))
& $py @args
exit $LASTEXITCODE
