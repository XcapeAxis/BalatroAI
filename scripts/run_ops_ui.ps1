param(
  [Alias("Host")]
  [string]$ListenHost = "127.0.0.1",
  [int]$Port = 8765,
  [switch]$Detach,
  [string]$TrainingPython = ""
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
  throw "[ops-ui] training python resolver returned empty output"
}
$resolver = $resolverJson | ConvertFrom-Json
$py = [string]$resolver.selected.python
if (-not $py.Trim()) {
  throw "[ops-ui] training python resolver did not return a python path"
}

$env:BALATRO_OPS_UI_PATH = ("http://" + $ListenHost + ":" + $Port + "/")
$args = @("-B", "-m", "trainer.ops_ui.server", "--host", $ListenHost, "--port", "$Port")

if ($Detach) {
  $jobsRoot = Join-Path $ProjectRoot "docs\\artifacts\\p53\\ops_ui\\jobs"
  if (-not (Test-Path $jobsRoot)) { New-Item -ItemType Directory -Path $jobsRoot -Force | Out-Null }
  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $stdoutPath = Join-Path $jobsRoot ("ops_ui_server_" + $stamp + ".stdout.log")
  $stderrPath = Join-Path $jobsRoot ("ops_ui_server_" + $stamp + ".stderr.log")
  $process = Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory $ProjectRoot -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru -WindowStyle Hidden
  Write-Host ("[ops-ui] detached pid=" + [string]$process.Id + " url=" + $env:BALATRO_OPS_UI_PATH)
  Write-Host ("[ops-ui] stdout=" + $stdoutPath)
  Write-Host ("[ops-ui] stderr=" + $stderrPath)
  exit 0
}

Write-Host ("[ops-ui] python: " + $py)
Write-Host ("[ops-ui] python_env_name: " + [string]$resolver.selected.env_name)
Write-Host ("[ops-ui] python_env_source: " + [string]$resolver.selection_reason)
Write-Host ("[ops-ui] cmd: " + $py + " " + ($args -join " "))
& $py @args
exit $LASTEXITCODE
