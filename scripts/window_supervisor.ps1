param(
  [switch]$List,
  [ValidateSet("visible", "minimized", "hidden", "offscreen", "restore")]
  [string]$Mode = "",
  [Alias("Pid")]
  [int]$ProcessId = 0,
  [string[]]$ProcessName = @("Balatro"),
  [string]$TitleContains = "",
  [switch]$IncludeAuxiliary,
  [string]$OutRoot = "docs/artifacts/p53/window_supervisor",
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
  throw "[window_supervisor] training python resolver returned empty output"
}
$resolver = $resolverJson | ConvertFrom-Json
$py = [string]$resolver.selected.python
if (-not $py.Trim()) {
  throw "[window_supervisor] training python resolver did not return a python path"
}

$args = @("-B", "-m", "trainer.runtime.window_supervisor", "--out-root", $OutRoot, "--json")
if ($List -or -not $Mode.Trim()) {
  $args += "--list"
} else {
  $args += @("--mode", $Mode)
}
if ($ProcessId -gt 0) { $args += @("--pid", "$ProcessId") }
foreach ($name in @($ProcessName)) {
  if ([string]::IsNullOrWhiteSpace([string]$name)) { continue }
  $args += @("--process-name", [string]$name)
}
if ($TitleContains.Trim()) { $args += @("--title-contains", $TitleContains) }
if ($IncludeAuxiliary) { $args += "--include-auxiliary" }

Write-Host ("[window_supervisor] python: " + $py)
Write-Host ("[window_supervisor] cmd: " + $py + " " + ($args -join " "))
& $py @args
exit $LASTEXITCODE
