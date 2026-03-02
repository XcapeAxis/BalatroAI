param(
  [string]$SinceTag = "",
  [string]$SinceRun = "",
  [string]$Candidate = "",
  [string]$OutDir = "docs/artifacts/p27/release_train/latest",
  [object]$DryRun = $false
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

function Get-PythonExe {
  $py = Join-Path $ProjectRoot ".venv_trainer\Scripts\python.exe"
  if (Test-Path $py) { return $py }
  return "python"
}

function Invoke-GitText([string[]]$GitArgs) {
  $out = @(& git @GitArgs 2>$null)
  if ($LASTEXITCODE -ne 0) { return "" }
  return (($out | ForEach-Object { [string]$_ }) -join "`n").Trim()
}

function Resolve-SinceTag([string]$Requested) {
  if (-not [string]::IsNullOrWhiteSpace($Requested)) {
    $exists = Invoke-GitText -GitArgs @("rev-parse", "--verify", ("refs/tags/" + $Requested))
    if (-not [string]::IsNullOrWhiteSpace($exists)) { return $Requested }
  }
  $latest = Invoke-GitText -GitArgs @("describe", "--tags", "--abbrev=0")
  if (-not [string]::IsNullOrWhiteSpace($latest)) { return $latest }
  $fallback = Invoke-GitText -GitArgs @("tag", "--sort=-creatordate")
  if ([string]::IsNullOrWhiteSpace($fallback)) { return "" }
  return ($fallback.Split("`n")[0].Trim())
}

function Convert-ToBool([object]$Value) {
  if ($Value -is [bool]) { return [bool]$Value }
  $token = [string]$Value
  if ([string]::IsNullOrWhiteSpace($token)) { return $true }
  switch -Regex ($token.Trim().ToLowerInvariant()) {
    "^(true|1|yes|y)$" { return $true }
    "^(false|0|no|n)$" { return $false }
    default { return $false }
  }
}

$py = Get-PythonExe
$resolvedTag = Resolve-SinceTag -Requested $SinceTag
$dryRunFlag = Convert-ToBool -Value $DryRun

$args = @(
  "-B",
  "-m", "trainer.experiments.release_train",
  "--out-dir", $OutDir,
  "--trends-root", "docs/artifacts/trends",
  "--artifacts-root", "docs/artifacts"
)
if (-not [string]::IsNullOrWhiteSpace($resolvedTag)) { $args += @("--since-tag", $resolvedTag) }
if (-not [string]::IsNullOrWhiteSpace($SinceRun)) { $args += @("--since-run", $SinceRun) }
if (-not [string]::IsNullOrWhiteSpace($Candidate)) { $args += @("--candidate", $Candidate) }
if ($dryRunFlag) { $args += "--dry-run" }

Write-Host ("[release-train] python=" + $py)
Write-Host ("[release-train] since_tag=" + $(if ($resolvedTag) { $resolvedTag } else { "N/A" }))
Write-Host ("[release-train] since_run=" + $(if ($SinceRun) { $SinceRun } else { "N/A" }))
Write-Host ("[release-train] out_dir=" + (Join-Path $ProjectRoot $OutDir))
Write-Host ("[release-train] dry_run=" + [string]$dryRunFlag)

$output = @(& $py @args 2>&1)
$code = $LASTEXITCODE
if ($output) { $output | ForEach-Object { Write-Host $_ } }
if ($code -ne 0) {
  throw ("release train failed with exit code " + $code)
}
