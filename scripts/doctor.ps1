param(
  [ValidateSet("auto", "cpu", "cuda")]
  [string]$Mode = "auto",
  [string]$ExplicitPython = "",
  [string]$ExplicitEnv = "",
  [switch]$QueueAttentionOnBlock,
  [switch]$SkipConfigCheck,
  [string]$BaseUrl = "http://127.0.0.1:12346",
  [string]$OutRoot = "docs/artifacts/p58",
  [ValidateSet("human", "json", "path")]
  [string]$Emit = "human"
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
  throw "[doctor] no bootstrap python found"
}

$bootstrap = Resolve-BootstrapPython
$args = @()
$args += $bootstrap.prefix
$args += @("-B", "-m", "trainer.runtime.doctor", "--mode", $Mode, "--out-root", $OutRoot, "--base-url", $BaseUrl)
if ($ExplicitPython.Trim()) { $args += @("--explicit-python", $ExplicitPython) }
if ($ExplicitEnv.Trim()) { $args += @("--explicit-env", $ExplicitEnv) }
if ($QueueAttentionOnBlock) { $args += "--queue-attention-on-block" }
if ($SkipConfigCheck) { $args += "--skip-config-check" }

$jsonText = (& $bootstrap.exe @args | Out-String).Trim()
if (-not $jsonText) {
  throw "[doctor] trainer.runtime.doctor returned empty output"
}

try {
  $payload = $jsonText | ConvertFrom-Json
} catch {
  throw ("[doctor] failed to parse JSON output: " + $_.Exception.Message)
}

if ($Emit -eq "json") {
  Write-Output $jsonText
} elseif ($Emit -eq "path") {
  Write-Output ([string]$payload.json_path)
} else {
  Write-Host ("[doctor] repo_root=" + [string]$payload.repo_root)
  Write-Host ("[doctor] status=" + [string]$payload.status + " recommended_mode=" + [string]$payload.recommended_mode + " ready=" + [string]$payload.ready_for_continuation)
  Write-Host ("[doctor] selected_python=" + [string]$payload.resolver.selected.python)
  Write-Host ("[doctor] training_env=" + [string]$payload.resolver.selected.env_name + " source=" + [string]$payload.resolver.selection_reason)
  Write-Host ("[doctor] bootstrap_state=" + [string]$payload.bootstrap_state_path)
  Write-Host ("[doctor] report_json=" + [string]$payload.json_path)
  Write-Host ("[doctor] report_md=" + [string]$payload.md_path)
  if (@($payload.blocking_reasons).Count -gt 0) {
    Write-Host "[doctor] blocking_reasons:"
    foreach ($item in @($payload.blocking_reasons)) {
      Write-Host ("  - " + [string]$item)
    }
  }
  if (@($payload.warnings).Count -gt 0) {
    Write-Host "[doctor] warnings:"
    foreach ($item in @($payload.warnings)) {
      Write-Host ("  - " + [string]$item)
    }
  }
  if (@($payload.next_steps).Count -gt 0) {
    Write-Host "[doctor] next_steps:"
    foreach ($item in @($payload.next_steps)) {
      Write-Host ("  - " + [string]$item)
    }
  }
  $attentionItemRef = ""
  if ($payload.PSObject.Properties.Name -contains "attention_item_ref") {
    $attentionItemRef = [string]$payload.attention_item_ref
  }
  if ($attentionItemRef) {
    Write-Host ("[doctor] attention_item=" + $attentionItemRef)
  }
}

if ([string]$payload.status -eq "blocked") {
  exit 1
}
exit 0
