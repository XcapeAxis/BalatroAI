param(
  [Alias("Host")]
  [string]$ListenHost = "127.0.0.1",
  [int]$Port = 8050,
  [switch]$Detach,
  [switch]$OpenBrowser,
  [string]$TrainingPython = "",
  [int]$StartupTimeoutSec = 15
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

function Resolve-ListenAddress([string]$BindHost) {
  if ($BindHost -in @("0.0.0.0", "*")) { return [System.Net.IPAddress]::Any }
  if ($BindHost -eq "::") { return [System.Net.IPAddress]::IPv6Any }
  if ($BindHost -eq "localhost") { return [System.Net.IPAddress]::Loopback }
  $ip = $null
  if ([System.Net.IPAddress]::TryParse($BindHost, [ref]$ip)) {
    return $ip
  }
  $resolved = [System.Net.Dns]::GetHostAddresses($BindHost) | Where-Object { $_.AddressFamily -eq [System.Net.Sockets.AddressFamily]::InterNetwork } | Select-Object -First 1
  if ($resolved) { return $resolved }
  throw ("[mvp-demo] failed to resolve host: " + $BindHost)
}

function Test-PortAvailable([string]$BindHost, [int]$Port) {
  $probeHost = $BindHost
  if ($probeHost -in @("0.0.0.0", "*", "localhost")) { $probeHost = "127.0.0.1" }
  if ($probeHost -eq "::") { $probeHost = "::1" }
  $client = $null
  try {
    $client = [System.Net.Sockets.TcpClient]::new()
    $async = $client.BeginConnect($probeHost, $Port, $null, $null)
    if ($async.AsyncWaitHandle.WaitOne(250)) {
      $client.EndConnect($async)
      return $false
    }
  } catch {
  } finally {
    if ($client -ne $null) {
      $client.Close()
    }
  }
  $address = Resolve-ListenAddress $BindHost
  $listener = $null
  try {
    $listener = [System.Net.Sockets.TcpListener]::new($address, $Port)
    $listener.Server.ExclusiveAddressUse = $true
    $listener.Start()
    return $true
  } catch {
    return $false
  } finally {
    if ($listener -ne $null) {
      $listener.Stop()
    }
  }
}

function Test-HealthEndpoint([string]$HealthUrl) {
  try {
    $response = Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 2
    return ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300)
  } catch {
    return $false
  }
}

function Wait-DetachedDemoReady($Process, [string]$HealthUrl, [int]$TimeoutSec, [string]$StdErrPath, [string]$BindHost, [int]$Port) {
  $deadline = (Get-Date).AddSeconds([Math]::Max(1, $TimeoutSec))
  while ((Get-Date) -lt $deadline) {
    $Process.Refresh()
    if ($Process.HasExited) {
      $stderr = ""
      if (Test-Path $StdErrPath) {
        $stderr = (Get-Content -LiteralPath $StdErrPath -Raw -ErrorAction SilentlyContinue)
      }
      if ($stderr -match "WinError 10013|Address already in use|\[Errno 98\]|Only one usage of each socket address") {
        throw ("[mvp-demo] port " + [string]$Port + " on " + $BindHost + " is already in use")
      }
      $summary = ("[mvp-demo] detached process exited early with code " + [string]$Process.ExitCode)
      if ($stderr.Trim()) {
        $compact = (($stderr -replace "\s+", " ").Trim())
        $summary += (": " + $compact)
      }
      throw $summary
    }
    if (Test-HealthEndpoint $HealthUrl) {
      return
    }
    Start-Sleep -Milliseconds 300
  }
  throw ("[mvp-demo] detached process did not become healthy within " + [string]$TimeoutSec + "s. Check stderr log: " + $StdErrPath)
}

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
$healthUrl = $url + "api/health"

if (-not (Test-PortAvailable -BindHost $ListenHost -Port $Port)) {
  throw ("[mvp-demo] port " + [string]$Port + " on " + $ListenHost + " is already in use")
}

if ($Detach) {
  $runtimeRoot = Join-Path $ProjectRoot "docs\artifacts\mvp\runtime"
  if (-not (Test-Path $runtimeRoot)) { New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null }
  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $stdoutPath = Join-Path $runtimeRoot ("mvp_demo_" + $stamp + ".stdout.log")
  $stderrPath = Join-Path $runtimeRoot ("mvp_demo_" + $stamp + ".stderr.log")
  $process = Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory $ProjectRoot -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru -WindowStyle Hidden
  Wait-DetachedDemoReady -Process $process -HealthUrl $healthUrl -TimeoutSec $StartupTimeoutSec -StdErrPath $stderrPath -BindHost $ListenHost -Port $Port
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
