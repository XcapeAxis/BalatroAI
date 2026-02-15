param(
    [string]$VenvPath = ".venv_trainer",
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$VenvAbs = Join-Path $ProjectRoot $VenvPath

$script:UsePyLauncher = $false
$script:PyTag = ""
$script:PythonExe = "python"

function Test-PyVersion {
    param([string]$Tag)
    try {
        & py $Tag -V *> $null
        return ($LASTEXITCODE -eq 0)
    }
    catch {
        return $false
    }
}

function Invoke-SelectedPython {
    param([string[]]$Args)
    if ($script:UsePyLauncher) {
        & py $script:PyTag @Args
    }
    else {
        & $script:PythonExe @Args
    }
    return $LASTEXITCODE
}

function Throw-OnError {
    param([int]$Code, [string]$Message)
    if ($Code -ne 0) {
        throw $Message
    }
}

Write-Host "[setup] Project root: $ProjectRoot"

$currentVersion = "unknown"
$currentLevel = "unknown"
try {
    $currentVersion = (& python -c "import sys; print(sys.version.split()[0])" 2>$null)
    $currentLevel = (& python -c "import sys; print(sys.version_info.releaselevel)" 2>$null)
}
catch {
    # Keep defaults and attempt py launcher fallback below.
}

Write-Host "[setup] Current python: $currentVersion (releaselevel=$currentLevel)"

$preferStable = $false
if ($currentVersion -like "3.15*") {
    Write-Warning "Detected Python 3.15 runtime. PyTorch wheels are often unavailable/unstable on 3.15 alpha. Prefer stable Python 3.12/3.14 for trainer venv."
    $preferStable = $true
}
elseif ($currentLevel -ne "final") {
    Write-Warning "Detected non-final Python runtime. Recommended: Python 3.12/3.14 stable for trainer."
    $preferStable = $true
}

if ($preferStable -or $currentVersion -eq "unknown") {
    if (Test-PyVersion "-3.12") {
        $script:UsePyLauncher = $true
        $script:PyTag = "-3.12"
        Write-Host "[setup] Using py -3.12 for venv creation."
    }
    elseif (Test-PyVersion "-3.14") {
        $script:UsePyLauncher = $true
        $script:PyTag = "-3.14"
        Write-Host "[setup] Using py -3.14 for venv creation."
    }
    else {
        Write-Warning "No py -3.12 / -3.14 detected. Falling back to current python."
    }
}

if ($Recreate -and (Test-Path $VenvAbs)) {
    Write-Host "[setup] Removing existing venv: $VenvAbs"
    Remove-Item -Recurse -Force $VenvAbs
}

if (-not (Test-Path $VenvAbs)) {
    Write-Host "[setup] Creating venv: $VenvAbs"
    $rc = Invoke-SelectedPython @("-m", "venv", $VenvAbs)
    Throw-OnError $rc "Failed to create venv at $VenvAbs"
}
else {
    Write-Host "[setup] Reusing existing venv: $VenvAbs"
}

$VenvPy = Join-Path $VenvAbs "Scripts\python.exe"
if (-not (Test-Path $VenvPy)) {
    throw "venv python not found: $VenvPy"
}

Write-Host "[setup] Upgrading pip"
& $VenvPy -m pip install -U pip
Throw-OnError $LASTEXITCODE "pip upgrade failed"

Write-Host "[setup] Installing trainer requirements"
& $VenvPy -m pip install -r (Join-Path $ProjectRoot "trainer\requirements.txt")
Throw-OnError $LASTEXITCODE "requirements install failed"

Write-Host "[setup] Installing torch (CUDA cu118)"
& $VenvPy -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if ($LASTEXITCODE -ne 0) {
    Write-Warning "CUDA wheel install failed. Falling back to CPU wheel."
    & $VenvPy -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    Throw-OnError $LASTEXITCODE "PyTorch CPU install failed"
}

Write-Host ""
Write-Host "[setup] Done."
Write-Host "Activate venv:" -ForegroundColor Green
Write-Host "  Set-Location `"$ProjectRoot`""
Write-Host "  . .\.venv_trainer\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Run E2E smoke:" -ForegroundColor Green
Write-Host "  .\trainer\scripts\smoke_e2e.ps1 --base-urls \"http://127.0.0.1:12346\""
