param(
    [string]$BaseUrls,
    [int]$Episodes = 2,
    [string]$DatasetOut = "trainer_data/smoke_e2e.jsonl",
    [string]$RunDir = "trainer_runs/smoke_e2e",
    [int]$MaxStepsPerEpisode = 120,
    [bool]$RestartOnFail = $true
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$VenvActivate = Join-Path $ProjectRoot ".venv_trainer\Scripts\Activate.ps1"
$VenvPy = Join-Path $ProjectRoot ".venv_trainer\Scripts\python.exe"

if (-not (Test-Path $VenvPy)) {
    throw "Missing venv python: $VenvPy. Run trainer/scripts/setup_trainer_env.ps1 first."
}
if (-not (Test-Path $VenvActivate)) {
    throw "Missing venv activation script: $VenvActivate"
}

if ([string]::IsNullOrWhiteSpace($BaseUrls)) {
    $BaseUrls = Read-Host "Enter --base-urls (example: http://127.0.0.1:12346,http://127.0.0.1:12347)"
}
if ([string]::IsNullOrWhiteSpace($BaseUrls)) {
    throw "--base-urls is required for smoke_e2e"
}

$firstUrl = ($BaseUrls -split ",")[0].Trim()
if ([string]::IsNullOrWhiteSpace($firstUrl)) {
    throw "Invalid --base-urls"
}

Push-Location $ProjectRoot
try {
    . $VenvActivate
    Write-Host "[smoke] Project root: $ProjectRoot"
    Write-Host "[smoke] Activated venv: $VenvActivate"
    Write-Host "[smoke] Using python: $VenvPy"

    Write-Host "[smoke] Step 1/6: index base self-check"
    & $VenvPy "trainer/scripts/check_index_base.py" --base-url $firstUrl
    if ($LASTEXITCODE -ne 0) { throw "check_index_base failed" }

    Write-Host "[smoke] Step 2/6: rollout"
    $rolloutArgs = @(
        "trainer/rollout.py",
        "--base-urls", $BaseUrls,
        "--episodes", "$Episodes",
        "--out", $DatasetOut,
        "--max-steps-per-episode", "$MaxStepsPerEpisode"
    )
    if ($RestartOnFail) {
        $rolloutArgs += "--restart-on-fail"
    }
    & $VenvPy @rolloutArgs
    if ($LASTEXITCODE -ne 0) { throw "rollout failed" }

    Write-Host "[smoke] Step 3/6: dataset summary + validate"
    & $VenvPy "trainer/dataset.py" --path $DatasetOut --summary --validate
    if ($LASTEXITCODE -ne 0) { throw "dataset validation failed" }

    Write-Host "[smoke] Step 4/6: train_bc (1 epoch)"
    & $VenvPy "trainer/train_bc.py" --train-jsonl $DatasetOut --epochs 1 --batch-size 32 --out-dir $RunDir
    if ($LASTEXITCODE -ne 0) { throw "train_bc failed" }

    $best = Join-Path $ProjectRoot (Join-Path $RunDir "best.pt")
    $last = Join-Path $ProjectRoot (Join-Path $RunDir "last.pt")
    $modelPath = $null
    if (Test-Path $best) {
        $modelPath = $best
    }
    elseif (Test-Path $last) {
        $modelPath = $last
    }
    else {
        throw "No model checkpoint found in $RunDir"
    }

    Write-Host "[smoke] Step 5/6: offline eval"
    & $VenvPy "trainer/eval.py" --offline --model $modelPath --dataset $DatasetOut
    if ($LASTEXITCODE -ne 0) { throw "offline eval failed" }

    Write-Host "[smoke] Step 6/6: infer_assistant --once (no execute)"
    & $VenvPy "trainer/infer_assistant.py" --base-url $firstUrl --model $modelPath --topk 3 --once
    if ($LASTEXITCODE -ne 0) { throw "infer_assistant failed" }

    Write-Host "[smoke] E2E smoke completed successfully." -ForegroundColor Green
    Write-Host "[smoke] Dataset: $DatasetOut"
    Write-Host "[smoke] Model:   $modelPath"
}
finally {
    Pop-Location
}
