# P58 Windows Bootstrap & Portability Hardening

## Goal

P58 makes it practical to continue BalatroAI on another Windows workstation without rebuilding the project workflow by hand. The target is not a separate bootstrap side-path; the target is to make the existing P22 / campaigns / dashboard / Ops UI flow portable.

## Standard Environment Layout

P58 keeps a two-environment layout:

- `.venv_trainer`
  - CPU-safe fallback
  - safe for config checks, docs, dashboard, ops UI, and non-CUDA execution
- `.venv_trainer_cuda`
  - CUDA-first training environment
  - used when the workstation has a healthy NVIDIA driver and Torch CUDA stack

This matches the P50 runtime design:

- CPU env stays as the conservative fallback
- CUDA env is isolated for real training
- all major entrypoints resolve the training interpreter automatically

## Recommended Python and Host Assumptions

Recommended baseline:

- Windows host
- Git available in `PATH`
- Python launcher (`py`) or a direct Python executable
- NVIDIA driver + `nvidia-smi` for CUDA mode

P58 does not install system drivers or Visual Studio build tools. It assumes those are handled outside the repo.

## Core Scripts

Main entrypoints:

- `scripts/setup_windows.ps1`
- `scripts/setup_cpu_env.ps1`
- `scripts/setup_cuda_env.ps1`
- `scripts/doctor.ps1`
- `trainer/runtime/bootstrap_env.py`
- `trainer/runtime/doctor.py`
- `trainer/runtime/python_resolver.py`

Related mainline entrypoints:

- `scripts/run_p22.ps1`
- `scripts/run_regressions.ps1`
- `scripts/run_dashboard.ps1`
- `scripts/run_ops_ui.ps1`
- `scripts/wait_for_service_ready.ps1`

## First-Run Flow on a New Windows Machine

Recommended first-run sequence:

```powershell
git clone https://github.com/XcapeAxis/BalatroAI.git
cd BalatroAI

powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -Mode auto -SkipSmoke
powershell -ExecutionPolicy Bypass -File scripts\doctor.ps1
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -DryRun
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Quick
```

For slower hosts or unattended runs, use `safe_run.ps1`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\safe_run.ps1 -TimeoutSec 1800 -- powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -Mode auto -SkipSmoke
powershell -ExecutionPolicy Bypass -File scripts\safe_run.ps1 -TimeoutSec 7200 -- powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Quick
```

## Setup Modes

`scripts/setup_windows.ps1` supports:

- `-Mode cpu`
  - build only `.venv_trainer`
  - use when the machine has no CUDA path or when the goal is a safe docs/ops/CPU-only setup
- `-Mode cuda`
  - require a visible NVIDIA runtime
  - build `.venv_trainer_cuda` and fail if CUDA prerequisites are missing
- `-Mode auto`
  - prefer CUDA when `nvidia-smi` is available
  - otherwise fall back to CPU

Additional useful flags:

- `-ForceRecreate`
- `-SkipSmoke`
- `-PythonPath <path>`
- `-NoGitCheck`

## What Bootstrap Actually Does

`setup_windows.ps1` performs the following steps:

1. check Python and Git availability
2. detect CUDA visibility when requested
3. create or repair `.venv_trainer`
4. create or repair `.venv_trainer_cuda` when appropriate
5. install or confirm project dependencies, including `PyYAML`
6. run config sidecar sync/check
7. write bootstrap state under `docs/artifacts/p58/bootstrap/`
8. print the resolved next-step commands

Primary artifacts:

- `docs/artifacts/p58/bootstrap/latest_bootstrap_state.json`
- `docs/artifacts/p58/bootstrap/latest_bootstrap_state.md`
- `docs/artifacts/p55/config_sidecar_sync/<run_id>/sidecar_sync_report.json`

## Doctor / Health Check

`scripts/doctor.ps1` answers three practical questions:

1. Is this repo on a usable branch and in a usable state?
2. Which training interpreter will the main scripts select?
3. Is this machine ready for continued work, and if not, why not?

Checks include:

- Git availability, branch, and working-tree cleanliness
- available Python interpreters
- resolver result and environment source
- Torch version and CUDA availability
- `PyYAML` presence
- P55 config sidecar check
- dashboard / ops UI / registry artifact presence
- live readiness probe when applicable

Primary outputs:

- `docs/artifacts/p58/latest_doctor.json`
- `docs/artifacts/p58/latest_doctor.md`
- `docs/artifacts/p58/doctor_<timestamp>.json`
- `docs/artifacts/p58/doctor_<timestamp>.md`

## Interpreting Doctor Output

Status meanings:

- `ready`
  - machine is healthy for continuation
- `warning`
  - mainline work can continue, but there are conditions worth reviewing
- `blocked`
  - the environment should not continue without repair or human review

Recommended mode:

- `cuda_mainline`
  - use the CUDA env as the default path
- `cpu_safe`
  - continue in CPU mode
- `blocked`
  - no healthy project environment is available

Typical next steps:

- `scripts\setup_windows.ps1 -Mode auto`
- `scripts\run_p22.ps1 -Quick`
- `scripts\run_ops_ui.ps1`

## Unified Interpreter Resolution

P58 keeps the multi-env complexity out of the common scripts.

The resolver now distinguishes:

- explicit override
- bootstrap-selected env
- `.venv_trainer_cuda`
- `.venv_trainer`
- system fallback as last resort

Scripts using the unified resolver:

- `scripts/run_p22.ps1`
- `scripts/run_regressions.ps1`
- `scripts/run_dashboard.ps1`
- `scripts/run_ops_ui.ps1`
- `scripts/wait_for_service_ready.ps1`

This means new-machine handoff does not require memorizing which `python.exe` to call.

## P22 / P57 / UI Integration

P58 is integrated into the mainline instead of sitting beside it:

- `run_p22.ps1` runs doctor/config checks before execution
- P22 summaries now include:
  - doctor report path
  - bootstrap state path
  - resolved training env
  - setup mode
- P57 overnight flow now includes an `environment_doctor` stage
- dashboard surfaces the latest environment state
- Ops UI exposes an environment page with doctor and bootstrap outputs

## Continue on Another Windows PC

Minimal handoff checklist:

1. clone the repo
2. run `scripts\setup_windows.ps1 -Mode auto -SkipSmoke`
3. run `scripts\doctor.ps1`
4. if doctor is not blocked, run `scripts\run_p22.ps1 -Quick`
5. open:
   - `docs/artifacts/dashboard/latest/index.html`
   - `http://127.0.0.1:8765/`
   - `docs/artifacts/p58/latest_doctor.md`

If the machine is CPU-only, that is still acceptable for:

- config checks
- P22 dry-run
- dashboard / Ops UI
- a subset of quick validation

If the machine is CUDA-capable, the same mainline commands continue to work without changing the operator workflow.

## Common Failures and Fixes

### `setup_windows.ps1` falls back to CPU

Check:

- `nvidia-smi`
- CUDA torch install inside `.venv_trainer_cuda`
- latest bootstrap state

Use:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -Mode cuda -SkipSmoke
```

### Doctor reports `blocked`

Common causes:

- no healthy project env
- missing Torch
- missing `PyYAML`
- config sidecar check failed

Recommended first action:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -Mode auto -SkipSmoke
```

### P22 chooses the wrong interpreter

Inspect:

- `docs/artifacts/p58/latest_doctor.json`
- `docs/artifacts/p58/bootstrap/latest_bootstrap_state.json`
- `summary_table.json` runtime fields:
  - `training_env_source`
  - `training_env_name`

### Dashboard or Ops UI is empty

Rebuild and reopen:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_dashboard.ps1
powershell -ExecutionPolicy Bypass -File scripts\run_ops_ui.ps1
```

## Validation Artifacts from This Milestone

Key P58 validation outputs are written under:

- `docs/artifacts/p58/bootstrap_validation_<timestamp>.json`
- `docs/artifacts/p58/bootstrap_validation_<timestamp>.md`
- `docs/artifacts/p58/python_resolver_validation_<timestamp>.md`
- `docs/artifacts/p58/doctor_<timestamp>.json`
- `docs/artifacts/p58/ops_ui_smoke_<timestamp>.md`

These files are intended to support real handoff, not just milestone completion.

## Related Docs

- `docs/EXPERIMENTS_P22.md`
- `docs/P49_GPU_MAINLINE_AND_DASHBOARD.md`
- `docs/P50_CUDA_ENVIRONMENT.md`
- `docs/P50_GPU_TROUBLESHOOTING.md`
- `docs/P55_CONFIG_HARDENING.md`
- `docs/P57_OVERNIGHT_AUTONOMY_PROTOCOL.md`
