# P53 Background Execution + Ops UI

P53 adds a local operations layer on top of the existing P49/P50/P51/P52 runtime stack:

- the LOVE/Balatro window becomes a managed runtime resource instead of a manual desktop artifact
- background-mode selection is validated before being treated as a default
- a localhost-only ops console reads existing P22/P49/P51/P52 artifacts and exposes low-risk controls with audit logs

P53 is intentionally local-first. It does not introduce cloud orchestration, remote multi-user auth, or a second storage system.

## Window Modes

Primary modules:

- `trainer/runtime/window_supervisor.py`
- `scripts/window_supervisor.ps1`
- `trainer/runtime/background_mode_validation.py`

Supported modes:

- `visible`: normal debug mode
- `minimized`: standard minimized window
- `hidden`: hidden window when the current validation still marks it safe
- `offscreen`: active window moved outside the visible desktop, used as the default background mode
- `restore`: return the managed game window to a visible state

Window operations write:

- `docs/artifacts/p53/window_supervisor/<timestamp>/window_state.json`
- `docs/artifacts/p53/window_supervisor/<timestamp>/window_ops.log`
- `docs/artifacts/p53/window_supervisor/latest/window_state.json`

Each operation records:

- `window_mode_before`
- `window_mode_after`
- `target_window`
- `operation_success`
- `error_reason`

## Background Validation Strategy

P53 does not assume that hiding the window is harmless. Validation runs the same repeatable smoke under multiple modes and compares the result against the visible baseline.

Compared modes in the current implementation:

- `visible`
- `offscreen`
- `minimized`
- `hidden`

Compared signals:

- service readiness before and after the mode switch
- fixed-seed smoke completion
- report signatures against the visible baseline
- mode-switch success/failure and any degradation

Artifacts:

- `docs/artifacts/p53/background_mode_validation/<run_id>/background_mode_validation.json`
- `docs/artifacts/p53/background_mode_validation/<run_id>/background_mode_validation.md`
- `docs/artifacts/p53/background_mode_validation/latest/background_mode_validation.json`

Current validated default:

- `window_mode = offscreen`
- `window_mode_fallback = offscreen`

Current note:

- the latest smoke also passed under `hidden` and `minimized`
- `offscreen` still remains the default because it keeps the window active while reducing desktop interference
- if a later validation marks `hidden` or `minimized` unstable, `resolve_effective_window_mode(...)` will downgrade to the fallback instead of silently keeping the requested mode

## Runtime / P22 Integration

Runtime defaults live in:

- `configs/runtime/runtime_defaults.yaml`
- `configs/runtime/runtime_defaults.json`

Relevant fields:

- `window_mode`
- `window_mode_fallback`
- `window_restore_on_failure`
- `window_restore_on_exit`
- `validate_background_mode_before_run`

P22 integration points:

- `scripts/run_p22.ps1`
- `scripts/run_regressions.ps1`
- `scripts/wait_for_service_ready.ps1`
- `trainer/runtime/service_readiness.py`
- `trainer/experiments/orchestrator.py`
- `trainer/experiments/report.py`

P53 experiment rows:

- `p53_background_ops_smoke`
- `p53_background_ops_nightly`

New summary/report fields:

- `window_mode`
- `background_validation_ref`
- `ops_ui_path`

P53 campaign stages:

- `background_mode_validation`
- `promotion_queue_update`
- `dashboard_build`
- `ops_ui_metadata`

Representative command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -RunP53
```

Optional overrides:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -RunP53 -WindowMode offscreen
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -RunP53 -Nightly
```

## Ops UI Pages

Primary modules:

- `trainer/ops_ui/server.py`
- `trainer/ops_ui/state_loader.py`
- `trainer/ops_ui/routes.py`
- `trainer/ops_ui/templates/*`
- `scripts/run_ops_ui.ps1`

The UI reads existing artifacts instead of creating a parallel data model.

Primary sources:

- `docs/artifacts/p22/runs/**/summary_table.json`
- `docs/artifacts/p49/readiness/**/service_readiness_report.json`
- `docs/artifacts/registry/checkpoints_registry.json`
- `**/promotion_queue.json`
- `**/campaign_state.json`
- `**/*progress*.jsonl`
- `docs/artifacts/dashboard/latest/index.html`
- `docs/artifacts/p53/window_supervisor/latest/window_state.json`
- `docs/artifacts/p53/background_mode_validation/latest/background_mode_validation.json`

Pages:

- `Overview`: latest P22, readiness, dashboard link, current window mode, recent campaigns, recent progress
- `Campaigns`: campaign id, stage status, resume command, state artifact
- `Checkpoint Registry`: family, status, created_at, source run, artifact ref
- `Promotion Queue`: promotion-review backlog and refs
- `Runs / Metrics`: progress rows including learner/rollout device, GPU memory, ETA, warnings
- `Background Execution / Windows`: managed window list, dominant mode, validation table, safe mode-switch buttons
- `Jobs / Audit`: UI-triggered jobs plus audited direct actions

## Safe Controls and Audit

The ops UI is intentionally limited to low-risk local actions:

- start `P22 Quick`
- start `P53 Smoke`
- resume the latest resumable campaign
- rebuild the static dashboard
- refresh registry + promotion-queue snapshots
- switch window mode

Audit log:

- `docs/artifacts/p53/ops_audit/ops_audit.jsonl`

Job records for detached UI-triggered commands:

- `docs/artifacts/p53/ops_ui/jobs/*.json`

The server listens on `127.0.0.1` by default and does not expose stop/kill controls.

## Dashboard Integration

P53 extends the existing dashboard rather than replacing it.

The static dashboard now includes:

- current window mode
- latest background validation result
- ops UI URL
- recent P53 campaign states
- recent ops audit rows

Representative output:

- `docs/artifacts/dashboard/latest/index.html`

The ops UI links back to the same dashboard artifact instead of rebuilding overlapping charts.

## Commands

Window inspection / mode switch:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\window_supervisor.ps1 -List
powershell -ExecutionPolicy Bypass -File scripts\window_supervisor.ps1 -Mode offscreen
powershell -ExecutionPolicy Bypass -File scripts\window_supervisor.ps1 -Mode restore
```

Validation:

```powershell
python -m trainer.runtime.background_mode_validation --base-url http://127.0.0.1:12346
```

P53 smoke:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -RunP53
```

Ops UI:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_ops_ui.ps1
powershell -ExecutionPolicy Bypass -File scripts\run_ops_ui.ps1 -Detach
```

## Known Limitations

- window supervision is currently Windows-only
- the validation smoke proves operational compatibility for the checked workload, not every future long-horizon task
- `hidden` and `minimized` are supported only as long as the latest validation still marks them safe
- the UI is local-only and intentionally light on controls; destructive process management remains out of scope
- jobs shown in the UI are actions triggered from the UI itself, not every background process on the workstation
