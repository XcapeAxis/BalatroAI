# STATUS PUBLISHING P27

P27 introduces a normalized status publishing layer that converts gate and trend artifacts into reusable status payloads.

## Entry Point

```powershell
.venv_trainer\Scripts\python.exe -m trainer.experiments.status_publish --trends-root docs\artifacts\trends --artifacts-root docs\artifacts --out-root docs\artifacts\status
```

## Inputs

- `docs/artifacts/trends/trend_rows.jsonl`
- `docs/artifacts/trends/trend_index_summary.json`
- latest `docs/artifacts/p*/**/report_p*_gate.json`
- latest `regression_alert_report.json` from P26 artifacts
- latest `champion.json` / `candidate.json` / `release_state.json` (if present)
- repository metadata from git (`branch`, mainline status, tree state)

## Outputs

- `docs/artifacts/status/latest_status.json`
- `docs/artifacts/status/latest_status.md`
- `docs/artifacts/status/latest_badges.json`
- `docs/artifacts/status/latest_dashboard_data.json`
- `docs/artifacts/status/status_publish_summary.json`

## README Integration

`scripts/update_readme_badges.ps1` consumes:

- `docs/artifacts/status/latest_badges.json`
- `docs/artifacts/status/latest_status.json`

And patches:

- `<!-- BADGES:START --> ... <!-- BADGES:END -->`
- `<!-- STATUS:START --> ... <!-- STATUS:END -->`

Modes:

- dry-run preview:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\update_readme_badges.ps1 -DryRun
```

- apply patch:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\update_readme_badges.ps1 -Apply
```

## Dashboard Data Contract

`latest_dashboard_data.json` contains:

- latest gate snapshot
- trend signal and key metric snapshots
- regression alert counters
- champion/candidate/release summaries
- recent runs list
- gate history
