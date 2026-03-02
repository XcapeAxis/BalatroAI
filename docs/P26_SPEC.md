# P26 SPEC: Benchmark Trend Warehouse + Alerting + Release Summary + Nightly Entry

## RunP26.functional

Must satisfy:

- `RunP25` baseline passes.
- P26 scripts do not break existing orchestrator/campaign/nightly entry points.

Artifacts:

- `gate_functional.json`

## RunP26.trends

Must satisfy:

- trend warehouse index succeeds
- `docs/artifacts/trends/trend_rows.jsonl` exists
- `docs/artifacts/trends/trend_rows.csv` exists
- `docs/artifacts/trends/trend_index_summary.json` exists
- regression alert smoke succeeds with report files
- release summary smoke succeeds with md/json outputs

Artifacts:

- `gate_trends.json`
- `alerts_latest/regression_alert_report.{json,md}`
- `alerts_latest/regression_alert_table.csv`
- `release_summary_p26.md`
- `release_summary_p26.json`

## RunP26.ops

Must satisfy:

- nightly scheduler wrapper dry-run entry is runnable (`scripts/run_p26.ps1 -DryRun`)
- scheduler quick run is runnable (`scripts/run_p26.ps1 -Quick`)
- scheduler manifest/status/summary files are produced

Artifacts:

- `gate_ops.json`
- `scheduler_run_manifest.json`
- `scheduler_stage_status.json`
- `scheduler_summary.md`

## RunP26.docs_status

Must satisfy:

- `scripts/generate_readme_status.ps1` succeeds
- `docs/generated/README_STATUS.md` includes trend fields:
  - `trend_warehouse_status`
  - `recent_trend_signal`

Artifacts:

- `gate_docs_status.json`

## RunP26 Overall Report

`report_p26_gate.json` is PASS only when all sections pass:

- `functional`
- `trends`
- `ops`
- `docs_status`
