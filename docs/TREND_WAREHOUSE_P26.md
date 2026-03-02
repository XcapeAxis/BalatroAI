# P26 Trend Warehouse

## Goal

Provide a durable cross-run benchmark history layer over `docs/artifacts/` so gates can be compared over time and used for automated alerting/release notes.

## Data Coverage

Indexer scans milestone roots under `docs/artifacts/`:

- `p22`
- `p23`
- `p24`
- `p25`
- `p26+` (automatically when present)

## Unified Row Schema

Output rows are written to `trend_rows.jsonl` and `trend_rows.csv` with this schema:

- `timestamp`
- `milestone`
- `run_id`
- `artifact_path`
- `gate_name`
- `strategy`
- `seed_set_name`
- `metric_name`
- `metric_value`
- `unit`
- `status`
- `source_file`
- `git_commit`
- `flake_status`
- `risk_status`

## Outputs

`docs/artifacts/trends/`:

- `trend_rows.jsonl`
- `trend_rows.csv`
- `trend_index_summary.json`

## CLI

```powershell
.venv_trainer\Scripts\python.exe -m trainer.experiments.index_artifacts --scan-root docs\artifacts --out-root docs\artifacts\trends --append
```

Supported flags:

- `--append` incremental indexing on top of existing warehouse
- `--rebuild` full rebuild from artifact history
- `--latest-only` smoke-friendly reduced scan
- query filters:
  - `--query-milestone`
  - `--query-strategy`
  - `--query-gate`
  - `--query-run-id`
  - `--query-out`

## Notes

- The indexer intentionally favors stable numeric metrics from gate/report/summary files.
- Per-seed payloads are not expanded into trend rows to keep the warehouse light.
- `trend_index_summary.json` is the canonical scan manifest for row counts and milestone coverage.
