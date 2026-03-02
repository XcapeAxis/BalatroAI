# P26 Release / Benchmark Notes Spec

## Scope

`trainer/experiments/release_notes.py` generates a publish-ready Markdown + JSON summary for benchmark and gate deltas since:

- a tag (`--since-tag`), or
- a run id (`--since-run`)

## Required Inputs

- Git history (`git log`)
- Trend warehouse rows (`docs/artifacts/trends/trend_rows.jsonl`)
- Optional risk context (`docs/artifacts/p26/alerts_latest/regression_alert_report.json`)

## CLI

```powershell
.venv_trainer\Scripts\python.exe -m trainer.experiments.release_notes --since-tag sim-p23-seed-governance-v1 --out docs\artifacts\p26\release_summary_p26.md --include-commits --include-benchmarks --include-risks
```

Supported arguments:

- `--since-tag <tag>`
- `--since-run <run_id>`
- `--out <path>`
- `--include-commits`
- `--include-benchmarks`
- `--include-risks`
- `--trends-root <path>`

If `--since-tag` does not exist, the script falls back to the most recent available tag.

## Outputs

- `release_summary_p26.md`
- `release_summary_p26.json`

## Minimum Content

- Executive summary
- What changed (capability/script/doc hints)
- Benchmark deltas
- Gate status changes
- Failure bucket changes
- Reliability notes
- Recommended next action (`promote` / `hold` / `investigate`)
