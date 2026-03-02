# P24 Campaign Manager

## Purpose
P24 introduces campaign-level orchestration on top of P23 experiment runs:

- multi-stage experiment operations (`exploration` / `validation` / `promotion` / `nightly`)
- stage dependencies and failure policy (`continue` / `fail_fast`)
- campaign-level budget and status tracking
- campaign post-actions (coverage, triage, bisect-lite, ranking, dashboard, champion recommendation)

## Config Files
- `configs/experiments/campaigns/p24_quick.yaml`
- `configs/experiments/campaigns/p24_nightly.yaml`

## Runner
Entry:

```powershell
python -B -m trainer.experiments.campaign_runner --campaign-config configs/experiments/campaigns/p24_quick.yaml --out-root docs/artifacts/p24
```

Key outputs under `docs/artifacts/p24/runs/<run_id>/`:
- `campaign_plan.json`
- `campaign_status.json`
- `campaign_summary.json`
- `campaign_summary.md`
- `telemetry.jsonl`
- copied per-experiment artifacts (`<stage_id>__<exp_id>/`)

## Stage Model
Each stage supports:
- `stage_id`
- `purpose`
- `matrix_ref`
- `mode`
- `include` / `exclude`
- `seed_set_name`
- `seed_limit`
- `failure_policy`
- `depends_on`
- `max_experiments`

## Resume
`--resume` reuses latest campaign run directory and allows stage-level continuation.

