# Seeds and Reproducibility (P34)

## Purpose

This document defines how P22 experiments choose seeds, how overrides are applied, and where exact seed usage is persisted for replayable results.

## Seed Policy Layers

1. External policy (optional): `seed_policy_config` in `configs/experiments/p22.yaml` can point to a P23 policy file.
2. Local policy (default in P34): `configs/experiments/p22.yaml -> seed_policy`.
3. Experiment explicit seeds: `matrix[].seeds` in the same config.
4. CLI override: `scripts/run_p22.ps1 -Seeds "AAAAAAA,BBBBBBB,CCCCCCC"` (highest precedence).

## Default Local Seed Sets

Defined in `configs/experiments/p22.yaml`:

- `regression_smoke`: short stable set for gate/smoke.
- `train_default`: broader fixed set for training comparisons.
- `eval_default`: medium set for evaluation comparisons.
- `nightly_extra_random`: deterministic generated extras appended in nightly mode.

Quick mode remains multi-seed:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Quick
```

`-Quick` applies `--seed-limit 2` by default, so runtime is bounded but no longer single-seed.

## Artifact Contract

For each run (`docs/artifacts/p22/runs/<run_id>/`):

- `run_plan.json`
  - `seed_policy_version`
  - `cli_seed_override`
  - `experiments_with_seeds[]` with per-experiment planned seeds
- `<exp_id>/seeds_used.json`
  - actual seeds used
  - `seed_set_name`, `seed_hash`, `seed_policy_version`
- `summary_table.{csv,json,md}`
  - includes `seed_set_name`, `seed_hash`, `seeds_used`, `seed_count`

## Telemetry Contract (P34)

- run-level stream: `telemetry.jsonl` (`schema: p34_telemetry_event_v1`)
- experiment stream: `<exp_id>/progress.jsonl` (`schema: p34_progress_event_v1`)

Common fields:

- `run_id`, `exp_id`, `seed`
- `phase` (`orchestrator` / `stage` / `eval`)
- `stage` (e.g. `sanity`, `gate`, `eval`, `done`)
- `status`, `step_or_epoch`
- `metrics`
- `elapsed_sec`, `wall_time_sec`, `message`

## Reproduce a Prior Run

1. Select a historical `run_id`.
2. Read `run_plan.json` and the target `<exp_id>/seeds_used.json`.
3. Re-run with matching experiment and seeds:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Only quick_baseline -Seeds "AAAAAAA,BBBBBBB"
```

4. Compare `summary_table.json` and per-experiment `progress.jsonl`.

## Practical Recommendations

- Gate/smoke: 2-4 seeds.
- Candidate ranking: >=8 seeds.
- Nightly trend checks: eval set + deterministic extras.
- Always cite `run_id`, `seed_hash`, config path, and commit hash in external reports.
