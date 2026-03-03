# P22 Experiment Orchestrator

## Goal
P22 is the matrix runner for reproducible policy experiments:

- matrix-driven execution
- seed-governed comparisons
- resumable runs
- machine-readable artifacts + summary tables
- champion/candidate decision support

## Entrypoints

PowerShell wrapper (recommended):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Quick
```

Python entrypoint:

```powershell
python -B -m trainer.experiments.orchestrator --config configs/experiments/p22.yaml --out-root docs/artifacts/p22
```

## Quick Command Reference

| Scenario | Command | Notes |
|---|---|---|
| Plan only | `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -DryRun` | validates matrix + writes plan/report |
| Fast smoke | `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Quick` | 6 experiments x 2 seeds (includes P31/P33/P36 selfsup entries) |
| Multi-seed quick compare | `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Only quick_baseline,quick_candidate -Seeds "AAAAAAA,BBBBBBB,CCCCCCC"` | 2 strategies x 3 seeds |
| Fast smoke (verbose) | `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Quick -VerboseLogs` | adds per-seed/per-stage console logs |
| Nightly style | `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Nightly -Resume` | larger seed set + resume |
| Single experiment | `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Only quick_baseline -Resume` | rerun one exp id |
| Limit seeds | `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Quick -SeedLimit 2` | local-cost control |
| Custom seeds | `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Only quick_baseline -Seeds "AAAAAAA,BBBBBBB,CCCCCCC"` | explicit reproducibility override (recorded to artifacts) |

## Config Structure (`configs/experiments/p22.yaml`)

Key sections:

- `schema`: config schema identifier.
- `budget`:
  - `max_wall_time_minutes`
  - `max_episodes`
  - `max_seeds`
  - `max_concurrent_workers`
- `seeds`:
  - `base_seed`
  - `regression_fixed`
  - `extra_random`
- `seed_policy` (P34 local policy):
  - `version`
  - `regression_smoke`
  - `train_default`
  - `eval_default`
  - `nightly_extra_random`
- `matrix[]`:
  - `id`, `name`
  - `backend`, `policy`
  - `experiment_type` (optional, e.g. `selfsup_pretrain`, `selfsup_p33`, `selfsup_future_value`, `selfsup_action_type`)
  - `seed_mode` (`regression_fixed` or `nightly`)
  - `seeds` (optional explicit seed override list)
  - `gate_flag` (passed to `scripts/run_regressions.ps1`)
  - `stages` (`sanity/gate/dataset/train/eval`)
  - `eval` settings
- `evaluation`:
  - `primary_metric`
  - `gates.baseline_gate`
  - `promotion.min_delta`

## Seed Policy and "AAAAAAA" Clarification

- Fixed seeds are intentionally used in regression-style comparisons for stability.
- P22 supports multi-seed experiments by default; quick and nightly modes materialize seed sets from config policy.
- P34 default policy sets:
  - `regression_smoke`: `AAAAAAA, BBBBBBB, CCCCCCC, DDDDDDD`
  - `train_default`: `AAAAAAA .. HHHHHHH`
  - `eval_default`: `AAAAAAA .. FFFFFFF`
- Nightly mode extends the selected base set with deterministic extra random seeds.
- `scripts/run_p22.ps1 -Quick` keeps runtime stable by applying `--seed-limit 2` while still using more than one seed.
- Real seeds used in each experiment are written to:
  - `docs/artifacts/p22/runs/<run_id>/<exp_id>/seeds_used.json`
- Planned seeds per experiment are written to:
  - `docs/artifacts/p22/runs/<run_id>/run_plan.json -> experiments_with_seeds[]`
- CLI override seeds are supported via `-Seeds`/`--seeds` and persisted with source `cli.seeds_override`.
- Do not infer seed usage from a single default token; always inspect `seeds_used.json`.
- Result robustness guidance:
  - 2 seeds: fast smoke and gate sanity only.
  - 8+ seeds: meaningful model comparison in local development.
  - fixed + extra_random: nightly trend monitoring and anti-overfit checks.

## P31 Self-Supervised Integration

P22 now supports `experiment_type: selfsup_pretrain`.

Current reference row in `configs/experiments/p22.yaml`:

- `id: quick_selfsup_pretrain`
- `experiment_type: selfsup_pretrain`
- explicit seeds: `AAAAAAA, BBBBBBB, CCCCCCC` (`-Quick` applies `--seed-limit 2`)
- selfsup config: `configs/experiments/p31_selfsup.yaml`

Behavior:

- orchestrator invokes `trainer.selfsup_train.run_selfsup_training(...)` directly.
- `run_manifest.json` records selfsup config and declared data sources.
- per-seed selfsup outputs are written under experiment run directories.

Selfsup output paths:

- `docs/artifacts/p22/runs/<run_id>/quick_selfsup_pretrain/selfsup_runs/seed_*/summary.json`
- `docs/artifacts/p22/runs/<run_id>/quick_selfsup_pretrain/progress.jsonl`
- `docs/artifacts/p22/runs/<run_id>/summary_table.{csv,json,md}`

## P33 Self-Supervised Plumbing Integration (Experimental)

P22 now supports `experiment_type: selfsup_p33` for the minimal P33 data->train plumbing path.

Current reference row in `configs/experiments/p22.yaml`:

- `id: quick_selfsup_p33`
- `experiment_type: selfsup_p33`
- explicit seeds: `AAAAAAA, BBBBBBB, CCCCCCC` (`-Quick` applies `--seed-limit 2`)
- selfsup config: `configs/experiments/p33_selfsup.yaml`

Behavior:

- orchestrator invokes `trainer.self_supervised.train.run_p33_selfsup_training(...)` directly.
- run manifest records the selfsup type, config path, and declared data sources.
- per-seed outputs are written under `selfsup_p33_runs/seed_*`.

Output paths:

- `docs/artifacts/p22/runs/<run_id>/quick_selfsup_p33/selfsup_p33_runs/seed_*/summary.json`
- `docs/artifacts/p22/runs/<run_id>/quick_selfsup_p33/progress.jsonl`
- `docs/artifacts/p33/selfsup_dataset_stats.json`
- `docs/artifacts/p33/selfsup_training_summary_<timestamp>.json`

## P36 Self-Supervised Core Integration

P36 adds two explicit self-supervised tasks under the same P22 matrix system:

- `quick_selfsup_future_value` (`experiment_type: selfsup_future_value`)
- `quick_selfsup_action_type` (`experiment_type: selfsup_action_type`)

Reference configs:

- `configs/experiments/p36_selfsup_future_value.yaml`
- `configs/experiments/p36_selfsup_action_type.yaml`

Task intent:

- `future_value`: predict short-horizon future chips delta (`delta_chips_k`) from state features.
- `action_type`: inverse-dynamics style prediction of next high-level action type from `(state_t, state_t+1)`.

Artifacts per seed:

- `.../quick_selfsup_future_value/selfsup_p36_future_runs/seed_*/metrics.json`
- `.../quick_selfsup_action_type/selfsup_p36_action_runs/seed_*/metrics.json`

P22 `summary_table.*` still exposes one normalized row per experiment with:

- `seed_set_name`, `seeds_used`, `seed_count`
- `final_loss` mapped from the task-specific validation loss
- standard comparison columns (`score`, `avg_ante`, `win_rate`, etc.) for ranking continuity

## P32 Action-Fidelity Integration

P32 adds position-sensitive action fidelity checks to the broader experiment/ops workflow.

- action contract docs:
  - `docs/P32_REAL_ACTION_CONTRACT_STATUS.md`
  - `docs/P32_REAL_ACTION_CONTRACT_SPEC.md`
- shop/rng micro-alignment report:
  - `docs/P32_SHOP_RNG_ALIGNMENT.md`
- gate entry:
  - `powershell -ExecutionPolicy Bypass -File scripts\run_regressions.ps1 -RunP32`

Notes:

- P22 experiment rows remain policy/eval oriented; P32 is a fidelity gate that ensures position actions can be represented and replay-validated.
- Position-sensitive strategy experiments should explicitly include action-space assumptions (allow/deny reorder actions) in their experiment row metadata.

## Runtime Observability (During Execution)

P22 emits both per-experiment and run-level observability artifacts:

- run-level:
  - `docs/artifacts/p22/runs/<run_id>/telemetry.jsonl`
  - `docs/artifacts/p22/runs/<run_id>/live_summary_snapshot.json`
  - `docs/artifacts/p22/runs/<run_id>/summary_table.{csv,json,md}`
- per experiment:
  - `docs/artifacts/p22/runs/<run_id>/<exp_id>/run_manifest.json`
  - `docs/artifacts/p22/runs/<run_id>/<exp_id>/progress.jsonl`
  - `docs/artifacts/p22/runs/<run_id>/<exp_id>/status.json`
  - `docs/artifacts/p22/runs/<run_id>/<exp_id>/seeds_used.json`

Telemetry schema notes (P34):

- run-level `telemetry.jsonl` uses `schema: p34_telemetry_event_v1`.
- experiment-level `progress.jsonl` uses `schema: p34_progress_event_v1`.
- canonical fields: `run_id`, `exp_id`, `seed`, `phase`, `stage`, `status`, `step_or_epoch`, `metrics`, `elapsed_sec`, `wall_time_sec`, `message`.
- `summary_table.*` now also exposes `seed_set_name`, `seed_hash`, `seeds_used`, `final_win_rate`, and `final_loss` (loss fields populated for self-supervised rows).

## How to Read Champion/Candidate Outputs

P22 writes ranking and decision artifacts used by later gates:

- run-level report: `docs/artifacts/p22/runs/<run_id>/report_p23.json`
- out-root rolling decision files:
  - `docs/artifacts/p22/champion.json`
  - `docs/artifacts/p22/candidate.json`
  - `docs/artifacts/p22/release_state.json` (when release flow is enabled)

Recommended interpretation flow:

1. use `summary_table.md` for quick ranking by primary metric
2. inspect per-experiment `seed_count` and `seeds_used` before trusting deltas
3. read `report_p23.json` decision fields for promote/hold rationale
4. validate with higher-seed reruns before changing default strategy

## Multi-Seed Quick Start Example (2x3)

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Only quick_baseline,quick_candidate -Seeds "AAAAAAA,BBBBBBB,CCCCCCC"
```

Expected artifacts:

- `run_plan.json` contains `experiments_with_seeds[]` with exactly 3 seeds per selected experiment
- each experiment has `seeds_used.json` with `seed_policy_version=explicit.cli_override`
- `summary_table.md` includes two rows (`quick_baseline`, `quick_candidate`) with aggregated metrics and `seed_count=3`

Practical readout:

- compare `mean`/`avg_ante` and `win_rate` across rows
- check `std` + failure counts to detect unstable candidates
- treat single-seed wins as smoke only; prefer multi-seed consistency

## P32 Self-Supervised Skeleton Integration (P35)

A dedicated config/wrapper is now available for representation pretrain stubs:

- config: `configs/experiments/p32_self_supervised.yaml`
- wrapper: `scripts/run_p32_self_supervised.ps1`
- docs: `docs/EXPERIMENTS_P32_SELF_SUPERVISED.md`

This line is intentionally experimental and does not replace BC/DAgger gates.

Quick viewer helper:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\show_p22_live.ps1
```

Console UX notes:

- default output includes `Experiment i/N` start lines and per-experiment completion summaries.
- completion summaries report key metrics: `avg_ante`, `median_ante`, `win_rate`, `hand_top1`, `hand_top3`, `shop_top1`, `illegal_action_rate`.
- `-VerboseLogs` adds per-seed and per-stage detailed command progress.
- P36 rows print task losses in-line (`selfsup_p36_future_val_loss` / `selfsup_p36_action_val_loss`) through the shared `final_loss` channel.

## Adding a New Experiment Row

1. Open `configs/experiments/p22.yaml`.
2. Add one `matrix` entry with:
   - unique `id`
   - explicit `seed_mode`
   - correct `gate_flag`
   - stage toggles (`dataset/train` on only when needed)
3. Keep initial budget small for first iteration (`max_seeds`, `SeedLimit`).
4. Run dry-run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -DryRun -Only <new_exp_id>
```

5. Execute quick smoke:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Only <new_exp_id> -SeedLimit 2
```

## Reproduce an Existing Experiment (`quick_baseline`)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Only quick_baseline -Resume -SeedLimit 2
```

Then inspect:

- `docs/artifacts/p22/runs/<run_id>/quick_baseline/run_manifest.json`
- `docs/artifacts/p22/runs/<run_id>/quick_baseline/progress.jsonl`
- `docs/artifacts/p22/runs/<run_id>/quick_baseline/seeds_used.json`
- `docs/artifacts/p22/runs/<run_id>/run_plan.json` (`experiments_with_seeds[]`)
- `docs/artifacts/p22/runs/<run_id>/summary_table.md`

Reproduce selfsup row only:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Only quick_selfsup_pretrain -SeedLimit 2
```

Reproduce P33 selfsup row only:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Only quick_selfsup_p33 -SeedLimit 2
```

## Related Docs

- [../README.md](../README.md)
- [REPRODUCIBILITY_P25.md](REPRODUCIBILITY_P25.md)
- [SEEDS_AND_REPRODUCIBILITY.md](SEEDS_AND_REPRODUCIBILITY.md)
- [SEED_POLICY_P23.md](SEED_POLICY_P23.md)
- [EXPERIMENTS_P31.md](EXPERIMENTS_P31.md)
- [EXPERIMENTS_P32_SELF_SUPERVISED.md](EXPERIMENTS_P32_SELF_SUPERVISED.md)
- [P36_SELF_SUP_LEARNING.md](P36_SELF_SUP_LEARNING.md)
