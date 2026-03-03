# P36 Self-Supervised Learning Core

## Goal

P36 adds a unified self-supervised core for representation pretraining without changing BC/DAgger production paths:

- unified trajectory-to-sample data contract (`trainer/selfsup/data.py`)
- shared dense encoder abstraction (`trainer/models/encoder.py`)
- two concrete tasks:
  - future value prediction (`delta_chips_k` regression)
  - inverse dynamics action type prediction

This line is designed for representation quality and data reuse, not direct deployment as a full game-playing policy.

## Data Sources

P36 reuses stable artifacts instead of live engine internals:

- sim traces (for example `sim/tests/fixtures_runtime/oracle_p0_v6_regression`)
- real-like traces from alignment fixtures (for example `docs/artifacts/p32/**/oracle_trace_real.jsonl`)
- optional P13 drift fixtures when non-empty

Dataset builder entrypoint:

```powershell
python -B -m trainer.selfsup.build_selfsup_dataset `
  --sources "oracle:sim/tests/fixtures_runtime/oracle_p0_v6_regression" "real:docs/artifacts/p32" `
  --out-dir docs/artifacts/p36/selfsup_datasets/<run_id> `
  --max-trajectories-per-source 8 `
  --max-samples 1600 `
  --lookahead-k 3
```

Outputs:

- `dataset.jsonl`
- `summary.json`
- `summary.md`
- index append: `docs/artifacts/p36/SELF_SUP_DATASETS_INDEX.md`

## SelfSupSample Contract

Each sample contains:

- `state`:
  - `phase`, `action_type`
  - dense `vector`
  - optional `state_hashes`
- `aux`:
  - source/stake/seed context
  - immediate `score_delta_t` and `reward_t`
- `future`:
  - `delta_chips_k`
  - `terminal_within_k`
  - `next_state_vector`
  - `next_action_type`
- `meta`:
  - `run_id`, `trajectory_id`, `step_idx`, source tags

## Task 1: Future Value Prediction

Definition:

- input: `state_t` dense features
- target: `future.delta_chips_k`
- loss: MSE

Entrypoint:

```powershell
python -B -m trainer.selfsup.train_future_value --config configs/experiments/p36_selfsup_future_value.yaml
```

Key artifacts (`docs/artifacts/p36/future_value/<run_id>/`):

- `metrics.json`
- `loss_curve.csv`
- `progress.jsonl`
- `future_value_epoch*.pt`

## Task 2: Action Type Prediction (Inverse Dynamics)

Definition:

- input: `(state_t, state_t+1)` encoded by shared encoder
- target: `future.next_action_type`
- loss: cross-entropy

Entrypoint:

```powershell
python -B -m trainer.selfsup.train_action_type --config configs/experiments/p36_selfsup_action_type.yaml
```

Key artifacts (`docs/artifacts/p36/action_type/<run_id>/`):

- `metrics.json`
- `loss_curve.csv`
- `progress.jsonl`
- `action_type_epoch*.pt`

## P22 Integration

P22 now schedules P36 rows directly from `configs/experiments/p22.yaml`:

- `quick_selfsup_future_value`
- `quick_selfsup_action_type`

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_p22.ps1 -Quick
```

For each experiment, seeds are recorded in:

- `docs/artifacts/p22/runs/<run_id>/<exp_id>/seeds_used.json`

## Replay Pipeline v1 Entry

For replay-contract-first pretraining smoke (real/sim unified rows + validity filtering):

```powershell
python -B -m trainer.replay.storage --real-roots docs/artifacts/p32 docs/artifacts/p13 --sim-roots sim/tests/fixtures_runtime docs/artifacts/p32/smoke_position_fixture --out-dir docs/artifacts/p36/replay/smoke_latest --max-episodes-per-source 6
python -B -m trainer.experiments.selfsup_train --config configs/experiments/p22_selfsup_smoke.yaml
```

The second command defaults to `valid_only=true` and reports `invalid_fraction` in `summary.json`.

Task metrics are emitted under each experiment directory and normalized into `summary_table.*`.

## Seed and Reproducibility Notes

- P36 rows use explicit seed lists in matrix config.
- Quick mode applies `--seed-limit 2` by default for bounded runtime but still multi-seed.
- Always trust `seeds_used.json` as executed truth.

## Limits and Risks

- P36 learns embeddings and predictive heads; it does not directly replace policy learning.
- Small fixture datasets can overfit and produce unstable metrics.
- P13 real fixture traces may be empty in some runs; fallback to real-like P32 traces is used for continuity.
- Full policy gain still depends on BC/DAgger/search/RL integration after encoder pretraining quality is validated.

## Related Docs

- [EXPERIMENTS_P22.md](EXPERIMENTS_P22.md)
- [EXPERIMENTS_P31.md](EXPERIMENTS_P31.md)
- [EXPERIMENTS_P33.md](EXPERIMENTS_P33.md)
- [SEEDS_AND_REPRODUCIBILITY.md](SEEDS_AND_REPRODUCIBILITY.md)
- [SIM_ALIGNMENT_STATUS.md](SIM_ALIGNMENT_STATUS.md)
