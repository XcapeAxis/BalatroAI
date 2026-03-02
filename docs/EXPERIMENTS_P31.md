# P31 Experiments: Self-Supervised Backbone

## Objective
P31 introduces a trajectory-native self-supervised backbone so training no longer depends on teacher labels as the only source.  
Current heads:

- `predict_score_delta`: regress `score_observed.delta`
- `predict_hand_type`: classify hand type when available

## Data Sources

Configured in [`configs/experiments/p31_selfsup.yaml`](../configs/experiments/p31_selfsup.yaml):

- oracle traces from `sim/tests/fixtures_runtime/oracle_p3_jokers_v1_regression`
- oracle traces from `sim/tests/fixtures_runtime/oracle_p8_shop_v1_regression`
- optional P13 fixture root `docs/artifacts/p13` (loaded when action traces exist)

Trajectory schema source:

- [`trainer/data/trajectory.py`](../trainer/data/trajectory.py)
- `DecisionStep` + `Trajectory` shared by self-supervised and future BC/DAgger/RL flows

## Run Minimal Training

```powershell
python -B trainer/selfsup_train.py --config configs/experiments/p31_selfsup.yaml --max-steps 100
```

Outputs (under `trainer_runs/p31_selfsup/<run_id>/`):

- `progress.jsonl`
- `selfsup_encoder_epoch*.pt`
- `summary.json`
- `summary.md`

## Run Through P22 Orchestrator

P22 now contains experiment `quick_selfsup_pretrain` in [`configs/experiments/p22.yaml`](../configs/experiments/p22.yaml).

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Quick
```

Per-run outputs:

- `docs/artifacts/p22/runs/<run_id>/quick_selfsup_pretrain/run_manifest.json`
- `docs/artifacts/p22/runs/<run_id>/quick_selfsup_pretrain/seeds_used.json`
- `docs/artifacts/p22/runs/<run_id>/quick_selfsup_pretrain/progress.jsonl`
- `docs/artifacts/p22/runs/<run_id>/summary_table.{csv,json,md}`

## How to Read Metrics

Self-supervised summaries include:

- `val_loss`: combined weighted objective
- `val_score_delta_mae`: lower is better
- `val_hand_type_acc`: higher is better

In orchestrator summary tables, compatibility metrics (`score`, `avg_ante_reached`, etc.) are projected from selfsup metrics so ranking/reporting stays schema-compatible with existing P22 tooling.

## Latest Quick Reference

- quick run id: `20260303-005315`
- summary table: `docs/artifacts/p22/runs/20260303-005315/summary_table.md`
- selfsup run root: `docs/artifacts/p22/runs/20260303-005315/quick_selfsup_pretrain/`
