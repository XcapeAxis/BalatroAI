# P32 Self-Supervised / Representation Stub Experiments (P35)

## Motivation

BC and DAgger depend on action labels and teacher quality.  
The P32 self-supervised line adds a lightweight representation/pretrain path that can consume replay traces directly, so future RL/self-play work is less tied to expensive labels.

## Current Scope (Stub)

This is intentionally a skeleton, not a production model pipeline:

- config-first experiment entry: `configs/experiments/p32_self_supervised.yaml`
- orchestrator experiment type: `pretrain_repr`
- per-seed execution path: `trainer/self_supervised/run_pretrain.py`
- outputs run plan + summary tables + per-experiment telemetry under `docs/artifacts/p32_selfsup/runs/<run_id>/...`

The current stub:

- loads trajectories from existing artifacts (`docs/artifacts/p32`, `docs/artifacts/p13`, etc.)
- builds compact state/action features using existing `trainer/self_supervised/datasets.py`
- computes simple majority-label baseline metrics (`val_acc`, `val_loss`) instead of full contrastive/world-model training
- writes reproducible summaries for orchestrated comparison

## Entrypoints

Dry-run plan:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p32_self_supervised.ps1 -DryRun
```

Quick run (multi-seed smoke):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p32_self_supervised.ps1 -Quick
```

Direct stub invocation:

```powershell
python -B -m trainer.self_supervised.run_pretrain --config configs/experiments/p32_self_supervised.yaml
```

## Artifact Layout

Run-level:

- `docs/artifacts/p32_selfsup/runs/<run_id>/run_plan.json`
- `docs/artifacts/p32_selfsup/runs/<run_id>/summary_table.{csv,json,md}`
- `docs/artifacts/p32_selfsup/runs/<run_id>/telemetry.jsonl`

Experiment-level:

- `docs/artifacts/p32_selfsup/runs/<run_id>/<exp_id>/run_manifest.json`
- `docs/artifacts/p32_selfsup/runs/<run_id>/<exp_id>/progress.jsonl`
- `docs/artifacts/p32_selfsup/runs/<run_id>/<exp_id>/seeds_used.json`
- `docs/artifacts/p32_selfsup/runs/<run_id>/<exp_id>/selfsup_p32_runs/seed_*/summary.json`

Shared stub dataset outputs:

- `trainer_data/p32/selfsup_stub_dataset.jsonl`
- `docs/artifacts/p32/selfsup_stub_dataset_stats.json`
- `docs/artifacts/p32/selfsup_stub_training_summary_latest.json`

## Seed Strategy

Default sets are defined in `configs/experiments/p32_self_supervised.yaml -> seed_policy`:

- `regression_smoke`
- `train_default`
- `eval_default`

Guidance:

- quick validation: 2-3 seeds
- local model comparison: >=5 seeds
- always use `run_plan.json` + `seeds_used.json` as source of truth

## Known Limits

- no contrastive loss / masked modeling yet
- no joint optimization with BC/DAgger heads yet
- no direct policy improvement claim from this stub

## Next Steps

- add true representation objectives (contrastive + temporal consistency)
- export learned encoder weights for BC/DAgger warm-start
- integrate representation quality metrics into P22 ranking outputs
