# P33 Self-Supervised Experiments

## Goal

P33 adds a minimal self-supervised training plumbing that is independent from BC/DAgger mainline training gates:

- trace/fixture trajectories -> normalized self-supervised dataset
- lightweight model training
- artifactized summary outputs for reproducibility

This path is intentionally **experimental** and is used to validate data/infra wiring first, not to claim immediate policy strength gains.

## Task Definition (Current)

- task id: `next_score_delta_bucket`
- input: per-step state/action features from trajectory records
- target: bucketized next-step `score_delta` (`low/mid/high`)
- auxiliary signal in dataset: `terminal_within_horizon` (binary, horizon default 3 steps)

No human annotation is required; labels are derived from observed simulator/oracle outcomes.

## Entrypoints

Direct python run:

```powershell
python -B trainer/experiments/selfsupervised_p33.py --config configs/experiments/p33_selfsup.yaml
```

PowerShell wrapper:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_p33_selfsup.ps1
```

P22 integration (optional row):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_p22.ps1 -Only quick_selfsup_p33 -SeedLimit 2
```

## Config

Primary config:

- `configs/experiments/p33_selfsup.yaml`

JSON sidecar for environments without PyYAML:

- `configs/experiments/p33_selfsup.json`

Key fields:

- `task.bucket_thresholds`
- `task.horizon_steps`
- `data.sources`
- `data.max_samples`
- `model.hidden_dim`
- `training.{epochs,batch_size,lr,val_ratio}`

## Data Sources

Default P33 sources:

- `docs/artifacts/p13` (`p13_drift_fixture`)
- `docs/artifacts/p32` (`oracle_traces`)

Each run writes:

- dataset jsonl: `trainer_data/p33/selfsup_dataset.jsonl`
- dataset stats: `docs/artifacts/p33/selfsup_dataset_stats.json`

## Training Outputs

Per-run outputs:

- run dir: `trainer_runs/p33_selfsup/<timestamp>/`
- `progress.jsonl`
- `selfsup_p33_epoch*.pt`
- `summary.json`

Global summary artifact:

- `docs/artifacts/p33/selfsup_training_summary_<timestamp>.json`

## Seed Guidance

- Regression/gate tasks: fixed explicit seeds, typically with `SeedLimit=2` for cost-controlled smoke.
- Comparison tasks: use larger fixed sets (8+) and inspect `seeds_used.json` from P22 runs.
- For P33 standalone runs, set `training.seed` explicitly in config or pass `--seed`.

## Known Limitations

- Current feature extractor is compact and intentionally simple.
- Dataset quality depends on available trajectory artifacts; empty/low-quality traces reduce signal.
- Metric quality indicates plumbing health only; it is not a direct champion-promotion criterion.

