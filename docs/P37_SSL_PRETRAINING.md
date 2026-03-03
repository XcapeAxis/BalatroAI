# P37 SSL Pretraining v1 (State/Trait Encoder)

## Goal

P37 SSL v1 builds a reusable state encoder from existing trace artifacts and wires it into P22 as first-class experiments:

- trace/episode -> SSL pair dataset
- contrastive pretraining on `(s_t, s_{t+1})`
- downstream probe on frozen embeddings
- multi-seed orchestration + artifactized reports

This is a research baseline, not a fully optimized representation stack.

## Data Pipeline

Code path:

- dataset builder: `trainer/experiments/ssl_dataset.py`
- source adapters reused from: `trainer/selfsup/data.py`

Supported sources:

- oracle traces (sim fixtures)
- real-like trace roots (for example P32 artifact roots)
- existing prebuilt dataset jsonl (if provided)

Each training sample stores:

- `obs`: dense state vector
- `next_obs`: next-step dense vector
- `delta_chips`: scalar future delta
- `reward_bucket`: coarse label (`neg/zero/pos`) for probe

Default artifact dataset path:

- `docs/artifacts/p37/ssl_datasets/latest/dataset.jsonl`

## Model and Objective

Code path:

- encoder model: `trainer/models/ssl_state_encoder.py`
- trainer: `trainer/experiments/ssl_trainer.py`

Architecture:

- `StateEncoder` reuses `BalatroEncoder` backbone.
- `SSLProjectionHead` maps encoder latent to contrastive projection space.

Objective (v1):

- next-step contrastive loss (InfoNCE style, in-batch negatives)
- track `train_loss`, `val_loss`, `val_pos_cos`, embedding std diagnostics

Trainer outputs:

- `metrics.json`
- `progress.jsonl`
- `loss_curve.csv`
- `ssl_encoder_epoch*.pt`
- `ssl_encoder_best.pt`

## P22 Integration

New rows in `configs/experiments/p22.yaml`:

- `quick_ssl_pretrain_v1` (`experiment_type: ssl_pretrain`)
- `quick_ssl_probe_v1` (`experiment_type: ssl_probe`)
- `ssl_pretrain_medium_v1` (longer/night run option)

Orchestrator dispatch:

- `trainer/experiments/orchestrator.py`
  - `_run_ssl_pretrain_seed_experiment(...)`
  - `_run_ssl_probe_seed_experiment(...)`

Quick command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Quick -Only quick_ssl_pretrain_v1,quick_ssl_probe_v1
```

## Downstream Probe

Code path:

- `trainer/experiments/ssl_probe.py`

Method:

- baseline: random frozen encoder + linear probe
- ssl warm-start: pretrained frozen encoder + linear probe
- task: predict reward bucket from state embedding

Probe outputs:

- `probe_metrics.json`
- `probe_report.md`

## Replay Relationship

- P37 SSL v1 directly consumes trajectory-style state transitions.
- P36 replay pipeline remains the stricter action-contract path for replay-first pretraining.
- Both can coexist:
  - P36 replay for contract/validity guarantees
  - P37 SSL for rapid encoder iteration on state-transition objectives

## Future Directions

- stronger augmentations and hard-negative sampling
- full PPO/A2C warm-start studies with frozen/unfrozen encoder variants
- larger multi-seed probe matrices in P22 night runs
- MCTS + prior / AlphaZero-like experiments using pretrained state encoder
