# P29 Train Batch

`trainer/train_batch_p29.py` executes multi-job candidate training from one YAML.

## Required Outputs

- `train_batch_manifest.json`
- `train_batch_summary.json`
- `train_batch_summary.md`
- per-candidate logs + summaries

## Candidate Types

- `bc_pv`: failure-prioritized BC/PV training
- `distill`: deploy-student distillation from targeted samples
- `hybrid_tuning`: search/risk threshold tuning candidate
- `rl_finetune`: short-budget RL finetune (allow-fail optional)

The manifest is consumed by `trainer/run_ablation.py --from-train-batch-manifest`.
