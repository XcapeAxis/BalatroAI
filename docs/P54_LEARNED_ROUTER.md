# P54 Learned Router / Meta-Controller v1

P54 upgrades the P48 rule-based hybrid controller with a lightweight learned router while keeping the original rule router as a safe fallback.

The milestone is intentionally conservative:

- the learned router is trained from routing traces plus arena/triage outcomes
- deployment still supports `rule`, `learned`, and `learned_with_rule_guard`
- real arena outcomes and regression triage remain the promotion authority

## Why P54 Exists

P48 proved that state-aware routing across controllers is useful and explainable, but its thresholds are hand-tuned. P54 adds a learnable meta-controller so routing can adapt to observed controller outcomes without discarding the P48 rule path.

The practical goal is not "fully autonomous routing." The goal is:

- use existing routing telemetry as supervision
- learn a compact controller-choice model
- keep guarded fallback behavior when uncertainty or feature coverage is weak

## Controller Set

The v1 learned router supports the same controller family used by P48:

- `policy_baseline`
- `policy_plus_wm_rerank`
- `search_baseline` when available
- `heuristic_baseline` when available

Arena-facing policy ids used for P54 compare:

- `hybrid_controller_rule`
- `hybrid_controller_learned`
- `hybrid_controller_learned_with_rule_guard`

## Routing Dataset Design

Primary modules:

- `trainer/hybrid/router_schema.py`
- `trainer/hybrid/router_labels.py`
- `trainer/hybrid/router_dataset.py`

Each dataset row stores at least:

- `sample_id`
- `routing_features`
- `available_controllers`
- `chosen_controller_rule`
- `target_controller_label`
- `target_controller_scores`
- `slice_*`
- `seed`
- `source_run_id`
- `campaign_run_id`
- `candidate_checkpoint_id`
- `champion_checkpoint_id`
- `arena_ref`
- `triage_ref`
- `valid_for_training`
- `label_source`
- `label_confidence`

Label provenance is explicit so the learned router can distinguish strong arena-backed labels from weak rule-derived labels.

### Labeling Strategy

P54 v1 uses a pragmatic mixed labeling strategy:

1. Prefer arena or ablation evidence when a controller clearly outperforms alternatives for a comparable slice/state bucket.
2. Fall back to rule-router decisions when only weak supervision exists.
3. Persist `label_source` and `label_confidence` so weak labels can be down-weighted or filtered later.

The dataset builder degrades gracefully when routing traces or arena artifacts are sparse. It writes stats and preview files instead of failing on missing historical coverage.

### Dataset Artifacts

Output root:

- `docs/artifacts/p54/router_dataset/<run_id>/`

Key files:

- `router_dataset_manifest.json`
- `router_dataset_stats.json`
- `router_dataset_stats.md`
- `router_samples_preview.json`
- `seeds_used.json`

Smoke entrypoint:

```powershell
python -m trainer.hybrid.router_dataset --quick
```

## Learned Router Model and Training

Primary modules:

- `trainer/hybrid/learned_router_model.py`
- `trainer/hybrid/learned_router_train.py`
- `trainer/hybrid/learned_router_eval.py`

The v1 model is intentionally small:

- lightweight MLP-style tabular feature model
- masked logits so unavailable controllers cannot be selected
- class weighting / imbalance handling for controller-heavy datasets
- calibration-friendly probability output

Training outputs:

- train/val split
- top-1 accuracy
- top-k accuracy when meaningful
- per-controller confusion summary
- per-slice evaluation
- checkpoint save + progress stream

Output root:

- `docs/artifacts/p54/router_train/<run_id>/`

Key files:

- `train_manifest.json`
- `metrics.json`
- `progress.jsonl`
- `best_checkpoint.txt`
- `confusion_matrix.json`
- `slice_eval.json`
- `seeds_used.json`

Smoke entrypoint:

```powershell
python -m trainer.hybrid.learned_router_train --quick
```

P54 defaults to the P50 CUDA-first resolver path. On the validated local setup this resolves to `.venv_trainer_cuda` and `learner_device=cuda:0`.

## Safe Deployment Modes

P54 keeps the router deployment contract explicit:

1. `rule`
2. `learned`
3. `learned_with_rule_guard`

The guarded mode falls back to the P48 rule router when one or more of these conditions hold:

- predicted controller is unavailable or invalid
- routing features are incomplete
- learned-router confidence is below threshold
- OOD-like score is high
- world-model uncertainty or other high-risk routing signals are elevated

Every routing event is logged with explainability fields including:

- selected router mode
- predicted probabilities
- rule choice
- learned choice
- guard trigger flag and reason
- final controller choice
- key routing feature values
- final action source

Inference traces are written to:

- `docs/artifacts/p54/router_inference/<run_id>/routing_trace.jsonl`

## Arena Ablation Design

P54 compares at least:

- `policy_baseline`
- `policy_plus_wm_rerank`
- `hybrid_controller_rule`
- `hybrid_controller_learned`
- `hybrid_controller_learned_with_rule_guard`

Optional:

- `search_baseline`

Outputs are slice-aware, checkpoint-aware, and campaign-aware:

- `summary_table.{json,md,csv}`
- `slice_eval.json`
- `routing_summary.json`
- `promotion_decision.json`
- `triage_report.json`

Output root:

- `docs/artifacts/p54/arena_ablation/<run_id>/`

The compare is intentionally arena-first. A router checkpoint is only interesting if it survives the same P39/P41 evaluation path as the rest of the project.

## Registry and Campaign Integration

P54 registers learned-router checkpoints through the P51 registry/state-machine path.

Registry expectations:

- `family = learned_router`
- `training_mode = p54_learned_router`
- dataset/train/eval/arena/triage refs are attached as available

Common status path:

- `draft`
- `smoke_passed`
- `arena_passed`
- `promotion_review`

P54 campaigns are stage-aware and resumeable with the following stage ids:

- `build_router_dataset`
- `train_learned_router`
- `eval_learned_router`
- `arena_ablation`
- `triage`
- `promotion_queue_update`
- `dashboard_build`

Resume validation artifact:

- `docs/artifacts/p54/campaign_resume_validation_20260307-161918.md`

## P22 Integration

P54 P22 rows:

- `p54_learned_router_smoke`
- `p54_learned_router_nightly`

Recommended commands:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -RunP54
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -RunP54 -Resume
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Quick
```

P22 summaries for P54 now surface:

- router checkpoint ids
- campaign state path
- registry snapshot path
- promotion queue path
- dashboard path
- training python
- device profile
- learner device

## Dashboard Surface

The dashboard now exposes P54-specific sections when artifacts are present:

- routing dataset size and label stats
- learned-router training progress
- latest learned-router checkpoints
- guard trigger rate
- controller-selection distribution
- campaign stage status
- promotion queue summary

Primary output:

- `docs/artifacts/dashboard/latest/index.html`

## Known Limitations

- label quality is only as good as the available arena/routing evidence; weak labels still exist in v1
- small or narrow datasets can cause controller collapse, which is why triage reports controller-overuse signals
- OOD detection is heuristic and should be treated as a safety aid, not a proof of generalization
- the learned router remains a lightweight model; it is not meant to replace deeper search/planning research
- promotion still depends on arena + triage results, not training accuracy

## Related Docs

- [P48_ADAPTIVE_HYBRID_CONTROLLER.md](P48_ADAPTIVE_HYBRID_CONTROLLER.md)
- [P39_POLICY_ARENA.md](P39_POLICY_ARENA.md)
- [P41_CLOSED_LOOP_V2.md](P41_CLOSED_LOOP_V2.md)
- [P49_GPU_MAINLINE_AND_DASHBOARD.md](P49_GPU_MAINLINE_AND_DASHBOARD.md)
- [P51_CHECKPOINT_REGISTRY_AND_CAMPAIGNS.md](P51_CHECKPOINT_REGISTRY_AND_CAMPAIGNS.md)
- [EXPERIMENTS_P22.md](EXPERIMENTS_P22.md)


## Legacy Alias

- P52 remains the legacy implementation label for historical artifacts and prior campaign outputs.
- P54 is the current first-class experiment/docs/artifact surface for the same learned-router stack.
