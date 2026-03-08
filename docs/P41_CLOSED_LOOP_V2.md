# P41 Closed-loop Improvement v2

P41 upgrades P40 from a runnable loop into an explainable and stability-oriented loop:

- replay lineage and lineage health checks
- shared slice labels across replay and arena
- curriculum-based staged replay mixing
- slice-aware champion gating with bootstrap/CI safeguards
- regression triage with source/slice/curriculum attribution

P41 keeps P40's conservative promotion stance: recommendation-only, no automatic champion swap.
P42 reuses this exact closed-loop shell and promotes RL candidate training (`rl_ppo_lite`) to default mainline mode.
P45 can plug into the same shell as an auxiliary world-model asset for wm-assisted arena comparisons.
P46 reuses the same lineage, replay-mix, arena, and triage surfaces for imagined replay augmentation.
P47 reuses the same arena and triage surfaces for uncertainty-aware world-model rerank ablations.
P48 reuses the same arena and triage surfaces for explainable hybrid-controller routing ablations.
P54 reuses the same arena and triage surfaces for learned-router dataset labeling, guarded deployment checks, and rule-vs-learned ablations.

## Architecture

```mermaid
flowchart LR
  A["Replay Sources\nP10/P13/P36/P39-failure"] --> B["ReplayMixer + Lineage\nreplay_mix_manifest.json"]
  B --> C["Lineage Check\nlineage_health.json/.md"]
  B --> D["CandidateTrain + Curriculum\ncurriculum_plan.json\ncurriculum_applied.jsonl"]
  E["Failure Mining\nfrom P39 arena artifacts"] --> B
  D --> F["Candidate checkpoint + metrics"]
  F --> G["P39 Arena Runner\n(candidate vs champion)"]
  G --> H["Champion Rules v2\nslice-aware + CI/bootstrap"]
  H --> I["Promotion decision\ncandidate_decision + slice_breakdown"]
  I --> J["Regression Triage\ntriage_report.json/.md"]
  C --> J
  D --> J
```

## Module Inputs / Outputs

| Module | Inputs | Outputs |
|---|---|---|
| `trainer.closed_loop.replay_mixer` | `configs/experiments/p41_replay_mix_*.yaml`; source roots | `docs/artifacts/p41/replay_mixer/<run_id>/replay_mix_manifest.json`, `replay_mix_stats.json`, `replay_mix_stats.md`, `seeds_used.json` |
| `trainer.closed_loop.replay_lineage` | replay records selected by mixer | `docs/artifacts/p41/replay_lineage/<run_id>/replay_lineage_summary.json`, `replay_lineage_summary.md` |
| `trainer.closed_loop.check_replay_lineage` | replay manifest | `lineage_health.json`, `lineage_health.md` |
| `trainer.closed_loop.failure_mining` | P39 `episode_records.jsonl`, `summary_table.json`, `bucket_metrics.json`, optional `candidate_decision.json` | `docs/artifacts/p41/failure_mining/<run_id>/failure_pack_manifest.json`, `failure_pack_stats.json`, `failure_pack_stats.md` |
| `trainer.closed_loop.candidate_train` | replay mix manifest + candidate config + optional curriculum config | `docs/artifacts/p41/candidate_train/<run_id>/candidate_train_manifest.json`, `metrics.json`, `progress.jsonl`, `curriculum_plan.json`, `curriculum_applied.jsonl`, `best_checkpoint.txt` |
| `trainer.policy_arena.arena_runner` | arena config/policies/seeds | `docs/artifacts/p41/closed_loop_runs/<run_id>/arena_runs/<arena_run_id>/summary_table.json`, `bucket_metrics.json`, `episode_records.jsonl` |
| `trainer.policy_arena.champion_rules` | arena summary + optional episode records | `promotion_decision.json/.md`, `slice_decision_breakdown.json/.md` |
| `trainer.closed_loop.regression_triage` | closed-loop refs + baseline refs + lineage/curriculum outputs | `triage_report.json`, `triage_report.md` |

## Replay Lineage Contract

P41 replay manifest entries include these key fields:

- `sample_id`
- `source_type` (`p10_long_episode`, `p13_dagger_or_real`, `selfsup_replay`, `arena_failures`, ...)
- `source_run_id`
- `source_artifact_path`
- `source_seed`
- `episode_id`
- `step_id`
- `generation_method`
- `valid_for_training`
- `lineage_version`

P46 extends the same lineage contract for synthetic replay rows with:

- `source_type = imagined_world_model`
- `world_model_checkpoint`
- `imagination_horizon`
- `uncertainty_score`
- `uncertainty_gate_passed`
- `root_sample_id`
- `imagined_step_idx`

Lineage summary reports:

- source share ratios
- seed coverage
- episode coverage
- missing-field rates

Lineage health checker reports:

- required-field missing ratio
- source path existence checks (`warning` when cleaned/missing)
- overall status (`ok` / `warn` / `error`)

## Training Mode Contract (P43)

P41/P42 closed-loop outputs now persist lane-aware training metadata:

- `training_mode`
- `training_mode_category` (`mainline` / `legacy_baseline`)
- `fallback_used`
- `fallback_reason`
- `legacy_paths_used`

These fields are emitted in candidate manifests, closed-loop `run_manifest.json`, and summary tables.

## Auxiliary World Model Contract (P45)

Closed-loop manifests now also reserve `auxiliary_assets` for optional P45 dependencies:

- `world_model_enabled`
- `world_model_checkpoint`
- `world_model_assist_mode`
- `world_model_weight`
- `world_model_uncertainty_penalty`

This keeps the dependency chain explicit when a candidate or arena compare uses `heuristic_wm_assist` or another wm-assisted policy variant.

## Model-Based Search Contract (P47)

P47 keeps replay/training untouched but extends arena/triage manifests with:

- `wm_assist_enabled`
- `wm_checkpoint`
- `wm_eval_ref`
- `assist_mode=rerank`
- `policy_assist_map_json`
- `candidate_policy` / `baseline_policy` for rerank ablations

## Adaptive Hybrid Controller Contract (P48)

P48 keeps replay/training unchanged but extends arena/triage manifests with:

- `hybrid_controller`
- `controller_registry_json`
- `routing_summary_json`
- `wm_assist_enabled`
- controller-selection traces under `router_traces/<run_id>/routing_trace.jsonl`

## Learned Router Contract (P54)

P54 keeps replay/candidate training unchanged but extends arena/triage manifests with:

- `router_dataset_manifest_json`
- `router_train_manifest_json`
- `router_checkpoint_id`
- `router_checkpoint_path`
- `router_mode`
- `routing_summary_json`
- `routing_trace_jsonl`
- `guard_trigger_rate`

## Learned Router Calibration + Canary Contract (P56)

P56 keeps replay/candidate training unchanged but extends arena, triage, registry, and promotion manifests with:

- `calibration_metrics_json`
- `reliability_bins_json`
- `guard_tuning_results_json`
- `recommended_guard_config_json`
- `canary_eval_summary_json`
- `deployment_mode_recommendation`

## Imagination Augmentation Contract (P46)

When P46 replay augmentation is enabled, candidate and replay artifacts also record:

- `imagined_enabled`
- `imagined_filter_mode`
- `imagined_fraction`
- replay-source stats for `imagined_world_model`

This lets triage separate model-generated replay effects from real-source shifts.

## Unified Slice Labels

P41 uses one label rule set (`trainer/common/slices.py`) for replay and arena:

- `slice_stage`: `early` / `mid` / `late` / `unknown`
- `slice_resource_pressure`: `low` / `medium` / `high` / `unknown`
- `slice_action_type`: `play` / `discard` / `shop` / `consumable` / `transition` / `unknown`
- mechanism-sensitive labels:
  - `slice_position_sensitive`: `true` / `false` / `unknown`
  - `slice_stateful_joker_present`: `true` / `false` / `unknown` (stub-compatible)

Missing upstream fields degrade to `unknown` instead of hard failures.

## Curriculum Scheduler

P41 supports staged replay weighting during candidate training:

1. `stabilize`: prioritize `p10_long_episode` + `p13_dagger_or_real`, low `arena_failures`
2. `hardening`: increase `arena_failures` and high-pressure slices
3. `balance`: rebalance to avoid overfitting failure tails

Each phase logs applied plan to `curriculum_applied.jsonl` (including seeds and effective source/slice weights).

## Slice-aware Champion Rules v2

Champion decision combines global constraints and slice-local statistics:

- hard safety gates: invalid/timeout regression caps
- global uplift checks: score/win deltas
- slice-aware CI/bootstrap checks:
  - compute candidate-champion diffs per slice
  - compute bootstrap CI for key metrics
  - if critical slice significantly degrades, downgrade to `observe/reject`
  - if sample count is low, mark `ci_status=insufficient_samples` and stay conservative

Decision artifacts:

- `promotion_decision.json/.md`
- `slice_decision_breakdown.json/.md`

## Regression Triage

Triggered after arena decision, triage reports:

1. overall degraded metrics
2. top regressed slices
3. likely source attribution (replay sources and source seeds from lineage)
4. curriculum drift vs baseline run
5. data quality anomalies (`lineage_health`, `invalid_for_training` shifts)

If baseline is missing, triage emits `baseline_missing` with a non-crashing report.

P46-specific triage fields:

- `imagined_source_impact`
- `top_degrading_imagined_slices`
- `suggested_imagination_adjustments`

P47-specific triage fields:

- `world_model_assist_impact`
- `uncertainty_penalty_sensitivity`
- `horizon_sensitivity`
- `top_degrading_slices_with_wm`

P48-specific triage fields:

- `routing_decision_impact`
- `controller_selection_distribution`
- `top_degrading_slices_for_hybrid`
- `search_budget_sensitivity`
- `wm_uncertainty_gate_impact`

P54-specific triage fields:

- `learned_router_impact`
- `top_degrading_slices_for_learned_router`
- `top_degrading_slices_for_guarded_router`
- `learned_router_selection_overuse`
- `learned_router_guard_condition_correlation`
- `learned_router_routing_variants`

P56-specific triage fields:

- `guard_effectiveness`
- `canary_effectiveness`
- `top_degrading_slices_for_canary`
- `controller_selection_distribution`
- `controller_overuse_or_collapse_signals`
- `deployment_mode_recommendation`

## Run Commands

Quick smoke (standalone):

```powershell
python -m trainer.closed_loop.closed_loop_runner --config configs/experiments/p41_closed_loop_v2_smoke.yaml --quick
```

P22 quick (includes P41 smoke row):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Quick
```

P42 RL candidate quick (same closed-loop shell, RL training mode):

```powershell
python -m trainer.closed_loop.closed_loop_runner --config configs/experiments/p42_closed_loop_rl_smoke.yaml --quick
```

Closed-loop run with world-model-assisted arena compare uses the same entrypoint:

```powershell
python -m trainer.closed_loop.closed_loop_runner --config configs/experiments/p42_closed_loop_rl_smoke.yaml --quick
```

Add a `world_model` block to the closed-loop config with `enabled/checkpoint/assist_mode/weight/uncertainty_penalty` to turn on the auxiliary asset path.

Nightly style:

```powershell
python -m trainer.closed_loop.closed_loop_runner --config configs/experiments/p41_closed_loop_v2_nightly.yaml
```

Lineage check:

```powershell
python -m trainer.closed_loop.check_replay_lineage --manifest docs/artifacts/p41/replay_mixer/<run_id>/replay_mix_manifest.json --out-dir docs/artifacts/p41/replay_lineage/<run_id>
```

Slice label smoke:

```powershell
python -m trainer.closed_loop.slice_smoke
```

## Artifact Layout

- `docs/artifacts/p41/replay_mixer/<run_id>/`
- `docs/artifacts/p41/replay_lineage/<run_id>/`
- `docs/artifacts/p41/failure_mining/<run_id>/`
- `docs/artifacts/p41/candidate_train/<run_id>/`
- `docs/artifacts/p41/closed_loop_runs/<run_id>/`
  - `run_manifest.json`
  - `run_manifest.json -> auxiliary_assets.world_model_*`
  - `promotion_decision.json/.md`
  - `slice_decision_breakdown.json/.md`
  - `triage_report.json/.md`
  - `summary_table.{json,csv,md}`
- `docs/artifacts/p41/slice_smoke_<timestamp>.json`

## Known Gaps / Degrade Paths

- Missing replay sources keep the loop runnable (`stub/warn`) but reduce attribution quality.
- Slice CI/bootstrap conclusions on low-count slices degrade to `insufficient_samples`.
- `model_policy` inference quality depends on checkpoint/runtime availability.
- Candidate training is mainline-first (RL/selfsup order), while BC/DAgger remain opt-in legacy baselines.
- P41 does not auto-apply champion replacement; recommendation remains manual-gated.
- P42 extends candidate training with PPO-lite but keeps the same recommendation-only promotion boundary.
- P45 world-model assist remains optional and heuristic-only; P41/P42 still rely on real arena outcomes for final gating.
- P46 imagined replay attribution is best-effort and only as good as world-model uncertainty plus replay-lineage completeness.
- P47 rerank attribution is summary-table based and does not claim causal proof; it is meant to highlight suspicious horizons and uncertainty settings for follow-up runs.
