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
| Fast smoke | `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Quick` | mainline smoke set with multi-seed materialization (includes P31/P33/P36 + P37 SSL + RL smoke + P39 arena smoke row + P40/P41/P42 + P45 world-model smoke row + P46 imagination smoke row + P47 model-based search smoke row + P48 hybrid-controller smoke row + P49 GPU-mainline row + P50 real-CUDA validation row + P51 registry/campaign row + P54 learned-router row + P56 calibration/canary row + P53 background-execution row) |
| Fast smoke + legacy probe | `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Quick -IncludeLegacy` | adds opt-in legacy BC/DAgger probe row(s) |
| Legacy only quick | `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Quick -LegacyOnly` | runs only legacy baseline probe row(s) |
| Multi-seed quick compare | `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Only quick_baseline,quick_candidate -Seeds "AAAAAAA,BBBBBBB,CCCCCCC"` | 2 strategies x 3 seeds |
| Fast smoke (verbose) | `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Quick -VerboseLogs` | adds per-seed/per-stage console logs |
| Nightly style | `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Nightly -Resume` | larger seed set + resume |
| P45 smoke only | `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -RunP45` | runs `p45_world_model_smoke` with orchestrator defaults |
| P46 smoke only | `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -RunP46` | runs `p46_imagination_smoke` with orchestrator defaults |
| P47 smoke only | `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -RunP47` | runs `p47_wm_search_smoke` with orchestrator defaults |
| P48 smoke only | `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -RunP48` | runs `p48_hybrid_controller_smoke` with orchestrator defaults |
| P49 smoke only | `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -RunP49` | runs `p49_gpu_mainline_smoke` with readiness guard + dashboard build |
| P50 smoke only | `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -RunP50` | runs `p50_gpu_validation_smoke` with CUDA-first python resolver + readiness guard + dashboard build |
| P51 smoke only | `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -RunP51` | runs `p51_registry_smoke` with checkpoint registry snapshots, promotion queue export, dashboard build, and campaign state artifacts |
| P54 smoke only | `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -RunP54` | runs `p54_learned_router_smoke` with router dataset build, learned-router training, guarded ablation compare, checkpoint registration, campaign state, promotion queue, and dashboard build |
| P56 smoke only | `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -RunP56` | runs `p56_router_calibration_smoke` with multi-seed benchmark, calibration, guard tuning, canary eval, registry refs, promotion queue refresh, and dashboard build |
| P53 smoke only | `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -RunP53` | runs `p53_background_ops_smoke` with background-mode validation, window-mode recording, campaign state, ops-ui metadata, promotion queue refresh, and dashboard build |
| P53 smoke + explicit mode | `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -RunP53 -WindowMode offscreen` | keeps the requested background mode explicit in the run summary |
| Start local Ops UI | `powershell -ExecutionPolicy Bypass -File scripts\run_ops_ui.ps1` | starts the localhost-only P53 operations console on `127.0.0.1:8765` by default |
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
  - `category` (`mainline`, `legacy_baseline`, `required_validation`, ...)
  - `default_enabled` (`false` rows are excluded unless explicitly requested)
  - `backend`, `policy`
  - `experiment_type` (optional, e.g. `selfsup_pretrain`, `selfsup_p33`, `selfsup_future_value`, `selfsup_action_type`, `ssl_pretrain`, `ssl_probe`, `rl_selfplay`, `policy_arena`, `closed_loop_improvement`, `closed_loop_improvement_v2`, `closed_loop_rl_candidate`, `world_model_train`, `world_model_eval`, `world_model_assist_compare`, `imagination_augmented_candidate`, `p46_imagination`, `world_model_rerank_eval`, `p47_wm_search`, `hybrid_controller_eval`, `p48_hybrid_controller`, `checkpoint_registry_campaign`, `p51_registry_campaign`, `router_dataset_build`, `learned_router_train`, `learned_router_ablation`, `p54_learned_router_campaign`, `p56_router_calibration_campaign`, `p53_background_ops_campaign`)
  - `seed_mode` (`regression_fixed` or `nightly`)
  - `seeds` (optional explicit seed override list)
  - `window_mode`, `window_mode_fallback` (when the row manages the real game window)
  - `gate_flag` (passed to `scripts/run_regressions.ps1`)
  - `stages` (`sanity/gate/dataset/train/eval`)
  - `eval` settings
- `evaluation`:
  - `primary_metric`
  - `gates.baseline_gate`
  - `promotion.min_delta`

## Training Lane Policy (P43)

- Default lane: `mainline` (+ `required_validation` support rows).
- Legacy lane: `legacy_baseline` rows are opt-in and disabled by default.
- New wrapper flags:
  - `-IncludeLegacy`: include `legacy_baseline` rows
  - `-LegacyOnly`: run only `legacy_baseline` rows
- `run_plan.json` now records:
  - `selected_categories`
  - `include_legacy`
  - `legacy_only`
  - per-experiment `category` and `default_enabled`
- `summary_table.{csv,json,md}` now includes category/default columns for triage.

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

## P37 SSL Pretraining Integration (State/Trait Encoder v1)

P37 adds two new orchestrated rows focused on representation pretraining and downstream probe:

- `quick_ssl_pretrain_v1` (`experiment_type: ssl_pretrain`)
- `quick_ssl_probe_v1` (`experiment_type: ssl_probe`)
- optional longer run: `ssl_pretrain_medium_v1`

Reference configs:

- `configs/experiments/p37_ssl_pretrain.yaml`
- `configs/experiments/p37_ssl_probe.yaml`

Task intent:

- `ssl_pretrain`: next-step contrastive objective on `(s_t, s_{t+1})` pairs, backed by trace-derived samples.
- `ssl_probe`: frozen-encoder linear probe on reward-bucket labels, with baseline-vs-SSL warm-start comparison.

Artifacts per seed:

- `.../quick_ssl_pretrain_v1/ssl_pretrain_runs/seed_*/metrics.json`
- `.../quick_ssl_probe_v1/ssl_probe_runs/seed_*/probe_metrics.json`

Recent smoke run (preliminary): `run_id=20260303-225348`

- seeds used: `AAAAAAA`, `BBBBBBB` (see each `seeds_used.json`)
- `quick_ssl_pretrain_v1` summary:
  - mean score: `3.1985`
  - final_loss: `2.2710`
- `quick_ssl_probe_v1` summary:
  - mean score: `3.4923`
  - final_loss: `0.9125`
  - per-seed probe can be mixed at this scale (small-data preliminary):
    - seed `AAAAAAA`: baseline_acc `0.9000`, ssl_acc `0.9000`, delta `0.0000`

## P36 Replay Pipeline v1 (Unified Real/Sim Replay Contract)

P36 replay v1 introduces a stricter data path for pretraining experiments:

- replay ingestion modules: `trainer/replay/schema.py`, `trainer/replay/ingest_real.py`, `trainer/replay/ingest_sim.py`, `trainer/replay/storage.py`
- pretrain entrypoint: `python -B -m trainer.experiments.selfsup_train --config configs/experiments/p22_selfsup_smoke.yaml`
- objective set: `mask`, `next_delta`, `hybrid`

Replay rows include `valid_for_training` / `invalid_reason` and training defaults to `valid_only=true`.

Quick smoke flow:

```powershell
python -B -m trainer.replay.storage --real-roots docs/artifacts/p32 docs/artifacts/p13 --sim-roots sim/tests/fixtures_runtime docs/artifacts/p32/smoke_position_fixture --out-dir docs/artifacts/p36/replay/smoke_latest --max-episodes-per-source 6
python -B -m trainer.experiments.selfsup_train --config configs/experiments/p22_selfsup_smoke.yaml
```

Generated outputs:

- replay dataset: `docs/artifacts/p36/replay/*/replay_steps.jsonl`
- replay summary: `docs/artifacts/p36/replay/*/replay_summary.{json,md}`
- selfsup run summary: `docs/artifacts/p36/selfsup_replay/<run_id>/summary.json`

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

## P39 Policy Arena Integration

P39 introduces `experiment_type: policy_arena` so P22 can run unified multi-policy comparisons under the same seed/materialization pipeline.

Reference rows in `configs/experiments/p22.yaml`:

- `p39_policy_arena_smoke` (quick/gate)
- `p39_policy_arena_nightly` (nightly)

Key eval fields:

- `policies`: policy ids (`heuristic_baseline`, `search_expert`, `model_policy`, ...)
- `episodes_per_seed`
- `max_steps`
- `mode` (`long_episode` for v1)
- `candidate_policy` / `champion_policy`
- `enable_champion_rules`

Generated artifacts:

- `docs/artifacts/p22/runs/<run_id>/p39_summary.json`
- `docs/artifacts/p22/runs/<run_id>/p39_policy_arena_smoke/policy_arena_runs/seed_*/arena_runs/<seed_run_id>/summary_table.json`
- `docs/artifacts/p22/runs/<run_id>/p39_policy_arena_smoke/policy_arena_runs/seed_*/arena_runs/<seed_run_id>/bucket_metrics.json`
- optional champion decision outputs under `.../champion_eval/`

## P53 Background Execution + Ops UI Integration

P53 adds `experiment_type: p53_background_ops_campaign` so P22 can validate background execution and publish the local ops-console metadata as part of the same orchestrated run.

Reference rows in `configs/experiments/p22.yaml`:

- `p53_background_ops_smoke`
- `p53_background_ops_nightly`

Wrapper flags:

- `-RunP53`
- `-WindowMode`
- `-WindowModeFallback`
- `-StartOpsUI`

Runtime/report additions:

- `window_mode`
- `background_validation_ref`
- `ops_ui_path`

Generated artifacts:

- `docs/artifacts/p22/runs/<run_id>/p53_background_ops_smoke/campaign_runs/seed_*/campaign_state.json`
- `docs/artifacts/p22/runs/<run_id>/p53_background_ops_smoke/campaign_runs/seed_*/checkpoint_registry_snapshot.json`
- `docs/artifacts/p22/runs/<run_id>/p53_background_ops_smoke/campaign_runs/seed_*/promotion_queue.json`
- `docs/artifacts/p53/background_mode_validation/<run_id>/background_mode_validation.json`
- `docs/artifacts/p53/ops_ui/latest/ops_ui_state.json`
- `docs/artifacts/p53/ops_audit/ops_audit.jsonl`

Operational notes:

- the orchestrator records the effective window mode, not just the requested one
- background-mode validation is reused by `run_p22.ps1`, readiness reporting, the dashboard, and the ops UI
- the ops UI reads existing artifacts instead of maintaining a parallel state store

## P40 Closed-loop Improvement Integration

P40 introduces `experiment_type: closed_loop_improvement` so P22 can orchestrate replay-mix -> failure-mining -> candidate-train -> arena-gated recommendation in one experiment row.

Reference rows in `configs/experiments/p22.yaml`:

- `p40_closed_loop_smoke` (quick/gate)
- `p40_closed_loop_nightly` (nightly)

Key eval fields:

- `config`: closed-loop config path (`configs/experiments/p40_closed_loop_smoke.yaml` / `...nightly.yaml`)
- `quick`: whether to run reduced budgets inside the loop
- `timeout_sec`: per-seed closed-loop timeout
- `candidate_policy` / `champion_policy`: promotion comparison focus

Generated artifacts:

- `docs/artifacts/p22/runs/<run_id>/p40_summary.json`
- `docs/artifacts/p22/runs/<run_id>/p40_closed_loop_smoke/closed_loop_runs/seed_*/run_manifest.json`
- `docs/artifacts/p22/runs/<run_id>/p40_closed_loop_smoke/closed_loop_runs/seed_*/promotion_decision.json`
- `docs/artifacts/p22/runs/<run_id>/p40_closed_loop_smoke/closed_loop_runs/seed_*/summary_table.json`

Operational notes:

- `scripts/run_p22.ps1 -Quick` includes `p40_closed_loop_smoke` by default.
- `seeds_used.json` is still recorded per experiment; closed-loop internals also emit per-module seed manifests.
- P40 v1 does not auto-replace champion metadata; output is recommendation-only.

## P41 Closed-loop Improvement v2 Integration

P41 introduces `experiment_type: closed_loop_improvement_v2` so P22 can orchestrate replay-lineage -> failure-mining -> curriculum-based candidate-train -> slice-aware arena gating -> regression triage.

Reference rows in `configs/experiments/p22.yaml`:

- `p41_closed_loop_v2_smoke` (quick/gate)
- `p41_closed_loop_v2_nightly` (nightly)

Key eval fields:

- `config`: closed-loop config path (`configs/experiments/p41_closed_loop_v2_smoke.yaml` / `...nightly.yaml`)
- `quick`: reduced-budget mode for local smoke
- `timeout_sec`: per-seed closed-loop timeout
- `candidate_policy` / `champion_policy`: promotion comparison focus
- regression triage switch is configured inside the closed-loop config (`regression_triage.enabled`)

Generated artifacts:

- `docs/artifacts/p22/runs/<run_id>/p41_summary.json`
- `docs/artifacts/p22/runs/<run_id>/p41_closed_loop_v2_smoke/closed_loop_runs/seed_*/run_manifest.json`
- `docs/artifacts/p22/runs/<run_id>/p41_closed_loop_v2_smoke/closed_loop_runs/seed_*/promotion_decision.json`
- `docs/artifacts/p22/runs/<run_id>/p41_closed_loop_v2_smoke/closed_loop_runs/seed_*/triage_report.json`
- `docs/artifacts/p22/runs/<run_id>/p41_closed_loop_v2_smoke/closed_loop_runs/seed_*/summary_table.json`

Operational notes:

- `scripts/run_p22.ps1 -Quick` includes `p41_closed_loop_v2_smoke` by default.
- multi-seed materialization remains in `seeds_used.json` per experiment.
- per-seed `triage_report.json` includes both `source_attribution` and `seed_attribution`.
- P41 keeps conservative recommendation-only promotion behavior; no automatic champion replacement.

## P42 RL Candidate Pipeline Integration

P42 introduces `experiment_type: closed_loop_rl_candidate` so P22 can orchestrate RL candidate training inside the existing closed-loop shell.

Reference rows in `configs/experiments/p22.yaml`:

- `p42_rl_candidate_smoke` (quick/gate)
- `p42_rl_candidate_nightly` (nightly)

Key eval fields:

- `config`: closed-loop config path (`configs/experiments/p42_closed_loop_rl_smoke.yaml` / `...nightly.yaml`)
- `quick`: reduced-budget mode for local smoke
- `timeout_sec`: per-seed closed-loop timeout
- `candidate_policy` / `champion_policy`: arena comparison focus

Generated artifacts:

- `docs/artifacts/p22/runs/<run_id>/p42_summary.json`
- `docs/artifacts/p22/runs/<run_id>/p42_rl_candidate_smoke/closed_loop_runs/seed_*/run_manifest.json`
- `docs/artifacts/p22/runs/<run_id>/p42_rl_candidate_smoke/closed_loop_runs/seed_*/promotion_decision.json`
- `docs/artifacts/p22/runs/<run_id>/p42_rl_candidate_smoke/closed_loop_runs/seed_*/triage_report.json`
- `docs/artifacts/p22/runs/<run_id>/p42_rl_candidate_smoke/closed_loop_runs/seed_*/candidate_train/rl_train/reward_config.json`
- `docs/artifacts/p22/runs/<run_id>/p42_rl_candidate_smoke/closed_loop_runs/seed_*/candidate_train/rl_train/warnings.log`

Operational notes:

- `scripts/run_p22.ps1 -Quick` includes `p42_rl_candidate_smoke` by default.
- multi-seed materialization remains mandatory (`seeds_used.json` per experiment).
- P42 reuses P41 slice-aware champion rules and regression triage outputs.
- RL candidate path is v1/research-grade and still guarded by recommendation-only promotion flow.

## P45 World Model / Latent Planning Integration

P45 introduces `experiment_type: world_model_train` so P22 can orchestrate dataset build -> world-model train -> eval -> wm-assisted arena compare in one experiment row.

Reference rows in `configs/experiments/p22.yaml`:

- `p45_world_model_smoke` (quick/gate)
- `p45_world_model_nightly` (nightly)

Key eval fields:

- `config`: world-model config path (`configs/experiments/p45_world_model_smoke.yaml` / `...nightly.yaml`)
- `quick`: reduced-budget mode for local smoke
- `candidate_policy`: wm-assisted arena policy id (defaults to `heuristic_wm_assist`)
- `champion_policy`: baseline comparison policy id

Generated artifacts:

- `docs/artifacts/p22/runs/<run_id>/p45_world_model_smoke/world_model_runs/seed_*/seed_*/train_manifest.json`
- `docs/artifacts/p22/runs/<run_id>/p45_world_model_smoke/world_model_runs/seed_*/seed_*/metrics.json`
- `docs/artifacts/p22/runs/<run_id>/p45_world_model_smoke/world_model_runs/seed_*/seed_*/progress.jsonl`
- referenced eval/assist artifacts under `docs/artifacts/p45/wm_eval/<run_id>/` and `docs/artifacts/p45/wm_assist_compare/<run_id>/`

Summary-table metrics surfaced by orchestrator:

- `world_model_reward_prediction_error`
- `world_model_latent_transition_error`
- `world_model_uncertainty_pearson`
- `world_model_best_checkpoint`

Operational notes:

- `scripts/run_p22.ps1 -Quick` includes `p45_world_model_smoke` by default.
- `scripts/run_p22.ps1 -RunP45` selects `p45_world_model_smoke` or `p45_world_model_nightly`.
- multi-seed materialization remains mandatory (`seeds_used.json` per experiment).
- P45 remains research-grade: the arena compare uses the world model as a one-step heuristic, not as a simulator replacement.

## P46 Imagination / Dyna-style Integration

P46 introduces `experiment_type: imagination_augmented_candidate` so P22 can orchestrate world-model bootstrap/checkpoint lookup -> imagination rollout -> replay mixing -> candidate ablation -> arena compare -> triage in one experiment row.

Reference rows in `configs/experiments/p22.yaml`:

- `p46_imagination_smoke` (quick/gate)
- `p46_imagination_nightly` (nightly)

Key eval fields:

- `config`: imagination config path (`configs/experiments/p46_imagination_smoke.yaml` / `...nightly.yaml`)
- `quick`: reduced-budget mode for local smoke
- recipe coverage for `real_only`, `real_plus_imagined`, and `real_plus_imagined_filtered`
- world-model bootstrap/checkpoint reuse policy

Generated artifacts:

- `docs/artifacts/p22/runs/<run_id>/p46_summary.json`
- `docs/artifacts/p22/runs/<run_id>/p46_imagination_smoke/imagination_runs/seed_*/pipeline_summary.json`
- referenced rollout outputs under `docs/artifacts/p46/imagination_rollouts/<run_id>/`
- referenced arena/triage outputs under `docs/artifacts/p46/{arena_compare,triage}/<run_id>/`

Summary-table metrics surfaced by orchestrator:

- `p46_real_only_score`
- `p46_filtered_score`
- `p46_filtered_delta_vs_real_only`
- `p46_imagined_acceptance_rate`
- `p46_imagined_sample_count`

Operational notes:

- `scripts/run_p22.ps1 -Quick` includes `p46_imagination_smoke` by default.
- `scripts/run_p22.ps1 -RunP46` selects `p46_imagination_smoke` or `p46_imagination_nightly`.
- multi-seed materialization remains mandatory (`seeds_used.json` per experiment).
- P46 keeps imagined replay auxiliary: short horizon, uncertainty-gated, and fraction-capped.

## P47 World-Model Assisted Search Integration

P47 introduces `experiment_type: world_model_rerank_eval` so P22 can orchestrate candidate generation -> lookahead planner -> wm-rerank arena ablation -> champion decision -> triage in one experiment row.

Reference rows in `configs/experiments/p22.yaml`:

- `p47_wm_search_smoke` (quick/gate)
- `p47_wm_search_nightly` (nightly)

Key eval fields:

- `config`: rerank config path (`configs/experiments/p47_wm_search_smoke.yaml` / `...nightly.yaml`)
- `quick`: reduced-budget mode for local smoke
- `candidate_policy` / `champion_policy`
- per-policy rerank settings in `policy_assist_map.json`

Generated artifacts:

- `docs/artifacts/p22/runs/<run_id>/p47_summary.json`
- `docs/artifacts/p22/runs/<run_id>/p47_wm_search_smoke/model_based_search_runs/seed_*/pipeline_summary.json`
- referenced planner outputs under `docs/artifacts/p47/lookahead/<run_id>/`
- referenced arena/triage outputs under `docs/artifacts/p47/{arena_ablation,triage}/<run_id>/`

Summary-table metrics surfaced by orchestrator:

- `p47_baseline_score`
- `p47_candidate_score`
- `p47_best_variant_score`
- `p47_candidate_delta_vs_baseline`
- `p47_best_variant_delta_vs_baseline`

Operational notes:

- `scripts/run_p22.ps1 -Quick` includes `p47_wm_search_smoke` by default.
- `scripts/run_p22.ps1 -RunP47` selects `p47_wm_search_smoke` or `p47_wm_search_nightly`.
- multi-seed materialization remains mandatory (`seeds_used.json` per experiment).
- P47 is rerank-only and keeps real arena outcomes as the promotion authority.

## P48 Adaptive Hybrid Controller Integration

P48 introduces `experiment_type: hybrid_controller_eval` so P22 can orchestrate controller-registry export -> routing-feature smoke -> router smoke -> hybrid-controller arena ablation -> champion decision -> triage in one experiment row.

Reference rows in `configs/experiments/p22.yaml`:

- `p48_hybrid_controller_smoke` (quick/gate)
- `p48_hybrid_controller_nightly` (nightly)

Key eval fields:

- `config`: hybrid-controller config path (`configs/experiments/p48_hybrid_controller_smoke.yaml` / `...nightly.yaml`)
- `candidate_policy` / `champion_policy`
- router thresholds and controller availability via `policy_assist_map.json`

Generated artifacts:

- `docs/artifacts/p22/runs/<run_id>/p48_summary.json`
- `docs/artifacts/p22/runs/<run_id>/p48_hybrid_controller_smoke/hybrid_controller_runs/seed_*/pipeline_summary.json`
- referenced arena/triage outputs under `docs/artifacts/p48/{arena_ablation,triage}/<run_id>/`

Summary-table metrics surfaced by orchestrator:

- `p48_baseline_score`
- `p48_hybrid_score`
- `p48_wm_rerank_score`
- `p48_search_score`
- `p48_hybrid_delta_vs_baseline`

Operational notes:

- `scripts/run_p22.ps1 -Quick` includes `p48_hybrid_controller_smoke` by default.
- `scripts/run_p22.ps1 -RunP48` selects `p48_hybrid_controller_smoke` or `p48_hybrid_controller_nightly`.
- multi-seed materialization remains mandatory (`seeds_used.json` per experiment).
- P48 is an explainable router layer; real arena outcomes remain authoritative.

## P49 GPU Mainline Integration

P49 introduces `experiment_type: gpu_mainline_eval` so P22 can orchestrate a shared runtime-profile lane that bundles:

- service readiness guard
- CPU-rollout / GPU-learner defaults
- P42 RL closed-loop smoke
- P45 world-model smoke
- optional P44/P46 expansion in nightly config
- unified progress streams for dashboard consumption

Reference rows in `configs/experiments/p22.yaml`:

- `p49_gpu_mainline_smoke` (quick/gate)
- `p49_gpu_mainline_nightly` (nightly)

Key eval fields:

- `config`: `configs/experiments/p49_gpu_mainline_smoke.yaml` / `...nightly.yaml`
- `device_profile`: `single_gpu_mainline`
- dashboard output: `docs/artifacts/dashboard/latest/index.html`

Generated artifacts:

- `docs/artifacts/p22/runs/<run_id>/p49_gpu_mainline_smoke/gpu_mainline_runs/seed_*/gpu_mainline_summary.json`
- `docs/artifacts/p49/readiness/<run_id>/service_readiness_report.json`
- `docs/artifacts/dashboard/latest/index.html`

Operational notes:

- `scripts/run_p22.ps1 -Quick` includes `p49_gpu_mainline_smoke` by default.
- `scripts/run_p22.ps1 -RunP49` selects `p49_gpu_mainline_smoke` or `p49_gpu_mainline_nightly`.
- `scripts/run_p22.ps1` now prints readiness-report and dashboard paths after successful runs.
- no-CUDA hosts degrade to CPU and keep that downgrade in `runtime_profile.json`.

## P50 Real CUDA Validation Integration

P50 adds `experiment_type: p50_gpu_validation` so P22 can prove that the local training path really reached CUDA instead of only exercising CPU fallback.

Reference rows in `configs/experiments/p22.yaml`:

- `p50_gpu_validation_smoke`
- `p50_gpu_validation_nightly`

What P50 adds on top of P49:

- shared training-python resolver that prefers `.venv_trainer_cuda`
- explicit CUDA-required mode for `-RunP50`
- real-CUDA smoke for P42 RL candidate and P45 world model
- benchmark-derived profile guidance for 12 GB single-GPU hosts

Key artifacts:

- `docs/artifacts/p22/runs/<run_id>/p50_gpu_validation_smoke/run_manifest.json`
- `docs/artifacts/p22/runs/<run_id>/p50_gpu_validation_smoke/gpu_mainline_runs/seed_*/gpu_mainline_summary.json`
- `docs/artifacts/p50/gpu_diagnose_<timestamp>.json`
- `docs/artifacts/p50/benchmarks/<run_id>/benchmark_summary.md`

Summary/runtime fields now surfaced in P22 artifacts for this lane:

- `training_python`
- `device_profile`
- `learner_device`
- `dashboard_path`
- `readiness_report_path`

Operational notes:

- `scripts/run_p22.ps1 -Quick` includes `p50_gpu_validation_smoke` by default.
- `scripts/run_p22.ps1 -RunP50` requires CUDA in resolver selection and fails early if the CUDA env is not healthy.
- readiness reports remain under `docs/artifacts/p49/readiness/...` because P50 reuses the shared readiness bucket introduced in P49.

## P51 Checkpoint Registry + Resumeable Campaign Integration

P51 adds `experiment_type: checkpoint_registry_campaign` / `p51_registry_campaign` so P22 can treat checkpoint production and nightly resumption as first-class operational state instead of anonymous files plus blind reruns.

Reference rows in `configs/experiments/p22.yaml`:

- `p51_registry_smoke`
- `p51_resumeable_nightly`

What P51 adds on top of P49/P50:

- auto-registration of RL and world-model checkpoints into `docs/artifacts/registry/checkpoints_registry.json`
- auditable checkpoint status transitions (`draft`, `smoke_passed`, `arena_passed`, `promotion_review`, `promoted`, `archived`, `rejected`)
- stage-level campaign state under `campaign_state.json`
- promotion queue exports and dashboard sections for registry/campaign status
- resume behavior that skips completed safe stages when rerunning the same campaign root

Key artifacts:

- `docs/artifacts/registry/checkpoints_registry.json`
- `docs/artifacts/p22/runs/<run_id>/p51_registry_smoke/campaign_runs/seed_*/campaign_state.json`
- `docs/artifacts/p22/runs/<run_id>/p51_registry_smoke/campaign_runs/seed_*/checkpoint_registry_snapshot.json`
- `docs/artifacts/p22/runs/<run_id>/p51_registry_smoke/campaign_runs/seed_*/promotion_queue.json`
- `docs/artifacts/p22/runs/<run_id>/p51_registry_smoke/campaign_runs/seed_*/campaign_resume_report.md`

Summary/runtime fields surfaced in P22 rows for this lane:

- `campaign_state_path`
- `registry_snapshot_path`
- `promotion_queue_path`
- `resume_report_path`
- `produced_checkpoint_ids`
- `training_python`
- `device_profile`

Operational notes:

- `scripts/run_p22.ps1 -RunP51` runs the registry/campaign smoke path.
- `scripts/run_p22.ps1 -RunP51 -Resume` resumes the latest compatible campaign root under the selected out-root.
- full experiment-level resume still short-circuits already successful experiments; stage-level resume applies when the experiment needs to rerun and campaign state already exists.
- imported historical checkpoints can remain metadata-light until their producers re-emit them through the new registry path.

## P54 Learned Router / Meta-Controller Integration

P54 adds `experiment_type: p54_learned_router_campaign` so P22 can orchestrate the full learned-router flow as one resumable experiment family.

Reference rows in `configs/experiments/p22.yaml`:

- `p54_learned_router_smoke`
- `p54_learned_router_nightly`

What P54 adds on top of P48/P49/P50/P51:

- routing dataset build from P48/P39/P41/P51 outputs
- lightweight learned-router training with masked controller logits
- safe router-mode compare: `rule`, `learned`, `learned_with_rule_guard`
- learned-router checkpoint registration under `family=learned_router`
- learned-router campaign state, registry snapshots, promotion queue updates, and dashboard sections

Key artifacts:

- `docs/artifacts/p54/router_dataset/<run_id>/router_dataset_manifest.json`
- `docs/artifacts/p54/router_train/<run_id>/train_manifest.json`
- `docs/artifacts/p54/router_inference/<run_id>/routing_trace.jsonl`
- `docs/artifacts/p54/arena_ablation/<run_id>/summary_table.json`
- `docs/artifacts/p54/triage/<run_id>/triage_report.json`
- `docs/artifacts/p22/runs/<run_id>/p54_learned_router_smoke/campaign_runs/seed_*/campaign_state.json`

Summary/runtime fields surfaced in P22 rows for this lane:

- `produced_checkpoint_ids`
- `campaign_state_path`
- `registry_snapshot_path`
- `promotion_queue_path`
- `resume_report_path`
- `dashboard_path`
- `training_python`
- `device_profile`
- `learner_device`

Operational notes:

- `scripts/run_p22.ps1 -Quick` now includes `p54_learned_router_smoke`.
- `scripts/run_p22.ps1 -RunP54` selects `p54_learned_router_smoke` or `p54_learned_router_nightly`.
- `scripts/run_p22.ps1 -RunP54 -Resume` reuses the latest compatible campaign root and skips completed resume-safe stages.
- the learned router is still arena-gated; training accuracy or checkpoint existence alone is not a promotion signal.

## P56 Learned Router Calibration + Canary Integration

P56 adds `experiment_type: p56_router_calibration_campaign` so P22 can orchestrate learned-router benchmarking, calibration, guard tuning, canary evaluation, registry updates, and dashboard build as one resumable experiment family.

Reference rows in `configs/experiments/p22.yaml`:

- `p56_router_calibration_smoke`
- `p56_router_calibration_nightly`

What P56 adds on top of P54/P49/P50/P51/P53:

- multi-seed benchmark summaries for rule, learned, guarded, and canary router modes
- confidence-bucket calibration reports and ECE-style metrics
- guard-threshold sweeps plus `recommended_guard_config.json`
- canary-only usage, fallback, and slice-distribution summaries
- registry refs and P22 summary fields for calibration, guard, canary, and deployment recommendation

Key artifacts:

- `docs/artifacts/p56/router_benchmark/<run_id>/benchmark_summary.json`
- `docs/artifacts/p56/router_calibration/<run_id>/calibration_metrics.json`
- `docs/artifacts/p56/guard_tuning/<run_id>/recommended_guard_config.json`
- `docs/artifacts/p56/canary_eval/<run_id>/canary_eval_summary.json`
- `docs/artifacts/p56/arena_ablation/<run_id>/promotion_decision.json`
- `docs/artifacts/p22/runs/<run_id>/p56_router_calibration_smoke/campaign_runs/seed_*/campaign_state.json`

Summary/runtime fields surfaced in P22 rows for this lane:

- `produced_checkpoint_ids`
- `campaign_state_path`
- `registry_snapshot_path`
- `promotion_queue_path`
- `resume_report_path`
- `dashboard_path`
- `calibration_ref`
- `guard_tuning_ref`
- `canary_eval_ref`
- `deployment_mode_recommendation`

Operational notes:

- `scripts/run_p22.ps1 -Quick` now includes `p56_router_calibration_smoke`.
- `scripts/run_p22.ps1 -RunP56` selects `p56_router_calibration_smoke` or `p56_router_calibration_nightly`.
- `scripts/run_p22.ps1 -RunP56 -Resume` reuses the latest compatible campaign root and skips completed resume-safe stages.
- learned-router promotion remains arena-gated; calibration helps rank deployment modes, not bypass triage.

## Runtime Observability (During Execution)

P22 emits both per-experiment and run-level observability artifacts:

- run-level:
  - `docs/artifacts/p22/runs/<run_id>/telemetry.jsonl`
  - `docs/artifacts/p22/runs/<run_id>/progress.unified.jsonl`
  - `docs/artifacts/p22/runs/<run_id>/live_summary_snapshot.json`
  - `docs/artifacts/p22/runs/<run_id>/summary_table.{csv,json,md}`
- per experiment:
  - `docs/artifacts/p22/runs/<run_id>/<exp_id>/run_manifest.json`
  - `docs/artifacts/p22/runs/<run_id>/<exp_id>/progress.jsonl`
  - `docs/artifacts/p22/runs/<run_id>/<exp_id>/status.json`
  - `docs/artifacts/p22/runs/<run_id>/<exp_id>/seeds_used.json`
- dashboard:
  - `docs/artifacts/dashboard/latest/index.html`
  - `docs/artifacts/dashboard/latest/dashboard_data.json`

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

This line is intentionally experimental and remains part of the mainline lane; BC/DAgger are now legacy-baseline probes.

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
- [P40_CLOSED_LOOP_IMPROVEMENT.md](P40_CLOSED_LOOP_IMPROVEMENT.md)
- [P41_CLOSED_LOOP_V2.md](P41_CLOSED_LOOP_V2.md)
- [P42_RL_CANDIDATE_PIPELINE.md](P42_RL_CANDIDATE_PIPELINE.md)
- [P44_DISTRIBUTED_RL.md](P44_DISTRIBUTED_RL.md)
- [P45_WORLD_MODEL.md](P45_WORLD_MODEL.md)
- [P46_IMAGINATION_LOOP.md](P46_IMAGINATION_LOOP.md)
- [P47_MODEL_BASED_SEARCH.md](P47_MODEL_BASED_SEARCH.md)
- [P48_ADAPTIVE_HYBRID_CONTROLLER.md](P48_ADAPTIVE_HYBRID_CONTROLLER.md)
- [P49_GPU_MAINLINE_AND_DASHBOARD.md](P49_GPU_MAINLINE_AND_DASHBOARD.md)
- [P50_CUDA_ENVIRONMENT.md](P50_CUDA_ENVIRONMENT.md)
- [P50_GPU_TROUBLESHOOTING.md](P50_GPU_TROUBLESHOOTING.md)
- [P51_CHECKPOINT_REGISTRY_AND_CAMPAIGNS.md](P51_CHECKPOINT_REGISTRY_AND_CAMPAIGNS.md)
- [P54_LEARNED_ROUTER.md](P54_LEARNED_ROUTER.md)
- [P56_ROUTER_CALIBRATION_AND_CANARY.md](P56_ROUTER_CALIBRATION_AND_CANARY.md)
