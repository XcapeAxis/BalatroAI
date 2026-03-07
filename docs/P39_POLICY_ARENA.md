# P39 Policy Arena v1

## Why Policy Arena

Single-policy score snapshots are not enough for promotion decisions. P39 introduces a unified multi-policy arena so heuristic/search/model/hybrid policies can be compared under the same seeds, episode budgets, and reporting schema.

Goals:

- one adapter interface for heterogeneous policies
- multi-seed, multi-episode aggregate evaluation
- bucketed diagnostics beyond global mean score
- machine-readable champion/candidate recommendation input

## Components

- adapter contract: `trainer/policy_arena/policy_adapter.py`
- adapters:
  - `trainer/policy_arena/adapters/heuristic_adapter.py`
  - `trainer/policy_arena/adapters/search_adapter.py`
  - `trainer/policy_arena/adapters/model_adapter.py` (loads BC-style checkpoints when available; falls back to heuristic when unavailable)
  - `trainer/policy_arena/adapters/hybrid_adapter.py`
  - `trainer/policy_arena/adapters/world_model_assist_adapter.py`
  - `trainer/policy_arena/adapters/wm_rerank_adapter.py`
  - `trainer/hybrid/hybrid_controller.py` (`hybrid_controller_v1` via arena adapter factory)
- arena execution:
  - `trainer/policy_arena/arena_runner.py`
  - `trainer/policy_arena/arena_metrics.py`
  - `trainer/policy_arena/arena_report.py`
- candidate decision rules:
  - `trainer/policy_arena/champion_rules.py`

## Metrics

Global metrics per policy:

- `mean_total_score`, `std_total_score`
- `mean_chips`
- `mean_rounds_survived`
- `mean_episode_length`
- `win_rate` (proxy)
- `p10/p50/p90_total_score`
- `invalid_action_rate`
- `timeout_rate`
- economy counters (`mean_money_earned`, `mean_rerolls_count`, `mean_packs_opened`, `mean_consumables_used`)

Bucket metrics / slice metrics:

- ante buckets: `ante_1_2`, `ante_3_4`, `ante_5_plus`
- risk buckets (proxy): `resource_tight`, `resource_balanced`, `resource_relaxed`
- action-type buckets: `PLAY`, `DISCARD`, `SHOP`, `CONSUMABLE`, `PACK`, `POSITION`, `OTHER`
- position-sensitive state exposure: `yes/no`
- unified slice labels (shared with P41 replay pipeline):
  - `slice_stage` (`early/mid/late/unknown`)
  - `slice_resource_pressure` (`low/medium/high/unknown`)
  - `slice_action_type` (`play/discard/shop/consumable/transition/unknown`)
  - `slice_position_sensitive` (`true/false/unknown`)
  - `slice_stateful_joker_present` (`true/false/unknown` stub-compatible)
- shared slice rules can be smoke-checked via `python -m trainer.closed_loop.slice_smoke`

## Champion Rules (P39 Decision Layer)

`champion_rules.py` consumes arena `summary_table.json` and a champion baseline.

Decision logic:

- hard constraints:
  - invalid action rate increase must stay below threshold
  - timeout rate increase must stay below threshold
- soft promotion:
  - score improvement passes threshold, or
  - win-rate improves while score does not regress beyond tolerance
- sample-size guard:
  - if `seed_count < min_seeds`, return `observe` (not promote)
- slice-aware guard (P41-compatible):
  - evaluates candidate vs champion by slice using bootstrap CI
  - blocks/softens promotion when critical slices show significant degradation
  - returns `ci_status=insufficient_samples` for low-count slices and keeps conservative recommendation

Outputs:

- `candidate_decision.json`
- `candidate_decision.md`
- `slice_decision_breakdown.json`
- `slice_decision_breakdown.md`

## Run Commands

Standalone quick smoke:

```powershell
python -m trainer.policy_arena.arena_runner --quick
```

Standalone champion decision:

```powershell
python -m trainer.policy_arena.champion_rules --summary-json docs/artifacts/p39/arena_runs/<run_id>/summary_table.json --episode-records-jsonl docs/artifacts/p39/arena_runs/<run_id>/episode_records.jsonl --bucket-metrics-json docs/artifacts/p39/arena_runs/<run_id>/bucket_metrics.json --out-dir docs/artifacts/p39
```

P22 smoke row:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_p22.ps1 -Only p39_policy_arena_smoke -SeedLimit 2
```

P22 quick (includes P39 smoke row):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_p22.ps1 -Quick
```

World-model-assisted compare (P45 hook):

```powershell
python -m trainer.policy_arena.arena_runner --policies "heuristic_baseline,heuristic_wm_assist" --world-model-checkpoint docs/artifacts/p45/wm_train/<run_id>/best.pt --world-model-assist-mode one_step_heuristic --world-model-weight 0.35 --world-model-uncertainty-penalty 0.5 --seeds "AAAAAAA,BBBBBBB" --episodes-per-seed 1 --max-steps 120
```

P46 imagination ablation compare:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_p22.ps1 -RunP46
```

P47 world-model rerank compare:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_p22.ps1 -RunP47
```

P48 adaptive hybrid compare:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_p22.ps1 -RunP48
```

## World Model Assist (P45)

P45 adds an optional arena adapter that reranks legal actions with a one-step world-model score while keeping the simulator as the execution authority.

Supported policy ids:

- `heuristic_wm_assist`
- `baseline_wm_assist`
- `wm_assist`
- `world_model_assist`

CLI flags:

- `--world-model-checkpoint`
- `--world-model-assist-mode` (`one_step_heuristic` in v1)
- `--world-model-weight`
- `--world-model-uncertainty-penalty`

Behavior:

- base policy remains `heuristic_baseline`
- world-model score is uncertainty-penalized before reranking
- missing or invalid checkpoints degrade to heuristic baseline with `status=stub`
- run manifest records checkpoint path and assist parameters under `config` / `adapters`

P46 also reuses the arena for candidate-vs-candidate ablations by supplying a `policy_model_map.json` so names like `candidate_real_only` and `candidate_real_plus_imagined_filtered` behave like normal arena policies.

## Model-Based Rerank (P47)

P47 keeps the same arena execution path but adds per-policy rerank configs through `policy_assist_map.json`.

Supported patterns:

- `heuristic_wm_rerank_*`
- `search_wm_rerank_*`
- model-backed variants when a `model_path` is supplied

Recorded fields:

- `world_model_assist=true`
- `assist_mode=rerank`
- `world_model_checkpoint`
- `horizon`
- `uncertainty_penalty`

P47 ablations still run through normal arena episodes and champion rules. The world model only changes candidate ordering before action selection.

## Adaptive Hybrid Controller (P48)

P48 adds `hybrid_controller_v1`, which chooses between:

- `policy_baseline`
- `policy_plus_wm_rerank`
- `search_baseline`
- `heuristic_baseline`

The router is explainable and writes routing traces under `router_traces/<run_id>/routing_trace.jsonl`.

Arena manifests keep the hybrid policy explicit through:

- `policy_assist_map.json`
- `run_manifest.json -> hybrid_controller`
- `routing_summary.json`

## Learned Router / Guarded Router (P54)

P54 keeps the same arena substrate but extends the policy set with learned-routing variants:

- `hybrid_controller_rule`
- `hybrid_controller_learned`
- `hybrid_controller_learned_with_rule_guard`

Required compare set in P54 ablations:

- `policy_baseline`
- `policy_plus_wm_rerank`
- `hybrid_controller_rule`
- `hybrid_controller_learned`
- `hybrid_controller_learned_with_rule_guard`
- optional `search_baseline` when stable on the selected config

P54 outputs remain slice-aware and checkpoint-aware because the arena summary now carries learned-router checkpoint refs, controller-selection distributions, and guard-trigger summaries.

## Artifacts

Standalone arena:

- `docs/artifacts/p39/arena_runs/<run_id>/run_manifest.json`
- `docs/artifacts/p39/arena_runs/<run_id>/episode_records.jsonl`
- `docs/artifacts/p39/arena_runs/<run_id>/summary_table.{json,csv,md}`
- `docs/artifacts/p39/arena_runs/<run_id>/bucket_metrics.{json,md}`
- `docs/artifacts/p39/arena_runs/<run_id>/warnings.log`
- `run_manifest.json` includes `world_model_checkpoint`, `world_model_assist_mode`, `world_model_weight`, and `world_model_uncertainty_penalty` when wm-assist is enabled

Champion evaluation:

- `docs/artifacts/p39/champion_eval_<timestamp>/candidate_decision.json`
- `docs/artifacts/p39/champion_eval_<timestamp>/candidate_decision.md`
- `docs/artifacts/p39/champion_eval_<timestamp>/slice_decision_breakdown.json`
- `docs/artifacts/p39/champion_eval_<timestamp>/slice_decision_breakdown.md`

P22 integration:

- `docs/artifacts/p22/runs/<run_id>/p39_summary.json`
- `docs/artifacts/p22/runs/<run_id>/p39_policy_arena_smoke/**`

P40 closed-loop dependency:

- `trainer.closed_loop.failure_mining` consumes P39 arena outputs as hard-failure mining inputs.
- required files for full P40 mining path:
  - `docs/artifacts/p39/arena_runs/<run_id>/episode_records.jsonl`
  - `docs/artifacts/p39/arena_runs/<run_id>/summary_table.json`
  - `docs/artifacts/p39/arena_runs/<run_id>/bucket_metrics.json`
- optional file used when present:
  - `docs/artifacts/p39/champion_eval_<timestamp>/candidate_decision.json`
- if these are missing, P40 degrades to `status=stub`/`arena_status=skipped` instead of failing the full pipeline.

P42 RL candidate dependency:

- P42 closed-loop RL rows also use P39 arena outputs plus champion rules for post-training gating.
- P22 P42 rows emit per-seed P42 summaries that reference arena summary/decision paths for the same seed run.

P45 planning-hook dependency:

- `heuristic_wm_assist` is treated as a normal arena candidate, so champion rules and bucket metrics operate unchanged.
- P45 assist compare runs store the resulting arena summary path in `assist_compare_summary.json`.

P46 imagination dependency:

- P46 combined arena compare writes results under `docs/artifacts/p46/arena_compare/<run_id>/`.
- champion rules and slice metrics remain unchanged because imagined augmentation is evaluated through the same real simulator episodes.

P47 rerank dependency:

- P47 ablations write results under `docs/artifacts/p47/arena_ablation/<run_id>/`.
- `policy_assist_map.json` and `run_manifest.json` make world-model rerank settings explicit for each policy variant.

P48 hybrid dependency:

- P48 ablations write results under `docs/artifacts/p48/arena_ablation/<run_id>/`.
- `routing_summary.json` and `triage_report.json` expose controller-selection distribution and routing impact.

P54 learned-router dependency:

- P54 ablations write results under `docs/artifacts/p54/arena_ablation/<run_id>/`.
- `summary_table.{json,csv,md}`, `slice_eval.json`, and `routing_summary.json` compare rule vs learned vs guarded routing.
- promotion and triage outputs remain downstream consumers of real arena results instead of training loss alone.
