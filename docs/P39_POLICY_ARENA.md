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
  - `trainer/policy_arena/adapters/model_adapter.py` (`status=stub` when checkpoint is unavailable)
  - `trainer/policy_arena/adapters/hybrid_adapter.py`
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

Bucket metrics (v1):

- ante buckets: `ante_1_2`, `ante_3_4`, `ante_5_plus`
- risk buckets (proxy): `resource_tight`, `resource_balanced`, `resource_relaxed`
- action-type buckets: `PLAY`, `DISCARD`, `SHOP`, `CONSUMABLE`, `PACK`, `POSITION`, `OTHER`
- position-sensitive state exposure: `yes/no`

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

Outputs:

- `candidate_decision.json`
- `candidate_decision.md`

## Run Commands

Standalone quick smoke:

```powershell
python -m trainer.policy_arena.arena_runner --quick
```

Standalone champion decision:

```powershell
python -m trainer.policy_arena.champion_rules --summary-json docs/artifacts/p39/arena_runs/<run_id>/summary_table.json --out-dir docs/artifacts/p39
```

P22 smoke row:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_p22.ps1 -Only p39_policy_arena_smoke -SeedLimit 2
```

P22 quick (includes P39 smoke row):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_p22.ps1 -Quick
```

## Artifacts

Standalone arena:

- `docs/artifacts/p39/arena_runs/<run_id>/run_manifest.json`
- `docs/artifacts/p39/arena_runs/<run_id>/episode_records.jsonl`
- `docs/artifacts/p39/arena_runs/<run_id>/summary_table.{json,csv,md}`
- `docs/artifacts/p39/arena_runs/<run_id>/bucket_metrics.{json,md}`
- `docs/artifacts/p39/arena_runs/<run_id>/warnings.log`

Champion evaluation:

- `docs/artifacts/p39/champion_eval_<timestamp>/candidate_decision.json`
- `docs/artifacts/p39/champion_eval_<timestamp>/candidate_decision.md`

P22 integration:

- `docs/artifacts/p22/runs/<run_id>/p39_summary.json`
- `docs/artifacts/p22/runs/<run_id>/p39_policy_arena_smoke/**`

