# SIM Alignment Status

## Aligned Scope (Current)

### P0 v6
- Hand-level core scoring aligned to oracle observable delta (`score_observed.delta`, derived from `round.chips` change).
- Played-card selection and hand-type evaluation aligned for directed P0 fixtures.
- Rank chip mapping aligned for base cards (`A=11`, `K/Q/J/T=10`, `9..2` numeric).
- Resource counters aligned for hand/discard consumption in P0 fixture paths.
- Oracle/sim trace contract unified as post-action state per step.

### P1 v3 smoke
- Planet hand-level boosts injected into simulator scoring path used by P1 fixtures.
- Card modifier scoring injected for smoke targets: `bonus`, `mult`, `foil`, `holo`.
- Computed expected diagnostics retained for explanation/debug, not as primary oracle truth.

### P2 v1 smoke
- Single-joker scoring deltas aligned for 12 controlled joker fixtures.
- Joker contribution decomposition added (`joker_bonus_*`, `joker_breakdown`) for diagnostics.

### P2b v1 smoke
- 18 directed fixtures aligned across four groups:
  - stacked jokers
  - joker + planet
  - order/selection-sensitive
  - resource-sensitive
- `p2b_hand_score_observed_core` scope added for stable regression hashing.
- Batch runner supports resume/target subset/retry/health recovery and diff dump options.
- Analyzer output available: `score_mismatch_table_p2b.csv` / `.md`.

## Rounding/Arithmetic Rules (Observed)
- Core expected score is combined as chips term × mult term.
- For aligned smoke fixtures, post-composition score follows integer truncation (`int(total_score_raw)`) before observable delta comparison.
- Oracle is treated as source of truth when CSV text and runtime behavior diverge.

## Explicitly Not Covered Yet
- Probabilistic joker triggers and multi-step stochastic effects.
- Cross-round accumulating joker state machines and economy-coupled joker interactions.
- Vouchers, tags, complex blind modifiers, shop economy/parsing details.
- Pack/consume flows and non-hand tactical systems.
- Full macro parity outside directed fixture scopes.

## Regression Procedure
Run baseline regression gates:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_regressions.ps1
```

Run baseline plus P2b + analyzer:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_regressions.ps1 -RunP2b
```

Optional parameters:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_regressions.ps1 -BaseUrl http://127.0.0.1:12346 -OutRoot sim/tests/fixtures_runtime -Seed AAAAAAA -RunP2b
```

The runner executes:
- P0 batch: `oracle_p0_v6_regression`
- P1 smoke batch: `oracle_p1_smoke_v3_regression`
- Optional P2b batch: `oracle_p2b_smoke_v1_regression` + analyzer

## Artifact Policy
- Runtime outputs are regenerated artifacts under `sim/tests/fixtures_runtime` (and related runtime/log paths).
- These outputs are ignored by `.gitignore` and should not be committed.
- Keep only scripts/spec/docs/source fixtures in git.

## P22 Experiment Pipeline Note
- P22 introduces matrix-based experiment orchestration with resumable runs and seed-governed evaluation.
- Entry points:
  - `scripts/run_p22.ps1`
  - `python -B -m trainer.experiments.orchestrator`
- Runtime observability (run-level):
  - `docs/artifacts/p22/runs/<run_id>/telemetry.jsonl`
  - `docs/artifacts/p22/runs/<run_id>/live_summary_snapshot.json`
  - `docs/artifacts/p22/runs/<run_id>/summary_table.{csv,json,md}`
- Runtime observability (per experiment):
  - `docs/artifacts/p22/runs/<run_id>/<exp_id>/run_manifest.json`
  - `docs/artifacts/p22/runs/<run_id>/<exp_id>/progress.jsonl`
  - `docs/artifacts/p22/runs/<run_id>/<exp_id>/seeds_used.json`
- Seed clarification:
  - fixed seeds are used for strict regression comparability.
  - P22 quick gate keeps `>1` seeds by default (`seed_limit=2`) for stable but non-single-seed smoke.
  - nightly/expanded runs add deterministic extra seeds.
- source of truth is always `seeds_used.json`, not a presumed single default seed.
- Curated outputs are written under `docs/artifacts/p22/**` (ignored from git tracking by repo policy).

## P32 RealAction / Position-Fidelity Note

- P32 extends the action contract to include explicit position operations:
  - `REORDER_HAND`, `SWAP_HAND_CARDS`
  - `REORDER_JOKERS`, `SWAP_JOKERS`
- New replay scope `p32_real_action_position_observed_core` captures order-sensitive state so hand/joker ordering drift is detectable.
- Real-runtime translation supports these actions in env-client; unsupported runtime RPC methods are surfaced as degraded execution with explicit reason.
- `real_trace_to_fixture` now attempts reorder/swap inference from raw before/after snapshots when explicit action logs are missing.
- Reference docs:
  - `docs/P32_REAL_ACTION_CONTRACT_STATUS.md`
  - `docs/P32_REAL_ACTION_CONTRACT_SPEC.md`
  - `docs/P32_SHOP_RNG_ALIGNMENT.md`

## P31 Self-Supervised Backbone Note

- P31 adds a unified trajectory schema (`DecisionStep` / `Trajectory`) to reuse oracle/sim/real traces for self-supervised pretraining.
- Initial heads focus on `score_observed.delta` regression and `hand_type` classification.
- P22 now includes `quick_selfsup_pretrain` so self-supervised smoke experiments are first-class orchestrator runs.
- Details:
  - `docs/EXPERIMENTS_P31.md`
  - `docs/COVERAGE_P31_STATUS.md`

## P33 Action Replay + Self-Supervised Plumbing Note

- P33 introduces a unified action replay adapter (`trainer/actions/replay.py`) to normalize and execute high-level actions across sim/real backends.
- Current integration points include recorder/real executor/DAgger sim-augment paths, plus action normalization in real-trace fixture conversion.
- P33 also adds a minimal self-supervised experiment entry:
  - `trainer/experiments/selfsupervised_p33.py`
  - `configs/experiments/p33_selfsup.yaml`
  - P22 row: `quick_selfsup_p33`
- Reference docs:
  - `docs/EXPERIMENTS_P33.md`
  - `docs/COVERAGE_P33_STATUS.md`

## P35 Telemetry + Reproducibility Hardening Note

- P22 telemetry/event streams are now normalized for both run-level and per-experiment logs:
  - run-level: `schema=p34_telemetry_event_v1`
  - experiment-level: `schema=p34_progress_event_v1`
- Seed usage is explicitly materialized in:
  - `run_plan.json -> experiments_with_seeds[]`
  - `<exp_id>/seeds_used.json`
  - `summary_table.*` (`seed_set_name`, `seed_hash`, `seeds_used`)
- P35 also adds a P32 self-supervised representation stub line (`pretrain_repr`) via:
  - `configs/experiments/p32_self_supervised.yaml`
  - `scripts/run_p32_self_supervised.ps1`
  - `docs/EXPERIMENTS_P32_SELF_SUPERVISED.md`

## P36 Self-Supervised Core Note

- P36 introduces a unified self-supervised sample contract (`trainer/selfsup/data.py`) that reuses trace artifacts rather than engine internals.
- Two concrete tasks are now available:
  - future chips-delta prediction (`trainer/selfsup/train_future_value.py`)
  - inverse dynamics action-type prediction (`trainer/selfsup/train_action_type.py`)
- P22 matrix includes these rows directly:
  - `quick_selfsup_future_value`
  - `quick_selfsup_action_type`
- Seed execution remains explicit and artifactized through:
  - `run_plan.json -> experiments_with_seeds`
  - `<exp_id>/seeds_used.json`
  - `summary_table.* -> seeds_used/final_loss`
- Reference docs:
  - `docs/P36_SELF_SUP_LEARNING.md`
  - `docs/EXPERIMENTS_P22.md`

## P37 Single-Action Fidelity Note

- P37 extends the action contract with position/control actions used by real-session replay:
  - `MOVE_HAND_CARD`, `MOVE_JOKER`
  - `SWAP_HAND_CARD`, `SWAP_JOKER`
  - canonical shop/consumable actions: `SHOP_REROLL`, `SHOP_BUY`, `PACK_OPEN`, `CONSUMABLE_USE`
- New replay scope `p37_action_fidelity_core` hashes:
  - ordered hand UIDs
  - ordered joker UID/key slots
  - selected hand indices + last action type token
- Added dedicated batch builder:
  - `sim/oracle/batch_build_p37_action_fidelity.py`
- Gate entry:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_regressions.ps1 -RunP37
```

- Probability/weight parity audit artifacts:
  - `docs/P37_PROBABILITY_PARITY.md`
  - `docs/artifacts/p37/probability_audit_*.json`
  - `docs/artifacts/p37/probability_audit_*.md`
