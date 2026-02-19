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
