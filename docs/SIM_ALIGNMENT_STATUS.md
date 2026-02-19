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

## Explicitly Not Covered Yet
- Joker effects and joker internal state interactions.
- Vouchers, tags, complex blind modifiers, shop economy/parsing details.
- Pack/consume flows and non-hand tactical systems.
- Full macro parity outside directed fixture scopes.

## Regression Procedure
Run all current regression gates with one command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_regressions.ps1
```

Optional parameters:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_regressions.ps1 -BaseUrl http://127.0.0.1:12346 -OutRoot sim/tests/fixtures_runtime -Seed AAAAAAA
```

The runner executes:
- P0 batch: `oracle_p0_v6_regression`
- P1 smoke batch: `oracle_p1_smoke_v3_regression`

## Artifact Policy
- Runtime outputs are regenerated artifacts under `sim/tests/fixtures_runtime` (and related runtime/log paths).
- These outputs are ignored by `.gitignore` and should not be committed.
- Keep only scripts/spec/docs/source fixtures in git.
