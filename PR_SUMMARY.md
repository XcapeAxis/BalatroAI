# PR Summary: Sim Score Core Parity (P0 v6)

## What This Fix Achieves
This PR aligns simulator hand-level scoring with Oracle-observed `round.chips` deltas and removes high-noise diff sources in P0 regression.

Progression:
- v4/v5: frequent `diff_fail` on non-observable `last_base_*` or unstable identity-heavy projections.
- v5 (intermediate): alignment signal moved to observable `score_observed.delta`, but parity still failed due to scoring core and transition semantics.
- v6: **P0 regression reaches 8/8 pass** with stable post-action trace contract and deterministic fixture flow.

Final v6 stats (`sim/tests/fixtures_runtime/oracle_p0_v6/report_p0.json`):
- pass: 8
- diff_fail: 0
- oracle_fail: 0
- gen_fail: 0
- skipped: 0

## Key Changes by Module

### 1) Score Core
- `sim/core/rank_chips.py`
  - Introduced explicit base rank->chip mapping (`A=11`, `K/Q/J/T=10`, `9..2=n`).
- `sim/core/score_basic.py`
  - Reworked into breakdown-based scoring:
    - hand type detection,
    - scoring-card set selection,
    - base chips/mult,
    - `total_delta = (base + sum(scoring card chips)) * mult`.
  - Corrected edge behavior:
    - `<5` cards no longer misclassified as Straight/Flush,
    - Four-of-a-kind scores only the 4 matching cards,
    - High Card scores highest card only.
- `sim/core/engine.py`
  - Play scoring now consumes `evaluate_selected_breakdown()` `total_delta`.

### 2) Engine Step / State Transition Parity
- `sim/core/engine.py`
  - Improved invalid resource handling and fallback behavior consistency.
  - Added explicit `MENU` action semantics for parity with oracle traces.
  - Snapshot load hand-cap behavior normalized to observable hand-cap expectations in replay.

### 3) Hash Scope and Diff Signal Quality
- `sim/core/hashing.py`
  - Added/iterated `p0_hand_score_observed_core` toward observable-only fields.
  - Removed unstable identity-heavy comparison from this scope (e.g., full hand identity ordering) to avoid false mismatches.

### 4) Fixture Generation / Oracle Replay
- `sim/oracle/generate_p0_trace.py`
  - Ensured synthesized start state persistence for replay consistency.
  - Improved deterministic generation flow for round-eval target.

### 5) Debug / Diagnosis Tooling
- `sim/oracle/analyze_score_delta_mismatch.py`
  - Added decomposition table (CSV/MD) per target:
    - selected cards,
    - detected hand type,
    - base chips/mult,
    - rank chips,
    - predicted core,
    - oracle vs sim delta.
  - This isolated whether failures came from hand selection, hand type, rank chips, or formula.

## Design Decisions
- Use `round.chips` delta (`score_observed.delta`) as primary alignment signal:
  - It is stable and observable from live oracle state.
  - It avoids dependence on non-portable / missing `last_base_*` fields.
- Keep hash scopes layered:
  - P0 scopes compare only stable semantics needed for current milestone.
  - Identity-heavy zone comparison stays outside score parity gate to reduce noise.

## Regression Coverage and Current Limits
Covered in P0 v6:
- PLAY/DISCARD hand-level semantics,
- resource counters (`hands_left`, `discards_left`),
- core hand type + base scoring parity,
- directed fixtures with oracle differential checks.

Not covered yet:
- Joker trigger graph,
- full macro economy/shop decision parity,
- all modifier/edition/seal combinatorics,
- full RNG internal-state equivalence.

## Reproducible Commands (Executed)
```powershell
python -B sim\oracle\batch_build_p0_oracle_fixtures.py --base-url http://127.0.0.1:12346 --out-dir sim\tests\fixtures_runtime\oracle_p0_v6 --max-steps 160 --scope p0_hand_score_observed_core --seed AAAAAAA --resume --dump-on-diff sim\tests\fixtures_runtime\oracle_p0_v6\dumps

python -B sim\oracle\analyze_score_delta_mismatch.py --fixtures-dir sim\tests\fixtures_runtime\oracle_p0_v6
```

Artifacts:
- report: `sim/tests/fixtures_runtime/oracle_p0_v6/report_p0.json`
- analyzer markdown: `sim/tests/fixtures_runtime/oracle_p0_v6/score_mismatch_table.md`
- analyzer csv: `sim/tests/fixtures_runtime/oracle_p0_v6/score_mismatch_table.csv`

## Commit Trail (high level)
- debug analyzer: `e88e0ea`
- rank chip mapping: `4c78d88`
- hand detection & scoring-card fixes: `4f41a6f`
- core formula parity: `9a0b48f`
- observed scope stabilization: `77b2860`
- menu/fixture deterministic flow: `649c314`
