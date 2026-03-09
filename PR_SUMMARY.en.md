# PR Summary: Sim Score Core Parity (P0 v6)

> Language: [简体中文](PR_SUMMARY.zh-CN.md) | [English](PR_SUMMARY.en.md)

## What This Fix Achieves

This PR aligns simulator hand-level scoring with Oracle-observed `round.chips` deltas and removes high-noise diff sources in P0 regression.

Progression:

- v4/v5: frequent `diff_fail` on non-observable `last_base_*` or unstable identity-heavy projections
- v5 (intermediate): the primary signal moved to observable `score_observed.delta`, but parity still failed due to scoring-core and transition semantics
- v6: **P0 regression reaches 8/8 pass** with a stable post-action trace contract and deterministic fixture flow

Final v6 stats (`sim/tests/fixtures_runtime/oracle_p0_v6/report_p0.json`):

- pass: 8
- diff_fail: 0
- oracle_fail: 0
- gen_fail: 0
- skipped: 0

## Key Changes By Module

### 1) Score Core

- `sim/core/rank_chips.py`
  - introduced an explicit base rank -> chip mapping (`A=11`, `K/Q/J/T=10`, `9..2=n`)
- `sim/core/score_basic.py`
  - reworked the scoring path into:
    - hand-type detection
    - scoring-card set selection
    - base chips / mult
    - `total_delta = (base + sum(scoring card chips)) * mult`
  - corrected edge behavior:
    - `<5` cards are no longer misclassified as Straight / Flush
    - Four-of-a-kind now scores only the 4 matching cards
    - High Card now scores only the highest card
- `sim/core/engine.py`
  - play scoring now consumes `evaluate_selected_breakdown()` `total_delta`

### 2) Engine Step / State Transition Parity

- `sim/core/engine.py`
  - improved invalid-resource handling and fallback consistency
  - added explicit `MENU` action semantics for oracle-trace parity
  - normalized snapshot-load hand-cap behavior to observable replay expectations

### 3) Hash Scope And Diff Signal Quality

- `sim/core/hashing.py`
  - added and iterated `p0_hand_score_observed_core` toward observable-only fields
  - removed identity-heavy comparisons from this scope to avoid false mismatches

### 4) Fixture Generation / Oracle Replay

- `sim/oracle/generate_p0_trace.py`
  - ensured synthesized start-state persistence for replay consistency
  - improved deterministic generation flow for the round-eval target

### 5) Debug / Diagnosis Tooling

- `sim/oracle/analyze_score_delta_mismatch.py`
  - added a decomposition table (CSV / MD) per target:
    - selected cards
    - detected hand type
    - base chips / mult
    - rank chips
    - predicted core
    - oracle vs sim delta
  - this isolates whether failures came from selection, hand type, rank chips, or formula

## Design Decisions

- use `round.chips` delta (`score_observed.delta`) as the primary alignment signal:
  - it is observable and stable in live oracle state
  - it avoids dependence on non-portable or missing `last_base_*` fields
- keep hash scopes layered:
  - P0 scopes compare only the stable semantics needed for the milestone
  - identity-heavy comparison stays outside the parity gate to reduce noise

## Regression Coverage And Current Limits

Covered in P0 v6:

- PLAY / DISCARD hand-level semantics
- resource counters such as `hands_left` and `discards_left`
- core hand-type and base-scoring parity
- directed fixtures with oracle differential checks

Not covered yet:

- Joker trigger graph
- full macro economy / shop-decision parity
- all modifier / edition / seal combinatorics
- full RNG internal-state equivalence

## Reproducible Commands (Executed)

```powershell
python -B sim\oracle\batch_build_p0_oracle_fixtures.py --base-url http://127.0.0.1:12346 --out-dir sim\tests\fixtures_runtime\oracle_p0_v6 --max-steps 160 --scope p0_hand_score_observed_core --seed AAAAAAA --resume --dump-on-diff sim\tests\fixtures_runtime\oracle_p0_v6\dumps

python -B sim\oracle\analyze_score_delta_mismatch.py --fixtures-dir sim\tests\fixtures_runtime\oracle_p0_v6
```

Artifacts:

- report: `sim/tests/fixtures_runtime/oracle_p0_v6/report_p0.json`
- analyzer markdown: `sim/tests/fixtures_runtime/oracle_p0_v6/score_mismatch_table.md`
- analyzer csv: `sim/tests/fixtures_runtime/oracle_p0_v6/score_mismatch_table.csv`

## Commit Trail (High Level)

- debug analyzer: `e88e0ea`
- rank chip mapping: `4c78d88`
- hand detection & scoring-card fixes: `4f41a6f`
- core formula parity: `9a0b48f`
- observed scope stabilization: `77b2860`
- menu / fixture deterministic flow: `649c314`
