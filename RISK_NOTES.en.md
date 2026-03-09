# Risk Notes: P0 v6 Score Parity Changes

> Language: [简体中文](RISK_NOTES.zh-CN.md) | [English](RISK_NOTES.en.md)

## Primary Risks Introduced / Highlighted

1. Scoring-card selection assumptions in `sim/core/score_basic.py`
- High Card currently scores only the highest card
- Pair / Two Pair / Three / Four-kind scoring-card subsets are explicitly selected
- Risk: future rule layers such as Joker, enhancement-triggered inclusion, or retriggers may change scoring-card participation semantics

2. `<5` card hand-type boundaries
- Straight / Flush detection is now constrained to the 5-card evaluation path
- Risk: if a later mode or mechanic allows valid non-5-card interpretations, this path will diverge

3. Four-of-a-kind kicker handling
- Four-kind currently excludes kicker rank chips from the scoring-card sum
- Risk: later modifier / retrigger interactions may require blended treatment

4. `MENU` transition semantics in the simulator
- Explicit `MENU` handling was added for oracle-fixture parity
- Risk: if production flow depends on deeper menu-stack semantics, this may overfit current fixture paths

5. Snapshot hand-cap normalization
- Snapshot replay is normalized toward observable hand-cap behavior
- Risk: debug fixtures containing over-cap hands may be truncated semantically by draw logic

6. Scope design vs identity noise
- `p0_hand_score_observed_core` intentionally avoids identity-heavy hand-card comparison
- Risk: this may hide ordering or identity regressions that matter to later mechanics

## Potential Conflicts With Future Mechanics

1. Enhancements / Editions / Seals
- Current P0 parity excludes full modifier algebra from the hash gate
- Future conflict: parity may break once additive / multiplicative modifiers stack with retriggers

2. Joker graph and timing windows
- P0 core still does not model joker traversal timing and trigger ordering
- Future conflict: correct base score may still diverge from the final observed total once jokers apply across phases

3. Planet hand levels
- P0 base logic aligns current fixtures but does not hard-enforce the full hand-level progression state machine
- Future conflict: level-dependent base chips / mult may diverge if state transitions are not tracked per hand-type lineage

## Isolation Recommendations

1. Keep feature flags by mechanism family
- `score_core_base_only`
- `score_core_planet`
- `score_core_modifiers`
- `score_core_jokers`

2. Keep layered scopes
- `p0_hand_score_observed_core` for the stable base
- `p1_hand_score_observed_core` for base + selected mechanism
- mechanism-specific diagnostic projections such as planet-only / modifier-only for targeted diff

3. Preserve analyzer-driven gating
- continue using `analyze_score_delta_mismatch.py` before broad formula edits
- require decomposition-table evidence before changing the core formula

4. Enforce deterministic fixture construction
- keep one mechanism variable per fixture to avoid attribution ambiguity

## Practical Guardrails

- always run the P0 v6 batch after any P1+ change as a non-regression gate
- keep P1 smoke on a separate branch and commit by mechanism family for faster rollback
- tag known-good parity points (`sim-p0-v6-pass` was already reserved in this cycle)
