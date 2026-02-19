# Risk Notes: P0 v6 Score Parity Changes

## Primary Risks Introduced / Highlighted

1. Scoring-card selection assumptions in `sim/core/score_basic.py`
- High Card currently scores only highest card.
- Pair/Two Pair/Three/Four-kind scoring-card subsets are explicitly selected.
- Risk: future rule layers (Joker, enhancement-triggered inclusion, retriggers) may alter scoring-card participation semantics.

2. `<5` card hand-type boundaries
- Straight/Flush detection now constrained to 5-card evaluation in core path.
- Risk: if a game mode/mechanic allows alternative valid hand-type interpretation for non-5-card plays, this will diverge.

3. Four-of-a-kind kicker handling
- Four-kind currently excludes kicker rank chips from scoring-card sum.
- Risk: interactions with modifiers/retriggers could require blended treatment in later milestones.

4. MENU transition semantics in simulator
- Explicit `MENU` handling was added to match oracle fixture behavior.
- Risk: if production flow expects deeper menu stack semantics, this simplified transition may overfit fixture paths.

5. Snapshot hand-cap normalization
- Snapshot replay normalizes toward observable hand cap behavior.
- Risk: debug fixtures containing over-cap hands can be truncated semantically by draw logic assumptions.

6. Scope design vs. identity noise
- `p0_hand_score_observed_core` intentionally avoids identity-heavy hand-card comparison.
- Risk: this may hide some ordering/identity regressions that matter to later mechanisms.

## Potential Conflicts with Future Mechanics

1. Enhancements / Editions / Seals
- Current P0 parity excludes full modifier algebra in hash gate.
- Future conflict: score parity may break once additive/multiplicative modifiers stack with retriggers.

2. Joker graph and timing windows
- P0 core does not model joker traversal timing and trigger ordering.
- Future conflict: correct base score may still diverge from full observed total when jokers apply pre/post hand phases.

3. Planet hand levels
- P0 base logic aligns current observed fixtures but does not hard-enforce full level progression state machine.
- Future conflict: level-dependent base chips/mult may diverge if state transitions are not tracked per hand-type lineage.

## Isolation Recommendations

1. Keep feature flags by mechanism family
- `score_core_base_only`
- `score_core_planet`
- `score_core_modifiers`
- `score_core_jokers`

2. Keep layered scopes
- `p0_hand_score_observed_core` (stable base)
- `p1_hand_score_observed_core` (base + selected mechanism)
- mechanism-specific diagnostic projections (planet-only/modifier-only) for targeted diff.

3. Preserve analyzer-driven gating
- Continue using `analyze_score_delta_mismatch.py` before broad formula edits.
- Require decomposition table evidence before changing core formula.

4. Enforce deterministic fixture construction
- Keep fixture generation per mechanism isolated (single mechanism variable per fixture) to avoid attribution ambiguity.

## Practical Guardrails
- Always run P0 v6 batch after any P1+ change as non-regression gate.
- Keep P1 smoke in a separate branch and commit by mechanism family for fast rollback.
- Tag known-good parity points (`sim-p0-v6-pass` already reserved in this cycle).
