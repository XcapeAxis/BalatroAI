# P6 Template Backlog (Next Batch)

Generated at: `2026-02-23T16:07:27.175740+00:00`

- Selection rule: low/med difficulty first, then larger cluster size.

## 1. flat_or_conditional_mult_add (from p6_cluster_001, size=4)
- trigger_timing: `unknown`
- expected_scope: `score_observed.delta, round.hands_left, round.discards_left, timing:unknown`
- representative_jokers: `Ice Cream, Misprint, Popcorn, Stuntman`
- fixture_strategy:
  - use `add joker` + controlled hand construction
  - isolate single-step action (prefer one PLAY; fallback DISCARD for non-scoring)
  - compare on observed scope and keep identity/economy/rng noise excluded
- risk_notes: `low direct risk under current observed scoring scope`

## 2. conditional_xmult (from p6_cluster_003, size=2)
- trigger_timing: `on_score`
- expected_scope: `score_observed.delta, round.hands_left, round.discards_left, timing:on_score, zones.played[min_card_fields]`
- representative_jokers: `Ancient Joker, The Idol`
- fixture_strategy:
  - use `add joker` + controlled hand construction
  - isolate single-step action (prefer one PLAY; fallback DISCARD for non-scoring)
  - compare on observed scope and keep identity/economy/rng noise excluded
- risk_notes: `low direct risk under current observed scoring scope`

## 3. conditional_xmult (from p6_cluster_004, size=2)
- trigger_timing: `unknown`
- expected_scope: `score_observed.delta, round.hands_left, round.discards_left, timing:unknown, jokers.count`
- representative_jokers: `Baseball Card, Loyalty Card`
- fixture_strategy:
  - use `add joker` + controlled hand construction
  - isolate single-step action (prefer one PLAY; fallback DISCARD for non-scoring)
  - compare on observed scope and keep identity/economy/rng noise excluded
- risk_notes: `low direct risk under current observed scoring scope`

## 4. conditional_xmult (from p6_cluster_013, size=1)
- trigger_timing: `unknown`
- expected_scope: `score_observed.delta, round.hands_left, round.discards_left, timing:unknown, zones.played[min_card_fields]`
- representative_jokers: `Ramen`
- fixture_strategy:
  - use `add joker` + controlled hand construction
  - isolate single-step action (prefer one PLAY; fallback DISCARD for non-scoring)
  - compare on observed scope and keep identity/economy/rng noise excluded
- risk_notes: `low direct risk under current observed scoring scope`

