# P6 P3 Unsupported Joker Clusters

Generated at: `2026-02-23T16:07:27.145438+00:00`

- Unsupported total: **27**
- Cluster count: **16**
- Method: rule features (timing/statefulness/numeric patterns) + token-similarity greedy clustering.

## Top Clusters

### p6_cluster_001 | size=4 | difficulty=med
- template_candidate: `flat_or_conditional_mult_add`
- dominant_timing: `unknown`
- top_keywords: `chips, mult, hand, add, 100, available, start, size, 000, every, 23, 20`
- reason_distribution: `{"no_safe_template_match": 4}`
- statefulness_distribution: `{}`
- numeric_pattern_distribution: `{"plus_chips": 2, "plus_mult": 1}`
- required_signals: `score_observed.delta, round.hands_left, round.discards_left, timing:unknown`
- representatives:
  - Ice Cream
  - Misprint
  - Popcorn
  - Stuntman

### p6_cluster_002 | size=4 | difficulty=high
- template_candidate: `metadata_required_template`
- dominant_timing: `unknown`
- top_keywords: `available, start, deck, create, must, have, room, blind, selected, change, hand, adds`
- reason_distribution: `{"no_safe_template_match": 2, "insufficient_structured_fields": 2}`
- statefulness_distribution: `{"depends_on_consumable": 1}`
- numeric_pattern_distribution: `{"per_rank": 1}`
- required_signals: `score_observed.delta, round.hands_left, round.discards_left, timing:unknown, zones.played[min_card_fields], consumables.cards`
- representatives:
  - Marble Joker
  - Riff-Raff
  - Sixth Sense
  - Superposition

### p6_cluster_003 | size=2 | difficulty=med
- template_candidate: `conditional_xmult`
- dominant_timing: `on_score`
- top_keywords: `mult, suit, changes, round, 000, x1, end, available, start, rank, x2, every`
- reason_distribution: `{"no_safe_template_match": 2}`
- statefulness_distribution: `{}`
- numeric_pattern_distribution: `{"per_suit": 2, "xmult": 2, "per_rank": 1}`
- required_signals: `score_observed.delta, round.hands_left, round.discards_left, timing:on_score, zones.played[min_card_fields]`
- representatives:
  - Ancient Joker
  - The Idol

### p6_cluster_004 | size=2 | difficulty=med
- template_candidate: `conditional_xmult`
- dominant_timing: `unknown`
- top_keywords: `mult, available, start, uncommon, jokers, x1, x4, every, hands, remaining`
- reason_distribution: `{"no_safe_template_match": 2}`
- statefulness_distribution: `{"depends_on_joker_count": 1}`
- numeric_pattern_distribution: `{"xmult": 2}`
- required_signals: `score_observed.delta, round.hands_left, round.discards_left, timing:unknown, jokers.count`
- representatives:
  - Baseball Card
  - Loyalty Card

### p6_cluster_005 | size=2 | difficulty=high
- template_candidate: `complex_or_unknown_template`
- dominant_timing: `on_score`
- top_keywords: `retrigger, available, start, all, next, 10, hands`
- reason_distribution: `{"no_safe_template_match": 2}`
- statefulness_distribution: `{}`
- numeric_pattern_distribution: `{}`
- required_signals: `score_observed.delta, round.hands_left, round.discards_left, timing:on_score`
- representatives:
  - Hack
  - Seltzer

### p6_cluster_006 | size=2 | difficulty=high
- template_candidate: `complex_or_unknown_template`
- dominant_timing: `passive`
- top_keywords: `hand, size, discards, round, win, rounds, discard, run, 12, fewer, consecutive, playing`
- reason_distribution: `{"no_safe_template_match": 2}`
- statefulness_distribution: `{}`
- numeric_pattern_distribution: `{}`
- required_signals: `score_observed.delta, round.hands_left, round.discards_left, timing:passive`
- representatives:
  - Merry Andy
  - Troubadour

### p6_cluster_007 | size=2 | difficulty=high
- template_candidate: `complex_or_unknown_template`
- dominant_timing: `passive`
- top_keywords: `available, start, hand, size, every, counts, reduces, round`
- reason_distribution: `{"no_safe_template_match": 2}`
- statefulness_distribution: `{}`
- numeric_pattern_distribution: `{}`
- required_signals: `score_observed.delta, round.hands_left, round.discards_left, timing:passive`
- representatives:
  - Splash
  - Turtle Bean

### p6_cluster_008 | size=1 | difficulty=high
- template_candidate: `complex_or_unknown_template`
- dominant_timing: `on_discard`
- top_keywords: `upgrade, level, first, discarded, poker, hand, round, discard, sell, 50`
- reason_distribution: `{"no_safe_template_match": 1}`
- statefulness_distribution: `{}`
- numeric_pattern_distribution: `{}`
- required_signals: `score_observed.delta, round.hands_left, round.discards_left, timing:on_discard`
- representatives:
  - Burnt Joker

### p6_cluster_009 | size=1 | difficulty=high
- template_candidate: `shop_or_economy_passive`
- dominant_timing: `passive`
- top_keywords: `sell, this, create, free, double, tag, money, available, start`
- reason_distribution: `{"no_safe_template_match": 1}`
- statefulness_distribution: `{"depends_on_money": 1}`
- numeric_pattern_distribution: `{}`
- required_signals: `score_observed.delta, round.hands_left, round.discards_left, timing:passive, economy.money`
- representatives:
  - Diet Cola

### p6_cluster_010 | size=1 | difficulty=high
- template_candidate: `complex_or_unknown_template`
- dominant_timing: `on_score`
- top_keywords: `hand, deck, first, round, only, add, permanent, copy, draw, change, available, start`
- reason_distribution: `{"no_safe_template_match": 1}`
- statefulness_distribution: `{}`
- numeric_pattern_distribution: `{}`
- required_signals: `score_observed.delta, round.hands_left, round.discards_left, timing:on_score`
- representatives:
  - DNA

### p6_cluster_011 | size=1 | difficulty=high
- template_candidate: `complex_or_unknown_template`
- dominant_timing: `on_score`
- top_keywords: `retrigger, first, used, additional, times, beat, boss, blind, high, hand`
- reason_distribution: `{"no_safe_template_match": 1}`
- statefulness_distribution: `{}`
- numeric_pattern_distribution: `{"first_scoring_card": 1}`
- required_signals: `score_observed.delta, round.hands_left, round.discards_left, timing:on_score, zones.played[min_card_fields]`
- representatives:
  - Hanging Chad

### p6_cluster_012 | size=1 | difficulty=high
- template_candidate: `complex_or_unknown_template`
- dominant_timing: `passive`
- top_keywords: `chips, prevents, death, are, least, 25, required, self, destructs, add, lose, five`
- reason_distribution: `{"no_safe_template_match": 1}`
- statefulness_distribution: `{}`
- numeric_pattern_distribution: `{}`
- required_signals: `score_observed.delta, round.hands_left, round.discards_left, timing:passive`
- representatives:
  - Mr. Bones

### p6_cluster_013 | size=1 | difficulty=med
- template_candidate: `conditional_xmult`
- dominant_timing: `unknown`
- top_keywords: `mult, x2, loses, x0, 01, discarded, discard, available, start`
- reason_distribution: `{"no_safe_template_match": 1}`
- statefulness_distribution: `{}`
- numeric_pattern_distribution: `{"per_card": 1, "xmult": 1}`
- required_signals: `score_observed.delta, round.hands_left, round.discards_left, timing:unknown, zones.played[min_card_fields]`
- representatives:
  - Ramen

### p6_cluster_014 | size=1 | difficulty=high
- template_candidate: `complex_or_unknown_template`
- dominant_timing: `passive`
- top_keywords: `tarot, planet, spectral, may, appear, multiple, times, mult, add, reach, ante, level`
- reason_distribution: `{"no_safe_template_match": 1}`
- statefulness_distribution: `{}`
- numeric_pattern_distribution: `{}`
- required_signals: `score_observed.delta, round.hands_left, round.discards_left, timing:passive`
- representatives:
  - Showman

### p6_cluster_015 | size=1 | difficulty=high
- template_candidate: `complex_or_unknown_template`
- dominant_timing: `passive`
- top_keywords: `count, as, same, suit, hearts, diamonds, spades, clubs, have, more, wild, deck`
- reason_distribution: `{"no_safe_template_match": 1}`
- statefulness_distribution: `{}`
- numeric_pattern_distribution: `{"per_suit": 1}`
- required_signals: `score_observed.delta, round.hands_left, round.discards_left, timing:passive, zones.played[min_card_fields]`
- representatives:
  - Smeared Joker

### p6_cluster_016 | size=1 | difficulty=high
- template_candidate: `complex_or_unknown_template`
- dominant_timing: `unknown`
- top_keywords: `mult, adds, number, times, poker, hand, been, this, run, add, available, start`
- reason_distribution: `{"no_safe_template_match": 1}`
- statefulness_distribution: `{}`
- numeric_pattern_distribution: `{}`
- required_signals: `score_observed.delta, round.hands_left, round.discards_left, timing:unknown`
- representatives:
  - Supernova
