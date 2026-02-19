# P3 Joker Template Backlog

Generated at: `2026-02-19T14:05:49.655745+00:00`

- Unsupported jokers: **109**
- Clusters discovered: **47**
- Clustering method: token signature + greedy Jaccard (conservative, deterministic)

## Unsupported Top Reasons
- `no_safe_template_match`: 38
- `cross_round`: 33
- `economy`: 18
- `probabilistic`: 14
- `insufficient_fields`: 6

## Top Clusters

### cluster_001 | size=7 | difficulty=high
- suggested_template: `complex_or_unknown_rule`
- dominant_trigger: `passive`
- trigger_distribution: `{"passive": 7}`
- reason_distribution: `{"economy": 5, "cross_round": 1, "no_safe_template_match": 1}`
- top_tokens: `start, available, round, end, money, earn, deck, change, discard, sell, value, currently`
- representatives:
  - Cloud 9
  - Delayed Gratification
  - Drunkard
  - Egg
  - Gift Card
  - Golden Joker
  - Rocket

### cluster_002 | size=6 | difficulty=high
- suggested_template: `complex_or_unknown_rule`
- dominant_trigger: `unknown`
- trigger_distribution: `{"unknown": 6}`
- reason_distribution: `{"cross_round": 4, "no_safe_template_match": 2}`
- top_tokens: `mult, start, available, add, currently, shop, reroll, tarot, run, used, slot, x1`
- representatives:
  - Flash Card
  - Fortune Teller
  - Joker Stencil
  - Misprint
  - Popcorn
  - Red Card

### cluster_003 | size=5 | difficulty=low
- suggested_template: `face_mult_rule`
- dominant_trigger: `unknown`
- trigger_distribution: `{"unknown": 5}`
- reason_distribution: `{"cross_round": 5}`
- top_tokens: `currently, start, add, available, mult, deck, change, chips, money, remaining, 104, you`
- representatives:
  - Abstract Joker
  - Blue Joker
  - Bull
  - Canio
  - Ceremonial Dagger

### cluster_004 | size=5 | difficulty=med
- suggested_template: `discard_to_chips_rule`
- dominant_trigger: `mixed`
- trigger_distribution: `{"mixed": 5}`
- reason_distribution: `{"cross_round": 5}`
- top_tokens: `currently, start, available, mult, discard, add, every, discarded, round, x1, x0, consecutive`
- representatives:
  - Castle
  - Green Joker
  - Hit the Road
  - Obelisk
  - Ride the Bus

### cluster_005 | size=5 | difficulty=med
- suggested_template: `deterministic_scoring_rule`
- dominant_trigger: `on_scored`
- trigger_distribution: `{"on_scored": 4, "unknown": 1}`
- reason_distribution: `{"no_safe_template_match": 5}`
- top_tokens: `start, available, retrigger, all, every, chips, add, final, round, permanently, 100, hands`
- representatives:
  - Dusk
  - Hack
  - Hiker
  - Ice Cream
  - Seltzer

### cluster_006 | size=4 | difficulty=low
- suggested_template: `suit_mult_rule`
- dominant_trigger: `on_scored`
- trigger_distribution: `{"on_scored": 4}`
- reason_distribution: `{"no_safe_template_match": 2, "probabilistic": 1, "economy": 1}`
- top_tokens: `suit, mult, least, x1, round, changes, 30, deck, earn, end, start, available`
- representatives:
  - Ancient Joker
  - Bloodstone
  - Rough Gem
  - The Idol

### cluster_007 | size=4 | difficulty=high
- suggested_template: `complex_or_unknown_rule`
- dominant_trigger: `unknown`
- trigger_distribution: `{"unknown": 4}`
- reason_distribution: `{"cross_round": 3, "probabilistic": 1}`
- top_tokens: `mult, start, available, currently, x1, x0, 25, blind, every, time, deck, change`
- representatives:
  - Campfire
  - Constellation
  - Hologram
  - Madness

### cluster_008 | size=4 | difficulty=high
- suggested_template: `complex_or_unknown_rule`
- dominant_trigger: `unknown`
- trigger_distribution: `{"unknown": 4}`
- reason_distribution: `{"no_safe_template_match": 2, "probabilistic": 2}`
- top_tokens: `mult, start, available, round, been, poker, x3, destroyed, chance, end, deck, change`
- representatives:
  - Card Sharp
  - Cavendish
  - Gros Michel
  - Supernova

### cluster_009 | size=4 | difficulty=high
- suggested_template: `complex_or_unknown_rule`
- dominant_trigger: `unknown`
- trigger_distribution: `{"unknown": 4}`
- reason_distribution: `{"insufficient_fields": 3, "probabilistic": 1}`
- top_tokens: `must, create, room, start, available, tarot, selected, blind, straight, poker, every, discover`
- representatives:
  - Cartomancer
  - Riff-Raff
  - Superposition
  - Séance

### cluster_010 | size=4 | difficulty=high
- suggested_template: `complex_or_unknown_rule`
- dominant_trigger: `passive`
- trigger_distribution: `{"passive": 4}`
- reason_distribution: `{"economy": 2, "no_safe_template_match": 2}`
- top_tokens: `start, available, money, free, sell, reroll, shop, 20, go, debt, up, double`
- representatives:
  - Chaos the Clown
  - Credit Card
  - Diet Cola
  - Luchador

### cluster_011 | size=4 | difficulty=high
- suggested_template: `complex_or_unknown_rule`
- dominant_trigger: `passive`
- trigger_distribution: `{"passive": 4}`
- reason_distribution: `{"no_safe_template_match": 4}`
- top_tokens: `start, available, all, straights, made, flushes, size, considered, face, ex, allows, rank`
- representatives:
  - Four Fingers
  - Juggler
  - Pareidolia
  - Shortcut

### cluster_012 | size=4 | difficulty=high
- suggested_template: `complex_or_unknown_rule`
- dominant_trigger: `unknown`
- trigger_distribution: `{"unknown": 3, "mixed": 1}`
- reason_distribution: `{"cross_round": 4}`
- top_tokens: `only, currently, deck, appear, shop, there, mult, x0, x1, change, start, available`
- representatives:
  - Glass Joker
  - Lucky Cat
  - Steel Joker
  - Stone Joker

### cluster_013 | size=3 | difficulty=med
- suggested_template: `deterministic_scoring_rule`
- dominant_trigger: `on_scored`
- trigger_distribution: `{"on_scored": 2, "passive": 1}`
- reason_distribution: `{"probabilistic": 3}`
- top_tokens: `chance, start, available, tarot, must, create, room, face, money, give, booster, any`
- representatives:
  - 8 Ball
  - Business Card
  - Hallucination

### cluster_014 | size=3 | difficulty=high
- suggested_template: `complex_or_unknown_rule`
- dominant_trigger: `unknown`
- trigger_distribution: `{"unknown": 3}`
- reason_distribution: `{"cross_round": 3}`
- top_tokens: `mult, currently, least, you, add, deck, full, change, every, same, polychrome, money`
- representatives:
  - Bootstraps
  - Driver's License
  - Erosion

### cluster_015 | size=3 | difficulty=high
- suggested_template: `complex_or_unknown_rule`
- dominant_trigger: `unknown`
- trigger_distribution: `{"unknown": 3}`
- reason_distribution: `{"no_safe_template_match": 3}`
- top_tokens: `start, available, hands, selected, blind, discard, all, lose, discards, every, mult, remaining`
- representatives:
  - Burglar
  - Loyalty Card
  - Marble Joker

### cluster_016 | size=3 | difficulty=high
- suggested_template: `complex_or_unknown_rule`
- dominant_trigger: `on_discard`
- trigger_distribution: `{"on_discard": 3}`
- reason_distribution: `{"economy": 3}`
- top_tokens: `discard, start, available, money, earn, discarded, round, more, face, same, time, every`
- representatives:
  - Faceless Joker
  - Mail-In Rebate
  - Trading Card

### cluster_017 | size=3 | difficulty=high
- suggested_template: `complex_or_unknown_rule`
- dominant_trigger: `mixed`
- trigger_distribution: `{"mixed": 3}`
- reason_distribution: `{"cross_round": 2, "no_safe_template_match": 1}`
- top_tokens: `mult, discard, start, available, discarded, x0, currently, x1, loses, x2, 01, removes`
- representatives:
  - Ramen
  - Vampire
  - Yorick

### cluster_018 | size=3 | difficulty=high
- suggested_template: `complex_or_unknown_rule`
- dominant_trigger: `mixed`
- trigger_distribution: `{"mixed": 3}`
- reason_distribution: `{"cross_round": 3}`
- top_tokens: `start, add, available, currently, chips, contains, straight, 15, mult, two, pair, exactly`
- representatives:
  - Runner
  - Spare Trousers
  - Square Joker

### cluster_019 | size=2 | difficulty=high
- suggested_template: `complex_or_unknown_rule`
- dominant_trigger: `unknown`
- trigger_distribution: `{"unknown": 2}`
- reason_distribution: `{"insufficient_fields": 2}`
- top_tokens: `ability, copies, win, run, right, discard, leftmost, royal, flush`
- representatives:
  - Blueprint
  - Brainstorm

### cluster_020 | size=2 | difficulty=high
- suggested_template: `complex_or_unknown_rule`
- dominant_trigger: `passive`
- trigger_distribution: `{"passive": 2}`
- reason_distribution: `{"no_safe_template_match": 2}`
- top_tokens: `win, rounds, size, discards, round, discard, run, 12, fewer, consecutive, only, fine`
- representatives:
  - Merry Andy
  - Troubadour

## De-prioritized Buckets
- probabilistic: keep unsupported for now unless oracle evidence allows deterministic harness
- economy/shop: isolate later with dedicated economy trace scope
- cross_round_state: postpone until persistent counters/state machine are modeled
