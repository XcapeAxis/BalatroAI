# P37 Probability / Weight Parity Audit

## Scope

This audit makes shop/pack probability behavior reviewable instead of implicit.

- Focus:
  - SHOP reroll offers
  - PACK offers and PACK_OPEN choices
  - RNG replay source-of-truth (`seed/cursor/events/outcomes`)
- Non-goal:
  - Claiming official in-game closed-form drop rates from CSV alone.

## Weight Source Mapping

| Domain | Source of truth in code | Mechanics reference | Notes |
|---|---|---|---|
| Shop reroll offers | `sim/oracle/extract_rng_events.py` (`shop_offers`, `shop_roll`) + `sim/core/engine.py` RNG replay apply path | `balatro_mechanics/jokers.csv`, `tarot_cards.csv`, `planet_cards.csv` | Runtime offers are sampled from oracle and replayed into sim outcomes. |
| Voucher offers | `sim/oracle/extract_rng_events.py` (`voucher_offers`) + sim RNG replay apply path | `balatro_mechanics/vouchers.csv` | Same replay-driven parity path. |
| Pack offers | `sim/oracle/extract_rng_events.py` (`pack_offers`) + sim RNG replay apply path | `balatro_mechanics/booster_packs.csv` | Offer keys/types align through oracle outcomes. |
| Pack opened choices | `sim/oracle/extract_rng_events.py` (`pack_choices`) + sim RNG replay apply path | `balatro_mechanics/*_cards.csv` | PACK_OPEN choice set parity is replay-audited. |
| RNG contract | canonical `rng.seed/cursor/events` + `rng_replay.outcomes` in trace/action | N/A (runtime contract) | Alignment path is deterministic replay rather than independent duplicated weight sampler. |

## Statistical Audit Run (P37)

- Command:

```powershell
python -B sim/oracle/audit_p37_probability_parity.py --base-url http://127.0.0.1:12346 --seed P37PROBAUDIT2 --samples 240 --pack-interval 5 --out-dir docs/artifacts/p37
```

- Artifacts:
  - `docs/artifacts/p37/probability_audit_20260304-013136.json`
  - `docs/artifacts/p37/probability_audit_20260304-013136.md`

- Result summary:
  - `status=PASS`
  - `samples_collected_reroll=240`
  - `oracle_steps=244`, `sim_replayed_steps=244`
  - Core metrics (`shop_set/shop_key/voucher_key/pack_offer_set/pack_offer_key/pack_choice_key`) all reported `KL=0`, `L1=0`, `Chi-square=0` in this run.

## Interpretation

- For replay-governed parity scopes, sim and oracle distributions are identical in the sampled run because sim consumes oracle-extracted RNG outcomes deterministically.
- This validates *mechanics parity of observable outcomes* for shop/pack events in the current framework.

## Known Gaps

- `balatro_mechanics/*.csv` is metadata/reference and does not expose complete official runtime weight constants for every stochastic path.
- Current parity path is replay-driven. A fully independent native sim sampler with first-principles weights is still future work.
- Live oracle sampling depends on local balatrobot + valid Balatro executable paths.

