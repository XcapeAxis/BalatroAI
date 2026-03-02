# P32 Shop / RNG Alignment

## Goal

P32 extends alignment beyond deterministic replay hashes by adding runtime shop sampling and probability-shape checks.

## Method

Sampling tool:

```powershell
python -B sim/oracle/analyze_shop_probabilities.py --base-url http://127.0.0.1:12346 --seed P32SHOP --samples 300 --out-dir docs/artifacts/p32/<timestamp>/shop_rng
```

The tool:

1. boots a run to `SHOP` phase,
2. performs repeated rerolls,
3. records observed shop/voucher/pack offers per sample,
4. aggregates joker rarity frequencies,
5. compares observed rarity mix against a mechanics-CSV proxy distribution (non-official).

## Outputs

- `shop_probability_summary.json`
- `shop_probability_summary.md`
- `shop_probability_samples.csv`
- `shop_rarity_distribution.csv`

## Alignment Interpretation

- `PASS` means runtime sampling and report generation succeeded and produced stable artifacts.
- Rarity comparison is against an internal proxy baseline derived from `balatro_mechanics/jokers.csv` counts.
- This is **not** a claim of official drop-rate verification.

## Verified Points (P32)

- SHOP reroll sampling loop is automated and artifactized.
- Joker/pack/voucher offer distributions are persisted in machine-readable and human-readable forms.
- Gate integration enforces report generation and fail-fast behavior on sampling failure.

## Known Uncertainties

- Official Balatro shop weight internals are not guaranteed by this repository.
- Runtime implementation details may vary by game/mod version.
- CSV rarity proportions are used as a pragmatic proxy, not a canonical probability contract.

## Next Steps

1. Add optional chi-square threshold gating once stable historical baselines are accumulated.
2. Split per-deck/per-stake shop sampling profiles.
3. Correlate sampled distributions with P8 mismatch diagnostics to localize model gaps faster.
