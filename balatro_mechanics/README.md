# Balatro Mechanics Data

This folder is reserved for mechanics reference datasets (for example CSV exports from `Balatro_Mechanics_CSV_UPDATED_20260219`).

Purpose:
- Rule implementation cross-check.
- Oracle/simulator parity debugging support.
- Documentation of assumptions.

Non-goal for current trace-contract fixes:
- No runtime dependency from `sim` trace contract logic.
- No training-time hard dependency in this iteration.

Recommended structure:
- `balatro_mechanics/*.csv` for final curated tables.
- `balatro_mechanics/manifest.json` (or `manifest_*.json`) for dataset provenance/version notes.
- raw scraping/intermediate files stay in ignored subfolders.

Data policy:
- Keep source/provenance metadata in manifest.
- Avoid committing bulky temporary extraction outputs.
## Final vs Derived Boundary

- `balatro_mechanics/*.csv` and `balatro_mechanics/manifest*.json` are curated final inputs.
- `balatro_mechanics/derived/` stores deterministic derived artifacts produced by tooling (for example joker template maps and coverage summaries).
- Raw scraping notebooks, intermediate merges, and temporary exports belong in ignored folders (`raw/`, `intermediate/`, `tmp/`) and are not committed.

P3 artifacts generated from final CSV:
- `balatro_mechanics/derived/joker_template_map.json`
- `balatro_mechanics/derived/joker_template_unsupported.json`


P3 derived artifacts (kept in repo):
- `balatro_mechanics/derived/joker_template_map.json`
- `balatro_mechanics/derived/joker_template_unsupported.json`
- `balatro_mechanics/derived/p3_discovery_clusters.json`
- `balatro_mechanics/derived/p3_supported_targets.txt`

P3 runtime artifacts (not committed):
- `sim/tests/fixtures_runtime/oracle_p3_jokers_*`
- runtime dumps and transient trace outputs

