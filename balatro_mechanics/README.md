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
