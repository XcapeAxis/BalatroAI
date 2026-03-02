# P25 SPEC: README / Docs Productization

## Functional Gate (RunP25.functional)

- Baseline gate (`RunP24`) passes.
- Quick-start smoke command passes (`scripts/run_p23.ps1 -DryRun`).
- Existing script entry points remain runnable.

## Docs Gate (RunP25.docs)

Must satisfy all:

- README contains required sections:
  - Quick Start
  - What This Project Is (Value)
  - Scope and Boundaries (Not suitable)
  - Architecture Overview
  - Reproducibility
  - Example Outputs
  - Roadmap
  - Known Limitations
- Badge count >= 8 (shields.io)
- `scripts/generate_readme_status.ps1` runs successfully
- `docs/REPRODUCIBILITY_P25.md` exists and is linked in README

## Asset Completeness

Must satisfy all:

- `docs/assets/readme/` exists
- At least three output asset types are present:
  - log snippet
  - summary table snippet
  - architecture/dataflow visual source
- README embeds at least one real output snippet and links these assets.

## P25 Gate Outputs

RunP25 writes:

- `baseline_summary.json` / `.md`
- `readme_status_generation.json`
- `readme_lint_report.json`
- `quickstart_smoke.json`
- `gate_functional.json`
- `gate_docs.json`
- `report_p25_gate.json`
