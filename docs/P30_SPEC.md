# P30 SPEC - Docs/UX Hardening + P22 Observability

## Objective
Improve onboarding clarity and experiment operability without changing core gameplay semantics:

- README top-level clarity (badges + concise value statement)
- fast, verified Quick Start
- explicit project scope/boundaries
- architecture and data-flow explanation
- P22 reproducibility + runtime observability documentation
- status/coverage synchronization for milestone P30

## Scope

- Documentation and UX improvements only.
- No simulator parity logic change.
- No policy weight promotion in this milestone.

## Deliverables

1. README improvements:
   - top badges (Python/license/CI/workflow/seed/orchestrator/trend/docs/platform + optional stars/issues)
   - concise "what this is" summary
   - quick-start path with runnable commands
   - reproducible experiment notes linked to P22 artifacts
   - architecture/data-flow section with mermaid
2. P22 documentation:
   - expanded `docs/EXPERIMENTS_P22.md`
   - updated `docs/SIM_ALIGNMENT_STATUS.md` P22 pipeline note with observability files
3. Milestone docs:
   - `docs/COVERAGE_P30_STATUS.md`
   - this spec file

## Acceptance

- `scripts/run_regressions.ps1 -RunP22` passes (covers baseline + P22 gate path).
- `scripts/run_p22.ps1 -Quick` runs and writes summary/artifacts.
- README commands and output paths are valid in local repo.
- Git working tree clean after commit/validation.
