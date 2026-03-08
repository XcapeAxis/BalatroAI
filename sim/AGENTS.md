# Simulator Directory Rules

- Preserve simulator parity first. Semantic changes must be justified against fixtures, traces, or oracle behavior.
- Treat drift detection, replay fidelity, hashing, and trace stability as contracts, not incidental details.
- Do not change core simulator semantics without running the relevant regression coverage through `scripts\run_regressions.ps1`.
- Keep fixture scope explicit and auditable; avoid silent behavior changes that would invalidate historical artifacts.
- When behavior must change, update the matching tests or fixtures in the same change and document the reason.
