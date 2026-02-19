# PR Summary: Repo Hygiene and Untracked Sweep

## Why
This change set closes the loop on repository hygiene after the P5 voucher/pack automation work. The focus is to keep the repo reproducible for CI/regression use, prevent local runtime files from polluting git state, and keep evidence artifacts persistent.

## What Changed
- Added a complete hygiene audit report: `docs/REPO_HYGIENE_SWEEP.md`.
- Hardened cleanup behavior in `scripts/cleanup.ps1`:
  - Added `-KeepFixturesRuntime` (default false).
  - Preserves `docs/artifacts/**`.
  - Stops local `Balatro` / `balatrobot` / `uvx` processes before cleanup to reduce file-lock failures.
  - Cleans `logs`, `runtime`, `trainer_runs`, `trainer_data`, and simulator runtime paths by default.
- Ran and verified full gate: `scripts/run_regressions.ps1 -RunP5` remained green for P0..P5.
- Performed post-gate cleanup and verified p3/p4/p5 artifact reports remain available under `docs/artifacts/*`.

## Key Decisions
- Keep source and docs under version control.
- Keep local runtime outputs ignored and cleaned.
- Keep persistent run evidence only under `docs/artifacts/*`.
- Avoid history rewriting to minimize risk and preserve branch safety.

## Validation
- Gate status: P0/P1/P2/P2b/P3/P4/P5 all pass.
- Working tree ends clean (`git status --porcelain` empty).
- Size dropped from ~1.51 GB to ~0.52 GB after cleanup (venv retained).

## Rollback
All changes are in small commits and can be reverted commit-by-commit without history rewrite.
