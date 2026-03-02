# P22 Experiment Orchestrator

## Goal
P22 provides an experiment matrix runner and nightly pipeline:

- matrix-driven experiment execution
- budget and seed control
- resumable runs with machine-readable artifacts
- summary/report generation
- champion/candidate update

## Config
Edit:

- `configs/experiments/p22.yaml`

Key sections:

- `budget`: wall time / max seeds / worker cap
- `seeds`:
  - `regression_fixed`: fixed seed set for regression-style comparators
  - `extra_random`: nightly extra seeds
- `matrix`: experiment variants and stage toggles
- `evaluation`: primary metric + promotion threshold

## Commands
Quick plan only:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -DryRun
```

Quick execution (2 experiments, 2 seeds):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Quick
```

Nightly style:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Nightly
```

Direct python entrypoint:

```powershell
python -B -m trainer.experiments.orchestrator --config configs/experiments/p22.yaml --out-root docs/artifacts/p22
```

Common filters:

- `--resume`
- `--only exp_a,exp_b`
- `--exclude exp_c`
- `--dry-run`
- `--keep-intermediate`
- `--max-parallel N`

## Seed Policy
- Regression-like experiments use `seeds.regression_fixed`.
- Nightly mode uses `regression_fixed + extra_random`.
- Extra random seeds are deterministic from base seed + git commit + date bucket.
- Each experiment run writes `seeds_used.json` and per-seed metrics.

## Champion / Candidate
Orchestrator updates:

- `docs/artifacts/p22/champion.json`
- `docs/artifacts/p22/candidate.json`
- `docs/artifacts/p22/CHANGELOG_P22.md`

Promotion occurs only when:

- candidate run succeeds
- candidate metric improves beyond configured threshold

## Artifacts
Per run:

- `docs/artifacts/p22/runs/<run_id>/run_plan.json`
- `docs/artifacts/p22/runs/<run_id>/<exp_id>/run_manifest.json`
- `docs/artifacts/p22/runs/<run_id>/<exp_id>/progress.jsonl`
- `docs/artifacts/p22/runs/<run_id>/<exp_id>/exp_summary.json`
- `docs/artifacts/p22/runs/<run_id>/summary_table.{csv,json,md}`
- `docs/artifacts/p22/runs/<run_id>/report_p22.json`
- `docs/artifacts/p22/report_p22_<run_id>.md`

