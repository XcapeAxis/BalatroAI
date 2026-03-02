# P22 Experiment Orchestrator

## Goal
P22 is the matrix runner for reproducible policy experiments:

- matrix-driven execution
- seed-governed comparisons
- resumable runs
- machine-readable artifacts + summary tables
- champion/candidate decision support

## Entrypoints

PowerShell wrapper (recommended):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Quick
```

Python entrypoint:

```powershell
python -B -m trainer.experiments.orchestrator --config configs/experiments/p22.yaml --out-root docs/artifacts/p22
```

## Quick Command Reference

| Scenario | Command | Notes |
|---|---|---|
| Plan only | `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -DryRun` | validates matrix + writes plan/report |
| Fast smoke | `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Quick` | 2 experiments x 2 seeds |
| Fast smoke (verbose) | `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Quick -VerboseLogs` | adds per-seed/per-stage console logs |
| Nightly style | `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Nightly -Resume` | larger seed set + resume |
| Single experiment | `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Only quick_baseline -Resume` | rerun one exp id |
| Limit seeds | `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Quick -SeedLimit 2` | local-cost control |

## Config Structure (`configs/experiments/p22.yaml`)

Key sections:

- `schema`: config schema identifier.
- `budget`:
  - `max_wall_time_minutes`
  - `max_episodes`
  - `max_seeds`
  - `max_concurrent_workers`
- `seeds`:
  - `base_seed`
  - `regression_fixed`
  - `extra_random`
- `matrix[]`:
  - `id`, `name`
  - `backend`, `policy`
  - `seed_mode` (`regression_fixed` or `nightly`)
  - `gate_flag` (passed to `scripts/run_regressions.ps1`)
  - `stages` (`sanity/gate/dataset/train/eval`)
  - `eval` settings
- `evaluation`:
  - `primary_metric`
  - `gates.baseline_gate`
  - `promotion.min_delta`

## Seed Policy and "AAAAAAA" Clarification

- Fixed seeds are intentionally used in regression-style comparisons for stability.
- P22 supports multi-seed experiments by default; quick and nightly modes materialize seed sets from config/policy.
- Nightly mode extends the fixed set with deterministic extra random seeds.
- Current default fixed seed set in `configs/experiments/p22.yaml`:
  - `AAAAAAA, BBBBBBB, CCCCCCC, DDDDDDD, EEEEEEE, FFFFFFF, GGGGGGG, HHHHHHH`
- `scripts/run_p22.ps1 -Quick` keeps runtime stable by applying `--seed-limit 2` while still using more than one seed.
- Real seeds used in each experiment are written to:
  - `docs/artifacts/p22/runs/<run_id>/<exp_id>/seeds_used.json`
- Do not infer seed usage from a single default token; always inspect `seeds_used.json`.
- Result robustness guidance:
  - 2 seeds: fast smoke and gate sanity only.
  - 8+ seeds: meaningful model comparison in local development.
  - fixed + extra_random: nightly trend monitoring and anti-overfit checks.

## Runtime Observability (During Execution)

P22 emits both per-experiment and run-level observability artifacts:

- run-level:
  - `docs/artifacts/p22/runs/<run_id>/telemetry.jsonl`
  - `docs/artifacts/p22/runs/<run_id>/live_summary_snapshot.json`
  - `docs/artifacts/p22/runs/<run_id>/summary_table.{csv,json,md}`
- per experiment:
  - `docs/artifacts/p22/runs/<run_id>/<exp_id>/run_manifest.json`
  - `docs/artifacts/p22/runs/<run_id>/<exp_id>/progress.jsonl`
  - `docs/artifacts/p22/runs/<run_id>/<exp_id>/status.json`
  - `docs/artifacts/p22/runs/<run_id>/<exp_id>/seeds_used.json`

Quick viewer helper:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\show_p22_live.ps1
```

Console UX notes:

- default output includes `Experiment i/N` start lines and per-experiment completion summaries.
- completion summaries report key metrics: `avg_ante`, `median_ante`, `win_rate`, `hand_top1`, `hand_top3`, `shop_top1`, `illegal_action_rate`.
- `-VerboseLogs` adds per-seed and per-stage detailed command progress.

## Adding a New Experiment Row

1. Open `configs/experiments/p22.yaml`.
2. Add one `matrix` entry with:
   - unique `id`
   - explicit `seed_mode`
   - correct `gate_flag`
   - stage toggles (`dataset/train` on only when needed)
3. Keep initial budget small for first iteration (`max_seeds`, `SeedLimit`).
4. Run dry-run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -DryRun -Only <new_exp_id>
```

5. Execute quick smoke:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Only <new_exp_id> -SeedLimit 2
```

## Reproduce an Existing Experiment (`quick_baseline`)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Only quick_baseline -Resume -SeedLimit 2
```

Then inspect:

- `docs/artifacts/p22/runs/<run_id>/quick_baseline/run_manifest.json`
- `docs/artifacts/p22/runs/<run_id>/quick_baseline/progress.jsonl`
- `docs/artifacts/p22/runs/<run_id>/quick_baseline/seeds_used.json`
- `docs/artifacts/p22/runs/<run_id>/summary_table.md`

## Related Docs

- [../README.md](../README.md)
- [REPRODUCIBILITY_P25.md](REPRODUCIBILITY_P25.md)
- [SEED_POLICY_P23.md](SEED_POLICY_P23.md)
