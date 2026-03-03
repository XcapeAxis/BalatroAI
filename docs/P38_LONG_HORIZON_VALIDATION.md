# P38 Long-Horizon Validation

## Why Long-Horizon Consistency Is Required

P37 verifies single-step action semantics and position-sensitive replay parity. That is necessary but not sufficient for policy training safety:

- small probability drift can accumulate across many rounds
- cross-round joker/economy state may diverge only after long trajectories
- one rare ordering bug can stay invisible in short directed fixtures

P38 adds multi-episode, multi-seed stress validation so parity is checked at aggregate statistics level, not only per-step hash equality.

## P37 vs P38

- P37 focus: action contract fidelity (`MOVE_*`, `CONSUMABLE_USE`, order-sensitive hash scopes) and deterministic replay mismatch detection.
- P38 focus: long-horizon statistical world consistency between oracle and sim.
- P37 answer: "Did this step replay correctly?"
- P38 answer: "Do long episodes produce similar distributions and outcomes?"

## Hard vs Soft Gates

Hard fail:

- any episode with replay drift mismatch (`mismatch_count > 0`)
- or long-episode batch runner reports hard failures

Soft warning:

- aggregate numeric relative diff above threshold (default `5%`)
- categorical chi-square p-value below threshold (default `0.01`, when SciPy is available)

Soft warnings are reported but do not fail process exit code.

## Components

- long episode builder:
  - `sim/oracle/batch_build_p38_long_episode.py`
- aggregate analyzer:
  - `sim/oracle/analyze_p38_long_stats.py`
- plots:
  - `sim/oracle/plot_p38_stats.py`
- regression gate integration:
  - `scripts/run_regressions.ps1 -RunP38`
- orchestrator integration:
  - `experiment_type: long_consistency` in `trainer/experiments/orchestrator.py`
  - rows in `configs/experiments/p22.yaml`
  - run summary sink: `docs/artifacts/p22/runs/<run_id>/p38_summary.json`

## Manual Run

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_regressions.ps1 -RunP38
```

Primary artifact outputs:

- `docs/artifacts/p38/long_episode/<run_id>/episode_*.json`
- `docs/artifacts/p38/long_episode/<run_id>/report_p38_long_episode.json`
- `docs/artifacts/p38/analysis_<timestamp>/summary_stats.json`
- `docs/artifacts/p38/analysis_<timestamp>/summary_stats.md`
- `docs/artifacts/p38/analysis_<timestamp>/distribution_table.csv`
- `docs/artifacts/p38/plots/score_distribution.png`
- `docs/artifacts/p38/plots/rounds_distribution.png`

## Execution Record (2026-03-04)

Commands executed in this change set:

```powershell
python -m py_compile sim/oracle/batch_build_p38_long_episode.py sim/oracle/analyze_p38_long_stats.py sim/oracle/plot_p38_stats.py trainer/experiments/orchestrator.py
powershell -ExecutionPolicy Bypass -File scripts/run_regressions.ps1 -RunP38
python -B -m trainer.experiments.orchestrator --config configs/experiments/p22.yaml --out-root docs/artifacts/p22 --only p38_long_consistency_smoke --seed-limit 1
```

Result summary:

- `RunP38`: PASS
- long-episode run id: `20260304-030038`
- hard_fail_count: `0`
- soft_warn_count: `0`
- coverage marker: `docs/COVERAGE_P38_STATUS.md`
- P22 smoke integration: PASS (`run_id=20260304-031858`, `p38_summary.json` emitted)
