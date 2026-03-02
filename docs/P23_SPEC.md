# P23 SPEC: Seed Governance + Coverage/Flake + Nightly v2

## Scope
P23 upgrades experiment execution from "can run" to "trustworthy, explainable, reproducible, and nightly-ready."

## Gate Layers
### RunP23.functional
- `RunP22` baseline must pass.
- P23 quick orchestrator must run end-to-end and emit required artifacts.

### RunP23.experiments
- Seed policy validation must pass.
- Orchestrator quick run must pass.
- Each experiment must emit:
  - `run_manifest.json`
  - `progress.jsonl`
  - `seeds_used.json`
- `summary_table` and `report_p23` must exist.

### RunP23.reliability
- Flake smoke must run for one fixed experiment repeated three times.
- Flake report must exist.
- Gate fails on flake failures.
- Implicit single-seed defaults are disallowed by policy.

## Modes
- `quick`: 2-3 experiments, small seed budget.
- `gate`: RunP23 default.
- `nightly`: expanded matrix + nightly extra random seeds.
- `milestone`: larger fixed seed sets (500/1000) for milestone evaluation.

## Seed Governance
- Policy source: `configs/experiments/seeds_p23.yaml`
- Contract set: fixed small stable set.
- Perf gate set: deterministic fixed 100.
- Milestone sets: deterministic fixed 500/1000.
- Nightly adds reproducible random extras on top of fixed set.

## Required Runtime Outputs
- `telemetry.jsonl`
- `live_summary_snapshot.json`
- `coverage_summary.{json,md}` and `coverage_table.csv`
- `flake_report.{json,md}`
- gate json files:
  - `gate_functional.json`
  - `gate_experiments.json`
  - `gate_reliability.json`
  - `report_p23_gate.json`

