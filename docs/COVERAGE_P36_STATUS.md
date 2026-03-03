# COVERAGE P36 STATUS

- milestone: P36
- scope: self-supervised core + P22 integration + docs refresh

## Implemented

- unified self-supervised data contract:
  - `trainer/selfsup/data.py`
  - `trainer/selfsup/build_selfsup_dataset.py`
- shared encoder abstraction:
  - `trainer/models/encoder.py`
- two self-supervised tasks:
  - `future_value` (`trainer/selfsup/train_future_value.py`)
  - `action_type` (`trainer/selfsup/train_action_type.py`)
- orchestrator integration:
  - `experiment_type=selfsup_future_value`
  - `experiment_type=selfsup_action_type`
  - run artifacts include `seeds_used.json` and task metrics under each experiment run dir

## Data Sources (Current)

- sim traces from fixture runtime roots (for example `sim/tests/fixtures_runtime/oracle_p0_v6_regression`)
- real-like traces from P32 action-fidelity outputs (`docs/artifacts/p32/**/oracle_trace_real.jsonl`)
- P13 drift fixture path is supported by loader; availability depends on non-empty fixture traces

## Validation Snapshot

- `python -m py_compile` on new/updated P36 modules: PASS
- dataset build smoke: PASS (`docs/artifacts/p36/selfsup_datasets/<run_id>`)
- standalone task smoke:
  - `trainer.selfsup.train_future_value`: PASS
  - `trainer.selfsup.train_action_type`: PASS
- P22 quick with P36 rows: PASS

## Current Limits

- P36 trains representation/predictive heads, not a direct deploy policy.
- Metrics are currently normalized into P22 comparison columns for ranking continuity; absolute values are task-dependent.
- For robust policy claims, downstream BC/DAgger/search/RL and larger seed budgets remain required.

## Related

- `docs/P36_SELF_SUP_LEARNING.md`
- `docs/EXPERIMENTS_P22.md`
- `docs/SEEDS_AND_REPRODUCIBILITY.md`

