# P15 Status

## Scope
- Search-labeled sim dataset generation for hand and shop decisions.
- Policy-Value model training and offline evaluation.
- Gold-stake long-horizon comparison for `heuristic` vs `pv`.

## Pipeline
1. `trainer/rollout_search_p15.py`
2. `trainer/train_pv.py`
3. `trainer/eval_pv.py`
4. `trainer/eval_long_horizon.py --policy pv`
5. `scripts/run_p15_smoke.ps1` via `scripts/run_regressions.ps1 -RunP15`

## Artifacts
- Gate artifacts are persisted under `docs/artifacts/p15/<timestamp>/`.
- Runtime datasets/models remain local (`trainer_data/`, `trainer_runs/`) and are gitignored.

