# Risk Notes: Repo Hygiene Sweep

## Potential Risks
1. Process termination side effects
- `scripts/cleanup.ps1` now stops local `Balatro`, `balatrobot`, and `uvx` processes before cleanup.
- Impact: if a user is actively debugging a live session, cleanup will terminate it.
- Mitigation: run cleanup intentionally after regression runs.

2. Locked training data file
- `trainer_data/sim_p5_rollout.jsonl` may remain locked by another process and survive cleanup.
- Impact: disk usage noise, but no git pollution (ignored path).
- Mitigation: close the process holding the file and rerun cleanup.

3. Runtime regeneration cost
- Default cleanup removes `sim/tests/fixtures_runtime` unless `-KeepFixturesRuntime` is passed.
- Impact: future diff debug may need re-generation.
- Mitigation: use `-KeepFixturesRuntime` during active fixture debugging.

4. Artifact retention policy
- `docs/artifacts/**` is preserved to keep regression evidence.
- Impact: artifact volume can grow over time.
- Mitigation: periodic manual pruning by timestamp if needed.

## Non-Goals / Out of Scope
- No git history rewrite.
- No removal of the Python environment (`.venv_trainer`) unless `-RemoveVenv` is explicitly used.

## Recovery
- If cleanup removed useful runtime fixtures, rerun the corresponding batch scripts.
- If any gate regression appears after cleanup, rerun `scripts/run_regressions.ps1 -RunP5` to regenerate state and verify.
