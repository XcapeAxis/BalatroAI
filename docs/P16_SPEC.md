# P16 Spec

## Scope
- Real session capture (`record_real_session.py`) -> trace reconstruction (`real_trace_to_fixture.py`) -> drift check (`run_real_drift_fixture.py`) -> teacher labeling (`dagger_collect.py`) -> training (`train_bc.py`) -> evaluation (`eval_pv.py` + `eval_long_horizon.py`).
- Single entrypoint: `trainer/p16_loop.py`.
- Gate integration: `scripts/run_regressions.ps1 -RunP16`.
- Artifacts are persisted under `docs/artifacts/p16/<timestamp>/`.

## Hard Acceptance
- `RunP16` exit code must be `0`.
- P16 smoke:
  - `drift.mismatch_count == 0` (at least one fixture).
  - dataset `hand_records >= 200`, `shop_records >= 50`.
  - `invalid_rows <= 1`.
  - offline illegal rates (`hand`, `shop`) must be `<= 0.001`.
  - long horizon smoke must complete on fixed seeds and produce report files.
- All P16 reports must exist under `docs/artifacts/p16/<timestamp>/`.

## Failure Policy
- Drift mismatch: exit code `10`.
- Labeling quota/failure: exit code `11`.
- Service unavailable/restart failure: exit code `12`.
- Training/eval failure: exit code `13`.
- Failure always writes `report_p16.json` with failed stage and reason.

## Retry / Resume
- `--resume` skips any stage with existing `stage_report.json` status `PASS`.
- Failed stage can be retried without rerunning previous successful stages.

## Run Entrypoints
- Smoke:
  - `python -B trainer/p16_loop.py --mode smoke --resume`
- Full:
  - `python -B trainer/p16_loop.py --mode full --resume`
- Gate:
  - `powershell -ExecutionPolicy Bypass -File scripts/run_regressions.ps1 -RunP16`

## Repository Hygiene
- Runtime output must not be committed.
- Durable evidence is restricted to `docs/artifacts/**`.
- Cleanup keeps `docs/artifacts/**` and removes runtime temp data.
