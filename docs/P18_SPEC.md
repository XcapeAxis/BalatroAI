# P18 Spec

## Scope
- Add RL pilot after Search/BC/DAgger flow.
- Add curriculum sampling and mixed replay instrumentation.
- Add champion-challenger v2 risk guard.
- Add ablation automation (heuristic/pv/hybrid/rl) for 100/500 seeds.
- Keep layered gates: functional vs perf.

## Entrypoints
- `powershell -ExecutionPolicy Bypass -File scripts/run_regressions.ps1 -RunP18`
- `python -B trainer/train_rl.py --config trainer/config/p18_rl.yaml --mode smoke ...`
- `python -B trainer/run_ablation.py ...`

## Hard checks
- RunP17 stays green before P18.
- P18 smoke pipeline produces artifacts and exits 0.
- Perf gate v3 outputs `perf_gate_pass`, `risk_guard_pass`, `final_decision`.
- Cleanup preserves `docs/artifacts/**`.

## Failure policy
- Service failure: retry/recover or exit with non-zero and report.
- RL train failure: emit `gate_rl_smoke.json` with FAIL and reason.
- Perf failure does not invalidate functional gate unless `RunPerfGateV3` strict mode is used.

## Artifact root
- `docs/artifacts/p18/<timestamp>/`
