# P19 Spec

## Scope
- Risk-aware inference controller (`pv/hybrid/rl/heuristic` fallback) for hand+shop.
- Champion manager v3 with staged status (`candidate/probation/champion/rolled_back/rejected`).
- 100/1000 seed ablation support including `risk_aware`.
- DAgger v4 wrapper with source-policy/failure-prioritized summary.
- P19 layered gates wired through `scripts/run_p19_smoke.ps1` and `scripts/run_regressions.ps1 -RunP19`.

## Hard Checks
- **Functional gate**: P0–P18 all green; P19 smoke stages complete without crash and emit all required artifacts.
- **Perf gate** (100 seeds): PerfGate uses median_ante_delta and/or avg_ante_delta thresholds; optionally CI lower bound > 0. At least one candidate (`rl` or `risk_aware`) passes OR decision is explicit reject/hold with rationale.
- **Risk gate**: risk_controller_smoke_pass, calibration_smoke_pass, rollback_smoke_pass all true; canary_status is PASS or SKIP (SKIP does not fail the risk gate).
- **Milestone eval**: 1000-seed report must exist under P19 artifacts; hold_for_more_data is allowed with written rationale.

## Failure Plan
- Any stage failure writes `gate_functional.json` with reason and exits non-zero.
- Perf fail can be tolerated unless `-FailOnPerfGate` is passed.
- Real canary can be `SKIP` with explicit `canary_skip.json`.

## Calibration
- ECE and per-phase (hand/shop) metrics use softmax confidence as proxy when models do not output strict probabilities; see `trainer/eval_calibration.py` and `trainer/calibration.py`. Document this limitation in reports.

## Real Canary
- Divergence reporting uses a single-model top-k to synthesize pv/hybrid/rl/risk_aware policies (not true multi-model inference per state). See `trainer/real_shadow_canary.py` and canary_divergence_summary.json. When real is unreachable, canary_skip.json is written and the step ends with SKIP without failing the functional gate.

## Artifacts
- Root: `docs/artifacts/p19/<timestamp>/`
- Required files:
  - `report_p19.json`
  - `gate_functional.json`
  - `gate_perf.json`
  - `gate_risk.json` (must include risk_controller_smoke_pass, calibration_smoke_pass, canary_status, rollback_smoke_pass)
  - `PERF_GATE_SUMMARY.md`
  - `baseline_summary.json`, `baseline_summary.md`

