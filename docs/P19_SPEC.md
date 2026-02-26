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
- **Synthetic label**: Artifacts `canary_divergence_summary.json` and `canary_summary.json` include `metrics_source: "synthetic"` and `metrics_note` when divergence/risk_aware rates are derived from synthetic rules (not separate pv/rl/risk_aware model inference).

## How to run
- **Full P19 gate** (after P18):  
  `powershell -ExecutionPolicy Bypass -File scripts\run_regressions.ps1 -RunP19`
- **Quick P19 gate** (skip 1000-seed milestone):  
  `powershell -ExecutionPolicy Bypass -File scripts\run_regressions.ps1 -RunP19 -SkipMilestone1000`
- **P19 smoke only** (requires existing P18 artifacts and champion model):  
  `powershell -ExecutionPolicy Bypass -File scripts\run_p19_smoke.ps1`
- **Skip 1000-seed milestone** (faster, when running smoke script directly):  
  `scripts\run_p19_smoke.ps1 -SkipMilestone1000`
- **Perf gate only**:  
  `scripts\run_regressions.ps1 -RunP19 -RunPerfGateOnly`; fail on perf: `-FailOnPerfGate`

**Avoiding hangs when debugging:** Use `safe_run` so long runs are capped and output is logged.  
- **Windows (PowerShell):**  
  `powershell -ExecutionPolicy Bypass -File scripts\safe_run.ps1 -TimeoutSec 1200 -- powershell -ExecutionPolicy Bypass -File scripts\run_regressions.ps1 -RunP18`  
  Logs go to `.safe_run/logs/`; timeout returns exit code 124.  
- **Linux/macOS (bash):**  
  `./safe_run.sh --timeout 1200 "powershell -ExecutionPolicy Bypass -File scripts/run_regressions.ps1 -RunP18"`  
  Or run the actual shell/python command inside the quoted string.

## Seeds
- Fixed seed files: `balatro_mechanics/derived/eval_seeds_{20,100,500,1000}.txt`
- Generated with `System.Random(seed)` per spec: 20→20260224, 100→20260225, 500→20260226, 1000→20260227. One integer per line (1 to 2^31-2). Run `Ensure-Seeds` in `run_p19_smoke.ps1` or the inline Python in the original P19 spec to recreate if missing.

## Artifacts
- Root: `docs/artifacts/p19/<timestamp>/`
- Required files:
  - `report_p19.json`
  - `gate_functional.json`
  - `gate_perf.json`
  - `gate_risk.json` (must include risk_controller_smoke_pass, calibration_smoke_pass, canary_status, rollback_smoke_pass)
  - `PERF_GATE_SUMMARY.md`
  - `baseline_summary.json`, `baseline_summary.md`
- Optional / when run: `registry/`, `calibration_smoke/`, `rl_smoke/`, `ablation_100/`, `ablation_1000/`, `promotion_decision_100.json`, `failure_mining_rl/`, `dagger_v4_summary_*.json`, `real_canary_latest/` or `canary_skip.json`, `rollback_report.json` (on rollback).

