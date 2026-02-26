# P20 Spec: Release Candidate Pipeline

## Scope
- RC Package export/verify tooling and deploy package specification.
- Ensemble teacher distillation (pv+hybrid+rl+risk_aware) -> deploy student model.
- Determinism / replay audit for fixed seeds + fixed model.
- Champion Manager v4 with release channels (dev/canary/stable) and staged promote/rollback.
- 2000-seed milestone evaluation suite.
- Real micro A/B (controlled, skippable).
- P20 layered gates wired through `scripts/run_p20_smoke.ps1` and `scripts/run_regressions.ps1 -RunP20`.

## Hard Checks

### Functional Gate (RunP20.functional)
- P0-P19 all green.
- P20 smoke full flow completes without crash; all required artifacts emitted.

### Perf Gate (PerfGate v5, 100 seeds)
Candidate (deploy_student or rl/risk_aware) vs champion must satisfy at least one:
- `median_ante_reached` improvement >= +0.5
- `avg_ante_reached` improvement >= +0.3
- If CI available: `avg_ante_delta` 95% CI lower bound > 0

### Reliability Gate
Must satisfy all:
- `package_verify_pass` = true
- `determinism_audit_pass` = true (or documented exemption)
- risk-aware infer smoke runs
- canary/real_ab status is PASS or SKIP (SKIP must have artifact and reason)

### Release Gate
- champion_manager v4 outputs parseable decision (promote / hold / reject / rollback)
- If rollback triggered: registry state and current pointer updated

### Milestone Evaluation (2000 seeds)
- Must produce results (at least champion vs best_candidate)
- Record win_rate / avg ante / median ante / failure breakdown / CI (if available)
- `hold_for_more_data` allowed with written rationale

## How to Run
- **Full P20 gate** (after P19):
  `powershell -ExecutionPolicy Bypass -File scripts\run_regressions.ps1 -RunP20`
- **Skip 2000-seed milestone** (faster):
  `powershell -ExecutionPolicy Bypass -File scripts\run_regressions.ps1 -RunP20 -SkipMilestone2000`
- **P20 smoke only**:
  `powershell -ExecutionPolicy Bypass -File scripts\run_p20_smoke.ps1`
- **With safe_run wrapper**:
  `powershell -ExecutionPolicy Bypass -File scripts\safe_run.ps1 -TimeoutSec 3600 -- powershell -ExecutionPolicy Bypass -File scripts\run_regressions.ps1 -RunP20`

## Seeds
- Fixed seed files: `balatro_mechanics/derived/eval_seeds_{20,100,500,1000,2000}.txt`
- 2000 seeds: generated with `System.Random(20260228)` / Python `random.Random(20260228)`.

## Artifacts
- Root: `docs/artifacts/p20/<timestamp>/`
- Required files:
  - `report_p20.json`
  - `gate_functional.json`, `gate_perf.json`, `gate_reliability.json`, `gate_release.json`
  - `PERF_GATE_SUMMARY.md`, `RELIABILITY_GATE_SUMMARY.md`
  - `baseline_summary.json`, `baseline_summary.md`
  - `packages/champion_rc/` (RC package + verify report)
  - `registry/` (models registry, release state, current pointers)
  - `distill_smoke/` (distillation train/eval summaries)
  - `determinism_smoke/` (audit report)
  - `ablation_100/` (multi-strategy comparison)
  - `release_decision_100.json` + `.md`
  - `ablation_2000/` (milestone evaluation)
  - `real_ab_latest/` or `real_ab_skip.json`
