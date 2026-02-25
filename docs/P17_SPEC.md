# P17 Spec

## Scope
- Champion/Challenger operational loop on top of P16.
- Dataset/model registry persistence.
- PerfGate v2 (100 seeds) plus milestone eval (500 seeds).
- Failure-bucket-driven DAgger v3 loop.
- Real shadow canary (readonly, skippable when real backend unavailable).

## Gate Layers
- Functional gate: `RunP16` + P17 smoke pipeline completion.
- Perf gate: 100-seed champion vs challenger comparison.
- Milestone eval: 500-seed challenger eval (record-only, not daily hard gate).

## PerfGate v2 thresholds
- pass if any condition is true:
  - `delta_median_ante >= +0.5`
  - `delta_avg_ante >= +0.3`
  - bootstrap `CI95_low(delta_avg_ante) > 0` (when available)

## Artifacts
- Root: `docs/artifacts/p17/<timestamp>/`
- Must include:
  - gate summary files (`gate_functional.json`, `gate_perf.json`, `report_p17.json`)
  - eval outputs (100/500), compare summaries
  - registry jsonl files
  - failure mining outputs
  - canary summary or skip marker

## Failure handling
- Functional stage failure: non-zero exit.
- Perf gate fail: non-zero exit for `RunP17`, while retaining full artifacts for diagnosis.
- Real unavailable: canary stage writes `canary_skip.json` and does not fail functional gate.

