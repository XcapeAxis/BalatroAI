# P29 SPEC - Strength Sprint

## Scope

P29 converts platform capability into measurable policy strength gains via:

- weakness bucket mining v3
- weakness-driven targeted data generation
- batch candidate training (BC/PV + Distill + RL/Hybrid)
- large-scale eval (100 / 500 / 1000 seeds)
- ranking + candidate compare + flake reliability

## RunP29 Gates

### RunP29.functional

- baseline gate (RunP27 fallback from requested RunP28) must pass
- no regressions in P23-P28 data/status pipelines

### RunP29.data_flywheel

- weakness report generated with top buckets
- targeted dataset generated and validated
- dataset summary includes source composition + bucket coverage + invalid_rows

### RunP29.training_eval

- train_batch_p29 outputs manifest + summary
- at least one valid candidate
- ablation_100 + ranking + candidate compare generated
- explicit candidate conclusion produced (improve/hold/regress)

### RunP29.reliability

- best-candidate flake smoke executed (repeats=3)
- if flake fails, recommendation is downgraded (no default promote)

## Required Artifacts

- `docs/artifacts/p29/<timestamp>/baseline_summary.{json,md}`
- `docs/artifacts/p29/<timestamp>/todo_plan.md`
- `docs/artifacts/p29/<timestamp>/weakness_latest/*`
- `docs/artifacts/p29/<timestamp>/datasets_latest/*`
- `docs/artifacts/p29/<timestamp>/train_batch_latest/*`
- `docs/artifacts/p29/<timestamp>/eval/ablation_{100,500,1000}/*`
- `docs/artifacts/p29/<timestamp>/ranking_latest/*`
- `docs/artifacts/p29/<timestamp>/candidate_compare_latest/*`
- `docs/artifacts/p29/<timestamp>/flake/best_candidate_latest/*`
- `docs/artifacts/p29/<timestamp>/gate_*.json`
- `docs/artifacts/p29/<timestamp>/report_p29_gate.json`
