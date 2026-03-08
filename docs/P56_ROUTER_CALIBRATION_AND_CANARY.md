# P56 Router Calibration and Canary

P56 moves the P54 learned router from "it trains and runs" to "it can be compared, calibrated, and promoted conservatively." The milestone keeps the learned router lightweight and reuses the existing P49/P50/P51/P53/P54 stack instead of introducing a parallel serving path.

## Goals

P56 adds five practical capabilities on top of P54:

- multi-seed router benchmarking instead of single small-sample snapshots
- controller-confidence calibration analysis with bucketed reliability metrics
- explicit guard-threshold tuning for `learned_with_rule_guard`
- `canary_learned_router` deployment mode for low-risk sub-slices only
- first-class P22, registry, dashboard, and ops-ui visibility for those results

## Deployment Modes

Supported router modes are now:

1. `rule`
2. `learned`
3. `learned_with_rule_guard`
4. `canary_learned_router`

`canary_learned_router` only uses the learned router when the state is considered safe enough for rollout. Otherwise it falls back to the P48 rule router.

## Multi-Seed Benchmark Design

Primary modules:

- `trainer/hybrid/router_benchmark.py`
- `trainer/hybrid/router_benchmark_schema.py`

The benchmark harness evaluates the same learned-router checkpoint family across multiple seeds, configurable seed pools, and multiple router modes.

Required compare set:

- `rule`
- `learned`
- `learned_with_rule_guard`
- `canary_learned_router`

Optional baselines remain available when configured:

- `policy_baseline`
- `policy_plus_wm_rerank`

Benchmark outputs include:

- mean / std / median of the primary outcome metric
- catastrophic-failure counts
- per-slice summaries
- controller-selection distributions
- guard-trigger and canary-usage rates
- seed-level raw results
- checkpoint ids and config provenance

Artifacts:

- `docs/artifacts/p56/router_benchmark/<run_id>/benchmark_manifest.json`
- `docs/artifacts/p56/router_benchmark/<run_id>/benchmark_summary.json`
- `docs/artifacts/p56/router_benchmark/<run_id>/benchmark_summary.md`
- `docs/artifacts/p56/router_benchmark/<run_id>/seed_results.json`
- `docs/artifacts/p56/router_benchmark/<run_id>/slice_results.json`

## Calibration Design

Primary module:

- `trainer/hybrid/router_calibration.py`

P56 treats the learned router's controller-selection probabilities as a quantity to audit, not a number to trust by default.

The calibration pass reports:

- confidence buckets
- bucket accuracy / best-controller hit rate
- ECE-style summary metric
- per-controller confidence stats
- optional slice-aware calibration breakdowns
- a simple bias readout (`optimistic`, `conservative`, or `usable`)

Artifacts:

- `docs/artifacts/p56/router_calibration/<run_id>/calibration_metrics.json`
- `docs/artifacts/p56/router_calibration/<run_id>/reliability_bins.json`
- `docs/artifacts/p56/router_calibration/<run_id>/calibration_report.md`

## Guard Tuning Design

Primary module:

- `trainer/hybrid/router_guard_tuning.py`

P56 turns `learned_with_rule_guard` into a tunable deployment policy instead of a single hard-coded threshold pack.

The sweep currently tunes combinations of:

- `router_confidence_min`
- `wm_uncertainty_max`
- `feature_completeness_min`
- `high_risk_slice_force_rule`
- optional `ood_score_max`

Each candidate guard config is scored against:

- performance
- catastrophic-failure counts
- guard-trigger rate
- stability relative to learned-only and rule-only routing

Artifacts:

- `docs/artifacts/p56/guard_tuning/<run_id>/guard_tuning_results.json`
- `docs/artifacts/p56/guard_tuning/<run_id>/guard_tuning_results.md`
- `docs/artifacts/p56/guard_tuning/<run_id>/recommended_guard_config.json`

## Canary Logic

Primary integration points:

- `trainer/hybrid/router.py`
- `trainer/hybrid/hybrid_controller.py`

`canary_learned_router` routes through the learned policy only when the current state passes a conservative eligibility filter. Typical checks are:

- learned-router confidence above threshold
- world-model uncertainty below threshold
- feature completeness above threshold
- not in a known high-risk slice

Per-decision trace rows now record:

- `canary_eligible`
- `canary_used`
- `canary_reject_reason`
- `final_controller`
- `guard_triggered`

This keeps the canary path auditable at the same trace granularity as the older guarded router.

## Arena, Registry, and Campaign Integration

P56 reuses the existing arena-first governance path:

- arena compare now covers rule, learned, guarded, and canary variants
- triage adds learned-router impact, guard effectiveness, canary effectiveness, and degrading-slice attribution
- learned-router checkpoints stay under `family=learned_router`
- registry rows gain `calibration_ref`, `guard_tuning_ref`, `canary_eval_ref`, and `deployment_mode_recommendation`
- the promotion queue can recommend canary mode when learned-only routing is unstable

P56 campaign stages are resumeable and include:

- `build_router_dataset`
- `train_learned_router`
- `eval_calibration`
- `tune_guard_thresholds`
- `arena_ablation`
- `canary_eval`
- `triage`
- `promotion_queue_update`
- `dashboard_build`

Representative campaign artifacts:

- `docs/artifacts/p22/runs/<run_id>/p56_router_calibration_smoke/campaign_runs/seed_*/campaign_state.json`
- `docs/artifacts/p22/runs/<run_id>/p56_router_calibration_smoke/campaign_runs/seed_*/checkpoint_registry_snapshot.json`
- `docs/artifacts/p22/runs/<run_id>/p56_router_calibration_smoke/campaign_runs/seed_*/promotion_queue.json`

## Dashboard and Ops UI

P56 surfaces through the existing monitoring stack instead of a separate UI.

Dashboard additions include:

- learned-router dataset size and label-confidence stats
- calibration summary and reliability bins
- guard-tuning summary and recommended config
- canary usage, fallback rate, and latest deployment recommendation
- recent learned-router checkpoints and campaign stage status

Ops UI additions include:

- router calibration page
- router guard / canary page
- latest ablation summary with rule vs learned vs guarded vs canary comparisons
- promotion recommendation for deployment mode

## P22 Commands

Quick smoke:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -RunP56
```

Quick matrix (includes the smoke row):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Quick
```

Resume the latest compatible P56 campaign:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -RunP56 -Resume
```

Standalone targeted utilities:

```powershell
python -m trainer.hybrid.router_benchmark --config configs/experiments/p56_router_calibration_smoke.yaml --quick
python -m trainer.hybrid.router_calibration --config configs/experiments/p56_router_calibration_smoke.yaml
python -m trainer.hybrid.router_guard_tuning --config configs/experiments/p56_router_calibration_smoke.yaml
```

## Known Limitations

- calibration quality is still limited by available seeds, slices, and label quality from the underlying learned-router dataset
- canary thresholds are heuristic sweeps, not formal risk guarantees
- rare controller-collapse or OOD cases can still slip through small-budget smokes
- learned-router promotion remains recommendation-only; arena and human review stay authoritative

## Related Docs

- [P54_LEARNED_ROUTER.md](P54_LEARNED_ROUTER.md)
- [P48_ADAPTIVE_HYBRID_CONTROLLER.md](P48_ADAPTIVE_HYBRID_CONTROLLER.md)
- [P39_POLICY_ARENA.md](P39_POLICY_ARENA.md)
- [P41_CLOSED_LOOP_V2.md](P41_CLOSED_LOOP_V2.md)
- [P49_GPU_MAINLINE_AND_DASHBOARD.md](P49_GPU_MAINLINE_AND_DASHBOARD.md)
- [P51_CHECKPOINT_REGISTRY_AND_CAMPAIGNS.md](P51_CHECKPOINT_REGISTRY_AND_CAMPAIGNS.md)
- [P53_BACKGROUND_EXECUTION_AND_OPS_UI.md](P53_BACKGROUND_EXECUTION_AND_OPS_UI.md)
- [EXPERIMENTS_P22.md](EXPERIMENTS_P22.md)
