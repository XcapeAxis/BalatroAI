# P52 Campaign Resume Validation

Validated on `2026-03-07` against the dedicated out-root:

- run root: `docs/artifacts/p52/p22_resume_validation/runs/20260307-161256/`
- campaign state: `docs/artifacts/p52/p22_resume_validation/runs/20260307-161256/p52_learned_router_smoke/campaign_runs/seed_002_BBBBBBB/campaign_state.json`
- registry snapshot: `docs/artifacts/p52/p22_resume_validation/runs/20260307-161256/p52_learned_router_smoke/campaign_runs/seed_002_BBBBBBB/checkpoint_registry_snapshot.json`
- promotion queue: `docs/artifacts/p52/p22_resume_validation/runs/20260307-161256/p52_learned_router_smoke/campaign_runs/seed_002_BBBBBBB/promotion_queue.json`
- dashboard output: `docs/artifacts/dashboard/latest/index.html`

Validation procedure:

1. Ran `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -RunP52 -OutRoot docs\artifacts\p52\p22_resume_validation`.
2. Reopened the `dashboard_build` stage in the resulting `campaign_state.json`.
3. Ran `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -RunP52 -Resume -OutRoot docs\artifacts\p52\p22_resume_validation`.

Observed stage attempts after resume:

- `build_router_dataset`: `attempt_count=1`
- `train_learned_router`: `attempt_count=1`
- `eval_learned_router`: `attempt_count=1`
- `arena_ablation`: `attempt_count=1`
- `triage`: `attempt_count=1`
- `promotion_queue_update`: `attempt_count=1`
- `dashboard_build`: `attempt_count=2`

Interpretation:

- experiment-level resume no longer short-circuits successful P52 experiments before campaign-state inspection
- completed resume-safe stages stayed skipped
- the intentionally reopened tail stage reran and completed cleanly
- produced checkpoint ids, registry refs, and campaign refs were preserved across the resume
