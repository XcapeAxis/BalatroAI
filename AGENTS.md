# BalatroAI Agent Rules

## Repo-Wide Defaults
- Work on `main` and keep changes reviewable.
- When touching milestone code, start from `powershell -ExecutionPolicy Bypass -File scripts\run_regressions.ps1 -RunP22`.
- Prefer `scripts\safe_run.ps1` for long-running commands so timeouts, logs, and summaries are preserved.
- Default to the P61 validation pyramid: Tier 0/1 fast loop first, Tier 2 only when scope requires it, Tier 3 as deferred certification unless a task explicitly needs full certification now.
- Keep artifacts under `docs/artifacts/**`; do not scatter ad-hoc logs elsewhere.
- End each milestone with updated docs, retained artifacts, a clean working tree, and a pushed `main` when possible.

## Safe Defaults
- You may modify code, configs, docs, tests, campaign state, registry state, dashboard payloads, attention queue artifacts, and morning summaries when the task requires it.
- You may run repo entrypoints such as `scripts\run_p22.ps1`, `scripts\run_regressions.ps1`, `scripts\cleanup.ps1`, `scripts\run_dashboard.ps1`, `scripts\run_ops_ui.ps1`, `scripts\doctor.ps1`, and `scripts\run_autonomy.ps1`.
- You may generate recommendations, promotion evidence, and deployment suggestions, but do not apply live or destructive changes automatically.
- Do not recreate Python environments, install system dependencies, rewrite git history, or switch real promoted checkpoints unless a human explicitly requests it.
- If evidence is statistically weak, provenance is broken, or a stop condition is hit, stop that branch and write an attention item instead of pushing through.

## Main Entrypoints
- `scripts\run_regressions.ps1`: baseline and regression gates.
- `scripts\run_p22.ps1`: main experiment orchestrator, campaign launcher, and milestone smoke/nightly entry.
- `scripts\run_autonomy.ps1`: unified autonomy entry that reads AGENTS, decision policy, attention queue, and campaign state before deciding continue / resume / block.
- `scripts\run_fast_checks.ps1`: P61 fast loop entry that plans change-scope-aware Tier 0/1/2 validation and records deferred certification.
- `scripts\run_certification.ps1`: consumes the certification queue and runs deferred Tier 3 work without changing the validated code path.
- `scripts\doctor.ps1` and `scripts\setup_windows.ps1`: environment readiness and Windows handoff/bootstrap.
- `scripts\run_dashboard.ps1` and `scripts\run_ops_ui.ps1`: operator surfaces built on the latest artifacts.
- `scripts\cleanup.ps1` and `scripts\git_sync.ps1`: retention and mainline hygiene.

## State And Artifacts
- Use checkpoint registry and promotion queue as the source of truth for model/checkpoint lifecycle.
- Use campaign state files for resumable stage execution; do not bypass them with ad-hoc long runs.
- Human-required stops belong in `docs/artifacts/attention_required/`.
- Overnight or autonomy runs should refresh `docs/artifacts/morning_summary/`.
- Dashboard and Ops UI should surface the latest autonomy, environment, and campaign state instead of reading raw logs.

## Pointer Docs
- `docs/DECISION_POLICY.md`
- `docs/P57_OVERNIGHT_AUTONOMY_PROTOCOL.md`
- `docs/P51_CHECKPOINT_REGISTRY_AND_CAMPAIGNS.md`
- `docs/P53_BACKGROUND_EXECUTION_AND_OPS_UI.md`
- `docs/EXPERIMENTS_P22.md`
- `docs/P58_WINDOWS_BOOTSTRAP.md`
- `docs/P60_AGENTS_STANDARDIZATION_AND_AUTONOMY_ENTRY.md`
- `docs/P61_WORKFLOW_ACCELERATION.md`
- Directory-specific rules live in `trainer/AGENTS.md`, `sim/AGENTS.md`, `scripts/AGENTS.md`, `docs/AGENTS.md`, and `configs/AGENTS.md`.
