# BalatroAI Codex Working Rules

## Default Workflow
- Work directly on `main`; do not create long-lived feature branches unless a human explicitly asks.
- Start with a baseline gate when touching milestone code. The default baseline is `powershell -ExecutionPolicy Bypass -File scripts\run_regressions.ps1 -RunP22`.
- Prefer `scripts\safe_run.ps1` for long-running PowerShell commands so bootstrap, regressions, and nightly flows leave auditable logs and time out cleanly.
- Prefer small, reviewable commits grouped by subsystem.
- End each milestone with a clean working tree, updated docs, retained artifacts under `docs/artifacts/**`, and a pushed `main` branch when possible.

## Auto-Allowed Actions
- Modify code, configs, docs, and tests that are directly required for the task.
- Run repo-approved scripts such as `run_regressions`, `run_p22`, `cleanup`, `git_sync`, dashboard builders, ops UI helpers, campaign utilities, training, evaluation, arena, and triage flows.
- Update config sidecars, registry state, campaign state, dashboard payloads, attention queue artifacts, and morning summaries.
- Generate recommendations, candidate evaluations, and promotion suggestions as artifacts without directly applying high-risk production changes.
- Run environment inspection and readiness checks such as `scripts\doctor.ps1` and resolver probes.

## Human-Required Actions
- Any environment, driver, OS, or system dependency change.
- Creating or recreating local Python environments, installing Python packages, or switching CUDA/runtime stacks unless a human explicitly starts the bootstrap flow.
- Destructive git actions such as history rewrites, force-pushes, branch deletion, or mass source deletion.
- Switching a real champion or promoted checkpoint, or any action that changes a live/default deployment target.
- Route-selection or roadmap decisions based on statistically insufficient evidence.
- Unexplained major regressions, broken config provenance that cannot be auto-repaired, or actions that may damage local environment integrity.

## Stop Policy
- Stop the overnight/autonomy branch and create an attention item when a task requires human approval, a stop-condition in the decision policy is hit, or a critical stage cannot safely degrade.
- Continue with warnings only for explicit warning-class conditions recorded by the decision policy and campaign state.
- Do not silently retry across unresolved human gates. Resume must respect open attention items tied to blocked stages.

## Output Contract
- Long-running work should persist a `campaign_state.json` when stage-based execution is used.
- Every substantial run should produce a human-readable summary artifact.
- When human review is required, write a structured attention item plus queue summary under `docs/artifacts/attention_required/`.
- Overnight/autonomy runs should finish with a morning summary under `docs/artifacts/morning_summary/`.
- Bootstrap and doctor flows should persist the latest environment state under `docs/artifacts/p58/`.
