# Scripts Directory Rules

- PowerShell entrypoints should stay thin: resolve environment, call the authoritative Python/module or repo workflow, and emit clear paths/results.
- Reuse shared entrypoints instead of creating near-duplicate wrappers. `run_p22`, `run_regressions`, `run_autonomy`, `doctor`, `cleanup`, `run_dashboard`, and `run_ops_ui` are the primary surfaces.
- Long-running script actions should prefer `scripts\safe_run.ps1` so logs, timeouts, and summaries remain auditable.
- New script parameters must preserve compatibility with existing automation, dashboard, and Ops UI flows.
- If a script launches training or campaign work, pass through resolver, decision-policy, provenance, and autonomy metadata instead of hiding it.
