# Mainline-Only Git Workflow

## Principles
- Workflow mode: `MAINLINE_ONLY`.
- Default behavior: commit directly to the detected main branch (`origin/HEAD`, fallback `main`, then `master`).
- Feature branches are not created by default.

## Standard Pre-Commit Flow
1. Run `powershell -ExecutionPolicy Bypass -File scripts\git_mainline_status.ps1`.
2. Run gate command based on scope:
   `RunFast`, `RunFull`, `RunPerfGate`, or milestone gate `RunPn`.
3. Commit directly on main branch.
4. Optionally create a tag for milestone snapshot.
5. Run `git_sync` dry-run, then real run when needed:
   - `powershell -ExecutionPolicy Bypass -File scripts\git_sync.ps1 -DryRun:$true`
   - `powershell -ExecutionPolicy Bypass -File scripts\git_sync.ps1 -DryRun:$false`

## When To Use Tag Instead Of Branch
- Use tags for milestone checkpoints (for example `repo-mainline-only-p21`).
- Use tags for reviewable snapshots before risky refactors.
- Use tags for rollout / rollback anchors.

## When Gate Is Mandatory
- Any commit touching runtime behavior or gate scripts must run at least one functional gate before commit.
- Milestone commits must run corresponding milestone gate (`RunPn`) before tagging.

## Rollback Strategy
- Rollback by commit with `git revert`.
- Rollback by locating a known good tag and reverting from there.
- Do not rely on temporary branches for rollback.

## Codex Behavior Constraints
- Default: no feature branch creation.
- Default: commit directly on main branch.
- Exception: create a branch only when user explicitly requests branch-based work.
