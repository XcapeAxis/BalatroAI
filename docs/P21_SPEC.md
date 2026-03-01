# P21 Spec: Mainline-Only Workflow Refactor

## Scope
- Refactor repository workflow to `MAINLINE_ONLY`.
- Remove all local branches except detected main branch.
- Align scripts, docs, and gates with direct-to-main commit policy.
- Keep git history traceable via commits and tags (no history rewrite).

## Main Branch Detection Priority
1. `git symbolic-ref refs/remotes/origin/HEAD`
2. local `main`
3. local `master`
4. fail with diagnostics (exit code `21`)

## Hard Checks

### Functional Gate (`RunP21.functional`)
- Baseline functional gate (current highest `RunP*`) runs and passes.
- Post-cleanup short gate (`RunFast`) passes.
- `RunP21` overall status is `PASS`.

### Git Workflow Gate (`RunP21.git_workflow`)
- Local branches contain only detected main branch.
- `git_mainline_status.ps1` reports `can_commit_now=true` when working tree is clean.
- `git_sync.ps1 -DryRun` report is generated.
- Branch cleanup reports exist:
  - `branch_cleanup_before.json`
  - `branch_cleanup_after.json`
  - `branch_cleanup_summary.md`

### Safety Constraints
- Tags are not deleted.
- History rewrite is not performed.
- Deleted branch tips remain traceable in cleanup report (`branch_head_commits`).

## Failure Plans
- Main branch detection fail: diagnostics + exit code `21`; no delete.
- Partial deletion fail: continue other deletes, collect `delete_errors[]`, final exit `22`.
- Post-cleanup gate fail: keep artifacts, mark `post_cleanup_regression_fail=true`, rollback guidance via commit/tag.
- `git_sync` dry-run remote/network fail (for example 502): tolerated for P21 if script logic is valid.

## Artifacts
- Root: `docs/artifacts/p21/<timestamp>/`
- Required:
  - `baseline_summary.json`, `baseline_summary.md`
  - `branch_cleanup_before.json`
  - `branch_cleanup_after.json`
  - `branch_cleanup_summary.md`
  - `git_mainline_status.json`
  - `git_sync_dryrun_report.json`
  - `gate_functional.json`
  - `gate_git_workflow.json`
  - `report_p21.json`
  - `report_p21.md`
