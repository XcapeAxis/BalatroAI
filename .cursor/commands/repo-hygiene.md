# repo-hygiene

## When to use
Check repo state: uncommitted changes, suggested cleanups, and a short change summary. No destructive edits by default.

## Inputs
- (Optional) Scope: e.g. "only list", "suggest cleanup", "summarize diffs".

## Steps
1. **Status**: Run `git status` (and optionally `git diff --stat`). Summarize.
2. **Suggest**: List possible cleanups (e.g. untracked large files, stale branches, temp files). Do not delete; report only.
3. **Summary**: Short list of current state and recommended next steps (e.g. "commit X", "add Y to .gitignore").

## Outputs
- Git status summary (branch, modified/untracked files).
- Suggested cleanups (no deletions unless explicitly requested).
- Short change summary if applicable.

## Verification
- Git status (or equivalent) was actually run and reflected in the output.

## Uncertainty handling
- If a path is ambiguous (e.g. "safe to delete"): do not delete; suggest and ask.

## Prohibited actions
- Do not delete files or rewrite history unless the user explicitly asks.
- Do not run destructive commands (e.g. `rm -rf`, `git push --force`) without explicit request.
