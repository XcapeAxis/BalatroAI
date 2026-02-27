# SOP: Repo hygiene and sync

## When to use
Checking repo state: uncommitted changes, suggested cleanups, and a short summary. No destructive actions by default.

## Inputs
- Optional: scope ("only list", "suggest cleanup", "summarize diffs").

## Outputs
- Git status summary (branch, modified/untracked files).
- Suggested cleanups (e.g. add to .gitignore, remove temp files). No deletions unless requested.
- Short change summary and recommended next steps.

## Steps (SOP)
1. **Status**: Run `git status` (and optionally `git diff --stat`). Summarize branch and file state.
2. **Suggest**: List possible cleanups (untracked large files, stale branches, temp dirs). Do not delete; report only.
3. **Summary**: Short list of current state and next steps (e.g. "commit X", "add Y to .gitignore").

## Acceptance criteria
- Git status (or equivalent) was run and reflected in the output.
- No file deletions or history rewrites unless explicitly requested.

## Common failures
- Deleting files without asking: only suggest; do not run `rm` or force-push.
- Skipping status: always run and show status.

## When not to use
- Implementing a feature (use implement-feature-with-gates).
- Debugging a failure (use debug-regression).

## Migration note
If the project later enables Cursor Skills, this SOP can be moved to `.cursor/skills/repo-hygiene-and-sync/SKILL.md` with the same structure and frontmatter (name, description).
