---
name: repo-hygiene-and-sync
description: Check repo hygiene — git status, cleanup suggestions, change summary. Report only; no destructive actions unless user explicitly requests.
---

# Repo hygiene and sync

## When to use
- User asks for "repo hygiene", "cleanup", "git status", or "what should we clean up".
- Before or after a batch of changes: summarize state and suggest next steps.

## Input/Output
- **Input**: Optional scope (e.g. "suggest .gitignore updates", "list untracked").
- **Output**: Git status summary, list of suggested cleanups (with rationale), optional list of candidate deletions (only if user asked; no actual delete).

## Steps (SOP)
1. Run `git status` (and optionally `git diff --stat`). Summarize branch, dirty/clean, untracked, notable paths.
2. Suggest cleanups: .gitignore additions, obsolete file candidates, merge/squash suggestions. Do not delete or modify unless user explicitly asked to delete.
3. If user asked "suggest deletions": list paths and reason; do not run `rm` or `git rm` without explicit confirmation.

## Acceptance criteria
- Status and suggestions are based on real `git status`/diff output. No invented state. No destructive actions without explicit request.

## Common failures
- **Inventing status**: Always run git and report actual output.
- **Deleting without ask**: Only suggest; do not delete unless user said "delete" or "remove" with confirmation.
- **History rewrite**: Do not suggest rebase/force-push unless user asked.

## When not to use
- User asked to "commit" or "push" (do that; no need for full hygiene SOP).
- User asked for a specific refactor (use refactor-safe).
