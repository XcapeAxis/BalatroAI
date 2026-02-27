---
name: debug-regression
description: Debug a regression or bug with reproduce, hypotheses, minimal instrumentation, root cause, fix, and regression verification. Use when something "used to work" or a test fails.
---

# Debug regression

## When to use
- User reports a regression, failing test, or "wrong behavior" with a way to reproduce.
- Task is "find and fix the bug" with a clear before/after or test.

## Input/Output
- **Input**: Failure description (expected vs actual), repro steps or test command, optional env/version.
- **Output**: Root cause, fix (files + change), and regression verification (re-run repro; must pass).

## Steps (SOP)
1. **Reproduce**: Run the given steps or test. Confirm failure.
2. **Hypotheses**: List possible causes (order by likelihood). No code change yet.
3. **Instrument**: Minimal logging/asserts or small repro to narrow down.
4. **Root cause**: Identify the single cause. State it clearly.
5. **Fix**: Smallest change that fixes the issue.
6. **Regress**: Re-run same repro and related tests. Confirm pass.

## Acceptance criteria
- Root cause stated. Fix applied. Reproduction (or test) run again and passes. No "fixed" without re-running.

## Common failures
- **Fixing without root cause**: Do not patch blindly; confirm cause first.
- **Skipping regression**: Always re-run the same repro after fix.
- **Over-fixing**: Change only what is needed; avoid refactors in the same edit.

## When not to use
- New feature (use implement-feature-with-gates or plan-then-implement).
- Refactor (use refactor-safe).
- No repro (help user get a minimal repro first).
