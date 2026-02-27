# SOP: Debug regression

## When to use
Investigating a bug or regression: reproduce, find root cause, fix, and confirm with regression.

## Inputs
- Symptom (what broke, when, where).
- Optional: suspected area, last known good, or partial repro steps.

## Outputs
- Reproduce steps (and command if applicable).
- Root cause with evidence.
- List/diff of changed files.
- Regression verification (command + result).

## Steps (SOP)
1. **Reproduce**: Document exact steps, environment, and inputs that trigger the failure. Run and confirm.
2. **Hypotheses**: List likely causes (recent change, config, data, dependency) by likelihood.
3. **Instrument**: Add minimal logs, asserts, or a small repro script. No large refactors.
4. **Confirm**: Pinpoint root cause with evidence before changing behavior.
5. **Fix**: Apply minimal code change to address the cause.
6. **Regress**: Re-run relevant tests or the repro; confirm the symptom is gone and no new failures.

## Acceptance criteria
- Root cause stated with evidence.
- Regression step run and passed (or explicitly skipped with reason).

## Common failures
- Fixing before confirming cause: always confirm with evidence first.
- Skipping regression: always re-run tests or repro after fix.
- Large "fix" that changes unrelated code: keep changes minimal.

## When not to use
- New feature work (use implement-feature-with-gates).
- Refactor or cleanup without a specific failure (use refactor-safe or repo-hygiene).
