# debug-regression

## When to use
Investigating a regression or bug: reproduce, find root cause, fix, and regress.

## Inputs
- Symptom (what broke, when, where).
- (Optional) Suspected area, last known good, or repro steps.

## Steps
1. **Reproduce**: Document steps, env, and inputs that trigger the failure.
2. **Rank hypotheses**: List likely causes (e.g. recent change, config, data) by likelihood.
3. **Minimal instrumentation**: Add logs, asserts, or a small repro script—no large refactors.
4. **Confirm root cause**: Pinpoint the cause with evidence before changing behavior.
5. **Fix**: Apply minimal code change to fix the cause.
6. **Regression**: Re-run relevant tests or checks; confirm symptom is gone and nothing else broke.

## Outputs
- Reproduce steps (and command if applicable).
- Root cause (with evidence).
- Diff or list of changed files.
- Regression verification (command + result).

## Verification
- Regression step must run and pass (or be explicitly skipped with reason).

## Uncertainty handling
- If cause is uncertain: say so and list remaining hypotheses. Do not assert "fixed" without regression.

## Prohibited actions
- Do not change behavior before confirming root cause.
- Do not skip regression verification.
