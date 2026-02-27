---
name: implement-feature-with-gates
description: Implement a feature with explicit verification gates (tests or smoke). Use when adding behavior that must pass tests or acceptance checks before done.
---

# Implement feature with gates

## When to use
- User asks for a new feature and expects tests or a clear acceptance check.
- Task implies "implement + verify" (e.g. "add X and ensure tests pass").

## Input/Output
- **Input**: Feature description, optional acceptance criteria or test path.
- **Output**: Changed files, gate command, verification result (pass/fail), and follow-ups.

## Steps (SOP)
1. Clarify acceptance gate (e.g. `pytest -q`, smoke script, manual check). If unclear, propose one.
2. List files to add/change. Implement the feature with minimal edits.
3. Add or update tests (or smoke) that encode acceptance. Run the gate.
4. Report: files changed, command run, result. If fail, fix or report and stop.

## Acceptance criteria
- At least one gate was run (tests or smoke).
- Result is reported (pass/fail + command). No "done" without running the gate.

## Common failures
- **Gate not run**: Always run the stated gate and paste or summarize output.
- **Tests not updated**: New behavior should be covered by a test or explicit check; add or extend tests when a test framework exists.
- **Scope creep**: Implement only what was asked; suggest follow-ups separately.

## When not to use
- Pure refactor (use refactor-safe).
- Bug fix with known repro (use debug-regression).
- Exploratory or one-off script with no gate (use plan-then-implement and state "no gate").
