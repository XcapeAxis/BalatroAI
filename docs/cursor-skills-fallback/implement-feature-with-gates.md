# SOP: Implement feature with gates

## When to use
Implementing a feature that must pass tests or explicit acceptance criteria before being considered done.

## Inputs
- Feature description and acceptance criteria (or "definition of done").
- Optional: files/modules in scope, existing test paths.

## Outputs
- Changed/added files list.
- Copy-pasteable acceptance command.
- Pass/fail result of the gate.

## Steps (SOP)
1. Clarify scope: which files to add/change; which tests or checks already exist.
2. Implement the feature with minimal, readable changes.
3. Add or update tests to cover acceptance; document the acceptance command and criteria.
4. Run the acceptance command (e.g. `pytest path/to/test.py`, or a script). Record result.
5. If gate fails: fix and re-run until pass, or report failure and blockers.

## Acceptance criteria
- At least one gate (test run or script) must be executed.
- Output must state clearly: gate pass or fail; if fail, what failed.

## Common failures
- Tests not run: always run the gate and report.
- Acceptance criteria vague: define concrete pass/fail before claiming done.
- Disabling tests to pass: do not remove or skip tests to make the gate pass.

## When not to use
- Pure refactor with no new behavior (use refactor-safe flow).
- Exploratory or one-off scripts with no acceptance definition.
