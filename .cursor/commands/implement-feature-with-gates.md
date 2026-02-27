# implement-feature-with-gates

## When to use
Implement a feature and ensure it is guarded by tests and/or acceptance criteria.

## Inputs
- Feature description and acceptance criteria (or "definition of done").
- (Optional) Files/modules in scope and existing tests to extend.

## Steps
1. **Scope**: Identify files to add/change and existing tests (if any).
2. **Implement**: Add or change code to meet the acceptance criteria. Minimal, readable changes.
3. **Gates**: Add or update tests; document the acceptance command and pass criteria.
4. **Run gates**: Execute the acceptance command (e.g. pytest, script). Report pass/fail.

## Outputs
- List of changed/added files.
- Acceptance command (copy-pasteable).
- Acceptance criteria (short list).
- Gate result (pass/fail and any failures).

## Verification
- At least one acceptance gate must run. If tests exist: run them. If not: document a manual or script-based check.

## Uncertainty handling
- If acceptance is ambiguous: propose concrete criteria and get confirmation before claiming done.

## Prohibited actions
- Do not mark feature "done" without running the acceptance gate.
- Do not remove or disable existing tests to make the gate pass.
