# refactor-safe

## When to use
Refactor code without changing external behavior (same inputs → same outputs, same API).

## Inputs
- Target (file, module, or refactor type: rename, extract, simplify).
- Explicit scope: what must not change (e.g. public API, tests, CLI behavior).

## Steps
1. **Define invariants**: List what must hold before and after (e.g. "all existing tests pass", "CLI flags unchanged").
2. **Refactor**: Apply changes. Prefer small, reviewable steps.
3. **Verify**: Run the invariant checks (tests, smoke, static check). Compare before/after if applicable.

## Outputs
- Invariants (short list).
- Changed files.
- Verification commands and results (before/after if relevant).

## Verification
- All stated invariants must be checked. Typically: full test run or equivalent.

## Uncertainty handling
- If behavior is unclear: add a small test or script to lock the invariant before refactoring.

## Prohibited actions
- Do not change public behavior or remove tests to "simplify" the refactor.
- Do not skip the invariant verification step.
