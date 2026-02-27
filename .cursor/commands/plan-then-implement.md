# plan-then-implement

## When to use
Standard entry for any non-trivial task: plan first, then implement, then verify.

## Inputs
- Task description and any constraints I provide.
- (Optional) Affected modules or files I specify.

## Steps
1. **Plan**: State goal, list files to change, verification method, and risks. Get alignment if scope is large.
2. **Implement**: Make minimal changes to achieve the goal. Prefer local edits.
3. **Verify**: Run at least one check (tests, smoke, static check, or small run). Report pass/fail.
4. **Summary**: Deliver the fixed output structure (changed files, commands, verification result, risks).

## Outputs
- **Plan**: Goal, files to change, how to verify, risks.
- **Changed files**: Paths only.
- **Commands**: Copy-pasteable commands used.
- **Verification result**: What was run and outcome.
- **Risks / follow-ups**: Short note.

## Verification
- At least one of: test run, lint, smoke command, or small repro. Must be explicit.

## Uncertainty handling
- If unclear: state assumptions and constraints. Do not guess; ask or document.

## Prohibited actions
- Do not claim "fixed" or "done" before running verification.
- Do not refactor beyond what the task requires.
