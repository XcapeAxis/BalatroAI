# Manual Cursor GUI Checklist

Use this list in Cursor Settings (UI names may vary by version).

## Model and mode

- [ ] **Default model**: Use Auto for daily work; switch to a specific model for critical or sensitive tasks; use Max only when you need large context.
- [ ] **Ask vs Agent**: Prefer Ask for planning and review; use Agent for implementation; use Ask again to review diffs and outcomes.

## Indexing

- [ ] **Codebase Indexing**: Ensure it is enabled so the agent can use project context.
- [ ] **Index scope**: Check that indexed paths are reasonable (e.g. exclude venv, `node_modules`, large data dirs). Align with `.cursorignore`.
- [ ] **Confirm `.cursorignore`**: After editing `.cursorignore`, verify it’s applied (e.g. open a file under an ignored path and confirm it’s not offered as context, or check settings for “Ignore” / “Cursor ignore” if available).

## Skills (if your Cursor version supports project or user skills)

- [ ] **Skills**: Enable or confirm project/user skills if you use `.cursor/skills/` or equivalent. If not available, use the SOPs in `docs/cursor-skills-fallback/` as reference.

## Terminal and execution

- [ ] **Terminal permissions**: Prefer confirming before running high-risk commands (e.g. overwrite, delete, network). Review Cursor’s execution permissions for scripts and shell.

## Version and UI

- [ ] **UI differences**: If an option is missing or named differently, check the current Cursor release notes or docs; names and locations can change between versions.
