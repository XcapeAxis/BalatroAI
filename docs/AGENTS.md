# Docs Directory Rules

- `README.md` is the navigation layer; detailed behavior belongs in milestone or subsystem docs.
- Milestone docs should be named by milestone and topic, and should describe what shipped, how to run it, where artifacts land, and current limitations.
- Keep command examples real. Do not document flags, scripts, artifact paths, or routes that do not exist in the repo.
- When code changes alter workflow or operator surfaces, update the corresponding docs in the same change.
- Validation workflow changes must update `README.md`, `docs/EXPERIMENTS_P22.md`, and any milestone docs that describe autonomy or certification behavior.
- Prefer concise, technical explanations over promotional language. Documentation should help someone continue the project on another machine.
