# Trainer Directory Rules

- Keep training, evaluation, registry, campaign, and monitoring changes on the same mainline; do not create a parallel experiment stack.
- Prefer existing experiment configs in `configs/experiments/` and runtime policy/config files in `configs/runtime/`.
- Route long-running work through campaign state, progress events, registry snapshots, and summary artifacts; do not bypass these with one-off result files.
- Respect the existing structure: simulator-facing data flows, self-supervised training, RL, world model, hybrid controller, learned router, registry, monitoring, and autonomy modules already have dedicated homes.
- Device/runtime selection should come from the resolver and doctor/bootstrap flows; do not hardcode a different interpreter or device policy inside trainer code.
- If a change affects promotion, autonomy, or deployment recommendations, ensure the result is visible in P22 summaries, dashboard data, and Ops UI state.
