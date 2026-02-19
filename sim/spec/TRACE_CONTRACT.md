# Trace Contract (Post-Action)

This project uses a **post-action trace contract**.

- `action_trace` has `N` actions, indexed `i = 0..N-1`.
- `oracle_trace` and `sim_trace` must each have exactly `N` states.
- `state[i]` must be the state **after** executing `action_trace[i]`.
- `start_snapshot` is stored separately and is **not** part of `trace[0]`.
- If pre-action state is needed for debugging, store it in a separate artifact (for example `pre_state`), not in the main trace list.

Implication for diff:
- Compare `oracle_trace[i]` vs `sim_trace[i]` directly by the same index.
