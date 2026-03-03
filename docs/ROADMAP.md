# BalatroAI Roadmap

## Milestone Tree

| Milestone | Theme | Status |
|---|---|---|
| P0-P13 | Oracle/sim baseline alignment and real-session drift loop | done |
| P22-P27 | Experiment orchestration, seed governance, campaign/release ops | done |
| P29-P37 | Data flywheel, action replay unification, self-supervised and single-action fidelity | done |
| P38 | Long-horizon statistical consistency framework (multi-seed stress + aggregate parity) | done |
| P39 | Policy Arena v1 (multi-policy adapters, bucketed evaluation, champion rule input) | in progress |

## Current Focus: P39

1. Maintain one adapter contract across heuristic/search/model/hybrid policy families.
2. Keep arena outputs machine-readable for policy promotion workflows.
3. Stabilize P22 integration (`experiment_type: policy_arena`) in quick/nightly presets.
4. Expand bucket coverage from v1 proxies to richer mechanism-aware slices.

## Next Milestones

| Milestone | Target | Exit Signal |
|---|---|---|
| P40 | Full UI-level parity and long-horizon policy robustness | Policy behavior remains stable under longer horizons and richer UI/action mechanics. |
| P41 | Self-supervised transfer and self-play scaling | Representation + policy lines show reproducible seed-robust gains with gate-backed evidence. |

## Constraints

- Real/oracle validation requires local Balatro + balatrobot runtime availability.
- Aggregate parity interpretation must remain seed/config/version scoped.
- Replay-level exactness can coexist with partially unknown native closed-form weight formulas; reports must disclose this boundary.
