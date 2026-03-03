# BalatroAI Roadmap

## Milestone Tree

| Milestone | Theme | Status |
|---|---|---|
| P0-P13 | Oracle/sim baseline alignment and real-session drift loop | done |
| P22-P27 | Experiment orchestration, seed governance, campaign/release ops | done |
| P29-P37 | Data flywheel, action replay unification, self-supervised and single-action fidelity | done |
| P38 | Long-horizon statistical consistency framework (multi-seed stress + aggregate parity) | in progress |

## Current Focus: P38

1. Long episode stress runner with multi-seed support and per-episode artifacts.
2. Aggregate stats comparator (`score`, `rounds`, `economy`, `shop/pack/joker distributions`).
3. Gate split:
   - hard fail for replay drift mismatches
   - soft warning for aggregate statistical drift
4. P22 orchestrator integration (`experiment_type: long_consistency`) and run-level summary output.

## Next Milestones

| Milestone | Target | Exit Signal |
|---|---|---|
| P39 | Full UI-action parity coverage (drag/drop, multi-select, booster interactions) | Complex real sessions replay in sim with stable zero-drift under fidelity scopes. |
| P40 | Long-horizon self-play and self-supervised transfer scale-up | Representation + policy lines show reproducible seed-robust gains with gate-backed evidence. |

## Constraints

- Real/oracle validation requires local Balatro + balatrobot runtime availability.
- Aggregate parity interpretation must remain seed/config/version scoped.
- Replay-level exactness can coexist with partially unknown native closed-form weight formulas; reports must disclose this boundary.
