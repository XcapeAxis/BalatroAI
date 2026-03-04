# BalatroAI Roadmap

## Milestone Tree

| Milestone | Theme | Status |
|---|---|---|
| P0-P13 | Oracle/sim baseline alignment and real-session drift loop | done |
| P22-P27 | Experiment orchestration, seed governance, campaign/release ops | done |
| P29-P37 | Data flywheel, action replay unification, self-supervised and single-action fidelity | done |
| P38 | Long-horizon statistical consistency framework (multi-seed stress + aggregate parity) | done |
| P39 | Policy Arena v1 (multi-policy adapters, bucketed evaluation, champion rule input) | done |
| P40 | Closed-loop improvement v1 (replay mix + failure mining + arena-gated promotion recommendation) | done |
| P41 | Closed-loop improvement v2 (lineage + curriculum + slice-aware gating + regression triage) | done |

## Current Focus: P42 Preparation

1. Keep P41 quick/nightly gates stable and artifact-complete under P22 orchestration.
2. Raise slice sample coverage so CI/bootstrap outputs are decisive more often.
3. Improve candidate policy quality while preserving conservative promotion controls.
4. Prepare longer-horizon robustness checks and UI-level parity work for P42.

## Next Milestones

| Milestone | Target | Exit Signal |
|---|---|---|
| P42 | Full UI-level parity and long-horizon policy robustness | Policy behavior remains stable under longer horizons and richer UI/action mechanics. |
| P43 | Self-supervised transfer and self-play scaling | Representation + policy lines show reproducible seed-robust gains with gate-backed evidence. |

## Constraints

- Real/oracle validation requires local Balatro + balatrobot runtime availability.
- Aggregate parity interpretation must remain seed/config/version scoped.
- Replay-level exactness can coexist with partially unknown native closed-form weight formulas; reports must disclose this boundary.
