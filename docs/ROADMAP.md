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
| P42 | RL candidate pipeline v1 (env adapter + rollout collector + PPO-lite + closed-loop integration) | active |

## Current Focus: P42 Stabilization

1. Stabilize P42 RL smoke/nightly rows under P22 orchestration.
2. Improve PPO-lite training quality while preserving action-mask and invalid-action safety controls.
3. Expand slice coverage so P41/P42 CI/bootstrap decisions are less frequently sample-limited.
4. Keep recommendation-only promotion discipline while candidate quality improves.

## Next Milestones

| Milestone | Target | Exit Signal |
|---|---|---|
| P43 | RL scaling and richer opponent/self-play schedules | RL candidate line shows repeatable seed-robust gains over heuristic/search baselines. |
| P44 | Self-supervised transfer + long-horizon robustness scaling | Representation + policy lines sustain gains under larger-budget multi-seed gating. |

## Constraints

- Real/oracle validation requires local Balatro + balatrobot runtime availability.
- Aggregate parity interpretation must remain seed/config/version scoped.
- Replay-level exactness can coexist with partially unknown native closed-form weight formulas; reports must disclose this boundary.
