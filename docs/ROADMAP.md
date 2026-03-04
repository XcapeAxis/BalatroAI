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
| P42 | RL candidate pipeline v1 (env adapter + rollout collector + PPO-lite + closed-loop integration) | done |
| P43 | Training strategy refocus (mainline selfsup+RL, BC/DAgger demoted to legacy baseline) | active |

## Current Focus: P43 Mainline Consolidation

1. Keep P22 defaults fully aligned with mainline-only execution and explicit legacy opt-in.
2. Increase stability/quality of RL-first candidate training while preserving lineage/triage clarity.
3. Expand self-supervised warm-start integration beyond stub mode in closed-loop candidate training.
4. Maintain lightweight but reliable legacy BC/DAgger smoke for baseline integrity checks.

## Next Milestones

| Milestone | Target | Exit Signal |
|---|---|---|
| P44 | RL scaling and richer opponent/self-play schedules | RL candidate line shows repeatable seed-robust gains over heuristic/search baselines. |
| P45 | Self-supervised transfer + long-horizon robustness scaling | Representation + policy lines sustain gains under larger-budget multi-seed gating. |

## Constraints

- Real/oracle validation requires local Balatro + balatrobot runtime availability.
- Aggregate parity interpretation must remain seed/config/version scoped.
- Replay-level exactness can coexist with partially unknown native closed-form weight formulas; reports must disclose this boundary.
