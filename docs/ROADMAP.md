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
| P43 | Training strategy refocus (mainline selfsup+RL, BC/DAgger demoted to legacy baseline) | done |
| P44 | Distributed RL training (rollout workers + curriculum + multi-seed gating) | done |
| P45 | World model / latent planning v1 (dataset + dynamics + uncertainty + planning hook) | active |

## Current Focus: P45 World Model Hardening

1. Improve uncertainty calibration quality on larger multi-seed P45 runs.
2. Extend wm-assisted compare beyond heuristic baseline into search and RL candidate evaluation.
3. Keep P22 quick/nightly summaries aligned with world-model metrics and auxiliary-asset lineage.
4. Explore future model-based hooks without weakening simulator-first promotion gates.

## Near-Term After P45

- uncertainty-aware imagined rollouts beyond one-step heuristic
- tighter P42/P45 coupling through auxiliary losses and rollout-value proxies
- higher-budget nightly comparisons for wm-assisted arena variants

## Constraints

- Real/oracle validation requires local Balatro + balatrobot runtime availability.
- Aggregate parity interpretation must remain seed/config/version scoped.
- Replay-level exactness can coexist with partially unknown native closed-form weight formulas; reports must disclose this boundary.
