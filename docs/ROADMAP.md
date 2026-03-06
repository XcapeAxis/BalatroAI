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
| P45 | World model / latent planning v1 (dataset + dynamics + uncertainty + planning hook) | done |
| P46 | Dyna / imagination loop v1 (short imagined rollouts + replay augmentation + ablation gating) | done |
| P47 | Uncertainty-aware model-based search v1 (candidate rerank + short lookahead + arena ablation) | active |

## Current Focus: P47 Model-Based Search Hardening

1. Improve world-model uncertainty calibration for rerank decisions on larger multi-seed arena budgets.
2. Stress-test horizon and uncertainty-penalty sensitivity before enabling broader default use.
3. Keep P22 quick/nightly summaries aligned with planner traces, arena ablations, and triage outputs.
4. Preserve simulator-first promotion gates while exploring deeper model-based hooks.

## Near-Term After P47

- broader imagined-root coverage beyond the current replay families
- tighter P42/P45 coupling through auxiliary losses and rollout-value proxies
- RL-policy candidate-source support for rerank-assisted inference
- careful expansion beyond horizon-1/2 while preserving uncertainty controls and arena-first evaluation

## Constraints

- Real/oracle validation requires local Balatro + balatrobot runtime availability.
- Aggregate parity interpretation must remain seed/config/version scoped.
- Replay-level exactness can coexist with partially unknown native closed-form weight formulas; reports must disclose this boundary.
