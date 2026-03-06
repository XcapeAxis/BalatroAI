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
| P47 | Uncertainty-aware model-based search v1 (candidate rerank + short lookahead + arena ablation) | done |
| P48 | Adaptive hybrid controller v1 (state-aware routing across policy/search/wm-rerank) | active |

## Current Focus: P48 Adaptive Hybrid Routing

1. Stress-test router thresholds and controller-selection heuristics on larger multi-seed arena budgets.
2. Improve the balance between search cost, policy confidence, and wm uncertainty gating.
3. Keep P22 quick/nightly summaries aligned with routing traces, arena ablations, and triage outputs.
4. Preserve simulator-first promotion gates while exploring future learned-router extensions.

## Near-Term After P48

- broader imagined-root coverage beyond the current replay families
- tighter P42/P45/P47 coupling through auxiliary losses, rollout-value proxies, and RL-policy routing
- learned router experiments once controller telemetry is richer and more stable
- careful expansion beyond fixed-budget search while preserving uncertainty controls and arena-first evaluation

## Constraints

- Real/oracle validation requires local Balatro + balatrobot runtime availability.
- Aggregate parity interpretation must remain seed/config/version scoped.
- Replay-level exactness can coexist with partially unknown native closed-form weight formulas; reports must disclose this boundary.
