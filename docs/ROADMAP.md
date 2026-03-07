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
| P48 | Adaptive hybrid controller v1 (state-aware routing across policy/search/wm-rerank) | done |
| P49 | GPU mainline + CPU rollout/GPU learner + dashboard + readiness guard | done |
| P50 | Real CUDA bring-up + GPU validation + nightly benchmark profiles | done |
| P51 | Checkpoint registry + resumeable nightly campaigns + promotion queue | done |
| P52 | Learned router / guarded meta-controller + arena ablation + campaign integration | done |
| P54 | Learned router / guarded meta-controller + P22/dashboard/ops-ui first-class integration | done |
| P53 | Background execution + local ops UI + window supervision | done |
| P55 | Config loading hardening + YAML/JSON sidecar sync + full P54 nightly validation | done |

## Current Focus: Post-P55 Quality and Durability

1. Maintain YAML/JSON sidecar parity — run `sync_config_sidecars.ps1` after every YAML edit.
2. Expand P54 nightly budget and multi-seed routing-label calibration.
3. Harden P53 background-mode validation beyond smoke.
4. Tighten promotion semantics with explicit human review on top of the state machine.
5. Keep config provenance, readiness guarding, dashboard generation, and ops-audit logging as default night-ops safety rails.

## Near-Term After P53

- broader imagined-root coverage beyond the current replay families
- tighter P42/P45/P47 coupling through auxiliary losses, rollout-value proxies, and RL-policy routing
- longer real-CUDA benchmark sweeps beyond smoke-sized learners
- stronger checkpoint deduplication / retention policy on top of the new registry
- richer campaign restart policies beyond simple latest-run resume
- learned-router calibration against broader controller telemetry, especially controller-collapse and OOD guard behavior
- richer local ops workflows such as better job history, safer queued actions, and stronger window-state diagnostics
- careful expansion beyond fixed-budget search while preserving uncertainty controls and arena-first evaluation
- richer GPU telemetry and eventual multi-GPU support once single-GPU runtime remains stable

## Constraints

- Real/oracle validation requires local Balatro + balatrobot runtime availability.
- Aggregate parity interpretation must remain seed/config/version scoped.
- Replay-level exactness can coexist with partially unknown native closed-form weight formulas; reports must disclose this boundary.
