# BalatroAI Roadmap

## Milestone Map (Condensed)

| Band | Focus | Representative Milestones |
|---|---|---|
| Foundations | Oracle/sim parity and deterministic replay scopes | P0-P8 |
| Episode & Economy | Long horizon episode checks, probability/econ controls | P9-P11 |
| Real Integration | Real session recording, drift checks, DAgger data loop | P13 |
| Experiment Ops | Matrix orchestration, seed governance, campaign ranking | P22-P24 |
| Docs/Ops Hardening | Status publishing, dashboard, release train, UX hardening | P25-P30 |
| Decision & Action Stack | Unified decision stack, real-action contract alignment | P31-P32 |
| Representation Track | Self-supervised plumbing, shared encoder tasks, replay-aligned training entry | P33-P36 |

## Current Priorities

1. Keep parity gates green (P0-P13, P22, P8/P10).
2. Expand multi-seed experiment coverage and telemetry observability.
3. Upgrade self-supervised line from stub to transferable encoder pretrain.
4. Tighten champion/candidate decision confidence with richer reliability checks.

## Near-Term Execution Items

- P35: README/docs consistency and self-supervised orchestrator skeleton.
- P36: unified self-supervised data contract + future/action task heads + P22 matrix integration.
- P37 target: plug encoder into BC/DAgger warm-start path and compare by seed sets.

## Known Constraints

- Real-runtime workflows depend on local Balatro + balatrobot environment.
- Metrics are seed/budget/version dependent and must cite artifacts.
- Several advanced mechanics remain under iterative parity expansion.
