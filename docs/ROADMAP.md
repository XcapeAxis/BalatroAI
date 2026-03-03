# BalatroAI Roadmap

## Milestone Tree

| Milestone | Theme | Status |
|---|---|---|
| P0-P2b | Hand-score core + joker smoke parity | done |
| P3-P8 | Joker/consumable/shop/rng coverage expansion | done |
| P9-P11 | Long-episode + probability/economy observed scopes | done |
| P13 | Real-session recording + drift fixtures | done |
| P22-P27 | Experiment orchestration, seed governance, campaign/release ops | done |
| P29-P36 | Data flywheel, action replay, self-supervised tracks | done |
| P37 | Single-action fidelity + mechanics parity audit framework (real↔sim) | in progress (this change set) |

## P37 Delivery Focus

1. Position-sensitive action semantics as first-class actions:
   - `MOVE_HAND_CARD`, `MOVE_JOKER`, `CONSUMABLE_USE`, `SHOP_REROLL/SHOP_BUY/PACK_OPEN`.
2. Real trace capture/conversion that preserves actionable replay details (including inferred move sequences when explicit drag RPC is absent).
3. Dedicated fidelity scope (`p37_action_fidelity_core`) and directed batch gate (`-RunP37`).
4. Probability parity audit path with artifactized json/md outputs under `docs/artifacts/p37/`.

## Next Milestones

| Milestone | Target | Exit Signal |
|---|---|---|
| P38 | Full UI-action parity (drag/drop, multi-select, booster interaction) | Real sessions can replay complex UI flows with zero drift in action-fidelity scopes. |
| P39 | Long-horizon self-play with oracle-supervised correction loops | Multi-round strategy deltas become measurable with stable parity and replay contracts. |
| P40 | Large-scale self-supervised representation + cross-seed transfer | Encoder pretraining materially improves BC/DAgger/RL sample efficiency under fixed gates. |

## Known Constraints

- Real/oracle paths require local Balatro + balatrobot runtime availability.
- Replay parity can be exact while underlying native weight formulas are still partially unknown; this must be disclosed when reporting results.
- All benchmark claims remain seed/config/version scoped and must cite artifacts.

