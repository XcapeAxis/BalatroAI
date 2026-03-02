# P32 RealAction Contract Status

## Scope

P32 defines a unified single-step action contract shared by simulator execution, real-runtime executor translation, and real-session trace conversion/replay.

## Action Coverage Matrix

| Action Type | Example | Sim Engine | Real Executor / Env Client | P13/P14/P32 Fixture Round-trip | Notes |
|---|---|---|---|---|---|
| `PLAY_HAND` | `PLAY(indices=[0,1,2])` | Y | Y | Y | Existing core action path. |
| `DISCARD` | `DISCARD(indices=[3])` | Y | Y | Y | Existing core action path. |
| `SELECT_BLIND` | `SELECT(index=0)` | Y | Y | Y | Existing pre-round action path. |
| `SHOP_BUY` | `BUY(params={card:0})` | Y | Y | Y | Existing shop path; depends on runtime availability. |
| `SHOP_REROLL` | `REROLL` | Y | Y | Y | Existing shop path. |
| `SHOP_PACK` | `PACK(params={card:0})` | Y | Y | Y | Existing booster open path. |
| `SHOP_SELL` | `SELL(params={joker:0})` | Y | Y | Y | Existing shop path. |
| `USE_CONSUMABLE` | `USE(params={...})` | Y | Y | Y | Existing path; target granularity depends on runtime fields. |
| `ReorderHand` | `REORDER_HAND(permutation=[...])` | Y | Y* | Y | Added in P32. `Y*` means env-client supports it, runtime RPC may degrade if method unsupported. |
| `SwapHandCards` | `SWAP_HAND_CARDS(i,j)` | Y | Y* | Y | Added in P32; same degradation rule as above. |
| `ReorderJokers` | `REORDER_JOKERS(permutation=[...])` | Y | Y* | Y | Added in P32; same degradation rule as above. |
| `SwapJokers` | `SWAP_JOKERS(i,j)` | Y | Y* | Y | Added in P32; same degradation rule as above. |
| `EPISODE_CONTROL` | `START/MENU/NEXT_ROUND/CASH_OUT` | Y | Y | Y | Existing control actions. |

## PositionSensitiveMechanics

Source: `balatro_mechanics/derived/position_sensitive_effects.json`

- scanned rows: `237`
- hand-order-sensitive effects: `4`
- joker-slot-sensitive effects: `5`
- position-targeted effects: `13`

Examples (hand order / left-right sensitive):

- `Brainstorm`: copies leftmost Joker ability.
- `Hanging Chad`: retriggers first played scoring card.
- `Photograph`: first played face card multiplier.
- `Death (XIII)`: left card becomes right card (explicit rearrange semantics).

Examples (Joker slot sensitive):

- `Blueprint`: copies Joker to the right.
- `Ceremonial Dagger`: destroys Joker to the right.
- `Joker Stencil` / `Antimatter`: explicit Joker-slot semantics.

Examples (position-targeted consumables):

- Spectral cards with "selected card" targeting (`Aura`, `Cryptid`, `Deja Vu`, etc.).
- Tarot cards with selected-card transforms (`Justice`, `The Devil`, `The Lovers`, etc.).

## Runtime/Toolchain Notes

- `trainer/env_client.py` now accepts reorder/swap action types and attempts RPC translation.
- If runtime RPC lacks reorder methods, env client records `degraded_reason` instead of crashing.
- `trainer/real_trace_to_fixture.py` now infers reorder/swap from raw before/after states when no explicit action was logged.
- P32 adds `state_hash_p32_real_action_position_observed_core` to detect order-sensitive drift.

## Explicit Gaps / Degraded Paths

- Some balatrobot runtimes currently do not expose reorder RPC methods; these paths are marked degraded at runtime.
- High-speed manual drag operations can still be missed when before/after snapshots are too sparse.
- Complex dual-change steps (simultaneous hand + joker reorder in one frame) are conservatively left uninterpreted by inference.

## Next Steps

1. Add native reorder RPC support in runtime service for non-degraded real execution.
2. Extend trace inference with richer gesture reconstruction for multi-object drag chains.
3. Add dedicated real-session fixtures with guaranteed position operations once runtime support is confirmed.
