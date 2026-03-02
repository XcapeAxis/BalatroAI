# P32 RealAction Contract Spec

## Design Goals

- One action schema for simulator, real runtime executor translation, and trace replay.
- Deterministic simulator execution for every action payload.
- Extensible fields for position-sensitive operations and targeted consumables.

## Canonical Action Envelope

```json
{
  "schema_version": "action_v1",
  "phase": "SELECTING_HAND|SHOP|...",
  "action_type": "PLAY|DISCARD|BUY|...|REORDER_HAND|REORDER_JOKERS|SWAP_HAND_CARDS|SWAP_JOKERS",
  "params": {}
}
```

`params` remains optional for compatibility; top-level fields are also accepted for fixed-shape actions.

## Position Actions (P32)

### `ReorderHand`

```json
{
  "schema_version": "action_v1",
  "phase": "SELECTING_HAND",
  "action_type": "REORDER_HAND",
  "permutation": [1,0,2,3,4,5,6,7]
}
```

Semantics: `new_hand[k] = old_hand[permutation[k]]`.

Constraints:

- permutation length must equal current hand length.
- permutation must contain each index exactly once.

### `SwapHandCards`

```json
{
  "schema_version": "action_v1",
  "phase": "SELECTING_HAND",
  "action_type": "SWAP_HAND_CARDS",
  "i": 0,
  "j": 4
}
```

Semantics: swap hand positions `i` and `j`.

### `ReorderJokers`

```json
{
  "schema_version": "action_v1",
  "phase": "SELECTING_HAND",
  "action_type": "REORDER_JOKERS",
  "permutation": [2,0,1]
}
```

Semantics: reorder joker slots using the same permutation contract.

### `SwapJokers`

```json
{
  "schema_version": "action_v1",
  "phase": "SELECTING_HAND",
  "action_type": "SWAP_JOKERS",
  "i": 0,
  "j": 2
}
```

Semantics: swap joker slot indices `i` and `j`.

## Position-Targeted Consumables

P32 keeps targeted consumables backward compatible by allowing explicit target fields without forcing one rigid payload shape.

Recommended representation:

```json
{
  "schema_version": "action_v1",
  "phase": "SHOP",
  "action_type": "USE",
  "params": {
    "consumable": 0,
    "targets": [1,3],
    "target_side": "left"
  }
}
```

Notes:

- `targets` is optional and can represent hand/joker indices depending on consumable kind.
- `target_side` (`left|right|none`) captures left/right semantics when present.

## Determinism and Drift Contract

- Simulator must apply position actions deterministically for identical input state + action.
- Replay traces include:
  - `state_hash_p14_real_action_observed_core` (legacy real-action scope)
  - `state_hash_p32_real_action_position_observed_core` (order-sensitive scope)
- P32 replay gates use the P32 scope to detect hand/joker order drift.

## Real Runtime Translation Rules

- `trainer/env_client.RealBackend.step` maps P32 actions to runtime RPC calls when available.
- If runtime reports unknown method, action is marked degraded (`degraded_reason`) and state is polled without crash.
- `trainer/real_trace_to_fixture.py` can infer reorder/swap actions from raw before/after states when explicit action logs are absent.

## Compatibility

- Existing action types remain unchanged.
- P32 fields are additive in `sim/spec/action_v1.json`.
- Trace schema is additive via `state_hash_p32_real_action_position_observed_core`.
