# Canonical Specs

This folder contains versioned schemas used by both the oracle and simulator:

- `state_v1.json`: canonical game state payload.
- `action_v1.json`: canonical action payload.
- `trace_v1.json`: one-line trace record payload.

`state_v1` card canonical format (zone cards):

- `uid`
- `rank`
- `suit`
- `key`
- `modifier` (sorted unique string list)
- `state` (sorted unique string list)

Versioning rules:
- Backward-compatible field additions keep the same major version.
- Breaking changes require a new file name, e.g. `state_v2.json`.
- Hashing always uses canonical JSON serialization from `sim/core/serde.py`.
