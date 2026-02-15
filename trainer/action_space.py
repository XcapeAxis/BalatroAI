if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
from itertools import combinations

MAX_HAND = 10
MAX_SELECT = 5
PLAY = "PLAY"
DISCARD = "DISCARD"
ACTION_TYPES = (PLAY, DISCARD)

_ACTION_TABLE: dict[int, list[tuple[str, int]]] = {}
_INDEX_TABLE: dict[int, dict[tuple[str, int], int]] = {}


def _check_hand_size(hand_size: int) -> None:
    if hand_size < 1 or hand_size > MAX_HAND:
        raise ValueError(f"hand_size must be in [1, {MAX_HAND}], got {hand_size}")


def indices_to_mask(indices) -> int:
    mask = 0
    for idx in indices:
        if idx < 0:
            raise ValueError(f"card index must be >= 0, got {idx}")
        mask |= 1 << int(idx)
    return mask


def _build_for_hand_size(hand_size: int) -> tuple[list[tuple[str, int]], dict[tuple[str, int], int]]:
    actions: list[tuple[str, int]] = []

    max_pick = min(MAX_SELECT, hand_size)
    for k in range(1, max_pick + 1):
        for combo in combinations(range(hand_size), k):
            actions.append((PLAY, indices_to_mask(combo)))

    for k in range(0, max_pick + 1):
        for combo in combinations(range(hand_size), k):
            actions.append((DISCARD, indices_to_mask(combo)))

    index_map = {action: idx for idx, action in enumerate(actions)}
    return actions, index_map


for _hs in range(1, MAX_HAND + 1):
    table, table_index = _build_for_hand_size(_hs)
    _ACTION_TABLE[_hs] = table
    _INDEX_TABLE[_hs] = table_index


def mask_to_indices(mask_int: int, hand_size: int) -> list[int]:
    _check_hand_size(hand_size)
    if mask_int < 0:
        raise ValueError("mask_int must be >= 0")
    out = []
    for idx in range(hand_size):
        if mask_int & (1 << idx):
            out.append(idx)
    extra_bits = mask_int >> hand_size
    if extra_bits != 0:
        raise ValueError(f"mask contains bits outside hand_size={hand_size}")
    return out


def encode(hand_size: int, action_type: str, mask_int: int) -> int:
    _check_hand_size(hand_size)
    if action_type not in ACTION_TYPES:
        raise ValueError(f"unknown action_type: {action_type}")
    key = (action_type, mask_int)
    idx = _INDEX_TABLE[hand_size].get(key)
    if idx is None:
        raise ValueError(
            f"illegal action for hand_size={hand_size}: action_type={action_type}, mask={mask_int}"
        )
    return idx


def decode(hand_size: int, action_id: int) -> tuple[str, int]:
    _check_hand_size(hand_size)
    table = _ACTION_TABLE[hand_size]
    if action_id < 0 or action_id >= len(table):
        raise ValueError(f"action_id out of range for hand_size={hand_size}: {action_id}")
    return table[action_id]


def legal_action_ids(hand_size: int) -> list[int]:
    _check_hand_size(hand_size)
    return list(range(len(_ACTION_TABLE[hand_size])))


def legal_action_mask(hand_size: int, max_actions_value: int | None = None) -> list[int]:
    count = action_count(hand_size)
    cap = max_actions_value if max_actions_value is not None else max_actions()
    if cap < count:
        raise ValueError(f"max_actions_value={cap} < action_count={count}")
    return [1] * count + [0] * (cap - count)


def action_count(hand_size: int) -> int:
    _check_hand_size(hand_size)
    return len(_ACTION_TABLE[hand_size])


def max_actions() -> int:
    return action_count(MAX_HAND)


def action_entries(hand_size: int, action_type: str | None = None) -> list[tuple[int, str, int]]:
    _check_hand_size(hand_size)
    table = _ACTION_TABLE[hand_size]
    out: list[tuple[int, str, int]] = []
    for idx, (atype, mask_int) in enumerate(table):
        if action_type is not None and atype != action_type:
            continue
        out.append((idx, atype, mask_int))
    return out


def self_check() -> None:
    for hand_size in range(1, MAX_HAND + 1):
        entries = action_entries(hand_size)
        assert action_count(hand_size) == len(entries)
        for action_id, atype, mask in entries:
            decoded = decode(hand_size, action_id)
            assert decoded == (atype, mask)
            encoded = encode(hand_size, atype, mask)
            assert encoded == action_id
            idxs = mask_to_indices(mask, hand_size)
            assert indices_to_mask(idxs) == mask

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Inspect hand-level action space.")
    parser.add_argument("--hand-size", type=int, default=8)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--dump-first", type=int, default=10, help="How many actions to print.")
    args = parser.parse_args()

    if args.self_check:
        self_check()
        print("action_space self-check passed")

    hs = max(1, min(MAX_HAND, args.hand_size))
    entries = action_entries(hs)
    print(f"hand_size={hs} action_count={len(entries)} max_actions={max_actions()}")
    for action_id, atype, mask in entries[: max(0, args.dump_first)]:
        print(json.dumps({"action_id": action_id, "type": atype, "mask": mask, "indices": mask_to_indices(mask, hs)}))

