from __future__ import annotations

from typing import Any

from trainer import action_space, action_space_shop
from trainer.policy_arena.policy_adapter import phase_from_obs


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def hand_size_from_state(state: dict[str, Any]) -> int:
    hand_cards = (state.get("hand") or {}).get("cards") if isinstance(state.get("hand"), dict) else []
    return min(len(hand_cards or []), action_space.MAX_HAND)


def legal_hand_action_ids_for_state(state: dict[str, Any]) -> list[int]:
    hand_size = hand_size_from_state(state)
    if hand_size <= 0:
        return []
    round_info = state.get("round") if isinstance(state.get("round"), dict) else {}
    hands_left = _safe_int(round_info.get("hands_left"), 0)
    discards_left = _safe_int(round_info.get("discards_left"), 0)
    legal: list[int] = []
    for aid in action_space.legal_action_ids(hand_size):
        action_type, mask_int = action_space.decode(hand_size, aid)
        if action_type == action_space.PLAY and hands_left <= 0:
            continue
        if action_type == action_space.DISCARD and discards_left <= 0:
            continue
        if action_type == action_space.DISCARD and mask_int == 0 and hands_left > 0:
            continue
        legal.append(int(aid))
    return legal


def legal_hand_action_rows_for_state(
    state: dict[str, Any],
    *,
    max_entries: int = 64,
) -> list[dict[str, Any]]:
    hand_size = hand_size_from_state(state)
    if hand_size <= 0:
        return []
    out: list[dict[str, Any]] = []
    for aid in legal_hand_action_ids_for_state(state)[: max(0, int(max_entries))]:
        action_type, mask = action_space.decode(hand_size, int(aid))
        out.append(
            {
                "action_type": action_type,
                "indices": action_space.mask_to_indices(mask, hand_size),
                "id": int(aid),
            }
        )
    return out


def legal_shop_action_rows_for_state(
    state: dict[str, Any],
    *,
    max_entries: int = 32,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for aid in action_space_shop.legal_action_ids(state)[: max(0, int(max_entries))]:
        out.append({"id": int(aid), "action": action_space_shop.action_from_id(state, int(aid))})
    return out


def legal_action_rows_for_state(state: dict[str, Any]) -> list[dict[str, Any]] | None:
    phase = phase_from_obs(state)
    if phase == "SELECTING_HAND":
        return legal_hand_action_rows_for_state(state)
    if phase in action_space_shop.SHOP_PHASES:
        return legal_shop_action_rows_for_state(state)
    return None
