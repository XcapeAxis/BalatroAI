from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Any

from trainer import action_space


def _card_rank_value(card: dict[str, Any]) -> float:
    v = card.get("value") if isinstance(card.get("value"), dict) else {}
    rank = str(v.get("rank") or card.get("rank") or "")
    rank_map = {
        "A": 14,
        "K": 13,
        "Q": 12,
        "J": 11,
        "T": 10,
        "10": 10,
        "9": 9,
        "8": 8,
        "7": 7,
        "6": 6,
        "5": 5,
        "4": 4,
        "3": 3,
        "2": 2,
    }
    return float(rank_map.get(rank.upper(), 1))


def _score_hand_action(action_type: str, indices: list[int], cards: list[dict[str, Any]]) -> float:
    if not indices:
        return -1e9
    selected = [cards[i] for i in indices if 0 <= i < len(cards)]
    rank_sum = sum(_card_rank_value(c) for c in selected)
    if action_type == "PLAY":
        # Prefer larger, stronger plays first.
        return 100.0 + 8.0 * len(selected) + rank_sum
    # DISCARD: prefer removing weaker cards with smaller rank sum.
    return 20.0 + 2.0 * len(selected) - 0.8 * rank_sum


def generate_hand_candidates(
    state: dict[str, Any],
    *,
    max_candidates: int = 40,
    max_discard_groups: int = 10,
) -> list[dict[str, Any]]:
    hand_cards = (state.get("hand") or {}).get("cards") or []
    hand_size = min(len(hand_cards), action_space.MAX_HAND)
    if hand_size <= 0:
        return []

    legal_ids = action_space.legal_action_ids(hand_size)
    scored: list[tuple[float, dict[str, Any]]] = []
    round_info = state.get("round") if isinstance(state.get("round"), dict) else {}
    hands_left = int(round_info.get("hands_left") or 0)
    discards_left = int(round_info.get("discards_left") or 0)

    discard_kept = 0
    for aid in legal_ids:
        action_type, mask_int = action_space.decode(hand_size, int(aid))
        if action_type == "PLAY" and hands_left <= 0:
            continue
        if action_type == "DISCARD" and discards_left <= 0:
            continue
        indices = action_space.mask_to_indices(mask_int, hand_size)
        if action_type == "DISCARD":
            if discard_kept >= max_discard_groups:
                continue
            discard_kept += 1
        score = _score_hand_action(action_type, indices, hand_cards)
        scored.append((score, {"action_type": action_type, "indices": indices}))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for _, action in scored:
        key = (str(action["action_type"]), tuple(int(i) for i in action.get("indices") or []))
        if key in seen:
            continue
        seen.add(key)
        out.append(action)
        if len(out) >= max(1, int(max_candidates)):
            break

    # Always keep at least one wait fallback candidate.
    if not out:
        out.append({"action_type": "WAIT", "sleep": 0.01})
    return out
