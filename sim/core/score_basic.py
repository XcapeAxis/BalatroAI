from __future__ import annotations

from collections import Counter
from typing import Any, TypedDict

from sim.core.rank_chips import normalize_rank, rank_from_card, sum_rank_chips

RANK_NUM = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}

HAND_BASE: dict[str, tuple[float, float]] = {
    "HIGH_CARD": (5.0, 1.0),
    "PAIR": (10.0, 2.0),
    "TWO_PAIR": (20.0, 2.0),
    "THREE_OF_A_KIND": (30.0, 3.0),
    "STRAIGHT": (30.0, 4.0),
    "FLUSH": (35.0, 4.0),
    "FULL_HOUSE": (40.0, 4.0),
    "FOUR_OF_A_KIND": (60.0, 7.0),
    "STRAIGHT_FLUSH": (100.0, 8.0),
}


class ScoreBreakdown(TypedDict):
    hand_type: str
    base_chips: float
    base_mult: float
    scoring_cards: list[dict[str, Any]]
    scoring_rank_chips: float
    total_delta: float


def _card_rank(card: dict[str, Any]) -> str:
    rank = normalize_rank(card.get("rank"))
    if rank:
        return rank
    return rank_from_card(card)


def _card_suit(card: dict[str, Any]) -> str:
    suit = str(card.get("suit") or "").strip().upper()
    if suit:
        return suit[:1]
    key = str(card.get("key") or "").strip().upper()
    if "_" in key:
        s, _ = key.split("_", 1)
        return s[:1]
    return ""


def _is_straight(rank_values: list[int]) -> bool:
    if len(rank_values) != 5:
        return False
    vals = sorted(set(rank_values))
    if len(vals) != 5:
        return False
    if vals[-1] - vals[0] == 4:
        return True
    # A-2-3-4-5 wheel
    if vals == [2, 3, 4, 5, 14]:
        return True
    return False


def _rank_counts(cards: list[dict[str, Any]]) -> Counter[str]:
    return Counter(_card_rank(c) for c in cards)


def _select_high_card(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not cards:
        return []

    def _rank_value(card: dict[str, Any]) -> int:
        return int(RANK_NUM.get(_card_rank(card), 0))

    best = max(cards, key=_rank_value)
    return [best]


def evaluate_selected_breakdown(cards: list[dict[str, Any]]) -> ScoreBreakdown:
    if not cards:
        return {
            "hand_type": "INVALID",
            "base_chips": 0.0,
            "base_mult": 1.0,
            "scoring_cards": [],
            "scoring_rank_chips": 0.0,
            "total_delta": 0.0,
        }

    ranks = [_card_rank(c) for c in cards]
    suits = [_card_suit(c) for c in cards]
    rank_values = [int(RANK_NUM.get(r, 0)) for r in ranks]
    counts = _rank_counts(cards)
    count_values = sorted(counts.values(), reverse=True)

    is_flush = len(cards) == 5 and len(set(suits)) == 1
    is_straight = _is_straight(rank_values)

    hand_type = "HIGH_CARD"
    scoring_cards: list[dict[str, Any]] = _select_high_card(cards)

    if len(cards) == 5 and is_flush and is_straight:
        hand_type = "STRAIGHT_FLUSH"
        scoring_cards = list(cards)
    elif 4 in count_values and len(cards) >= 4:
        hand_type = "FOUR_OF_A_KIND"
        four_rank = next((rank for rank, cnt in counts.items() if cnt == 4), "")
        scoring_cards = [c for c in cards if _card_rank(c) == four_rank]
    elif len(cards) == 5 and count_values == [3, 2]:
        hand_type = "FULL_HOUSE"
        scoring_cards = list(cards)
    elif len(cards) == 5 and is_flush:
        hand_type = "FLUSH"
        scoring_cards = list(cards)
    elif len(cards) == 5 and is_straight:
        hand_type = "STRAIGHT"
        scoring_cards = list(cards)
    elif 3 in count_values and len(cards) >= 3:
        hand_type = "THREE_OF_A_KIND"
        triple_rank = next((rank for rank, cnt in counts.items() if cnt == 3), "")
        scoring_cards = [c for c in cards if _card_rank(c) == triple_rank]
    elif count_values.count(2) >= 2 and len(cards) >= 4:
        hand_type = "TWO_PAIR"
        pair_ranks = {rank for rank, cnt in counts.items() if cnt == 2}
        scoring_cards = [c for c in cards if _card_rank(c) in pair_ranks]
    elif 2 in count_values and len(cards) >= 2:
        hand_type = "PAIR"
        pair_rank = next((rank for rank, cnt in counts.items() if cnt == 2), "")
        scoring_cards = [c for c in cards if _card_rank(c) == pair_rank]

    base_chips, base_mult = HAND_BASE.get(hand_type, HAND_BASE["HIGH_CARD"])
    scoring_rank_chips = sum_rank_chips(scoring_cards)
    total_delta = (float(base_chips) + float(scoring_rank_chips)) * float(base_mult)

    return {
        "hand_type": hand_type,
        "base_chips": float(base_chips),
        "base_mult": float(base_mult),
        "scoring_cards": scoring_cards,
        "scoring_rank_chips": float(scoring_rank_chips),
        "total_delta": float(total_delta),
    }


def evaluate_selected(cards: list[dict[str, Any]]) -> tuple[str, float, float]:
    info = evaluate_selected_breakdown(cards)
    return str(info["hand_type"]), float(info["base_chips"]), float(info["base_mult"])
