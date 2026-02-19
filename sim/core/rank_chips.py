from __future__ import annotations

from typing import Any

RANK_CHIP_MAP: dict[str, int] = {
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
    "J": 10,
    "Q": 10,
    "K": 10,
    "A": 11,
}


def normalize_rank(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text == "T":
        return "10"
    return text


def rank_chip(value: Any) -> int:
    return int(RANK_CHIP_MAP.get(normalize_rank(value), 0))


def rank_from_card(card: dict[str, Any]) -> str:
    rank = normalize_rank(card.get("rank"))
    if rank:
        return rank
    key = str(card.get("key") or "").strip().upper()
    if "_" in key:
        _, raw_rank = key.split("_", 1)
        return normalize_rank(raw_rank)
    return ""


def sum_rank_chips(cards: list[dict[str, Any]]) -> float:
    return float(sum(rank_chip(rank_from_card(card)) for card in cards))
