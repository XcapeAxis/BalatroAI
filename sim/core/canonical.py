from __future__ import annotations

import re
from typing import Any


HAND_ALIASES = {
    "FOUR_KIND": "FOUR_OF_A_KIND",
    "FOUR_OF_KIND": "FOUR_OF_A_KIND",
    "HIGHCARD": "HIGH_CARD",
    "THREE_KIND": "THREE_OF_A_KIND",
    "THREE_OF_KIND": "THREE_OF_A_KIND",
    "STRAIGHTFLUSH": "STRAIGHT_FLUSH",
    "FULLHOUSE": "FULL_HOUSE",
}


def _normalize_hand_name(value: Any) -> str:
    text = str(value or "").strip().upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return HAND_ALIASES.get(text, text)


def _normalize_rank(rank: Any) -> str:
    text = str(rank or "").strip().upper()
    if text == "T":
        return "10"
    return text


def _normalize_suit(suit: Any) -> str:
    text = str(suit or "").strip().upper()
    if text:
        return text[0]
    return text


def _norm_rank_suit(card: dict[str, Any]) -> tuple[str, str]:
    value = card.get("value") if isinstance(card.get("value"), dict) else {}
    rank = _normalize_rank(card.get("rank") or value.get("rank"))
    suit = _normalize_suit(card.get("suit") or value.get("suit"))

    if (not rank or not suit) and card.get("key"):
        key = str(card.get("key") or "")
        if "_" in key:
            key_suit, key_rank = key.split("_", 1)
            if not suit:
                suit = _normalize_suit(key_suit)
            if not rank:
                rank = _normalize_rank(key_rank)
    return rank, suit


def _norm_uid(card: dict[str, Any], idx: int, zone: str) -> str:
    uid = card.get("id") or card.get("card_id") or card.get("uid") or card.get("key")
    if uid is None or str(uid) == "":
        return f"{zone}-{idx}"
    return str(uid)


def _norm_key(card: dict[str, Any], rank: str, suit: str) -> str:
    key = str(card.get("key") or "").strip()
    if key:
        return key
    if rank and suit:
        key_rank = "T" if rank == "10" else rank
        return f"{suit}_{key_rank}"
    return ""


def _norm_tags(raw_tags: Any) -> list[str]:
    if isinstance(raw_tags, list):
        tags: list[str] = []
        for item in raw_tags:
            if isinstance(item, (str, int, float, bool)):
                text = str(item).strip()
                if text:
                    tags.append(text)
    elif raw_tags is None:
        tags = []
    elif isinstance(raw_tags, (str, int, float, bool)):
        text = str(raw_tags).strip()
        tags = [text] if text else []
    else:
        tags = []
    return sorted(set(tags))


def _canonicalize_card(raw_card: Any, idx: int, zone_name: str) -> dict[str, Any] | None:
    if not isinstance(raw_card, dict):
        return None
    rank, suit = _norm_rank_suit(raw_card)
    modifier = _norm_tags(raw_card.get("modifier"))
    if not modifier:
        modifier = _norm_tags(raw_card.get("modifier_tags"))
    state = _norm_tags(raw_card.get("state"))
    if not state:
        state = _norm_tags(raw_card.get("state_tags"))

    return {
        "uid": _norm_uid(raw_card, idx, zone_name),
        "rank": rank,
        "suit": suit,
        "key": _norm_key(raw_card, rank, suit),
        "modifier": modifier,
        "state": state,
    }


def _canonicalize_zone_cards(raw_zone: Any, zone_name: str) -> list[dict[str, Any]]:
    if isinstance(raw_zone, dict) and isinstance(raw_zone.get("cards"), list):
        cards = raw_zone.get("cards")
    elif isinstance(raw_zone, list):
        cards = raw_zone
    else:
        cards = []

    out: list[dict[str, Any]] = []
    for idx, raw_card in enumerate(cards):
        canon = _canonicalize_card(raw_card, idx, zone_name)
        if canon is not None:
            out.append(canon)
    return out


def _canonicalize_hands(raw_hands: Any) -> dict[str, Any]:
    levels: dict[str, dict[str, float]] = {}
    if not isinstance(raw_hands, dict):
        return {"levels": levels}

    source = raw_hands.get("levels") if isinstance(raw_hands.get("levels"), dict) else raw_hands

    for raw_name, raw_info in source.items():
        hand_name = _normalize_hand_name(raw_name)
        if not hand_name:
            continue
        if isinstance(raw_info, dict):
            level = int(raw_info.get("level") or 1)
            chips = float(raw_info.get("chips") or 0.0)
            mult = float(raw_info.get("mult") or 1.0)
        else:
            try:
                level = int(raw_info)
            except Exception:
                level = 1
            chips = 0.0
            mult = 1.0
        levels[hand_name] = {
            "level": level,
            "chips": chips,
            "mult": mult,
        }
    return {"levels": levels}


def _canonicalize_consumables(raw_consumables: Any) -> dict[str, Any]:
    if not isinstance(raw_consumables, dict):
        return {"count": 0, "limit": 0, "highlighted_limit": 0, "cards": []}

    cards_raw = raw_consumables.get("cards")
    cards = cards_raw if isinstance(cards_raw, list) else []
    out_cards: list[dict[str, Any]] = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        out_cards.append(
            {
                "key": str(card.get("key") or "").strip().lower(),
                "label": str(card.get("label") or "").strip(),
                "set": str(card.get("set") or "").strip().upper(),
            }
        )

    return {
        "count": int(raw_consumables.get("count") or len(out_cards)),
        "limit": int(raw_consumables.get("limit") or 0),
        "highlighted_limit": int(raw_consumables.get("highlighted_limit") or 0),
        "cards": out_cards,
    }


def to_canonical_state(
    raw_state: dict[str, Any],
    *,
    rng_mode: str = "native",
    seed: str | None = None,
    rng_cursor: int = 0,
    rng_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    zones = {
        "deck": _canonicalize_zone_cards(raw_state.get("deck"), "deck"),
        "discard": _canonicalize_zone_cards(raw_state.get("discard"), "discard"),
        "hand": _canonicalize_zone_cards(raw_state.get("hand"), "hand"),
        "played": _canonicalize_zone_cards(raw_state.get("played"), "played"),
    }

    round_info = raw_state.get("round") or {}
    score_info = raw_state.get("score") or {}

    canonical = {
        "schema_version": "state_v1",
        "phase": str(raw_state.get("state") or "UNKNOWN"),
        "zones": zones,
        "hands": _canonicalize_hands(raw_state.get("hands")),
        "consumables": _canonicalize_consumables(raw_state.get("consumables")),
        "round": {
            "hands_left": int(round_info.get("hands_left") or 0),
            "discards_left": int(round_info.get("discards_left") or 0),
            "ante": int(raw_state.get("ante_num") or 0),
            "round_num": int(raw_state.get("round_num") or 0),
            "blind": str(round_info.get("blind") or raw_state.get("blind") or "small"),
        },
        "score": {
            "chips": float(round_info.get("chips") or score_info.get("chips") or 0),
            "mult": float(score_info.get("mult") or 1),
            "target_chips": float(score_info.get("target_chips") or raw_state.get("target_chips") or 0),
            "last_hand_type": str(score_info.get("last_hand_type") or ""),
            "last_base_chips": float(score_info.get("last_base_chips") or 0),
            "last_base_mult": float(score_info.get("last_base_mult") or 1),
        },
        "economy": {
            "money": float(raw_state.get("money") or 0),
        },
        "jokers": list(raw_state.get("jokers") or []),
        "rng": {
            "mode": rng_mode,
            "seed": seed,
            "cursor": int(rng_cursor),
            "events": list(rng_events or []),
        },
        "flags": {
            "done": bool(raw_state.get("done") or raw_state.get("state") == "GAME_OVER"),
            "won": bool(raw_state.get("won") or False),
        },
    }
    return canonical
