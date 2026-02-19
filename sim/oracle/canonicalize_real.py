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


def _rank_suit_from_card(card: dict[str, Any]) -> tuple[str, str]:
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


def _extract_zone_cards(raw: Any, zone_name: str) -> list[dict[str, Any]]:
    if isinstance(raw, dict) and isinstance(raw.get("cards"), list):
        cards = raw.get("cards")
    elif isinstance(raw, list):
        cards = raw
    else:
        cards = []

    out: list[dict[str, Any]] = []
    for idx, card in enumerate(cards):
        if not isinstance(card, dict):
            continue
        rank, suit = _rank_suit_from_card(card)
        key = str(card.get("key") or "")
        if not key and rank and suit:
            key_rank = "T" if rank == "10" else rank
            key = f"{suit}_{key_rank}"

        uid = str(card.get("id") or card.get("card_id") or card.get("uid") or key or f"{zone_name}-{idx}")

        modifier = _norm_tags(card.get("modifier"))
        if not modifier:
            modifier = _norm_tags(card.get("modifier_tags"))
        state = _norm_tags(card.get("state"))
        if not state:
            state = _norm_tags(card.get("state_tags"))

        out.append(
            {
                "uid": uid,
                "rank": rank,
                "suit": suit,
                "key": key,
                "modifier": modifier,
                "state": state,
            }
        )
    return out


def _infer_blind(raw_state: dict[str, Any]) -> tuple[str, float]:
    blind = str((raw_state.get("round") or {}).get("blind") or raw_state.get("blind") or "small")
    target = 0.0
    blinds = raw_state.get("blinds") or {}
    if isinstance(blinds, dict) and blinds:
        selected = None
        for key in ("small", "big", "boss"):
            info = blinds.get(key) or {}
            status = str(info.get("status") or "").upper()
            if status in {"CURRENT", "SELECT", "SELECTED"}:
                selected = key
                break
        if selected is None and blind in blinds:
            selected = blind
        if selected is None:
            selected = "small"
        blind = selected
        target = float((blinds.get(selected) or {}).get("score") or 0.0)
    return blind, target


def _extract_jokers(raw_state: dict[str, Any]) -> list[dict[str, Any]]:
    jokers = raw_state.get("jokers") or []
    if isinstance(jokers, dict):
        jokers = jokers.get("cards") or []

    out: list[dict[str, Any]] = []
    for idx, joker in enumerate(jokers):
        if isinstance(joker, str):
            out.append({"joker_id": joker, "counters": {}, "disabled": False})
            continue
        if not isinstance(joker, dict):
            out.append({"joker_id": f"joker-{idx}", "counters": {}, "disabled": False})
            continue

        counters = {}
        for k, v in joker.items():
            if isinstance(v, (int, float)) and any(token in str(k).lower() for token in ["count", "counter", "uses", "charges"]):
                counters[str(k)] = v

        out.append(
            {
                "joker_id": str(joker.get("key") or joker.get("id") or joker.get("name") or f"joker-{idx}"),
                "counters": counters,
                "disabled": bool(joker.get("disabled") or False),
            }
        )
    return out


def _extract_consumables(raw_state: dict[str, Any]) -> dict[str, Any]:
    cons = raw_state.get("consumables")
    if not isinstance(cons, dict):
        return {"count": 0, "limit": 0, "highlighted_limit": 0, "cards": []}

    cards = cons.get("cards") if isinstance(cons.get("cards"), list) else []
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
        "count": int(cons.get("count") or len(out_cards)),
        "limit": int(cons.get("limit") or 0),
        "highlighted_limit": int(cons.get("highlighted_limit") or 0),
        "cards": out_cards,
    }


def _extract_hands(raw_state: dict[str, Any]) -> dict[str, Any]:
    hands = raw_state.get("hands")
    levels: dict[str, dict[str, float]] = {}
    if isinstance(hands, dict):
        for raw_name, raw_info in hands.items():
            hand_name = _normalize_hand_name(raw_name)
            if not hand_name:
                continue
            if isinstance(raw_info, dict):
                level = int(raw_info.get("level") or 1)
                chips = float(raw_info.get("chips") or 0.0)
                mult = float(raw_info.get("mult") or 1.0)
            else:
                level = 1
                chips = 0.0
                mult = 1.0
            levels[hand_name] = {
                "level": level,
                "chips": chips,
                "mult": mult,
            }
    return {"levels": levels}


def canonicalize_real_state(
    raw_state: dict[str, Any],
    *,
    seed: str | None,
    rng_events: list[dict[str, Any]] | None = None,
    rng_cursor: int = 0,
) -> dict[str, Any]:
    blind, target = _infer_blind(raw_state)

    zones = {
        "deck": _extract_zone_cards(raw_state.get("deck"), "deck"),
        "discard": _extract_zone_cards(raw_state.get("discard"), "discard"),
        "hand": _extract_zone_cards((raw_state.get("hand") or {}).get("cards") or raw_state.get("hand"), "hand"),
        "played": _extract_zone_cards(raw_state.get("played") or raw_state.get("played_cards"), "played"),
    }

    round_info = raw_state.get("round") or {}
    score_chips = float(round_info.get("chips") or 0.0)
    score_mult = float(round_info.get("mult") or 1.0)

    canonical = {
        "schema_version": "state_v1",
        "phase": str(raw_state.get("state") or "UNKNOWN"),
        "zones": zones,
        "hands": _extract_hands(raw_state),
        "consumables": _extract_consumables(raw_state),
        "round": {
            "hands_left": int(round_info.get("hands_left") or 0),
            "discards_left": int(round_info.get("discards_left") or 0),
            "ante": int(raw_state.get("ante_num") or 0),
            "round_num": int(raw_state.get("round_num") or 0),
            "blind": blind,
        },
        "score": {
            "chips": score_chips,
            "mult": score_mult,
            "target_chips": float(target),
            "last_hand_type": str(round_info.get("last_hand_type") or round_info.get("hand_type") or ""),
            "last_base_chips": float(round_info.get("last_base_chips") or 0.0),
            "last_base_mult": float(round_info.get("last_base_mult") or 1.0),
        },
        "economy": {
            "money": float(raw_state.get("money") or 0.0),
        },
        "jokers": _extract_jokers(raw_state),
        "rng": {
            "mode": "oracle_stream",
            "seed": seed,
            "cursor": int(rng_cursor),
            "events": list(rng_events or []),
        },
        "flags": {
            "done": bool(str(raw_state.get("state") or "").upper() == "GAME_OVER"),
            "won": bool(raw_state.get("won") or False),
        },
    }
    return canonical
