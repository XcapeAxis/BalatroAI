if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import re
from typing import Any

from trainer.action_space import MAX_HAND

RANK_ORDER = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "10": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}

RANK_CHIP_FALLBACK = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "10": 10,
    "J": 10,
    "Q": 10,
    "K": 10,
    "A": 11,
}

SUIT_ORDER = {
    "C": 1,
    "D": 2,
    "H": 3,
    "S": 4,
}

_INT_PATTERN = re.compile(r"[-+]?\d+")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _parse_rank_suit(card: dict) -> tuple[str | None, str | None]:
    value = card.get("value") or {}
    rank = value.get("rank")
    suit = value.get("suit")

    if rank and suit:
        return str(rank), str(suit)

    key = str(card.get("key") or "")
    if "_" in key:
        suit_part, rank_part = key.split("_", 1)
        if not suit:
            suit = suit_part
        if not rank:
            rank = rank_part

    if rank is not None:
        rank = str(rank)
    if suit is not None:
        suit = str(suit)
    return rank, suit


def _parse_effect_to_chip_hint(effect: str | None, rank: str | None) -> int:
    if effect:
        text = str(effect)
        m = _INT_PATTERN.search(text)
        if m:
            try:
                return int(m.group(0))
            except Exception:
                pass
    rank_key = (rank or "").upper()
    return int(RANK_CHIP_FALLBACK.get(rank_key, 0))


def _modifier_flags(card: dict) -> tuple[int, int, int]:
    mods = card.get("modifier") or []
    state = card.get("state") or []
    text = " ".join(str(x).lower() for x in [*mods, *state])
    has_enhancement = 1 if any(k in text for k in ["enhance", "foil", "holo", "poly", "stone", "gold", "steel"]) else 0
    has_edition = 1 if any(k in text for k in ["edition", "foil", "holo", "polychrome"]) else 0
    has_seal = 1 if "seal" in text else 0
    return has_enhancement, has_edition, has_seal


def _blind_score(blinds: dict, key: str) -> int:
    info = (blinds or {}).get(key) or {}
    return _safe_int(info.get("score"), 0)


def extract_features(state: dict) -> dict:
    phase = state.get("state") or "UNKNOWN"
    hand_cards = (state.get("hand") or {}).get("cards") or []
    hand_size = min(len(hand_cards), MAX_HAND)

    rank_ids = [0] * MAX_HAND
    suit_ids = [0] * MAX_HAND
    chip_hint = [0] * MAX_HAND
    enh_flags = [0] * MAX_HAND
    edt_flags = [0] * MAX_HAND
    seal_flags = [0] * MAX_HAND
    pad_mask = [0] * MAX_HAND

    for i, card in enumerate(hand_cards[:MAX_HAND]):
        rank, suit = _parse_rank_suit(card)
        rank_ids[i] = RANK_ORDER.get((rank or "").upper(), 0)
        suit_ids[i] = SUIT_ORDER.get((suit or "").upper(), 0)

        effect = (card.get("value") or {}).get("effect")
        chip_hint[i] = _parse_effect_to_chip_hint(effect, rank)

        enh, edt, seal = _modifier_flags(card)
        enh_flags[i] = enh
        edt_flags[i] = edt
        seal_flags[i] = seal
        pad_mask[i] = 1

    round_info = state.get("round") or {}
    blinds = state.get("blinds") or {}

    context_values = {
        "hands_left": _safe_int(round_info.get("hands_left"), 0),
        "discards_left": _safe_int(round_info.get("discards_left"), 0),
        "round_chips": _safe_int(round_info.get("chips"), 0),
        "money": _safe_int(state.get("money"), 0),
        "reroll_cost": _safe_int(round_info.get("reroll_cost"), 0),
        "small_blind_score": _blind_score(blinds, "small"),
        "big_blind_score": _blind_score(blinds, "big"),
        "boss_blind_score": _blind_score(blinds, "boss"),
        "jokers_count": len(state.get("jokers") or []),
        "consumables_count": len(state.get("consumables") or []),
        "ante_num": _safe_int(state.get("ante_num"), 0),
        "round_num": _safe_int(state.get("round_num"), 0),
    }

    context = [
        context_values["hands_left"],
        context_values["discards_left"],
        context_values["round_chips"],
        context_values["money"],
        context_values["reroll_cost"],
        context_values["small_blind_score"],
        context_values["big_blind_score"],
        context_values["boss_blind_score"],
        context_values["jokers_count"],
        context_values["consumables_count"],
        context_values["ante_num"],
        context_values["round_num"],
    ]

    return {
        "schema_version": "feature_v2",
        "phase": phase,
        "hand_size": hand_size,
        "card_rank_ids": rank_ids,
        "card_suit_ids": suit_ids,
        "card_chip_hint": chip_hint,
        "card_has_enhancement": enh_flags,
        "card_has_edition": edt_flags,
        "card_has_seal": seal_flags,
        "hand_pad_mask": pad_mask,
        "context": context,
        "context_values": context_values,
    }


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Extract model features from a gamestate JSON file.")
    parser.add_argument("--state-json", required=True, help="Path to a json file containing gamestate dict.")
    args = parser.parse_args()

    state = json.loads(Path(args.state_json).read_text(encoding="utf-8"))
    feat = extract_features(state)
    print(json.dumps(feat, ensure_ascii=False, indent=2))
