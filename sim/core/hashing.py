import hashlib
from typing import Any

from sim.core.serde import canonical_dumps, to_builtin


def _zone_cards(zones: dict[str, Any], name: str) -> list[dict[str, Any]]:
    raw = zones.get(name)
    if isinstance(raw, list):
        return [c for c in raw if isinstance(c, dict)]
    return []


def _normalize_rank(raw_rank: Any) -> str:
    rank = str(raw_rank or "").strip().upper()
    if rank == "T":
        rank = "10"
    return rank


def _normalize_suit(raw_suit: Any) -> str:
    suit = str(raw_suit or "").strip().upper()
    return suit[:1] if suit else ""


def _normalize_tags(raw_tags: Any) -> list[str]:
    if isinstance(raw_tags, list):
        tags = [str(x).strip() for x in raw_tags if str(x).strip()]
    elif raw_tags is None:
        tags = []
    else:
        text = str(raw_tags).strip()
        tags = [text] if text else []
    return sorted(set(tags))


def _card_minimal(card: dict[str, Any]) -> dict[str, Any]:
    value = card.get("value") if isinstance(card.get("value"), dict) else {}
    rank = _normalize_rank(card.get("rank") or value.get("rank"))
    suit = _normalize_suit(card.get("suit") or value.get("suit"))

    key = str(card.get("key") or "").strip().upper()
    if not key and rank and suit:
        key_rank = "T" if rank == "10" else rank
        key = f"{suit}_{key_rank}"

    modifier = _normalize_tags(card.get("modifier"))
    if not modifier:
        modifier = _normalize_tags(card.get("modifier_tags"))

    state_tags = _normalize_tags(card.get("state"))
    if not state_tags:
        state_tags = _normalize_tags(card.get("state_tags"))

    return {
        "rank": rank,
        "suit": suit,
        "key": key,
        "modifier": modifier,
        "state": state_tags,
    }


def _zone_cards_min_sorted(zones: dict[str, Any], name: str) -> list[dict[str, Any]]:
    cards = [_card_minimal(c) for c in _zone_cards(zones, name)]
    cards.sort(key=canonical_dumps)
    return cards


def _zone_uids(zones: dict[str, Any], name: str) -> list[str]:
    out: list[str] = []
    for idx, card in enumerate(_zone_cards(zones, name)):
        uid = card.get("uid") or card.get("card_id") or card.get("id") or card.get("key")
        if uid is None or str(uid) == "":
            uid = f"{name}-{idx}"
        out.append(str(uid))
    return out


def _filter_hand_core(state: dict[str, Any]) -> dict[str, Any]:
    state = to_builtin(state)
    zones = state.get("zones") or {}
    round_info = state.get("round") or {}
    score = state.get("score") or {}
    economy = state.get("economy") or {}
    flags = state.get("flags") or {}

    return {
        "schema_version": state.get("schema_version"),
        "phase": state.get("phase"),
        "zones": {
            "hand": _zone_cards(zones, "hand"),
        },
        "round": {
            "hands_left": round_info.get("hands_left", 0),
            "discards_left": round_info.get("discards_left", 0),
            "ante": round_info.get("ante", 0),
            "round_num": round_info.get("round_num", 0),
            "blind": round_info.get("blind", "unknown"),
        },
        "score": {
            "chips": score.get("chips", 0),
            "mult": score.get("mult", 1),
            "target_chips": score.get("target_chips", 0),
        },
        "economy": {
            "money": economy.get("money", 0),
        },
        "flags": {
            "done": bool(flags.get("done", False)),
            "won": bool(flags.get("won", False)),
        },
    }


def _filter_score_core(state: dict[str, Any]) -> dict[str, Any]:
    state = to_builtin(state)
    zones = state.get("zones") or {}
    round_info = state.get("round") or {}
    score = state.get("score") or {}

    return {
        "schema_version": state.get("schema_version"),
        "phase": state.get("phase"),
        "zones": {
            "played": _zone_cards(zones, "played"),
        },
        "round": {
            "hands_left": round_info.get("hands_left", 0),
            "discards_left": round_info.get("discards_left", 0),
            "blind": round_info.get("blind", "unknown"),
            "ante": round_info.get("ante", 0),
            "round_num": round_info.get("round_num", 0),
        },
        "score": {
            "chips": score.get("chips", 0),
            "mult": score.get("mult", 1),
            "target_chips": score.get("target_chips", 0),
            "last_hand_type": score.get("last_hand_type", ""),
            "last_base_chips": score.get("last_base_chips", 0),
            "last_base_mult": score.get("last_base_mult", 1),
        },
    }


def _filter_p0_hand_score_core(state: dict[str, Any]) -> dict[str, Any]:
    state = to_builtin(state)
    zones = state.get("zones") or {}
    round_info = state.get("round") or {}
    score = state.get("score") or {}
    return {
        "schema_version": state.get("schema_version"),
        "zones": {
            "hand": _zone_cards_min_sorted(zones, "hand"),
            "played": _zone_cards_min_sorted(zones, "played"),
        },
        "round": {
            "hands_left": round_info.get("hands_left", 0),
            "discards_left": round_info.get("discards_left", 0),
        },
        "score": {
            "last_hand_type": score.get("last_hand_type", ""),
            "last_base_chips": score.get("last_base_chips", 0),
            "last_base_mult": score.get("last_base_mult", 1),
        },
    }


def _filter_p0_hand_score_observed_core(state: dict[str, Any]) -> dict[str, Any]:
    state = to_builtin(state)
    zones = state.get("zones") or {}
    round_info = state.get("round") or {}
    observed = state.get("score_observed") or {}
    phase = str(state.get("phase") or "")

    hand_count = len(_zone_cards(zones, "hand")) if phase == "SELECTING_HAND" else 0

    return {
        "schema_version": state.get("schema_version"),
        "zones": {
            "hand_count": hand_count,
        },
        "round": {
            "hands_left": round_info.get("hands_left", 0),
            "discards_left": round_info.get("discards_left", 0),
        },
        "score_observed": {
            "total": float(observed.get("total") or 0.0),
            "delta": float(observed.get("delta") or 0.0),
        },
    }


def _filter_p1_hand_score_observed_core(state: dict[str, Any]) -> dict[str, Any]:
    state = to_builtin(state)
    zones = state.get("zones") or {}
    round_info = state.get("round") or {}
    observed = state.get("score_observed") or {}
    phase = str(state.get("phase") or "")

    hand_count = len(_zone_cards(zones, "hand")) if phase == "SELECTING_HAND" else 0

    return {
        "schema_version": state.get("schema_version"),
        "zones": {
            "hand_count": hand_count,
        },
        "round": {
            "hands_left": round_info.get("hands_left", 0),
            "discards_left": round_info.get("discards_left", 0),
        },
        "score_observed": {
            "total": float(observed.get("total") or 0.0),
            "delta": float(observed.get("delta") or 0.0),
        },
    }


def _filter_p2_hand_score_observed_core(state: dict[str, Any]) -> dict[str, Any]:
    state = to_builtin(state)
    zones = state.get("zones") or {}
    round_info = state.get("round") or {}
    observed = state.get("score_observed") or {}
    phase = str(state.get("phase") or "")

    hand_count = len(_zone_cards(zones, "hand")) if phase == "SELECTING_HAND" else 0

    return {
        "schema_version": state.get("schema_version"),
        "zones": {
            "hand_count": hand_count,
        },
        "round": {
            "hands_left": round_info.get("hands_left", 0),
            "discards_left": round_info.get("discards_left", 0),
        },
        "score_observed": {
            "total": float(observed.get("total") or 0.0),
            "delta": float(observed.get("delta") or 0.0),
        },
    }


def _filter_p2b_hand_score_observed_core(state: dict[str, Any]) -> dict[str, Any]:
    state = to_builtin(state)
    zones = state.get("zones") or {}
    round_info = state.get("round") or {}
    observed = state.get("score_observed") or {}
    phase = str(state.get("phase") or "")

    hand_count = len(_zone_cards(zones, "hand")) if phase == "SELECTING_HAND" else 0

    return {
        "schema_version": state.get("schema_version"),
        "zones": {
            "hand_count": hand_count,
        },
        "round": {
            "hands_left": round_info.get("hands_left", 0),
            "discards_left": round_info.get("discards_left", 0),
        },
        "score_observed": {
            "total": float(observed.get("total") or 0.0),
            "delta": float(observed.get("delta") or 0.0),
        },
    }



def _filter_p3_hand_score_observed_core(state: dict[str, Any]) -> dict[str, Any]:
    state = to_builtin(state)
    zones = state.get("zones") or {}
    round_info = state.get("round") or {}
    observed = state.get("score_observed") or {}
    phase = str(state.get("phase") or "")

    hand_count = len(_zone_cards(zones, "hand")) if phase == "SELECTING_HAND" else 0

    return {
        "schema_version": state.get("schema_version"),
        "zones": {
            "hand_count": hand_count,
        },
        "round": {
            "hands_left": round_info.get("hands_left", 0),
            "discards_left": round_info.get("discards_left", 0),
        },
        "score_observed": {
            "total": float(observed.get("total") or 0.0),
            "delta": float(observed.get("delta") or 0.0),
        },
    }


def _extract_hand_levels(state: dict[str, Any]) -> dict[str, int]:
    hands = state.get("hands") or {}
    levels: dict[str, int] = {}

    if isinstance(hands, dict) and isinstance(hands.get("levels"), dict):
        source = hands.get("levels") or {}
        for raw_name, raw_info in source.items():
            name = str(raw_name or "").strip().upper()
            if not name:
                continue
            if isinstance(raw_info, dict):
                level = int(raw_info.get("level") or 1)
            else:
                try:
                    level = int(raw_info)
                except Exception:
                    level = 1
            levels[name] = level
        return levels

    if isinstance(hands, dict):
        for raw_name, raw_info in hands.items():
            name = str(raw_name or "").strip().upper()
            if not name:
                continue
            if isinstance(raw_info, dict):
                level = int(raw_info.get("level") or 1)
            else:
                try:
                    level = int(raw_info)
                except Exception:
                    level = 1
            levels[name] = level
    return levels


def _extract_consumable_keys(state: dict[str, Any]) -> dict[str, Any]:
    raw = state.get("consumables")
    if not isinstance(raw, dict):
        return {"count": 0, "limit": 0, "cards": []}

    cards_raw = raw.get("cards")
    cards = cards_raw if isinstance(cards_raw, list) else []
    keys: list[str] = []
    for card in cards:
        if isinstance(card, dict):
            key = str(card.get("key") or "").strip().lower()
            if key:
                keys.append(key)
    keys.sort()
    return {
        "count": int(raw.get("count") or len(keys)),
        "limit": int(raw.get("limit") or 0),
        "cards": keys,
    }



def _extract_market_keys(state: dict[str, Any], field: str) -> dict[str, Any]:
    raw = state.get(field)
    if not isinstance(raw, dict):
        return {"count": 0, "limit": 0, "cards": []}

    cards_raw = raw.get("cards")
    cards = cards_raw if isinstance(cards_raw, list) else []
    keys: list[str] = []
    for card in cards:
        if isinstance(card, dict):
            key = str(card.get("key") or "").strip().lower()
            if key:
                keys.append(key)
    keys.sort()
    return {
        "count": int(raw.get("count") or len(keys)),
        "limit": int(raw.get("limit") or 0),
        "cards": keys,
    }


def _extract_used_vouchers(state: dict[str, Any]) -> list[str]:
    raw = state.get("used_vouchers")
    out: list[str] = []
    if isinstance(raw, dict):
        out.extend(str(k).strip().lower() for k in raw.keys() if str(k).strip())
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                key = str(item.get("key") or item.get("id") or "").strip().lower()
                if key:
                    out.append(key)
            else:
                key = str(item).strip().lower()
                if key:
                    out.append(key)
    elif isinstance(raw, str):
        key = raw.strip().lower()
        if key:
            out.append(key)
    return sorted(set(out))


def _extract_joker_ids(state: dict[str, Any]) -> list[str]:
    raw = state.get("jokers")
    out: list[str] = []
    if isinstance(raw, dict):
        cards = raw.get("cards") if isinstance(raw.get("cards"), list) else []
        for card in cards:
            if isinstance(card, dict):
                key = str(card.get("key") or card.get("id") or "").strip().lower()
                if key:
                    out.append(key)
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                key = str(item.get("joker_id") or item.get("key") or item.get("id") or "").strip().lower()
                if key:
                    out.append(key)
            else:
                key = str(item).strip().lower()
                if key:
                    out.append(key)
    return sorted(out)


def _filter_p4_consumable_observed_core(state: dict[str, Any]) -> dict[str, Any]:
    state = to_builtin(state)
    zones = state.get("zones") or {}
    round_info = state.get("round") or {}
    observed = state.get("score_observed") or {}
    hand_cards = _zone_cards_min_sorted(zones, "hand")

    return {
        "schema_version": state.get("schema_version"),
        "zones": {
            "hand_count": len(hand_cards),
            "hand": hand_cards,
        },
        "round": {
            "hands_left": round_info.get("hands_left", 0),
            "discards_left": round_info.get("discards_left", 0),
        },
        "hands": {
            "levels": _extract_hand_levels(state),
        },
        "consumables": _extract_consumable_keys(state),
        "score_observed": {
            "total": float(observed.get("total") or 0.0),
            "delta": float(observed.get("delta") or 0.0),
        },
    }


def _filter_p5_modifier_observed_core(state: dict[str, Any]) -> dict[str, Any]:
    state = to_builtin(state)
    zones = state.get("zones") or {}
    round_info = state.get("round") or {}
    observed = state.get("score_observed") or {}
    hand_cards = _zone_cards_min_sorted(zones, "hand")
    modified_hand_count = sum(1 for c in hand_cards if c.get("modifier") or c.get("state"))

    return {
        "schema_version": state.get("schema_version"),
        "zones": {
            "hand_count": len(hand_cards),
            "modified_hand_count": int(modified_hand_count),
        },
        "round": {
            "hands_left": round_info.get("hands_left", 0),
            "discards_left": round_info.get("discards_left", 0),
        },
        "consumables": _extract_consumable_keys(state),
        "score_observed": {
            "total": float(observed.get("total") or 0.0),
            "delta": float(observed.get("delta") or 0.0),
        },
    }



def _filter_p5_voucher_pack_observed_core(state: dict[str, Any]) -> dict[str, Any]:
    state = to_builtin(state)
    zones = state.get("zones") or {}
    round_info = state.get("round") or {}
    observed = state.get("score_observed") or {}
    hand_count = len(_zone_cards(zones, "hand"))

    joker_ids = _extract_joker_ids(state)

    return {
        "schema_version": state.get("schema_version"),
        "zones": {
            "hand_count": hand_count,
        },
        "round": {
            "hands_left": round_info.get("hands_left", 0),
            "discards_left": round_info.get("discards_left", 0),
        },
        "jokers": {
            "count": len(joker_ids),
            "ids": joker_ids,
        },
        "consumables": _extract_consumable_keys(state),
        "vouchers": _extract_market_keys(state, "vouchers"),
        "packs": _extract_market_keys(state, "packs"),
        "used_vouchers": _extract_used_vouchers(state),
        "score_observed": {
            "total": float(observed.get("total") or 0.0),
            "delta": float(observed.get("delta") or 0.0),
        },
    }

def _filter_zones_core(state: dict[str, Any]) -> dict[str, Any]:
    state = to_builtin(state)
    zones = state.get("zones") or {}

    zone_view: dict[str, Any] = {}
    for name in ("deck", "discard", "hand"):
        uids = _zone_uids(zones, name)
        zone_view[name] = {
            "len": len(uids),
            "uids": uids,
        }

    return {
        "schema_version": state.get("schema_version"),
        "phase": state.get("phase"),
        "zones": zone_view,
    }


def _filter_zones_counts_core(state: dict[str, Any]) -> dict[str, Any]:
    state = to_builtin(state)
    zones = state.get("zones") or {}
    round_info = state.get("round") or {}
    counts: dict[str, int] = {}
    for name in ("deck", "discard", "hand"):
        counts[name] = len(_zone_cards(zones, name))
    return {
        "schema_version": state.get("schema_version"),
        "zones": counts,
        "round": {
            "hands_left": round_info.get("hands_left", 0),
            "discards_left": round_info.get("discards_left", 0),
        },
    }


def _filter_economy_core(state: dict[str, Any]) -> dict[str, Any]:
    state = to_builtin(state)
    economy = state.get("economy") or {}

    return {
        "schema_version": state.get("schema_version"),
        "phase": state.get("phase"),
        "economy": {
            "money": economy.get("money", 0),
            "interest": economy.get("interest", 0),
            "discount": economy.get("discount", 0),
            "reroll_cost": economy.get("reroll_cost", 0),
        },
    }


def _rng_event_key(event: Any) -> str:
    if isinstance(event, dict):
        for key in ("event", "type", "kind", "name", "action", "source"):
            value = event.get(key)
            if value is not None and str(value) != "":
                return f"{key}:{value}"
        keys = sorted(str(k) for k in event.keys())
        return "keys:" + ",".join(keys)
    if isinstance(event, list):
        return f"list:{len(event)}"
    if event is None:
        return "none"
    return f"scalar:{type(event).__name__}:{event}"


def _filter_rng_events_core(state: dict[str, Any]) -> dict[str, Any]:
    state = to_builtin(state)
    rng = state.get("rng") or {}
    raw_events = rng.get("events")
    events = raw_events if isinstance(raw_events, list) else []
    event_keys = [_rng_event_key(ev) for ev in events]

    return {
        "schema_version": state.get("schema_version"),
        "phase": state.get("phase"),
        "rng": {
            "mode": rng.get("mode", "native"),
            "seed": rng.get("seed"),
            "cursor": rng.get("cursor", 0),
            "event_keys": event_keys,
        },
    }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def state_hash_full(state: dict[str, Any]) -> str:
    return _sha256_text(canonical_dumps(state))


def state_hash_hand_core(state: dict[str, Any]) -> str:
    return _sha256_text(canonical_dumps(_filter_hand_core(state)))


def state_hash_score_core(state: dict[str, Any]) -> str:
    return _sha256_text(canonical_dumps(_filter_score_core(state)))


def state_hash_p0_hand_score_core(state: dict[str, Any]) -> str:
    return _sha256_text(canonical_dumps(_filter_p0_hand_score_core(state)))


def state_hash_p0_hand_score_observed_core(state: dict[str, Any]) -> str:
    return _sha256_text(canonical_dumps(_filter_p0_hand_score_observed_core(state)))


def state_hash_p1_hand_score_observed_core(state: dict[str, Any]) -> str:
    return _sha256_text(canonical_dumps(_filter_p1_hand_score_observed_core(state)))


def state_hash_p2_hand_score_observed_core(state: dict[str, Any]) -> str:
    return _sha256_text(canonical_dumps(_filter_p2_hand_score_observed_core(state)))


def state_hash_p2b_hand_score_observed_core(state: dict[str, Any]) -> str:
    return _sha256_text(canonical_dumps(_filter_p2b_hand_score_observed_core(state)))



def state_hash_p3_hand_score_observed_core(state: dict[str, Any]) -> str:
    return _sha256_text(canonical_dumps(_filter_p3_hand_score_observed_core(state)))


def state_hash_p4_consumable_observed_core(state: dict[str, Any]) -> str:
    return _sha256_text(canonical_dumps(_filter_p4_consumable_observed_core(state)))


def state_hash_p5_modifier_observed_core(state: dict[str, Any]) -> str:
    return _sha256_text(canonical_dumps(_filter_p5_modifier_observed_core(state)))


def state_hash_p5_voucher_pack_observed_core(state: dict[str, Any]) -> str:
    return _sha256_text(canonical_dumps(_filter_p5_voucher_pack_observed_core(state)))


def state_hash_zones_core(state: dict[str, Any]) -> str:
    return _sha256_text(canonical_dumps(_filter_zones_core(state)))


def state_hash_zones_counts_core(state: dict[str, Any]) -> str:
    return _sha256_text(canonical_dumps(_filter_zones_counts_core(state)))


def state_hash_economy_core(state: dict[str, Any]) -> str:
    return _sha256_text(canonical_dumps(_filter_economy_core(state)))


def state_hash_rng_events_core(state: dict[str, Any]) -> str:
    return _sha256_text(canonical_dumps(_filter_rng_events_core(state)))


def hand_core_projection(state: dict[str, Any]) -> dict[str, Any]:
    return _filter_hand_core(state)


def score_core_projection(state: dict[str, Any]) -> dict[str, Any]:
    return _filter_score_core(state)


def p0_hand_score_core_projection(state: dict[str, Any]) -> dict[str, Any]:
    return _filter_p0_hand_score_core(state)


def p0_hand_score_observed_core_projection(state: dict[str, Any]) -> dict[str, Any]:
    return _filter_p0_hand_score_observed_core(state)


def p1_hand_score_observed_core_projection(state: dict[str, Any]) -> dict[str, Any]:
    return _filter_p1_hand_score_observed_core(state)


def p2_hand_score_observed_core_projection(state: dict[str, Any]) -> dict[str, Any]:
    return _filter_p2_hand_score_observed_core(state)


def p2b_hand_score_observed_core_projection(state: dict[str, Any]) -> dict[str, Any]:
    return _filter_p2b_hand_score_observed_core(state)



def p3_hand_score_observed_core_projection(state: dict[str, Any]) -> dict[str, Any]:
    return _filter_p3_hand_score_observed_core(state)


def p4_consumable_observed_core_projection(state: dict[str, Any]) -> dict[str, Any]:
    return _filter_p4_consumable_observed_core(state)


def p5_modifier_observed_core_projection(state: dict[str, Any]) -> dict[str, Any]:
    return _filter_p5_modifier_observed_core(state)


def p5_voucher_pack_observed_core_projection(state: dict[str, Any]) -> dict[str, Any]:
    return _filter_p5_voucher_pack_observed_core(state)


def zones_core_projection(state: dict[str, Any]) -> dict[str, Any]:
    return _filter_zones_core(state)


def zones_counts_core_projection(state: dict[str, Any]) -> dict[str, Any]:
    return _filter_zones_counts_core(state)


def economy_core_projection(state: dict[str, Any]) -> dict[str, Any]:
    return _filter_economy_core(state)


def rng_events_core_projection(state: dict[str, Any]) -> dict[str, Any]:
    return _filter_rng_events_core(state)


