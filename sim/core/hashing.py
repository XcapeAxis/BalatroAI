import hashlib
from typing import Any

from sim.core.serde import to_builtin, canonical_dumps


def _zone_cards(zones: dict[str, Any], name: str) -> list[dict[str, Any]]:
    raw = zones.get(name)
    if isinstance(raw, list):
        return [c for c in raw if isinstance(c, dict)]
    return []


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


def _filter_zones_core(state: dict[str, Any]) -> dict[str, Any]:
    state = to_builtin(state)
    zones = state.get("zones") or {}

    zone_view: dict[str, Any] = {}
    for name in ("deck", "discard", "hand", "played"):
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


def state_hash_zones_core(state: dict[str, Any]) -> str:
    return _sha256_text(canonical_dumps(_filter_zones_core(state)))


def state_hash_economy_core(state: dict[str, Any]) -> str:
    return _sha256_text(canonical_dumps(_filter_economy_core(state)))


def state_hash_rng_events_core(state: dict[str, Any]) -> str:
    return _sha256_text(canonical_dumps(_filter_rng_events_core(state)))


def hand_core_projection(state: dict[str, Any]) -> dict[str, Any]:
    return _filter_hand_core(state)


def score_core_projection(state: dict[str, Any]) -> dict[str, Any]:
    return _filter_score_core(state)


def zones_core_projection(state: dict[str, Any]) -> dict[str, Any]:
    return _filter_zones_core(state)


def economy_core_projection(state: dict[str, Any]) -> dict[str, Any]:
    return _filter_economy_core(state)


def rng_events_core_projection(state: dict[str, Any]) -> dict[str, Any]:
    return _filter_rng_events_core(state)
