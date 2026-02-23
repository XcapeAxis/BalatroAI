from __future__ import annotations

from typing import Any

SHOP_PHASES = {"SHOP", "SMODS_BOOSTER_OPENED"}

# Fixed action head for shop-related decisions.
# Keep this compact and deterministic for BC smoke.
_ACTION_SPECS: list[tuple[str, dict[str, Any]]] = [
    ("NEXT_ROUND", {}),
    ("REROLL", {}),
    ("BUY", {"card": 0}),
    ("BUY", {"pack": 0}),
    ("BUY", {"voucher": 0}),
    ("SELL", {"joker": 0}),
    ("PACK", {"card": 0}),
    ("PACK", {"skip": True}),
    ("USE", {"consumable": 0}),
    ("WAIT", {}),
]


def max_actions() -> int:
    return len(_ACTION_SPECS)


def decode(action_id: int) -> tuple[str, dict[str, Any]]:
    idx = int(action_id)
    if idx < 0 or idx >= len(_ACTION_SPECS):
        raise ValueError(f"shop action_id out of range: {action_id}")
    atype, params = _ACTION_SPECS[idx]
    return atype, dict(params)


def _normalize_params(params: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(params, dict):
        return {}
    out: dict[str, Any] = {}
    for k, v in params.items():
        out[str(k)] = v
    return out


def encode(action_type: str, params: dict[str, Any] | None = None) -> int:
    at = str(action_type or "").upper()
    norm = _normalize_params(params)
    for i, (spec_at, spec_params) in enumerate(_ACTION_SPECS):
        if at != spec_at:
            continue
        if spec_params == norm:
            return i
    # Relaxed match on action type only if exact param preset not found.
    for i, (spec_at, _) in enumerate(_ACTION_SPECS):
        if at == spec_at:
            return i
    raise ValueError(f"unsupported shop action for encode: {action_type} {params}")


def _market_count(state: dict[str, Any], field: str) -> int:
    raw = state.get(field)
    if not isinstance(raw, dict):
        return 0
    cards = raw.get("cards")
    if isinstance(cards, list):
        return len(cards)
    return int(raw.get("count") or 0)


def _consumable_count(state: dict[str, Any]) -> int:
    raw = state.get("consumables")
    if not isinstance(raw, dict):
        return 0
    cards = raw.get("cards")
    if isinstance(cards, list):
        return len(cards)
    return int(raw.get("count") or 0)


def _joker_count(state: dict[str, Any]) -> int:
    jokers = state.get("jokers")
    if isinstance(jokers, list):
        return len(jokers)
    return 0


def legal_action_ids(state: dict[str, Any]) -> list[int]:
    phase = str(state.get("state") or "UNKNOWN")

    ids: list[int] = []

    if phase == "SHOP":
        ids.extend([0, 1, 9])  # NEXT_ROUND, REROLL, WAIT

        if _market_count(state, "shop") > 0:
            ids.append(2)
        if _market_count(state, "packs") > 0:
            ids.append(3)
        if _market_count(state, "vouchers") > 0:
            ids.append(4)
        if _joker_count(state) > 0:
            ids.append(5)
        if _consumable_count(state) > 0:
            ids.append(8)

    elif phase == "SMODS_BOOSTER_OPENED":
        ids.extend([6, 7, 9, 0])  # PACK(card/skip), WAIT, NEXT_ROUND

    else:
        ids.append(9)  # WAIT fallback

    # Keep deterministic order and uniqueness.
    seen: set[int] = set()
    out: list[int] = []
    for aid in ids:
        if 0 <= aid < len(_ACTION_SPECS) and aid not in seen:
            seen.add(aid)
            out.append(aid)
    return out


def action_from_id(state: dict[str, Any], action_id: int) -> dict[str, Any]:
    atype, params = decode(action_id)
    phase = str(state.get("state") or "UNKNOWN")

    # Build dynamic params from current state where needed.
    if phase == "SHOP":
        if atype == "BUY":
            if "card" in params:
                params["card"] = 0
            elif "pack" in params:
                params["pack"] = 0
            elif "voucher" in params:
                params["voucher"] = 0
        elif atype == "SELL":
            params["joker"] = 0
        elif atype == "USE":
            params["consumable"] = 0
    elif phase == "SMODS_BOOSTER_OPENED" and atype == "PACK":
        if "card" in params:
            params["card"] = 0

    action = {"action_type": atype}
    if params:
        action["params"] = dict(params)
    return action
