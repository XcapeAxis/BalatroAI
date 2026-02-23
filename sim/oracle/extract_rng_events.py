
from typing import Any


def _hand_keys(state: dict[str, Any]) -> list[str]:
    cards = (state.get("hand") or {}).get("cards") or []
    out = []
    for c in cards:
        if not isinstance(c, dict):
            continue
        out.append(str(c.get("key") or c.get("card_id") or c.get("id") or ""))
    return out


def _market_items(state: dict[str, Any], field: str) -> list[tuple[str, str, float, int]]:
    raw = state.get(field)
    if not isinstance(raw, dict):
        return []
    cards = raw.get("cards") if isinstance(raw.get("cards"), list) else []
    out: list[tuple[str, str, float, int]] = []
    for idx, card in enumerate(cards):
        if not isinstance(card, dict):
            continue
        key = str(card.get("key") or "").strip().lower()
        kind = str(card.get("set") or card.get("kind") or "").strip().upper()
        slot = int(card.get("slot_index") if isinstance(card.get("slot_index"), int) else idx)
        cost_obj = card.get("cost")
        cost = 0.0
        if isinstance(cost_obj, dict):
            cost = float(cost_obj.get("buy") or 0.0)
        else:
            try:
                cost = float(cost_obj or 0.0)
            except Exception:
                cost = 0.0
        out.append((kind, key, cost, slot))
    out.sort()
    return out


def _pack_choice_keys(state: dict[str, Any]) -> list[str]:
    candidates = [
        state.get("pack_choices"),
        state.get("pack"),
        state.get("booster"),
        state.get("booster_pack"),
    ]
    out: list[str] = []
    for cand in candidates:
        cards: list[Any] = []
        if isinstance(cand, list):
            cards = cand
        elif isinstance(cand, dict):
            raw_cards = cand.get("cards")
            if isinstance(raw_cards, list):
                cards = raw_cards
        for card in cards:
            if isinstance(card, dict):
                key = str(card.get("key") or "").strip().lower()
                if key:
                    out.append(key)
            elif isinstance(card, (str, int, float, bool)):
                text = str(card).strip().lower()
                if text:
                    out.append(text)
    return sorted(set(out))


def _active_blind_signal(state: dict[str, Any]) -> dict[str, Any]:
    round_info = state.get("round") if isinstance(state.get("round"), dict) else {}
    blind = str(round_info.get("blind") or state.get("blind") or "").strip().lower()
    blinds = state.get("blinds") if isinstance(state.get("blinds"), dict) else {}
    selected = ""
    selected_score = 0.0
    for key in ("small", "big", "boss"):
        info = blinds.get(key) if isinstance(blinds, dict) else None
        if not isinstance(info, dict):
            continue
        status = str(info.get("status") or "").upper()
        if status in {"SELECT", "CURRENT", "SELECTED"}:
            selected = key
            selected_score = float(info.get("score") or 0.0)
            break
    if not selected:
        selected = blind
    return {
        "blind": blind,
        "selected": selected,
        "selected_score": selected_score,
        "is_boss": bool((selected or blind) == "boss"),
    }


def _tags_signal(state: dict[str, Any]) -> list[str]:
    raw = state.get("tags")
    if isinstance(raw, list):
        items = raw
    elif raw is None:
        items = []
    else:
        items = [raw]
    out: list[str] = []
    for item in items:
        if isinstance(item, dict):
            key = str(item.get("key") or item.get("id") or item.get("name") or "").strip().lower()
            if key:
                out.append(key)
        elif isinstance(item, (str, int, float, bool)):
            text = str(item).strip().lower()
            if text:
                out.append(text)
    out.sort()
    return out


def extract_rng_events(prev_state: dict[str, Any] | None, cur_state: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(cur_state, dict):
        return []

    events: list[dict[str, Any]] = []
    prev_state = prev_state if isinstance(prev_state, dict) else {}

    cur_phase = str(cur_state.get("state") or "UNKNOWN")
    prev_phase = str(prev_state.get("state") or "UNKNOWN")
    if cur_phase != prev_phase:
        events.append(
            {
                "type": "phase_transition",
                "from_phase": prev_phase,
                "to_phase": cur_phase,
            }
        )

    cur_hand = _hand_keys(cur_state)
    prev_hand = _hand_keys(prev_state)
    if cur_hand and cur_hand != prev_hand:
        events.append(
            {
                "type": "hand_cards",
                "from_phase": prev_phase,
                "to_phase": cur_phase,
                "cards": cur_hand,
            }
        )

    cur_money = float(cur_state.get("money") or 0)
    prev_money = float(prev_state.get("money") or 0)
    if cur_money != prev_money:
        events.append({"type": "money_change", "delta": cur_money - prev_money})

    for field, label in (("shop", "shop_offers"), ("vouchers", "voucher_offers"), ("packs", "pack_offers")):
        cur_items = _market_items(cur_state, field)
        prev_items = _market_items(prev_state, field)
        if cur_items != prev_items:
            events.append(
                {
                    "type": label,
                    "items": [
                        {"kind": k, "key": key, "cost": cost, "slot": slot}
                        for (k, key, cost, slot) in cur_items
                    ],
                }
            )

    cur_pack_choices = _pack_choice_keys(cur_state)
    prev_pack_choices = _pack_choice_keys(prev_state)
    if cur_pack_choices != prev_pack_choices:
        events.append({"type": "pack_choices", "choices": cur_pack_choices})

    blind_cur = _active_blind_signal(cur_state)
    blind_prev = _active_blind_signal(prev_state)
    if blind_cur != blind_prev:
        events.append({"type": "blind_signal", "signal": blind_cur})

    tags_cur = _tags_signal(cur_state)
    tags_prev = _tags_signal(prev_state)
    if tags_cur != tags_prev:
        events.append({"type": "tags_signal", "tags": tags_cur})

    if not events:
        events.append({"type": "phase_snapshot", "phase": cur_phase})

    return events

