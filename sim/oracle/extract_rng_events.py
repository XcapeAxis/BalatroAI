
from typing import Any


def _hand_keys(state: dict[str, Any]) -> list[str]:
    cards = (state.get("hand") or {}).get("cards") or []
    out = []
    for c in cards:
        if not isinstance(c, dict):
            continue
        out.append(str(c.get("key") or c.get("card_id") or c.get("id") or ""))
    return out


def extract_rng_events(prev_state: dict[str, Any] | None, cur_state: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(cur_state, dict):
        return []

    events: list[dict[str, Any]] = []
    cur_phase = str(cur_state.get("state") or "UNKNOWN")
    prev_phase = str((prev_state or {}).get("state") or "UNKNOWN")

    cur_hand = _hand_keys(cur_state)
    prev_hand = _hand_keys(prev_state or {})
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
    prev_money = float((prev_state or {}).get("money") or 0)
    if cur_money != prev_money:
        events.append({"type": "money_change", "delta": cur_money - prev_money})

    return events

