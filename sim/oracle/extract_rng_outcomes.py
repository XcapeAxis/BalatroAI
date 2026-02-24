from __future__ import annotations

from typing import Any

from sim.oracle.extract_rng_events import extract_rng_events


def _action_type(action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict):
        return ""
    return str(action.get("action_type") or "").strip().upper()


def _round_chips(state: dict[str, Any] | None) -> float:
    if not isinstance(state, dict):
        return 0.0
    round_info = state.get("round") if isinstance(state.get("round"), dict) else {}
    try:
        return float(round_info.get("chips") or 0.0)
    except Exception:
        return 0.0


def _money(state: dict[str, Any] | None) -> float:
    if not isinstance(state, dict):
        return 0.0
    try:
        return float(state.get("money") or 0.0)
    except Exception:
        return 0.0


def _shop_offer_keys(state: dict[str, Any] | None) -> list[str]:
    if not isinstance(state, dict):
        return []
    shop = state.get("shop") if isinstance(state.get("shop"), dict) else {}
    cards = shop.get("cards") if isinstance(shop.get("cards"), list) else []
    keys: list[str] = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        key = str(card.get("key") or "").strip().lower()
        if key:
            keys.append(key)
    return sorted(keys)


def _expected_jokers(action: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(action, dict):
        return []
    ctx = action.get("expected_context") if isinstance(action.get("expected_context"), dict) else {}
    jokers = ctx.get("jokers") if isinstance(ctx.get("jokers"), list) else []
    out: list[dict[str, Any]] = []
    for item in jokers:
        if isinstance(item, dict):
            out.append(item)
    return out


def _is_prob_joker(j: dict[str, Any]) -> bool:
    mode = str(j.get("mode") or j.get("kind") or "").strip().lower()
    if mode in {"prob", "probabilistic", "prob_trigger"}:
        return True
    tags = j.get("tags") if isinstance(j.get("tags"), list) else []
    return any(str(x).strip().lower() in {"prob", "random"} for x in tags)


def _is_econ_joker(j: dict[str, Any]) -> bool:
    mode = str(j.get("mode") or j.get("kind") or "").strip().lower()
    if mode in {"econ", "economy", "shop", "econ_shop"}:
        return True
    tags = j.get("tags") if isinstance(j.get("tags"), list) else []
    return any(str(x).strip().lower() in {"econ", "economy", "shop"} for x in tags)


def extract_rng_outcomes(
    prev_state: dict[str, Any] | None,
    cur_state: dict[str, Any] | None,
    *,
    action: dict[str, Any] | None = None,
    step_id: int | None = None,
) -> list[dict[str, Any]]:
    outcomes = [dict(x) for x in extract_rng_events(prev_state, cur_state)]

    at = _action_type(action)
    money_delta = _money(cur_state) - _money(prev_state)
    score_delta = _round_chips(cur_state) - _round_chips(prev_state)
    scope = "shop" if at in {"BUY", "SELL", "REROLL", "PACK", "USE"} else "hand"

    # Explicit economy token for P11 scope/diff.
    if at in {"BUY", "SELL", "REROLL", "PACK", "USE", "CASH_OUT", "NEXT_ROUND"} or abs(money_delta) > 1e-9:
        outcomes.append(
            {
                "type": "econ_delta",
                "key": at.lower() if at else "unknown",
                "value": float(money_delta),
                "scope": scope,
                "step": int(step_id or 0),
            }
        )

    # Shop reroll token with stable offer-key signature.
    if at == "REROLL":
        outcomes.append(
            {
                "type": "shop_roll",
                "key": "reroll",
                "value": _shop_offer_keys(cur_state),
                "scope": "shop",
                "step": int(step_id or 0),
            }
        )

    # Oracle-guided token for probabilistic/econ tagged jokers from expected_context.
    for joker in _expected_jokers(action):
        key = str(joker.get("key") or joker.get("joker_key") or "").strip().lower()
        if not key:
            continue
        if _is_prob_joker(joker):
            outcomes.append(
                {
                    "type": "prob_trigger",
                    "key": key,
                    "value": bool(abs(score_delta) > 1e-9 or abs(money_delta) > 1e-9),
                    "scope": scope,
                    "step": int(step_id or 0),
                }
            )
        if _is_econ_joker(joker):
            outcomes.append(
                {
                    "type": "econ_delta",
                    "key": key,
                    "value": float(money_delta),
                    "scope": scope,
                    "step": int(step_id or 0),
                }
            )

    return outcomes

