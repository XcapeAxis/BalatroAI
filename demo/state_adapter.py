from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _rank_label(card: dict[str, Any]) -> str:
    value = card.get("value") if isinstance(card.get("value"), dict) else {}
    rank = str(card.get("rank") or value.get("rank") or "").strip().upper()
    return "10" if rank == "T" else rank


def _suit_label(card: dict[str, Any]) -> str:
    value = card.get("value") if isinstance(card.get("value"), dict) else {}
    suit = str(card.get("suit") or value.get("suit") or "").strip().upper()
    return suit[:1]


def card_view(card: dict[str, Any], *, index: int | None = None) -> dict[str, Any]:
    rank = _rank_label(card)
    suit = _suit_label(card)
    key = str(card.get("key") or "").strip()
    return {
        "index": index,
        "card_id": str(card.get("card_id") or card.get("uid") or key or f"card-{index}"),
        "rank": rank,
        "suit": suit,
        "label": f"{rank}{suit}",
        "key": key,
        "modifier_tags": [str(x) for x in (card.get("modifier_tags") or card.get("modifier") or []) if str(x).strip()],
        "state_tags": [str(x) for x in (card.get("state_tags") or card.get("state") or []) if str(x).strip()],
        "effect_text": str(card.get("effect_text") or (card.get("value") or {}).get("effect") or ""),
    }


def zone_cards(state: dict[str, Any], zone_name: str) -> list[dict[str, Any]]:
    raw = state.get(zone_name) if isinstance(state.get(zone_name), dict) else {}
    cards = raw.get("cards") if isinstance(raw.get("cards"), list) else []
    return [card_view(card, index=idx) for idx, card in enumerate(cards) if isinstance(card, dict)]


def resources_view(state: dict[str, Any]) -> dict[str, Any]:
    round_info = state.get("round") if isinstance(state.get("round"), dict) else {}
    score = state.get("score") if isinstance(state.get("score"), dict) else {}
    return {
        "hands_left": _safe_int(round_info.get("hands_left")),
        "discards_left": _safe_int(round_info.get("discards_left")),
        "money": _safe_float(state.get("money")),
        "round_chips": _safe_float(round_info.get("chips")),
        "score_chips": _safe_float(score.get("chips")),
        "target_chips": _safe_float(score.get("target_chips")),
        "blind": str(round_info.get("blind") or ""),
        "ante": _safe_int(state.get("ante_num"), 1),
        "round_num": _safe_int(state.get("round_num"), 1),
        "reroll_cost": _safe_int(round_info.get("reroll_cost")),
    }


def compute_resource_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_resources = resources_view(before)
    after_resources = resources_view(after)
    return {
        "hands_left": after_resources["hands_left"] - before_resources["hands_left"],
        "discards_left": after_resources["discards_left"] - before_resources["discards_left"],
        "money": round(after_resources["money"] - before_resources["money"], 4),
        "round_chips": round(after_resources["round_chips"] - before_resources["round_chips"], 4),
        "score_chips": round(after_resources["score_chips"] - before_resources["score_chips"], 4),
        "target_chips": round(after_resources["target_chips"] - before_resources["target_chips"], 4),
        "blind_before": before_resources["blind"],
        "blind_after": after_resources["blind"],
        "phase_before": str(before.get("state") or "UNKNOWN"),
        "phase_after": str(after.get("state") or "UNKNOWN"),
        "hand_count": len(zone_cards(after, "hand")) - len(zone_cards(before, "hand")),
        "discard_count": len(zone_cards(after, "discard")) - len(zone_cards(before, "discard")),
        "played_count": len(zone_cards(after, "played")) - len(zone_cards(before, "played")),
        "deck_count": len(zone_cards(after, "deck")) - len(zone_cards(before, "deck")),
        "joker_count": len(state_jokers(after)) - len(state_jokers(before)),
    }


def state_jokers(state: dict[str, Any]) -> list[dict[str, Any]]:
    raw = state.get("jokers")
    cards = raw if isinstance(raw, list) else []
    out: list[dict[str, Any]] = []
    for idx, card in enumerate(cards):
        if not isinstance(card, dict):
            continue
        out.append(
            {
                "index": idx,
                "key": str(card.get("key") or ""),
                "label": str(card.get("label") or card.get("name") or card.get("key") or "Joker"),
                "set": str(card.get("set") or "JOKER"),
            }
        )
    return out


def _selected_cards_from_action(action: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    indices = [int(x) for x in (action.get("indices") or [])]
    hand = zone_cards(state, "hand")
    selected: list[dict[str, Any]] = []
    for idx in indices:
        if 0 <= idx < len(hand):
            selected.append(hand[idx])
    return selected


def action_label(action: dict[str, Any], state: dict[str, Any]) -> str:
    action_type = str(action.get("action_type") or "WAIT").upper()
    if action_type in {"PLAY", "DISCARD"}:
        selected = _selected_cards_from_action(action, state)
        if not selected:
            return f"{action_type.title()} nothing"
        cards_text = ", ".join(card["label"] for card in selected)
        verb = "Play" if action_type == "PLAY" else "Discard"
        return f"{verb} {cards_text}"
    if action_type == "SELECT":
        return f"Select blind #{int(action.get('index', 0)) + 1}"
    if action_type == "CASH_OUT":
        return "Cash out"
    if action_type == "NEXT_ROUND":
        return "Start next round"
    if action_type == "START":
        return "Restart scenario"
    return action_type.replace("_", " ").title()


def build_state_payload(
    state: dict[str, Any],
    *,
    scenario: dict[str, Any],
    timeline: list[dict[str, Any]],
    mode: str,
    policy: str,
    model_name: str,
) -> dict[str, Any]:
    hand = zone_cards(state, "hand")
    discard = zone_cards(state, "discard")
    played = zone_cards(state, "played")
    deck = zone_cards(state, "deck")
    score = state.get("score") if isinstance(state.get("score"), dict) else {}
    blinds = state.get("blinds") if isinstance(state.get("blinds"), dict) else {}
    return {
        "timestamp": now_iso(),
        "phase": str(state.get("state") or "UNKNOWN"),
        "scenario": scenario,
        "mode": mode,
        "policy": policy,
        "model_name": model_name,
        "resources": resources_view(state),
        "score": {
            "chips": _safe_float(score.get("chips")),
            "mult": _safe_float(score.get("mult"), 1.0),
            "target_chips": _safe_float(score.get("target_chips")),
            "last_hand_type": str(score.get("last_hand_type") or ""),
            "last_base_chips": _safe_float(score.get("last_base_chips")),
            "last_base_mult": _safe_float(score.get("last_base_mult"), 1.0),
        },
        "zones": {
            "hand": hand,
            "discard": discard,
            "played": played,
            "deck_count": len(deck),
        },
        "jokers": state_jokers(state),
        "blinds": blinds,
        "timeline": list(timeline[-12:]),
    }

