from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


PHASE_LABELS = {
    "SELECTING_HAND": "选择手牌",
    "ROUND_EVAL": "回合结算",
    "BLIND_SELECT": "选择盲注",
    "SHOP": "商店",
    "MENU": "菜单",
    "GAME_OVER": "游戏结束",
    "HAND_PLAYED": "出牌过渡",
    "DRAW_TO_HAND": "补牌过渡",
    "NEW_ROUND": "新回合",
}

BLIND_LABELS = {
    "small": "小盲注",
    "big": "大盲注",
    "boss": "Boss 盲注",
}

SUIT_META = {
    "C": {"symbol": "♣", "label": "梅花", "tone": "black"},
    "D": {"symbol": "♦", "label": "方片", "tone": "red"},
    "H": {"symbol": "♥", "label": "红桃", "tone": "red"},
    "S": {"symbol": "♠", "label": "黑桃", "tone": "black"},
}


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


def phase_label(phase: str) -> str:
    return PHASE_LABELS.get(str(phase or "").upper(), str(phase or "未知阶段"))


def blind_label(blind: str) -> str:
    return BLIND_LABELS.get(str(blind or "").lower(), str(blind or "未知盲注"))


def card_view(card: dict[str, Any], *, index: int | None = None) -> dict[str, Any]:
    rank = _rank_label(card)
    suit = _suit_label(card)
    key = str(card.get("key") or "").strip()
    suit_meta = SUIT_META.get(suit, {"symbol": suit, "label": suit, "tone": "black"})
    modifier_tags = [str(x) for x in (card.get("modifier_tags") or card.get("modifier") or []) if str(x).strip()]
    state_tags = [str(x) for x in (card.get("state_tags") or card.get("state") or []) if str(x).strip()]
    return {
        "index": index,
        "card_id": str(card.get("card_id") or card.get("uid") or key or f"card-{index}"),
        "rank": rank,
        "suit": suit,
        "suit_symbol": suit_meta["symbol"],
        "suit_label": suit_meta["label"],
        "suit_text": suit_meta["label"],
        "suit_tone": suit_meta["tone"],
        "label": f"{rank}{suit_meta['symbol']}",
        "key": key,
        "modifier_tags": modifier_tags,
        "state_tags": state_tags,
        "effect_text": str(card.get("effect_text") or (card.get("value") or {}).get("effect") or ""),
        "status_text": " / ".join(modifier_tags + state_tags) if (modifier_tags or state_tags) else "标准牌",
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
        "blind_label": blind_label(str(round_info.get("blind") or "")),
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
        "blind_before": before_resources["blind_label"],
        "blind_after": after_resources["blind_label"],
        "phase_before": phase_label(str(before.get("state") or "UNKNOWN")),
        "phase_after": phase_label(str(after.get("state") or "UNKNOWN")),
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
            return "未选择牌组"
        cards_text = "、".join(card["label"] for card in selected)
        verb = "打出" if action_type == "PLAY" else "弃掉"
        return f"{verb} {cards_text}"
    if action_type == "SELECT":
        return f"选择第 {int(action.get('index', 0)) + 1} 个盲注"
    if action_type == "CASH_OUT":
        return "领取奖励并结算"
    if action_type == "NEXT_ROUND":
        return "进入下一回合"
    if action_type == "START":
        return "重新开始场景"
    return action_type.replace("_", " ").title()


def _risk_meta(state: dict[str, Any], scenario: dict[str, Any], jokers: list[dict[str, Any]]) -> dict[str, str]:
    resources = resources_view(state)
    score_now = max(0.0, _safe_float((state.get("score") or {}).get("chips")))
    target = max(1.0, _safe_float((state.get("score") or {}).get("target_chips"), 1.0))
    gap_ratio = max(0.0, (target - score_now) / target)
    hands_left = int(resources["hands_left"])
    discards_left = int(resources["discards_left"])
    high_pressure = hands_left <= 1 or discards_left <= 1
    has_joker = bool(jokers)

    if high_pressure and gap_ratio > 0.2:
        risk_level = "high"
        risk_label = "高风险"
        risk_hint = "资源偏紧，当前选择会显著影响能否过盲。"
    elif gap_ratio > 0.45 or hands_left <= 1 or discards_left <= 1:
        risk_level = "medium"
        risk_label = "中风险"
        risk_hint = "需要兼顾当前收益和下一手机会。"
    else:
        risk_level = "low"
        risk_label = "低风险"
        risk_hint = "当前资源较充足，适合展示基础决策逻辑。"

    focus_text = str(scenario.get("focus") or "")
    if has_joker or "Joker" in focus_text or "协同" in focus_text:
        decision_label = "Joker 协同"
    elif hands_left <= 1 and discards_left <= 1:
        decision_label = "资源抉择"
    elif discards_left > 0 and hands_left <= 1:
        decision_label = "弃牌转折"
    else:
        decision_label = "出牌选择"

    return {
        "risk_level": risk_level,
        "risk_label": risk_label,
        "risk_hint": risk_hint,
        "decision_label": decision_label,
    }


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
    jokers = state_jokers(state)
    resources = resources_view(state)
    score = state.get("score") if isinstance(state.get("score"), dict) else {}
    blinds = state.get("blinds") if isinstance(state.get("blinds"), dict) else {}
    meta = _risk_meta(state, scenario, jokers)
    return {
        "timestamp": now_iso(),
        "phase": str(state.get("state") or "UNKNOWN"),
        "phase_label": phase_label(str(state.get("state") or "UNKNOWN")),
        "phase_text": phase_label(str(state.get("state") or "UNKNOWN")),
        "scenario": scenario,
        "mode": mode,
        "mode_label": "自动演示" if mode == "autoplay" else "手动单步",
        "policy": policy,
        "policy_label": "模型" if policy == "model" else "启发式",
        "model_name": model_name,
        "meta": {
            "risk_level": meta["risk_level"],
            "risk_label": meta["risk_label"],
            "risk_hint": meta["risk_hint"],
            "decision_label": meta["decision_label"],
            "starter_hint": str(scenario.get("talk_track") or scenario.get("summary") or ""),
        },
        "insights": {
            "risk_level": meta["risk_label"],
            "risk_code": meta["risk_level"],
            "decision_type": meta["decision_label"],
            "decision_hint": str(scenario.get("talk_track") or scenario.get("summary") or ""),
            "risk_reason": meta["risk_hint"],
            "score_gap": max(0.0, _safe_float(score.get("target_chips")) - _safe_float(score.get("chips"))),
            "tags": [meta["risk_label"], meta["decision_label"], *list(scenario.get("tags") or [])],
        },
        "resources": resources,
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
        "jokers": jokers,
        "blinds": blinds,
        "timeline": list(timeline[-16:]),
    }
