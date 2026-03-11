from __future__ import annotations

from typing import Any


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _extract_state(sample: dict[str, Any]) -> dict[str, Any]:
    """Normalize different call sites into a state dict."""
    state = sample.get("state")
    if isinstance(state, dict):
        return state
    return sample


def _infer_ante(state: dict[str, Any]) -> int:
    ante = _safe_int(state.get("ante_num"), 0)
    if ante > 0:
        return ante
    round_num = _safe_int(state.get("round_num"), 0)
    if round_num > 0:
        return max(1, ((round_num - 1) // 3) + 1)
    round_block = _as_dict(state.get("round"))
    if round_block:
        round_num = _safe_int(round_block.get("round_num"), 0)
        if round_num > 0:
            return max(1, ((round_num - 1) // 3) + 1)
    return 0


def infer_slice_stage(sample: dict[str, Any]) -> str:
    """Classify episode progression as early/mid/late/unknown."""
    state = _extract_state(sample)
    ante = _infer_ante(state)
    if ante <= 0:
        round_num = _safe_int(state.get("round_num"), 0)
        if round_num <= 0:
            round_num = _safe_int(_as_dict(state.get("round")).get("round_num"), 0)
        if round_num <= 0:
            return "unknown"
        if round_num <= 4:
            return "early"
        if round_num <= 10:
            return "mid"
        return "late"
    if ante <= 2:
        return "early"
    if ante <= 5:
        return "mid"
    return "late"


def infer_slice_resource_pressure(sample: dict[str, Any]) -> str:
    """Classify resource pressure as low/medium/high/unknown."""
    state = _extract_state(sample)
    round_block = _as_dict(state.get("round"))
    hands_left = _safe_int(round_block.get("hands_left"), _safe_int(state.get("hands_left"), -1))
    discards_left = _safe_int(round_block.get("discards_left"), _safe_int(state.get("discards_left"), -1))
    money = _safe_float(state.get("money"), float("nan"))
    chips_gap = _safe_float(state.get("chips_gap"), float("nan"))
    blind_block = _as_dict(state.get("blind"))
    blind_chips = _safe_float(blind_block.get("chips"), float("nan"))
    score_block = _as_dict(state.get("score"))
    cur_chips = _safe_float(score_block.get("chips"), float("nan"))

    if chips_gap != chips_gap and blind_chips == blind_chips and cur_chips == cur_chips and blind_chips > 0.0:
        chips_gap = max(0.0, (blind_chips - cur_chips) / blind_chips)

    observed = 0
    if money == money:
        observed += 1
    if hands_left >= 0:
        observed += 1
    if discards_left >= 0:
        observed += 1
    if chips_gap == chips_gap:
        observed += 1
    if observed <= 0:
        return "unknown"

    high_reasons = 0
    low_reasons = 0
    if money == money:
        if money <= 3.0:
            high_reasons += 1
        elif money >= 15.0:
            low_reasons += 1
    if hands_left >= 0:
        if hands_left <= 1:
            high_reasons += 1
        elif hands_left >= 3:
            low_reasons += 1
    if discards_left >= 0:
        if discards_left <= 0:
            high_reasons += 1
        elif discards_left >= 2:
            low_reasons += 1
    if chips_gap == chips_gap:
        if chips_gap >= 0.7:
            high_reasons += 1
        elif chips_gap <= 0.2:
            low_reasons += 1

    if high_reasons >= 2 or (high_reasons >= 1 and observed <= 2):
        return "high"
    if low_reasons >= 2:
        return "low"
    return "medium"


def infer_slice_action_type(sample: dict[str, Any]) -> str:
    """Normalize action semantics into a small fixed slice vocabulary."""
    action_type = str(sample.get("action_type") or "").upper()
    phase = str(sample.get("phase") or "").upper()

    if action_type in {"PLAY"}:
        return "play"
    if action_type in {"DISCARD"}:
        return "discard"
    if action_type in {"CONSUMABLE_USE", "USE_CONSUMABLE"}:
        return "consumable"
    if action_type in {"SHOP_REROLL", "SHOP_BUY", "SELL", "NEXT_ROUND"} or phase == "SHOP":
        return "shop"
    if action_type in {
        "WAIT",
        "START_ROUND",
        "END_ROUND",
        "PACK_OPEN",
        "MOVE_HAND_CARD",
        "MOVE_JOKER",
        "SWAP_HAND_CARD",
        "SWAP_JOKER",
        "REORDER_HAND",
        "REORDER_JOKERS",
    }:
        return "transition"
    if "PACK" in phase or "BOOSTER" in phase:
        return "transition"
    return "unknown"


def infer_slice_position_sensitive(sample: dict[str, Any]) -> bool | str:
    """Detect position sensitivity; returns bool or 'unknown' when input is sparse."""
    state = _extract_state(sample)
    action_type = str(sample.get("action_type") or "").upper()
    if action_type in {
        "MOVE_HAND_CARD",
        "MOVE_JOKER",
        "SWAP_HAND_CARD",
        "SWAP_JOKER",
        "REORDER_HAND",
        "REORDER_JOKERS",
    }:
        return True

    tags = state.get("tags") if isinstance(state.get("tags"), list) else []
    if tags:
        joined = " ".join(str(x).lower() for x in tags)
        if any(k in joined for k in ("position", "order", "left", "right", "swap", "reorder")):
            return True

    raw_jokers = state.get("jokers")
    if isinstance(raw_jokers, list) and not raw_jokers:
        return False
    jokers = raw_jokers if isinstance(raw_jokers, list) else []
    if jokers:
        for joker in jokers:
            if not isinstance(joker, dict):
                continue
            key = str(joker.get("key") or joker.get("joker_id") or "").lower()
            if any(k in key for k in ("photograph", "hanging_chad", "blueprint", "brainstorm", "hack")):
                return True
        return False

    return "unknown"


def infer_slice_stateful_joker_present(sample: dict[str, Any]) -> bool | str:
    """Detect whether likely stateful jokers are present."""
    state = _extract_state(sample)
    raw_jokers = state.get("jokers")
    if not isinstance(raw_jokers, list):
        return "unknown"
    jokers = raw_jokers
    if not jokers:
        return False
    for joker in jokers:
        if not isinstance(joker, dict):
            continue
        key = str(joker.get("key") or joker.get("joker_id") or "").lower()
        if any(k in key for k in ("egg", "perkeo", "campfire", "green_joker", "square_joker", "ride_the_bus")):
            return True
        if joker.get("counter") is not None or joker.get("state") is not None or joker.get("chips") is not None:
            return True
    return False


def compute_slice_labels(sample: dict[str, Any]) -> dict[str, Any]:
    """Compute the unified P41 slice labels for replay + arena."""
    return {
        "slice_stage": infer_slice_stage(sample),
        "slice_resource_pressure": infer_slice_resource_pressure(sample),
        "slice_action_type": infer_slice_action_type(sample),
        "slice_position_sensitive": infer_slice_position_sensitive(sample),
        "slice_stateful_joker_present": infer_slice_stateful_joker_present(sample),
    }


def as_legacy_ante_bucket(slice_stage: str) -> str:
    token = str(slice_stage or "").lower()
    if token == "early":
        return "ante_1_2"
    if token == "mid":
        return "ante_3_4"
    if token == "late":
        return "ante_5_plus"
    return "ante_unknown"


def as_legacy_risk_bucket(slice_resource_pressure: str) -> str:
    token = str(slice_resource_pressure or "").lower()
    if token == "high":
        return "resource_tight"
    if token == "low":
        return "resource_relaxed"
    if token == "medium":
        return "resource_balanced"
    return "resource_unknown"


def as_legacy_action_bucket(slice_action_type: str) -> str:
    token = str(slice_action_type or "").lower()
    if token == "play":
        return "PLAY"
    if token == "discard":
        return "DISCARD"
    if token == "shop":
        return "SHOP"
    if token == "consumable":
        return "CONSUMABLE"
    if token == "transition":
        return "OTHER"
    return "OTHER"

