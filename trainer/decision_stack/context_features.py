from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _rank_to_chip(rank_token: Any) -> float:
    token = str(rank_token).strip().upper()
    if token in {"A"}:
        return 11.0
    if token in {"K", "Q", "J", "T"}:
        return 10.0
    try:
        n = int(token)
        if 2 <= n <= 10:
            return float(n)
    except Exception:
        pass
    return 0.0


@dataclass(frozen=True)
class ContextFeatures:
    phase: str
    ante_level: int
    hand_strength_proxy: float
    economy_level: float
    remaining_discards: int
    joker_synergy_score: float
    score_gap_ratio: float
    resource_burn_rate: float
    joker_volatility: float
    remaining_outs_ratio: float
    budget_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _phase_token(state: dict[str, Any]) -> str:
    raw = str(state.get("state") or state.get("phase") or "UNKNOWN").strip().upper()
    if not raw:
        return "UNKNOWN"
    return raw


def _ante_level(state: dict[str, Any]) -> int:
    round_info = state.get("round") if isinstance(state.get("round"), dict) else {}
    value = round_info.get("ante", state.get("ante", 1))
    try:
        ante = int(value)
    except Exception:
        ante = 1
    return max(1, ante)


def _hand_strength_proxy(state: dict[str, Any]) -> float:
    if "hand_strength_proxy" in state:
        return _clamp01(_safe_float(state.get("hand_strength_proxy"), 0.5))
    hand = (state.get("hand") or {}).get("cards") or []
    if not hand:
        return 0.35
    values: list[float] = []
    for card in hand:
        if not isinstance(card, dict):
            continue
        rank = card.get("rank")
        if rank is None:
            rank = card.get("r")
        values.append(_rank_to_chip(rank))
    if not values:
        return 0.35
    return _clamp01((sum(values) / max(1.0, len(values))) / 11.0)


def _economy_level(state: dict[str, Any]) -> float:
    if "economy_level" in state:
        return _clamp01(_safe_float(state.get("economy_level"), 0.5))
    money = _safe_float(state.get("money"), 0.0)
    return _clamp01(money / 30.0)


def _remaining_discards(state: dict[str, Any]) -> int:
    round_info = state.get("round") if isinstance(state.get("round"), dict) else {}
    value = round_info.get("discards_left", state.get("discards_left", 0))
    try:
        rem = int(value)
    except Exception:
        rem = 0
    return max(0, rem)


def _joker_synergy_score(state: dict[str, Any]) -> float:
    if "joker_synergy_score" in state:
        return _clamp01(_safe_float(state.get("joker_synergy_score"), 0.5))
    jokers = state.get("jokers")
    if not isinstance(jokers, list) or not jokers:
        return 0.2
    names = [str(j.get("name") or j.get("id") or "").strip().lower() for j in jokers if isinstance(j, dict)]
    uniq = len({x for x in names if x})
    density = len(names) / 5.0
    diversity = (uniq / len(names)) if names else 0.0
    return _clamp01(0.30 + 0.35 * _clamp01(density) + 0.35 * diversity)


def _score_gap_ratio(state: dict[str, Any]) -> float:
    if "score_gap_ratio" in state:
        return _clamp01(_safe_float(state.get("score_gap_ratio"), 0.5))
    round_info = state.get("round") if isinstance(state.get("round"), dict) else {}
    blind = state.get("blind") if isinstance(state.get("blind"), dict) else {}
    chips = _safe_float(round_info.get("chips"), 0.0)
    required = _safe_float(blind.get("chips_required", blind.get("required_chips", 0.0)), 0.0)
    if required <= 0.0:
        return 0.5
    gap = max(0.0, required - chips)
    return _clamp01(gap / max(1.0, required))


def _resource_burn_rate(state: dict[str, Any]) -> float:
    if "resource_burn_rate" in state:
        return _clamp01(_safe_float(state.get("resource_burn_rate"), 0.5))
    round_info = state.get("round") if isinstance(state.get("round"), dict) else {}
    hands_left = _safe_float(round_info.get("hands_left"), 0.0)
    hands_total = _safe_float(round_info.get("hands_total", 4.0), 4.0)
    discards_left = _safe_float(round_info.get("discards_left"), 0.0)
    discards_total = _safe_float(round_info.get("discards_total", 3.0), 3.0)
    hand_burn = 1.0 - _clamp01(hands_left / max(1.0, hands_total))
    discard_burn = 1.0 - _clamp01(discards_left / max(1.0, discards_total))
    return _clamp01((0.6 * hand_burn) + (0.4 * discard_burn))


def _joker_volatility(state: dict[str, Any], synergy_score: float) -> float:
    if "joker_volatility" in state:
        return _clamp01(_safe_float(state.get("joker_volatility"), 0.5))
    jokers = state.get("jokers")
    if not isinstance(jokers, list) or not jokers:
        return _clamp01(0.40 - 0.20 * synergy_score)
    vals: list[float] = []
    for joker in jokers:
        if isinstance(joker, dict) and "volatility" in joker:
            vals.append(_clamp01(_safe_float(joker.get("volatility"), 0.5)))
    if vals:
        return _clamp01(sum(vals) / len(vals))
    return _clamp01(0.55 - 0.25 * synergy_score)


def _remaining_outs_ratio(state: dict[str, Any]) -> float:
    if "remaining_outs_ratio" in state:
        return _clamp01(_safe_float(state.get("remaining_outs_ratio"), 0.5))
    rem = _safe_float(state.get("remaining_outs"), -1.0)
    total = _safe_float(state.get("total_outs"), -1.0)
    if rem >= 0.0 and total > 0.0:
        return _clamp01(rem / total)
    # Use discards as a weak proxy when explicit outs are unavailable.
    discards = _remaining_discards(state)
    return _clamp01((discards + 1.0) / 5.0)


def _budget_ms(state: dict[str, Any], default_budget_ms: float) -> float:
    if "budget_ms" in state:
        return max(1.0, _safe_float(state.get("budget_ms"), default_budget_ms))
    if "search_budget_ms" in state:
        return max(1.0, _safe_float(state.get("search_budget_ms"), default_budget_ms))
    return max(1.0, float(default_budget_ms))


def extract_context_features(state: dict[str, Any], default_budget_ms: float = 15.0) -> ContextFeatures:
    synergy = _joker_synergy_score(state)
    return ContextFeatures(
        phase=_phase_token(state),
        ante_level=_ante_level(state),
        hand_strength_proxy=_hand_strength_proxy(state),
        economy_level=_economy_level(state),
        remaining_discards=_remaining_discards(state),
        joker_synergy_score=synergy,
        score_gap_ratio=_score_gap_ratio(state),
        resource_burn_rate=_resource_burn_rate(state),
        joker_volatility=_joker_volatility(state, synergy_score=synergy),
        remaining_outs_ratio=_remaining_outs_ratio(state),
        budget_ms=_budget_ms(state, default_budget_ms=default_budget_ms),
    )

