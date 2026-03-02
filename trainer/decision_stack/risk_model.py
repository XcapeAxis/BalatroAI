from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .context_features import ContextFeatures


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


@dataclass(frozen=True)
class RiskEstimate:
    risk_score: float
    risk_bucket: str
    components: dict[str, float]
    weights: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_score": float(self.risk_score),
            "risk_bucket": str(self.risk_bucket),
            "components": dict(self.components),
            "weights": dict(self.weights),
        }


DEFAULT_WEIGHTS: dict[str, float] = {
    "score_gap": 0.42,
    "resource_burn": 0.23,
    "joker_volatility": 0.20,
    "remaining_outs": 0.15,
}

DEFAULT_THRESHOLDS: dict[str, float] = {
    "low_max": 0.35,
    "medium_max": 0.68,
}


def _pick_bucket(score: float, thresholds: dict[str, float]) -> str:
    low_max = _safe_float(thresholds.get("low_max"), DEFAULT_THRESHOLDS["low_max"])
    medium_max = _safe_float(thresholds.get("medium_max"), DEFAULT_THRESHOLDS["medium_max"])
    if score < low_max:
        return "low"
    if score < medium_max:
        return "medium"
    return "high"


def estimate_risk(
    features: ContextFeatures,
    config: dict[str, Any] | None = None,
) -> RiskEstimate:
    cfg = config or {}
    weights_cfg = cfg.get("weights") if isinstance(cfg.get("weights"), dict) else {}
    thresholds_cfg = cfg.get("thresholds") if isinstance(cfg.get("thresholds"), dict) else {}

    weights = {
        "score_gap": _safe_float(weights_cfg.get("score_gap"), DEFAULT_WEIGHTS["score_gap"]),
        "resource_burn": _safe_float(weights_cfg.get("resource_burn"), DEFAULT_WEIGHTS["resource_burn"]),
        "joker_volatility": _safe_float(weights_cfg.get("joker_volatility"), DEFAULT_WEIGHTS["joker_volatility"]),
        "remaining_outs": _safe_float(weights_cfg.get("remaining_outs"), DEFAULT_WEIGHTS["remaining_outs"]),
    }
    total_w = sum(max(0.0, float(v)) for v in weights.values())
    if total_w <= 0.0:
        weights = dict(DEFAULT_WEIGHTS)
        total_w = sum(weights.values())
    for key in list(weights.keys()):
        weights[key] = max(0.0, float(weights[key])) / total_w

    # remaining_outs is protective when high, riskier when low.
    outs_risk = _clamp01(1.0 - float(features.remaining_outs_ratio))
    components = {
        "score_gap": _clamp01(features.score_gap_ratio),
        "resource_burn": _clamp01(features.resource_burn_rate),
        "joker_volatility": _clamp01(features.joker_volatility),
        "remaining_outs": outs_risk,
    }

    raw_score = 0.0
    for key, weight in weights.items():
        raw_score += float(weight) * float(components[key])

    # Phase-aware bias: selecting hand is slightly higher tactical risk, shop is lower tactical risk.
    phase = str(features.phase).upper()
    phase_bias = 0.0
    if phase == "SELECTING_HAND":
        phase_bias = 0.035
    elif phase in {"SHOP", "SHOPPING"}:
        phase_bias = -0.02

    # Ante-based pressure grows gradually after ante 4.
    ante_bias = 0.0
    if int(features.ante_level) >= 5:
        ante_bias = min(0.10, 0.015 * (int(features.ante_level) - 4))

    risk_score = _clamp01(raw_score + phase_bias + ante_bias)
    bucket = _pick_bucket(risk_score, thresholds_cfg)

    return RiskEstimate(
        risk_score=risk_score,
        risk_bucket=bucket,
        components=components,
        weights=weights,
    )

