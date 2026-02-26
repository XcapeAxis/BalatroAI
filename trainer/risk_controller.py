from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CFG: dict[str, Any] = {
    "weights": {
        "uncertainty": 0.35,
        "disagreement": 0.30,
        "value_gap": 0.20,
        "illegal": 0.10,
        "ood": 0.05,
    },
    "hand": {
        "medium": 0.40,
        "high": 0.70,
        "low_risk_policy": "rl",
        "medium_risk_policy": "hybrid",
        "high_risk_policy": "search",
        "extreme_risk_policy": "heuristic",
    },
    "shop": {
        "medium": 0.45,
        "high": 0.75,
        "low_risk_policy": "rl",
        "medium_risk_policy": "hybrid",
        "high_risk_policy": "heuristic",
        "extreme_risk_policy": "heuristic",
    },
    "ood_context_abs_threshold": 6.0,
    "extreme_risk_threshold": 0.90,
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_config(path: str | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_CFG)
    if not path:
        return cfg
    p = Path(path)
    if not p.exists():
        return cfg
    text = p.read_text(encoding="utf-8")
    payload: dict[str, Any] = {}
    try:
        payload = json.loads(text)
    except Exception:
        try:
            import yaml  # type: ignore

            loaded = yaml.safe_load(text)
            if isinstance(loaded, dict):
                payload = loaded
        except Exception:
            payload = {}
    if not isinstance(payload, dict):
        return cfg
    return _deep_merge(cfg, payload)


@dataclass
class PolicySignal:
    name: str
    top1: int | None
    confidence: float
    value: float | None
    legal: bool


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _max_disagreement(signals: list[PolicySignal]) -> float:
    ids = [s.top1 for s in signals if s.top1 is not None and s.legal]
    if len(ids) <= 1:
        return 0.0
    uniq = len(set(ids))
    return _clip01((uniq - 1.0) / max(1.0, len(ids) - 1.0))


def _value_gap(signals: list[PolicySignal]) -> float:
    vals = [float(s.value) for s in signals if s.value is not None]
    if len(vals) <= 1:
        return 0.0
    vmax = max(vals)
    vmin = min(vals)
    # Smooth normalization; assumes value-head magnitudes are small.
    return _clip01(abs(vmax - vmin) / (1.0 + abs(vmax) + abs(vmin)))


def _ood_proxy(context: list[float], threshold: float) -> float:
    if not context:
        return 0.0
    if threshold <= 0:
        threshold = 1.0
    max_abs = max(abs(float(x)) for x in context)
    return _clip01((max_abs - threshold) / threshold)


def select_policy(
    *,
    phase_group: str,
    signals: list[PolicySignal],
    context: list[float],
    cfg: dict[str, Any],
    available_policies: set[str],
) -> dict[str, Any]:
    weights = cfg.get("weights") if isinstance(cfg.get("weights"), dict) else {}
    section = cfg.get(phase_group) if isinstance(cfg.get(phase_group), dict) else {}

    legal_signals = [s for s in signals if s.legal]
    if not legal_signals:
        return {
            "selected_policy": "heuristic" if "heuristic" in available_policies else sorted(available_policies)[0],
            "risk_score": 1.0,
            "fallback_reason": "no_legal_signal",
            "diagnostics": {
                "disagreement": 1.0,
                "uncertainty": 1.0,
                "value_gap": 1.0,
                "illegal_ratio": 1.0,
                "ood_score": 0.0,
            },
        }

    best_conf = max(float(s.confidence) for s in legal_signals)
    uncertainty = _clip01(1.0 - best_conf)
    disagreement = _max_disagreement(legal_signals)
    value_gap = _value_gap(legal_signals)
    illegal_ratio = _clip01(sum(1 for s in signals if not s.legal) / max(1.0, float(len(signals))))
    ood = _ood_proxy(context, float(cfg.get("ood_context_abs_threshold") or 6.0))

    risk = (
        float(weights.get("uncertainty") or 0.0) * uncertainty
        + float(weights.get("disagreement") or 0.0) * disagreement
        + float(weights.get("value_gap") or 0.0) * value_gap
        + float(weights.get("illegal") or 0.0) * illegal_ratio
        + float(weights.get("ood") or 0.0) * ood
    )
    risk = _clip01(risk)

    medium = float(section.get("medium") or 0.45)
    high = float(section.get("high") or 0.75)
    extreme = float(cfg.get("extreme_risk_threshold") or 0.90)

    if risk >= extreme:
        chosen = str(section.get("extreme_risk_policy") or "heuristic")
        reason = "extreme_risk"
    elif risk >= high:
        chosen = str(section.get("high_risk_policy") or "heuristic")
        reason = "high_risk"
    elif risk >= medium:
        chosen = str(section.get("medium_risk_policy") or "hybrid")
        reason = "medium_risk"
    else:
        chosen = str(section.get("low_risk_policy") or "rl")
        reason = "low_risk"

    fallback = ""
    if chosen not in available_policies:
        fallback = f"policy_unavailable:{chosen}"
        for candidate in ("rl", "hybrid", "pv", "search", "heuristic", "bc"):
            if candidate in available_policies:
                chosen = candidate
                break

    return {
        "selected_policy": chosen,
        "risk_score": risk,
        "fallback_reason": fallback or reason,
        "diagnostics": {
            "disagreement": disagreement,
            "uncertainty": uncertainty,
            "value_gap": value_gap,
            "illegal_ratio": illegal_ratio,
            "ood_score": ood,
            "signals": [
                {
                    "name": s.name,
                    "top1": s.top1,
                    "confidence": float(s.confidence),
                    "value": None if s.value is None else float(s.value),
                    "legal": bool(s.legal),
                }
                for s in signals
            ],
        },
    }
