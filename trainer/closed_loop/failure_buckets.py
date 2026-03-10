from __future__ import annotations

from collections import Counter
from typing import Any

KNOWN_FAILURE_BUCKETS: tuple[str, ...] = (
    "early_collapse",
    "resource_pressure_misplay",
    "discard_mismanagement",
    "position_sensitive_misplay",
    "stateful_joker_misplay",
    "shop_or_economy_misallocation",
    "risk_overcommit",
    "risk_undercommit",
    "low_score_survival",
    "invalid_or_wasted_decision",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _normalized_label(value: Any) -> str:
    return str(value or "").strip().lower()


def _bucket_count_map(row: dict[str, Any], category: str) -> dict[str, int]:
    bucket_counts = row.get("bucket_counts") if isinstance(row.get("bucket_counts"), dict) else {}
    payload = bucket_counts.get(category) if isinstance(bucket_counts.get(category), dict) else {}
    out: dict[str, int] = {}
    for label, count in payload.items():
        token = str(label).strip()
        if not token:
            continue
        out[token] = _safe_int(count, 0)
    return out


def _dominant_bucket_label(row: dict[str, Any], category: str, *, fallback_key: str = "") -> str:
    direct = row.get("slice_labels") if isinstance(row.get("slice_labels"), dict) else {}
    if fallback_key:
        token = _normalized_label(direct.get(fallback_key))
        if token:
            return token
    counts = _bucket_count_map(row, category)
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0].strip().lower()


def classify_failure_bucket(
    *,
    row: dict[str, Any],
    failure_types: set[str],
    high_risk_round_threshold: int,
    low_score_threshold: float,
) -> dict[str, Any]:
    rounds_survived = _safe_int(row.get("rounds_survived"), 0)
    total_score = _safe_float(row.get("total_score"), 0.0)
    invalid_rate = _safe_float(row.get("invalid_action_rate"), 0.0)
    timeout_rate = _safe_float(row.get("timeout_rate"), 0.0)
    slice_stage = _dominant_bucket_label(row, "slice_stage", fallback_key="slice_stage")
    resource_pressure = _dominant_bucket_label(row, "slice_resource_pressure", fallback_key="slice_resource_pressure")
    action_type = _dominant_bucket_label(row, "slice_action_type", fallback_key="slice_action_type")
    position_sensitive = _dominant_bucket_label(row, "slice_position_sensitive", fallback_key="slice_position_sensitive")
    stateful_joker_present = _dominant_bucket_label(
        row,
        "slice_stateful_joker_present",
        fallback_key="slice_stateful_joker_present",
    )
    risk_label = _dominant_bucket_label(row, "risk")

    invalid_like = {"invalid_action", "timeout", "execution_error", "episode_failure_status"} & set(failure_types)
    if invalid_like or invalid_rate > 0.0 or timeout_rate > 0.0:
        return {
            "failure_bucket": "invalid_or_wasted_decision",
            "bucket_reason": "invalid_or_execution_signal",
            "failure_bucket_candidates": ["invalid_or_wasted_decision"],
            "failure_bucket_signals": ["invalid_or_execution_signal"],
        }

    candidate_signals: list[dict[str, Any]] = []

    def _append_signal(bucket: str, reason: str, priority: int) -> None:
        candidate_signals.append(
            {
                "failure_bucket": str(bucket),
                "bucket_reason": str(reason),
                "priority": int(priority),
            }
        )

    if position_sensitive not in {"", "unknown"}:
        _append_signal("position_sensitive_misplay", f"slice_position_sensitive:{position_sensitive}", 100)

    if stateful_joker_present not in {"", "unknown", "absent", "false", "none"}:
        _append_signal("stateful_joker_misplay", f"slice_stateful_joker_present:{stateful_joker_present}", 96)

    if action_type == "shop":
        _append_signal("shop_or_economy_misallocation", "shop_dominant_failure", 90)

    if action_type == "discard":
        _append_signal("discard_mismanagement", "discard_dominant_failure", 88)

    if resource_pressure == "high" or risk_label == "resource_tight":
        if action_type == "play" and (
            rounds_survived <= max(2, int(high_risk_round_threshold) + 1)
            or total_score <= low_score_threshold
        ):
            _append_signal("risk_overcommit", "play_under_resource_tight", 86)
        _append_signal("resource_pressure_misplay", f"resource_pressure:{resource_pressure or risk_label}", 82)

    if resource_pressure in {"low", "medium"} or risk_label == "resource_relaxed":
        if action_type == "play":
            _append_signal(
                "risk_undercommit",
                f"play_without_conversion:{resource_pressure or risk_label or 'balanced'}",
                78,
            )
        elif total_score <= low_score_threshold:
            _append_signal(
                "risk_undercommit",
                f"resource_pressure:{resource_pressure or risk_label or 'balanced'}",
                72,
            )

    if rounds_survived <= max(1, int(high_risk_round_threshold)) and (
        slice_stage == "early" or total_score <= low_score_threshold
    ):
        _append_signal("early_collapse", f"rounds_survived<={max(1, int(high_risk_round_threshold))}", 65)

    if total_score <= low_score_threshold:
        _append_signal("low_score_survival", "fallback_low_score_survival", 40)

    if not candidate_signals:
        candidate_signals.append(
            {
                "failure_bucket": "low_score_survival",
                "bucket_reason": "fallback_low_score_survival",
                "priority": 1,
            }
        )

    candidate_signals.sort(
        key=lambda item: (
            -_safe_int(item.get("priority"), 0),
            str(item.get("failure_bucket") or ""),
        )
    )
    chosen = dict(candidate_signals[0])
    chosen["failure_bucket_candidates"] = [
        str(item.get("failure_bucket") or "") for item in candidate_signals if str(item.get("failure_bucket") or "").strip()
    ]
    chosen["failure_bucket_signals"] = [
        str(item.get("bucket_reason") or "") for item in candidate_signals if str(item.get("bucket_reason") or "").strip()
    ]
    chosen.pop("priority", None)
    return chosen


def scarce_failure_buckets(counts: dict[str, int] | Counter[str], *, threshold: int = 2) -> list[str]:
    normalized = {str(bucket): _safe_int(count, 0) for bucket, count in dict(counts).items()}
    scarce: list[str] = []
    for bucket in KNOWN_FAILURE_BUCKETS:
        if normalized.get(bucket, 0) < max(1, int(threshold)):
            scarce.append(bucket)
    return scarce
