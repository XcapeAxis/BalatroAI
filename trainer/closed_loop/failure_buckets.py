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
        }

    if rounds_survived <= max(1, int(high_risk_round_threshold)) and (
        slice_stage == "early" or total_score <= low_score_threshold
    ):
        return {
            "failure_bucket": "early_collapse",
            "bucket_reason": f"rounds_survived<={max(1, int(high_risk_round_threshold))}",
        }

    if position_sensitive not in {"", "unknown"}:
        return {
            "failure_bucket": "position_sensitive_misplay",
            "bucket_reason": f"slice_position_sensitive:{position_sensitive}",
        }

    if stateful_joker_present not in {"", "unknown", "absent", "false", "none"}:
        return {
            "failure_bucket": "stateful_joker_misplay",
            "bucket_reason": f"slice_stateful_joker_present:{stateful_joker_present}",
        }

    if action_type == "discard":
        return {
            "failure_bucket": "discard_mismanagement",
            "bucket_reason": "discard_dominant_failure",
        }

    if action_type == "shop":
        return {
            "failure_bucket": "shop_or_economy_misallocation",
            "bucket_reason": "shop_dominant_failure",
        }

    if resource_pressure == "high" or risk_label == "resource_tight":
        if action_type == "play" and rounds_survived <= max(2, int(high_risk_round_threshold) + 1):
            return {
                "failure_bucket": "risk_overcommit",
                "bucket_reason": "play_under_resource_tight",
            }
        return {
            "failure_bucket": "resource_pressure_misplay",
            "bucket_reason": f"resource_pressure:{resource_pressure or risk_label}",
        }

    if resource_pressure == "low" or risk_label == "resource_relaxed":
        return {
            "failure_bucket": "risk_undercommit",
            "bucket_reason": f"resource_pressure:{resource_pressure or risk_label}",
        }

    return {
        "failure_bucket": "low_score_survival",
        "bucket_reason": "fallback_low_score_survival",
    }


def scarce_failure_buckets(counts: dict[str, int] | Counter[str], *, threshold: int = 2) -> list[str]:
    normalized = {str(bucket): _safe_int(count, 0) for bucket, count in dict(counts).items()}
    scarce: list[str] = []
    for bucket in KNOWN_FAILURE_BUCKETS:
        if normalized.get(bucket, 0) < max(1, int(threshold)):
            scarce.append(bucket)
    return scarce
