from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return float(default)
    if math.isnan(result) or math.isinf(result):
        return float(default)
    return result


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def numeric_summary(values: list[float]) -> dict[str, float]:
    rows = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    if not rows:
        return {"count": 0, "mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(rows),
        "mean": float(statistics.fmean(rows)),
        "std": float(statistics.pstdev(rows)) if len(rows) > 1 else 0.0,
        "median": float(statistics.median(rows)),
        "min": float(min(rows)),
        "max": float(max(rows)),
    }


def distribution_rows(counter: Counter[str], *, key_name: str) -> list[dict[str, Any]]:
    total = max(1, sum(counter.values()))
    return [
        {key_name: key, "count": int(value), "ratio": float(value) / total}
        for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def is_catastrophic_record(record: dict[str, Any], catastrophic_cfg: dict[str, Any] | None = None) -> bool:
    cfg = dict(catastrophic_cfg or {})
    invalid_action_rate_min = _safe_float(cfg.get("invalid_action_rate_min"), 0.25)
    total_score_max = _safe_float(cfg.get("total_score_max"), 40.0)
    status = str(record.get("status") or "ok").strip().lower()
    invalid_action_rate = _safe_float(record.get("invalid_action_rate"), 0.0)
    total_score = _safe_float(record.get("total_score"), 0.0)
    if status not in {"ok", "passed", "success"}:
        return True
    if invalid_action_rate >= invalid_action_rate_min:
        return True
    return total_score <= total_score_max


def summarize_episode_rows(
    records: list[dict[str, Any]],
    *,
    catastrophic_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total_scores = [_safe_float(row.get("total_score"), 0.0) for row in records]
    invalid_rates = [_safe_float(row.get("invalid_action_rate"), 0.0) for row in records]
    win_rates = [_safe_float(row.get("win_proxy"), 0.0) for row in records]
    seed_counter: Counter[str] = Counter()
    status_counter: Counter[str] = Counter()
    for row in records:
        seed_counter[str(row.get("seed") or "unknown")] += 1
        status_counter[str(row.get("status") or "ok")] += 1
    catastrophic_count = sum(1 for row in records if is_catastrophic_record(row, catastrophic_cfg))
    total_summary = numeric_summary(total_scores)
    return {
        "episode_count": len(records),
        "seed_count": len(seed_counter),
        "seeds": sorted(seed_counter.keys()),
        "mean_total_score": float(total_summary.get("mean") or 0.0),
        "std_total_score": float(total_summary.get("std") or 0.0),
        "median_total_score": float(total_summary.get("median") or 0.0),
        "mean_invalid_action_rate": float(numeric_summary(invalid_rates).get("mean") or 0.0),
        "win_rate": float(numeric_summary(win_rates).get("mean") or 0.0),
        "catastrophic_failure_count": int(catastrophic_count),
        "status_distribution": distribution_rows(status_counter, key_name="status"),
    }


def aggregate_seed_results(
    records: list[dict[str, Any]],
    *,
    catastrophic_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        policy_id = str(row.get("policy_id") or "unknown")
        seed = str(row.get("seed") or "unknown")
        buckets[(policy_id, seed)].append(row)
    out: list[dict[str, Any]] = []
    for (policy_id, seed), rows in sorted(buckets.items()):
        summary = summarize_episode_rows(rows, catastrophic_cfg=catastrophic_cfg)
        out.append(
            {
                "policy_id": policy_id,
                "seed": seed,
                **summary,
            }
        )
    return out


def aggregate_slice_results(
    records: list[dict[str, Any]],
    *,
    slice_keys: list[str],
    catastrophic_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        labels = row.get("slice_labels") if isinstance(row.get("slice_labels"), dict) else {}
        policy_id = str(row.get("policy_id") or "unknown")
        for slice_key in slice_keys:
            slice_label = str(labels.get(slice_key) or "unknown")
            buckets[(policy_id, slice_key, slice_label)].append(row)
    out: list[dict[str, Any]] = []
    for (policy_id, slice_key, slice_label), rows in sorted(buckets.items()):
        summary = summarize_episode_rows(rows, catastrophic_cfg=catastrophic_cfg)
        out.append(
            {
                "policy_id": policy_id,
                "slice_key": slice_key,
                "slice_label": slice_label,
                "count": len(rows),
                "mean_total_score": float(summary.get("mean_total_score") or 0.0),
                "std_total_score": float(summary.get("std_total_score") or 0.0),
                "median_total_score": float(summary.get("median_total_score") or 0.0),
                "win_rate": float(summary.get("win_rate") or 0.0),
                "catastrophic_failure_count": int(summary.get("catastrophic_failure_count") or 0),
            }
        )
    return out


def aggregate_trace_results(trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        buckets[str(row.get("policy_id") or "unknown")].append(row)
    out: list[dict[str, Any]] = []
    for policy_id, rows in sorted(buckets.items()):
        selection_counter: Counter[str] = Counter()
        final_counter: Counter[str] = Counter()
        reject_counter: Counter[str] = Counter()
        guard_count = 0
        fallback_count = 0
        canary_eligible_count = 0
        canary_used_count = 0
        for row in rows:
            selection_counter[str(row.get("selected_controller") or "unknown")] += 1
            final_counter[str(row.get("final_controller") or row.get("selected_controller") or "unknown")] += 1
            guard_count += int(bool(row.get("guard_triggered")))
            fallback_count += int(bool(row.get("fallback_used")))
            canary_eligible_count += int(bool(row.get("canary_eligible")))
            canary_used_count += int(bool(row.get("canary_used")))
            reject_reason = str(row.get("canary_reject_reason") or "")
            if reject_reason:
                reject_counter[reject_reason] += 1
        total = max(1, len(rows))
        out.append(
            {
                "policy_id": policy_id,
                "decision_count": len(rows),
                "controller_selection_distribution": distribution_rows(selection_counter, key_name="controller_id"),
                "final_controller_distribution": distribution_rows(final_counter, key_name="controller_id"),
                "guard_trigger_rate": float(guard_count) / total,
                "fallback_rate": float(fallback_count) / total,
                "canary_eligible_rate": float(canary_eligible_count) / total,
                "canary_usage_rate": float(canary_used_count) / total,
                "canary_reject_reason_distribution": distribution_rows(reject_counter, key_name="reason"),
            }
        )
    return out
