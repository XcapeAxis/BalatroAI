from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = _mean(values)
    var = sum((x - mu) ** 2 for x in values) / len(values)
    return float(var ** 0.5)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    qv = min(1.0, max(0.0, float(q)))
    pos = (len(ordered) - 1) * qv
    lo = int(pos)
    hi = min(len(ordered) - 1, lo + 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _merge_nested_counts(dst: dict[str, dict[str, int]], src: dict[str, dict[str, Any]]) -> None:
    for bucket_name, bucket_vals in (src or {}).items():
        if not isinstance(bucket_vals, dict):
            continue
        target = dst.setdefault(str(bucket_name), {})
        for key, val in bucket_vals.items():
            key_text = str(key)
            target[key_text] = int(target.get(key_text, 0)) + int(_safe_float(val, 0.0))


def summarize_policy_rows(episode_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in episode_records:
        policy_id = str(record.get("policy_id") or "unknown")
        grouped[policy_id].append(record)

    rows: list[dict[str, Any]] = []
    for policy_id in sorted(grouped.keys()):
        records = grouped[policy_id]
        scores = [_safe_float(x.get("total_score")) for x in records]
        chips = [_safe_float(x.get("chips")) for x in records]
        rounds = [_safe_float(x.get("rounds_survived")) for x in records]
        money = [_safe_float(x.get("money_earned")) for x in records]
        rerolls = [_safe_float(x.get("rerolls_count")) for x in records]
        packs = [_safe_float(x.get("packs_opened")) for x in records]
        consumables = [_safe_float(x.get("consumables_used")) for x in records]
        lengths = [_safe_float(x.get("episode_length")) for x in records]
        invalid_rates = [_safe_float(x.get("invalid_action_rate")) for x in records]
        timeout_rates = [_safe_float(x.get("timeout_rate")) for x in records]
        wins = [_safe_float(x.get("win_proxy")) for x in records]

        status_counts = Counter(str(x.get("status") or "unknown") for x in records)
        seeds = sorted({str(x.get("seed") or "") for x in records if str(x.get("seed") or "")})

        rows.append(
            {
                "policy_id": policy_id,
                "status": "ok" if status_counts.get("ok", 0) == len(records) else "partial_fail",
                "episodes": int(len(records)),
                "seeds": seeds,
                "seed_count": int(len(seeds)),
                "mean_total_score": _mean(scores),
                "std_total_score": _std(scores),
                "mean_chips": _mean(chips),
                "mean_rounds_survived": _mean(rounds),
                "mean_episode_length": _mean(lengths),
                "win_rate": _mean(wins),
                "p10_total_score": _quantile(scores, 0.10),
                "p50_total_score": _quantile(scores, 0.50),
                "p90_total_score": _quantile(scores, 0.90),
                "invalid_action_rate": _mean(invalid_rates),
                "timeout_rate": _mean(timeout_rates),
                "mean_money_earned": _mean(money),
                "mean_rerolls_count": _mean(rerolls),
                "mean_packs_opened": _mean(packs),
                "mean_consumables_used": _mean(consumables),
                "status_counts": dict(status_counts),
            }
        )

    rows.sort(key=lambda x: float(x.get("mean_total_score") or 0.0), reverse=True)
    return rows


def summarize_bucket_metrics(episode_records: list[dict[str, Any]]) -> dict[str, Any]:
    per_policy: dict[str, dict[str, dict[str, int]]] = {}
    for record in episode_records:
        policy_id = str(record.get("policy_id") or "unknown")
        target = per_policy.setdefault(policy_id, {})
        _merge_nested_counts(target, record.get("bucket_counts") if isinstance(record.get("bucket_counts"), dict) else {})

    policies_payload: list[dict[str, Any]] = []
    for policy_id in sorted(per_policy.keys()):
        buckets = per_policy[policy_id]
        normalized: dict[str, list[dict[str, Any]]] = {}
        for bucket_name, counts in buckets.items():
            total = int(sum(int(v) for v in counts.values()))
            rows = []
            for key, count in sorted(counts.items(), key=lambda kv: (-int(kv[1]), kv[0])):
                rows.append(
                    {
                        "bucket": key,
                        "count": int(count),
                        "ratio": (float(count) / total) if total > 0 else 0.0,
                    }
                )
            normalized[bucket_name] = rows
        policies_payload.append({"policy_id": policy_id, "buckets": normalized})

    return {"schema": "p39_bucket_metrics_v1", "policies": policies_payload}

