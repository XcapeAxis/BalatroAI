from __future__ import annotations

import math
import statistics
from typing import Any


def aggregate_seed_metrics(seed_results: list[dict[str, Any]], primary_metric: str) -> dict[str, Any]:
    values: list[float] = []
    catastrophic_failures: list[dict[str, Any]] = []
    per_seed_rows: list[dict[str, Any]] = []

    for row in seed_results:
        seed = str(row.get("seed") or "")
        status = str(row.get("status") or "unknown").lower()
        metrics = row.get("metrics") or {}
        value = metrics.get(primary_metric)
        numeric = value if isinstance(value, (int, float)) else None
        if numeric is not None:
            values.append(float(numeric))

        out = {
            "seed": seed,
            "status": status,
            "primary_metric": numeric,
            "stage": row.get("stage"),
            "error": row.get("error"),
            "elapsed_sec": row.get("elapsed_sec"),
        }
        per_seed_rows.append(out)

        if status != "ok":
            catastrophic_failures.append(
                {
                    "seed": seed,
                    "stage": row.get("stage"),
                    "error": row.get("error") or "unknown",
                }
            )

    mean_v = statistics.mean(values) if values else math.nan
    std_v = statistics.pstdev(values) if len(values) >= 2 else 0.0

    return {
        "primary_metric": primary_metric,
        "count": len(seed_results),
        "count_valid_metric": len(values),
        "mean": mean_v,
        "std": std_v,
        "catastrophic_failure_count": len(catastrophic_failures),
        "catastrophic_failures": catastrophic_failures,
        "per_seed": per_seed_rows,
    }


def is_success(metric_summary: dict[str, Any]) -> bool:
    return int(metric_summary.get("catastrophic_failure_count") or 0) == 0

