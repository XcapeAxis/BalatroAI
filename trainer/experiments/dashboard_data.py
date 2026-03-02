from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from statistics import median
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_ts(value: Any) -> datetime | None:
    token = str(value or "").strip()
    if not token:
        return None
    try:
        return datetime.fromisoformat(token.replace("Z", "+00:00"))
    except Exception:
        return None


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def metric_snapshot(rows: list[dict[str, Any]], metric_name: str, tail: int = 20) -> dict[str, Any]:
    series: list[tuple[datetime, dict[str, Any], float]] = []
    for row in rows:
        if str(row.get("metric_name") or "") != metric_name:
            continue
        ts = parse_ts(row.get("timestamp"))
        value = as_float(row.get("metric_value"))
        if ts is None or value is None:
            continue
        series.append((ts, row, value))
    if not series:
        return {
            "metric_name": metric_name,
            "points": 0,
            "latest_value": 0.0,
            "baseline_median": 0.0,
            "delta": 0.0,
            "pct_change": 0.0,
            "sparkline_values": [],
            "latest_run_id": "",
            "latest_gate_name": "",
            "latest_timestamp": "",
        }

    series.sort(key=lambda item: item[0])
    latest_ts, latest_row, latest_value = series[-1]
    baseline_values = [item[2] for item in series[-11:-1]]
    baseline = median(baseline_values) if baseline_values else latest_value
    delta = latest_value - baseline
    pct_change = 0.0 if abs(baseline) <= 1e-9 else delta / abs(baseline)
    spark = [item[2] for item in series[-tail:]]
    return {
        "metric_name": metric_name,
        "points": len(series),
        "latest_value": latest_value,
        "baseline_median": baseline,
        "delta": delta,
        "pct_change": pct_change,
        "sparkline_values": spark,
        "latest_run_id": str(latest_row.get("run_id") or ""),
        "latest_gate_name": str(latest_row.get("gate_name") or ""),
        "latest_timestamp": latest_ts.isoformat(),
    }


def infer_trend_signal(alert_summary: dict[str, Any], snapshots: list[dict[str, Any]]) -> str:
    hard = int(alert_summary.get("hard_regression", 0) or 0)
    soft = int(alert_summary.get("soft_regression", 0) or 0)
    noisy = int(alert_summary.get("noisy_needs_more_data", 0) or 0)
    improvement = int(alert_summary.get("improvement", 0) or 0)
    if hard > 0 or soft > 0:
        return "regression"
    if noisy > 0:
        return "noisy"
    if improvement > 0:
        return "improving"
    if any(int(s.get("points", 0) or 0) > 0 for s in snapshots):
        return "stable"
    return "unknown"


def build_recent_runs(rows: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    per_run: dict[str, dict[str, Any]] = {}
    for row in rows:
        run_id = str(row.get("run_id") or "").strip()
        if not run_id:
            continue
        ts = parse_ts(row.get("timestamp"))
        if ts is None:
            continue
        info = per_run.setdefault(
            run_id,
            {
                "run_id": run_id,
                "timestamp": ts,
                "milestone": str(row.get("milestone") or ""),
                "avg_ante_reached": None,
                "median_ante_reached": None,
                "win_rate": None,
                "_gate_values": [],
                "_metric_ts": defaultdict(lambda: datetime.fromtimestamp(0, tz=timezone.utc)),
            },
        )
        if ts > info["timestamp"]:
            info["timestamp"] = ts
            info["milestone"] = str(row.get("milestone") or info["milestone"])

        metric_name = str(row.get("metric_name") or "")
        metric_value = as_float(row.get("metric_value"))
        if metric_name in {"avg_ante_reached", "median_ante_reached", "win_rate"} and metric_value is not None:
            latest_metric_ts = info["_metric_ts"][metric_name]
            if ts >= latest_metric_ts:
                info["_metric_ts"][metric_name] = ts
                info[metric_name] = metric_value
        if metric_name in {"gate_overall_pass", "gate_pass"} and metric_value is not None:
            info["_gate_values"].append(metric_value)

    rows_out: list[dict[str, Any]] = []
    for info in per_run.values():
        gate_values = list(info.get("_gate_values", []))
        gate_status = "UNKNOWN"
        if gate_values:
            gate_status = "PASS" if min(gate_values) >= 0.5 else "FAIL"
        rows_out.append(
            {
                "run_id": info["run_id"],
                "timestamp": info["timestamp"].isoformat(),
                "milestone": info["milestone"],
                "gate_status": gate_status,
                "avg_ante_reached": info["avg_ante_reached"],
                "median_ante_reached": info["median_ante_reached"],
                "win_rate": info["win_rate"],
            }
        )
    rows_out.sort(key=lambda item: item["timestamp"], reverse=True)
    return rows_out[:limit]


def build_gate_history(rows: list[dict[str, Any]], limit: int = 40) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in sorted(rows, key=lambda r: str(r.get("timestamp") or ""), reverse=True):
        metric_name = str(row.get("metric_name") or "")
        if metric_name not in {"gate_overall_pass", "gate_pass"}:
            continue
        run_id = str(row.get("run_id") or "")
        gate_name = str(row.get("gate_name") or "")
        strategy = str(row.get("strategy") or "")
        key = (run_id, gate_name, strategy)
        if key in seen:
            continue
        seen.add(key)
        value = as_float(row.get("metric_value")) or 0.0
        history.append(
            {
                "timestamp": str(row.get("timestamp") or ""),
                "run_id": run_id,
                "milestone": str(row.get("milestone") or ""),
                "gate_name": gate_name,
                "strategy": strategy,
                "status": "PASS" if value >= 0.5 else "FAIL",
                "metric_name": metric_name,
                "metric_value": value,
            }
        )
        if len(history) >= limit:
            break
    return history


def build_dashboard_data(
    *,
    sources: dict[str, Any],
    latest_gate: dict[str, Any],
    alert_summary: dict[str, Any],
    trend_rows: list[dict[str, Any]],
    champion_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    release_summary: dict[str, Any],
) -> dict[str, Any]:
    avg_snapshot = metric_snapshot(trend_rows, "avg_ante_reached")
    median_snapshot = metric_snapshot(trend_rows, "median_ante_reached")
    win_rate_snapshot = metric_snapshot(trend_rows, "win_rate")
    trend_signal = infer_trend_signal(alert_summary, [avg_snapshot, median_snapshot, win_rate_snapshot])
    return {
        "schema": "p27_dashboard_data_v1",
        "generated_at": now_iso(),
        "sources": sources,
        "latest_gate": latest_gate,
        "trend_signal": trend_signal,
        "trend_metrics": {
            "avg_ante_reached": avg_snapshot,
            "median_ante_reached": median_snapshot,
            "win_rate": win_rate_snapshot,
        },
        "regression_alerts": {
            "summary": alert_summary,
            "hard_regression": int(alert_summary.get("hard_regression", 0) or 0),
            "soft_regression": int(alert_summary.get("soft_regression", 0) or 0),
            "noisy_needs_more_data": int(alert_summary.get("noisy_needs_more_data", 0) or 0),
            "improvement": int(alert_summary.get("improvement", 0) or 0),
            "total_series": int(alert_summary.get("total_series", 0) or 0),
        },
        "champion": champion_summary,
        "candidate": candidate_summary,
        "release_state": release_summary,
        "recent_runs": build_recent_runs(trend_rows, limit=20),
        "gate_history": build_gate_history(trend_rows, limit=40),
    }
