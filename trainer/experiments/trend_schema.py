from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


TREND_COLUMNS = [
    "timestamp",
    "milestone",
    "run_id",
    "artifact_path",
    "gate_name",
    "strategy",
    "seed_set_name",
    "metric_name",
    "metric_value",
    "unit",
    "status",
    "source_file",
    "git_commit",
    "flake_status",
    "risk_status",
]


_METRIC_ALIASES = {
    "avg_ante": "avg_ante_reached",
    "mean": "avg_ante_reached",
    "median_ante": "median_ante_reached",
    "elapsed_sec": "runtime_seconds",
    "elapsed_seconds": "runtime_seconds",
    "runtime_sec": "runtime_seconds",
    "runtime_seconds": "runtime_seconds",
}


_METRIC_UNITS = {
    "avg_ante_reached": "ante",
    "median_ante_reached": "ante",
    "win_rate": "ratio",
    "runtime_seconds": "seconds",
    "seed_count": "count",
    "catastrophic_failure_count": "count",
    "stage_pass_count": "count",
    "experiment_count": "count",
    "weighted_score": "score",
    "risk_score": "score",
    "flake_score": "score",
    "trace_mismatch": "count",
    "avg_ante_std": "ante",
    "median_ante_span": "ante",
    "win_rate_std": "ratio",
}


_PASS_TOKENS = {"pass", "passed", "success", "completed", "ok", "true"}
_FAIL_TOKENS = {"fail", "failed", "error", "false"}
_SKIP_TOKENS = {"skip", "skipped", "unknown", "dry_run", "dry-run", "pending"}


def canonical_metric_name(metric_name: str) -> str:
    token = (metric_name or "").strip().lower()
    if not token:
        return "unknown_metric"
    return _METRIC_ALIASES.get(token, token)


def metric_unit(metric_name: str) -> str:
    return _METRIC_UNITS.get(canonical_metric_name(metric_name), "number")


def normalize_status(raw: Any) -> str:
    if isinstance(raw, bool):
        return "pass" if raw else "fail"
    token = str(raw or "").strip().lower()
    if token in _PASS_TOKENS:
        return "pass"
    if token in _FAIL_TOKENS:
        return "fail"
    if token in _SKIP_TOKENS:
        return "skip"
    return "skip"


def coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return None
    token = str(value).strip()
    if token == "":
        return None
    try:
        return float(token)
    except Exception:
        return None


@dataclass(frozen=True)
class TrendRow:
    timestamp: str
    milestone: str
    run_id: str
    artifact_path: str
    gate_name: str
    strategy: str
    seed_set_name: str
    metric_name: str
    metric_value: float
    unit: str
    status: str
    source_file: str
    git_commit: str
    flake_status: str
    risk_status: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metric_name"] = canonical_metric_name(payload["metric_name"])
        payload["unit"] = metric_unit(payload["metric_name"])
        payload["status"] = normalize_status(payload["status"])
        return payload
