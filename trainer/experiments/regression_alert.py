from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

try:  # pragma: no cover
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if not token:
            continue
        try:
            payload = json.loads(token)
        except Exception:
            continue
        if isinstance(payload, dict):
            out.append(payload)
    return out


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("config must be a mapping")
    return payload


def _metric_config(cfg: dict[str, Any], metric_name: str) -> dict[str, Any]:
    metrics = cfg.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    item = metrics.get(metric_name)
    return item if isinstance(item, dict) else {}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _status_token(raw: Any) -> str:
    return str(raw or "").strip().lower()


def _is_noisy(row: dict[str, Any], cfg: dict[str, Any]) -> tuple[bool, str]:
    noisy_cfg = cfg.get("noisy_filters")
    if not isinstance(noisy_cfg, dict):
        noisy_cfg = {}
    noisy_flakes = {str(x).strip().lower() for x in noisy_cfg.get("flake_statuses", []) if str(x).strip()}
    noisy_risks = {str(x).strip().lower() for x in noisy_cfg.get("risk_statuses", []) if str(x).strip()}
    noisy_statuses = {str(x).strip().lower() for x in noisy_cfg.get("statuses", []) if str(x).strip()}

    flake = _status_token(row.get("flake_status"))
    risk = _status_token(row.get("risk_status"))
    status = _status_token(row.get("status"))
    if flake and flake in noisy_flakes:
        return True, f"flake_status={flake}"
    if risk and risk in noisy_risks:
        return True, f"risk_status={risk}"
    if status and status in noisy_statuses:
        return True, f"status={status}"
    return False, ""


def _group_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    metric_name = str(row.get("metric_name") or "")
    gate_name = str(row.get("gate_name") or "")
    strategy = str(row.get("strategy") or "__all__")
    seed_set = str(row.get("seed_set_name") or "__all__")
    return metric_name, gate_name, strategy, seed_set


def _classify_group(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    rows_sorted = sorted(rows, key=lambda r: (str(r.get("timestamp") or ""), str(r.get("run_id") or "")))
    latest = rows_sorted[-1]
    metric_name = str(latest.get("metric_name") or "")
    mcfg = _metric_config(cfg, metric_name)

    baseline_window = int(mcfg.get("baseline_window", cfg.get("baseline_window", 5)))
    min_points = int(mcfg.get("min_points", cfg.get("min_points", 4)))
    hysteresis = _as_float(mcfg.get("hysteresis", cfg.get("hysteresis", 0.03)))
    hard_hysteresis = _as_float(mcfg.get("hard_hysteresis", max(hysteresis * _as_float(cfg.get("hard_multiplier", 2.0)), hysteresis)))
    direction = str(mcfg.get("direction", "higher_better")).strip().lower()

    # Keep pass rows for baseline stability.
    baseline_candidates = [
        r
        for r in rows_sorted[:-1]
        if _status_token(r.get("status")) in {"pass", "passed", "success", "completed", "ok"}
    ]
    baseline_tail = baseline_candidates[-baseline_window:] if baseline_window > 0 else baseline_candidates
    baseline_values = [_as_float(r.get("metric_value")) for r in baseline_tail]
    latest_value = _as_float(latest.get("metric_value"))
    baseline_value = median(baseline_values) if baseline_values else 0.0
    baseline_count = len(baseline_values)

    noisy, noisy_reason = _is_noisy(latest, cfg)
    classification = "no_signal"
    reason = "within_hysteresis"
    if baseline_count < min_points:
        classification = "no_signal"
        reason = f"insufficient_baseline_points={baseline_count}<{min_points}"
    elif noisy:
        classification = "noisy_needs_more_data"
        reason = noisy_reason
    else:
        if direction == "lower_better":
            worse_delta = latest_value - baseline_value
            improve_delta = baseline_value - latest_value
        else:
            worse_delta = baseline_value - latest_value
            improve_delta = latest_value - baseline_value
        if worse_delta >= hard_hysteresis:
            classification = "hard_regression"
            reason = f"worse_delta={worse_delta:.6f} >= hard_hysteresis={hard_hysteresis:.6f}"
        elif worse_delta >= hysteresis:
            classification = "soft_regression"
            reason = f"worse_delta={worse_delta:.6f} >= hysteresis={hysteresis:.6f}"
        elif improve_delta >= hysteresis:
            classification = "improvement"
            reason = f"improve_delta={improve_delta:.6f} >= hysteresis={hysteresis:.6f}"

    pct_change = 0.0
    if abs(baseline_value) > 1e-9:
        pct_change = (latest_value - baseline_value) / abs(baseline_value)

    return {
        "metric_name": metric_name,
        "gate_name": str(latest.get("gate_name") or ""),
        "strategy": str(latest.get("strategy") or ""),
        "seed_set_name": str(latest.get("seed_set_name") or ""),
        "milestone": str(latest.get("milestone") or ""),
        "run_id": str(latest.get("run_id") or ""),
        "timestamp": str(latest.get("timestamp") or ""),
        "latest_value": latest_value,
        "baseline_value": baseline_value,
        "baseline_count": baseline_count,
        "hysteresis": hysteresis,
        "hard_hysteresis": hard_hysteresis,
        "direction": direction,
        "classification": classification,
        "reason": reason,
        "status": str(latest.get("status") or ""),
        "flake_status": str(latest.get("flake_status") or ""),
        "risk_status": str(latest.get("risk_status") or ""),
        "pct_change": pct_change,
    }


def _build_report(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    include_metrics = {str(x).strip() for x in cfg.get("include_metrics", []) if str(x).strip()}
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        metric_name = str(row.get("metric_name") or "")
        if include_metrics and metric_name not in include_metrics:
            continue
        groups.setdefault(_group_key(row), []).append(row)

    records: list[dict[str, Any]] = []
    for key_rows in groups.values():
        if len(key_rows) < 2:
            continue
        records.append(_classify_group(key_rows, cfg))

    records.sort(
        key=lambda r: (
            str(r.get("classification") or ""),
            str(r.get("metric_name") or ""),
            str(r.get("gate_name") or ""),
            str(r.get("strategy") or ""),
        )
    )
    counts = {
        "hard_regression": 0,
        "soft_regression": 0,
        "improvement": 0,
        "no_signal": 0,
        "noisy_needs_more_data": 0,
    }
    for rec in records:
        cls = str(rec.get("classification") or "")
        if cls in counts:
            counts[cls] += 1

    return {
        "schema": "p26_regression_alert_report_v1",
        "generated_at": _now_iso(),
        "config": {
            "baseline_window": int(cfg.get("baseline_window", 5)),
            "min_points": int(cfg.get("min_points", 4)),
            "hysteresis": _as_float(cfg.get("hysteresis", 0.03)),
            "hard_multiplier": _as_float(cfg.get("hard_multiplier", 2.0)),
            "include_metrics": sorted(include_metrics),
            "noisy_filters": cfg.get("noisy_filters", {}),
        },
        "summary": {
            "total_series": len(records),
            **counts,
        },
        "records": records,
    }


def _write_outputs(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "regression_alert_report.json"
    md_path = out_dir / "regression_alert_report.md"
    csv_path = out_dir / "regression_alert_table.csv"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fieldnames = [
        "classification",
        "metric_name",
        "gate_name",
        "strategy",
        "seed_set_name",
        "milestone",
        "run_id",
        "latest_value",
        "baseline_value",
        "baseline_count",
        "hysteresis",
        "hard_hysteresis",
        "pct_change",
        "status",
        "flake_status",
        "risk_status",
        "reason",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for rec in report.get("records", []):
            writer.writerow({k: rec.get(k) for k in fieldnames})

    summary = report.get("summary", {})
    config = report.get("config", {})
    noisy = [r for r in report.get("records", []) if r.get("classification") == "noisy_needs_more_data"]
    lines = [
        "# P26 Regression Alert Report",
        "",
        f"- generated_at: {report.get('generated_at')}",
        f"- total_series: {summary.get('total_series', 0)}",
        f"- hard_regression: {summary.get('hard_regression', 0)}",
        f"- soft_regression: {summary.get('soft_regression', 0)}",
        f"- noisy_needs_more_data: {summary.get('noisy_needs_more_data', 0)}",
        f"- improvement: {summary.get('improvement', 0)}",
        f"- no_signal: {summary.get('no_signal', 0)}",
        "",
        "## Baseline Settings",
        f"- baseline_window: {config.get('baseline_window')}",
        f"- min_points: {config.get('min_points')}",
        f"- hysteresis: {config.get('hysteresis')}",
        f"- hard_multiplier: {config.get('hard_multiplier')}",
        "",
        "## Noisy Signals",
    ]
    if noisy:
        for rec in noisy[:20]:
            lines.append(
                f"- {rec.get('metric_name')} | {rec.get('gate_name')} | {rec.get('strategy')} | reason: {rec.get('reason')}"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Top Alerts"])
    top = [
        r
        for r in report.get("records", [])
        if r.get("classification") in {"hard_regression", "soft_regression", "improvement"}
    ]
    if top:
        for rec in top[:30]:
            lines.append(
                f"- {rec.get('classification')}: {rec.get('metric_name')} ({rec.get('gate_name')} / {rec.get('strategy')}) latest={rec.get('latest_value')} baseline={rec.get('baseline_value')}"
            )
    else:
        lines.append("- no significant changes")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "json": str(json_path),
        "md": str(md_path),
        "csv": str(csv_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P26 regression alerting from trend warehouse")
    parser.add_argument("--trends-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trends_root = Path(args.trends_root).resolve()
    config = _load_mapping(Path(args.config).resolve())
    trend_rows = _read_jsonl(trends_root / "trend_rows.jsonl")
    report = _build_report(trend_rows, config)
    paths = _write_outputs(report, Path(args.out_dir).resolve())
    print(
        json.dumps(
            {
                "status": "PASS",
                "series": report.get("summary", {}).get("total_series", 0),
                "summary": report.get("summary", {}),
                "paths": paths,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
