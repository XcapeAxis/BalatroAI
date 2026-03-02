"""P29 weakness mining v3: aggregate weak buckets from artifacts and prioritize fixes."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .weakness_taxonomy import (
    bucket_id,
    classify_alert,
    classify_seed_metrics,
    default_buckets,
    priority_score,
    template_for,
)

try:  # pragma: no cover
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if not token:
            continue
        try:
            payload = json.loads(token)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8").lstrip("\ufeff")
    if yaml is not None:
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if isinstance(payload, dict):
        return payload
    raise ValueError("config must be a mapping")


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _append_signal(bucket: dict[str, Any], signal: str) -> None:
    arr = bucket.setdefault("signals", [])
    if not isinstance(arr, list):
        arr = []
        bucket["signals"] = arr
    if signal not in arr and len(arr) < 8:
        arr.append(signal)


def _ensure_bucket(store: dict[str, dict[str, Any]], category: str) -> dict[str, Any]:
    key = bucket_id(category)
    if key in store:
        return store[key]
    tpl = template_for(category)
    row = {
        "bucket_id": key,
        "category": tpl.category,
        "frequency": 0.0,
        "penalty_sum": 0.0,
        "avg_penalty_proxy": 0.0,
        "affected_phase": tpl.affected_phase,
        "likely_fix_type": tpl.likely_fix_type,
        "impact_avg": 0.0,
        "impact_median": 0.0,
        "impact_win": 0.0,
        "priority_score": 0.0,
        "signals": [],
        "sources": set(),
    }
    store[key] = row
    return row


def _collect_latest_alert_report(artifacts_root: Path) -> Path | None:
    candidates = list((artifacts_root / "p26").rglob("regression_alert_report.json")) if (artifacts_root / "p26").exists() else []
    if not candidates:
        candidates = list(artifacts_root.rglob("regression_alert_report.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _collect_exp_summaries(artifacts_root: Path) -> list[Path]:
    roots: list[Path] = []
    latest = artifacts_root / "p24" / "runs" / "latest"
    if latest.exists():
        roots.append(latest)
    runs_root = artifacts_root / "p24" / "runs"
    if runs_root.exists():
        numbered = [p for p in runs_root.iterdir() if p.is_dir() and p.name[:8].isdigit()]
        if numbered:
            roots.append(max(numbered, key=lambda p: p.name))
    paths: list[Path] = []
    for root in roots:
        paths.extend(root.rglob("exp_summary.json"))
    # Keep latest per exp_id if duplicated.
    by_exp: dict[str, Path] = {}
    for p in paths:
        payload = read_json(p)
        exp_id = str(payload.get("exp_id") or p.parent.name)
        if exp_id not in by_exp or p.stat().st_mtime > by_exp[exp_id].stat().st_mtime:
            by_exp[exp_id] = p
    return list(by_exp.values())


def _collect_triage_report(artifacts_root: Path) -> Path | None:
    candidates = list((artifacts_root / "p24").rglob("triage_report.json")) if (artifacts_root / "p24").exists() else []
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def build_report(artifacts_root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    top_n = int(cfg.get("top_n") or 10)
    min_freq = float(cfg.get("min_frequency") or 0.1)
    penalty_scale = float(cfg.get("penalty_scale") or 1.0)
    seed_thresholds = cfg.get("seed_thresholds") if isinstance(cfg.get("seed_thresholds"), dict) else {}
    min_avg = float(seed_thresholds.get("min_avg_ante") or 3.30)
    min_median = float(seed_thresholds.get("min_median_ante") or 3.20)
    min_win = float(seed_thresholds.get("min_win_rate") or 0.35)

    buckets: dict[str, dict[str, Any]] = {}
    source_counts = {"alerts": 0, "per_seed": 0, "triage": 0, "trends": 0}
    alert_path = ""
    triage_path = ""

    alert_report = _collect_latest_alert_report(artifacts_root)
    if alert_report is not None:
        alert_path = str(alert_report.resolve())
        payload = read_json(alert_report)
        for rec in payload.get("records") if isinstance(payload.get("records"), list) else []:
            if not isinstance(rec, dict):
                continue
            cls = str(rec.get("classification") or "").lower()
            if cls not in {"hard_regression", "soft_regression", "regression"}:
                continue
            category = classify_alert(rec)
            row = _ensure_bucket(buckets, category)
            freq_weight = 2.0 if cls == "hard_regression" else 1.0
            row["frequency"] += freq_weight
            pct = _safe_float(rec.get("pct_change"))
            if pct is None:
                latest_v = _safe_float(rec.get("latest_value"))
                base_v = _safe_float(rec.get("baseline_value"))
                pct = abs((latest_v or 0.0) - (base_v or 0.0))
            row["penalty_sum"] += abs(float(pct or 0.0)) * penalty_scale
            metric = str(rec.get("metric_name") or "")
            if "avg_ante" in metric:
                row["impact_avg"] += abs(float(pct or 0.0))
            elif "median_ante" in metric:
                row["impact_median"] += abs(float(pct or 0.0))
            elif "win_rate" in metric:
                row["impact_win"] += abs(float(pct or 0.0))
            row["sources"].add("regression_alert")
            _append_signal(row, f"{metric}:{cls}")
            source_counts["alerts"] += 1

    for exp_summary_path in _collect_exp_summaries(artifacts_root):
        payload = read_json(exp_summary_path)
        exp_id = str(payload.get("exp_id") or exp_summary_path.parent.name)
        seed_metrics = payload.get("seed_metrics") if isinstance(payload.get("seed_metrics"), dict) else {}
        per_seed = seed_metrics.get("per_seed") if isinstance(seed_metrics.get("per_seed"), list) else []
        for seed_row in per_seed:
            if not isinstance(seed_row, dict):
                continue
            avg_ante = _safe_float(seed_row.get("avg_ante_reached"))
            median_ante = _safe_float(seed_row.get("median_ante"))
            win_rate = _safe_float(seed_row.get("win_rate"))
            categories = classify_seed_metrics(avg_ante=avg_ante, median_ante=median_ante, win_rate=win_rate)
            for category in categories:
                row = _ensure_bucket(buckets, category)
                row["frequency"] += 1.0
                penalty = 0.0
                if avg_ante is not None and avg_ante < min_avg:
                    penalty += (min_avg - avg_ante)
                    row["impact_avg"] += (min_avg - avg_ante)
                if median_ante is not None and median_ante < min_median:
                    penalty += (min_median - median_ante)
                    row["impact_median"] += (min_median - median_ante)
                if win_rate is not None and win_rate < min_win:
                    penalty += (min_win - win_rate)
                    row["impact_win"] += (min_win - win_rate)
                row["penalty_sum"] += penalty
                row["sources"].add("per_seed_eval")
                _append_signal(row, f"{exp_id}:{seed_row.get('seed')}")
                source_counts["per_seed"] += 1

    triage_report = _collect_triage_report(artifacts_root)
    if triage_report is not None:
        triage_path = str(triage_report.resolve())
        triage = read_json(triage_report)
        for row_data in triage.get("rows") if isinstance(triage.get("rows"), list) else []:
            if not isinstance(row_data, dict):
                continue
            category = str(row_data.get("category") or "unknown").lower()
            mapped = "simulator_coverage_gap" if category in {"unknown", "runtime_crash", "config_error", "service_instability"} else "policy_confusion"
            row = _ensure_bucket(buckets, mapped)
            row["frequency"] += 1.0
            row["penalty_sum"] += 0.15
            row["sources"].add("triage")
            _append_signal(row, f"triage:{category}")
            source_counts["triage"] += 1

    trend_rows_path = artifacts_root / "trends" / "trend_rows.jsonl"
    for row_data in read_jsonl(trend_rows_path)[-3000:]:
        status = str(row_data.get("status") or "").lower()
        flake_status = str(row_data.get("flake_status") or "").lower()
        risk_status = str(row_data.get("risk_status") or "").lower()
        metric_name = str(row_data.get("metric_name") or "").lower()

        mapped: str | None = None
        penalty = 0.05
        if flake_status == "flake_fail":
            mapped = "seed_fragility"
            penalty = 0.30
        elif risk_status in {"high_risk", "regression"}:
            mapped = "risk_overconservative"
            penalty = 0.20
        elif status in {"fail", "error"}:
            mapped = "action_illegality"
            penalty = 0.18
        elif metric_name == "runtime_seconds":
            val = _safe_float(row_data.get("metric_value"))
            if val is not None and val > 0.30:
                mapped = "runtime_latency"
                penalty = min(1.0, val)
        if mapped is None:
            continue
        row = _ensure_bucket(buckets, mapped)
        row["frequency"] += 0.5
        row["penalty_sum"] += penalty
        row["sources"].add("trends")
        _append_signal(row, f"{metric_name}:{status}")
        source_counts["trends"] += 1

    prepared: list[dict[str, Any]] = []
    for b in buckets.values():
        freq = max(min_freq, float(b.get("frequency") or 0.0))
        avg_penalty = float(b.get("penalty_sum") or 0.0) / max(freq, 1e-9)
        score = priority_score(category=str(b.get("category")), frequency=freq, avg_penalty_proxy=avg_penalty)
        prepared.append(
            {
                "bucket_id": str(b.get("bucket_id")),
                "category": str(b.get("category")),
                "frequency": round(freq, 4),
                "avg_penalty_proxy": round(avg_penalty, 6),
                "affected_phase": str(b.get("affected_phase")),
                "likely_fix_type": str(b.get("likely_fix_type")),
                "priority_score": round(score, 6),
                "impact": {
                    "avg_ante_loss_proxy": round(float(b.get("impact_avg") or 0.0), 6),
                    "median_ante_loss_proxy": round(float(b.get("impact_median") or 0.0), 6),
                    "win_rate_loss_proxy": round(float(b.get("impact_win") or 0.0), 6),
                },
                "sources": sorted([str(x) for x in b.get("sources", set())]),
                "signals": list(b.get("signals") or []),
            }
        )

    # Backfill to at least top_n with taxonomy defaults for stable shape.
    if len(prepared) < top_n:
        existing = {str(row.get("category")) for row in prepared}
        for row in default_buckets():
            if row["category"] in existing:
                continue
            prepared.append(row)
            if len(prepared) >= top_n:
                break

    prepared = sorted(prepared, key=lambda x: float(x.get("priority_score") or 0.0), reverse=True)
    top_buckets = prepared[: max(top_n, 10)]

    return {
        "schema": "p29_weakness_priority_report_v1",
        "generated_at": now_iso(),
        "artifacts_root": str(artifacts_root.resolve()),
        "config": cfg,
        "source_paths": {
            "alert_report": alert_path,
            "triage_report": triage_path,
            "trend_rows": str(trend_rows_path.resolve()),
        },
        "source_counts": source_counts,
        "total_buckets": len(prepared),
        "top_n": len(top_buckets),
        "top_buckets": top_buckets,
        "all_buckets": prepared,
    }


def write_outputs(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "weakness_priority_report.json"
    md_path = out_dir / "weakness_priority_report.md"
    csv_path = out_dir / "weakness_priority_table.csv"

    write_json(json_path, report)

    fieldnames = [
        "bucket_id",
        "category",
        "frequency",
        "avg_penalty_proxy",
        "affected_phase",
        "likely_fix_type",
        "priority_score",
        "avg_ante_loss_proxy",
        "median_ante_loss_proxy",
        "win_rate_loss_proxy",
        "sources",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in report.get("top_buckets") or []:
            impact = row.get("impact") if isinstance(row.get("impact"), dict) else {}
            writer.writerow(
                {
                    "bucket_id": row.get("bucket_id"),
                    "category": row.get("category"),
                    "frequency": row.get("frequency"),
                    "avg_penalty_proxy": row.get("avg_penalty_proxy"),
                    "affected_phase": row.get("affected_phase"),
                    "likely_fix_type": row.get("likely_fix_type"),
                    "priority_score": row.get("priority_score"),
                    "avg_ante_loss_proxy": impact.get("avg_ante_loss_proxy"),
                    "median_ante_loss_proxy": impact.get("median_ante_loss_proxy"),
                    "win_rate_loss_proxy": impact.get("win_rate_loss_proxy"),
                    "sources": ";".join(row.get("sources") or []),
                }
            )

    lines = [
        "# P29 Weakness Priority Report",
        "",
        f"- generated_at: `{report.get('generated_at')}`",
        f"- total_buckets: `{report.get('total_buckets')}`",
        f"- top_n: `{report.get('top_n')}`",
        "",
        "## Top Buckets",
    ]
    for idx, row in enumerate(report.get("top_buckets") or [], start=1):
        lines.append(
            f"{idx}. `{row.get('bucket_id')}` | {row.get('category')} | freq={row.get('frequency')} | penalty={row.get('avg_penalty_proxy')} | phase={row.get('affected_phase')} | fix={row.get('likely_fix_type')} | score={row.get('priority_score')}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path), "csv": str(csv_path)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P29 weakness bucket mining v3")
    p.add_argument("--artifacts-root", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    artifacts_root = Path(args.artifacts_root).resolve()
    config_path = Path(args.config).resolve()
    out_dir = Path(args.out_dir).resolve()

    cfg = load_mapping(config_path)
    report = build_report(artifacts_root=artifacts_root, cfg=cfg)
    paths = write_outputs(report, out_dir)
    print(
        json.dumps(
            {
                "status": "PASS",
                "out_dir": str(out_dir),
                "top_n": int(report.get("top_n") or 0),
                "top_bucket": (report.get("top_buckets") or [{}])[0].get("bucket_id") if report.get("top_buckets") else "",
                "json": paths["json"],
                "md": paths["md"],
                "csv": paths["csv"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

