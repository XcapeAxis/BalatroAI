from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _contains_any(text: str, patterns: list[str]) -> bool:
    t = text.lower()
    return any(p.lower() in t for p in patterns)


def _classify_failure(
    *,
    exp_id: str,
    status_payload: dict[str, Any],
    summary_payload: dict[str, Any],
    stage_payload: dict[str, Any],
) -> tuple[str, str, str]:
    status = str(status_payload.get("status") or summary_payload.get("status") or "").lower()
    reason = str(status_payload.get("reason") or "")
    stage = str(status_payload.get("stage") or "")
    stage_blob = json.dumps(stage_payload, ensure_ascii=False)

    if status in {"budget_cut", "timed_out"} or _contains_any(reason, ["budget", "timeout"]):
        return (
            "timeout_budget_cut",
            "experiment hit wall-time or per-experiment budget",
            "increase budget or reduce matrix/seed size",
        )

    if _contains_any(stage_blob, ["connection refused", "health check", "base_url unhealthy", "service start timeout"]):
        return (
            "service_instability",
            "service health/connectivity instability detected in stage logs",
            "restart service, rerun failing stage, inspect service logs",
        )

    if stage == "gate" or _contains_any(reason, ["gate"]):
        return (
            "gate_fail",
            "functional gate step failed",
            "inspect gate summary and fix baseline regressions before campaign rerun",
        )

    if stage == "sanity" and _contains_any(stage_blob, ["no such file", "unknown argument", "valueerror", "config"]):
        return (
            "config_error",
            "sanity/config validation indicates bad config or invocation",
            "fix stage config/matrix arguments and rerun",
        )

    if stage in {"train", "dataset", "eval"} and status in {"failed", "fail"} and _contains_any(
        stage_blob, ["traceback", "exception", "runtimeerror", "typeerror", "keyerror"]
    ):
        return (
            "runtime_crash",
            "runtime exception detected in execution stage",
            "inspect stack trace in stage results and patch runtime code path",
        )

    seed_metrics = summary_payload.get("seed_metrics") if isinstance(summary_payload.get("seed_metrics"), dict) else {}
    per_seed = seed_metrics.get("per_seed") if isinstance(seed_metrics.get("per_seed"), list) else []
    failed_seed_count = 0
    ok_seed_count = 0
    for row in per_seed:
        if not isinstance(row, dict):
            continue
        if str(row.get("status") or "").lower() == "ok":
            ok_seed_count += 1
        else:
            failed_seed_count += 1
    if failed_seed_count > 0 and ok_seed_count > 0:
        return (
            "seed_specific_failure",
            "only subset of seeds failed while others passed",
            "run seed bisect and inspect failing-seed traces",
        )

    avg_ante = seed_metrics.get("avg_ante_reached")
    median_ante = seed_metrics.get("median_ante")
    win_rate = seed_metrics.get("win_rate")
    try:
        avg_ante_v = float(avg_ante) if avg_ante is not None else None
    except Exception:
        avg_ante_v = None
    try:
        median_ante_v = float(median_ante) if median_ante is not None else None
    except Exception:
        median_ante_v = None
    try:
        win_rate_v = float(win_rate) if win_rate is not None else None
    except Exception:
        win_rate_v = None
    if avg_ante_v is not None and median_ante_v is not None and win_rate_v is not None:
        if avg_ante_v < 3.2 or median_ante_v < 3.0 or win_rate_v < 0.25:
            return (
                "metric_regression",
                "metrics below expected gate threshold",
                "compare against champion baseline and run ranking + triage follow-up",
            )

    return (
        "unknown",
        f"unable to infer dominant failure mode for {exp_id}",
        "inspect run_manifest/stage_results manually",
    )


def build_triage(run_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    category_counter: Counter[str] = Counter()

    for exp_dir in sorted([p for p in run_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        manifest = read_json(exp_dir / "run_manifest.json")
        if not manifest:
            continue
        exp_id = str(manifest.get("exp_id") or exp_dir.name)
        summary = read_json(exp_dir / "exp_summary.json")
        status_payload = read_json(exp_dir / "status.json")
        stage_payload = read_json(exp_dir / "stage_results.json")
        category, cause, action = _classify_failure(
            exp_id=exp_id,
            status_payload=status_payload,
            summary_payload=summary,
            stage_payload=stage_payload,
        )
        category_counter[category] += 1
        rows.append(
            {
                "exp_id": exp_id,
                "category": category,
                "probable_cause": cause,
                "suggested_next_action": action,
                "status": str(status_payload.get("status") or summary.get("status") or "unknown"),
                "evidence_paths": [
                    str(exp_dir / "run_manifest.json"),
                    str(exp_dir / "status.json"),
                    str(exp_dir / "stage_results.json"),
                    str(exp_dir / "exp_summary.json"),
                ],
            }
        )

    flake_report = read_json(run_root / "flake_report.json")
    if flake_report and str(flake_report.get("status") or "").upper() == "FAIL":
        category_counter["flake_failure"] += 1
        rows.append(
            {
                "exp_id": "GLOBAL",
                "category": "flake_failure",
                "probable_cause": "flake harness reported instability across repeated runs",
                "suggested_next_action": "run deterministic profile and isolate unstable seed/config subset",
                "status": "fail",
                "evidence_paths": [
                    str(run_root / "flake_report.json"),
                    str(run_root / "flake_report.md"),
                ],
            }
        )

    summary = {
        "schema": "p24_triage_report_v1",
        "generated_at": now_iso(),
        "run_root": str(run_root),
        "total_items": len(rows),
        "category_counts": dict(category_counter),
        "rows": rows,
    }
    return summary


def write_outputs(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "triage_report.json"
    md_path = out_dir / "triage_report.md"
    csv_path = out_dir / "triage_table.csv"

    write_json(json_path, report)

    fieldnames = [
        "exp_id",
        "category",
        "status",
        "probable_cause",
        "suggested_next_action",
        "evidence_paths",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in report.get("rows") or []:
            item = dict(row)
            item["evidence_paths"] = ";".join(item.get("evidence_paths") or [])
            writer.writerow(item)

    lines = [
        "# P24 Triage Report",
        "",
        f"- generated_at: `{report.get('generated_at')}`",
        f"- total_items: `{report.get('total_items')}`",
        f"- category_counts: `{report.get('category_counts')}`",
        "",
        "## Items",
    ]
    for row in report.get("rows") or []:
        lines += [
            f"- exp_id: `{row.get('exp_id')}`",
            f"  category: `{row.get('category')}`",
            f"  probable_cause: {row.get('probable_cause')}",
            f"  suggested_next_action: {row.get('suggested_next_action')}",
            f"  evidence_paths: {row.get('evidence_paths')}",
        ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path), "csv": str(csv_path)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P24 automatic triage")
    p.add_argument("--run-root", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    report = build_triage(run_root)
    paths = write_outputs(report, out_dir)
    print(json.dumps({"status": "PASS", "paths": paths}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

