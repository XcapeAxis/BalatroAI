from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KEY_METRICS = {
    "avg_ante_reached",
    "median_ante_reached",
    "win_rate",
    "runtime_seconds",
    "gate_overall_pass",
    "gate_pass",
}


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


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8")
    for candidate in (raw, raw.lstrip("\ufeff")):
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
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


def run_git(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    return proc.returncode, out


def latest_tag() -> str:
    code, out = run_git(["describe", "--tags", "--abbrev=0"])
    if code == 0 and out:
        return out.splitlines()[-1].strip()
    code, out = run_git(["tag", "--sort=-creatordate"])
    if code == 0 and out:
        return out.splitlines()[0].strip()
    return ""


def tag_exists(tag: str) -> bool:
    if not tag:
        return False
    return run_git(["rev-parse", "--verify", f"refs/tags/{tag}"])[0] == 0


def tag_commit_time(tag: str) -> datetime | None:
    if not tag:
        return None
    code, out = run_git(["show", "-s", "--format=%cI", tag])
    if code != 0 or not out:
        return None
    return parse_ts(out.splitlines()[-1].strip())


def cutoff_from_run(rows: list[dict[str, Any]], run_id: str) -> datetime | None:
    candidates = [parse_ts(r.get("timestamp")) for r in rows if str(r.get("run_id") or "") == run_id]
    candidates = [c for c in candidates if c is not None]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def latest_by_group(
    rows: list[dict[str, Any]],
    *,
    cutoff: datetime | None,
    older_equal: bool,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        metric = str(row.get("metric_name") or "")
        if metric not in KEY_METRICS:
            continue
        ts = parse_ts(row.get("timestamp"))
        if ts is None:
            continue
        if cutoff is not None:
            if older_equal and ts > cutoff:
                continue
            if not older_equal and ts <= cutoff:
                continue
        key = (
            metric,
            str(row.get("gate_name") or ""),
            str(row.get("strategy") or "__all__"),
        )
        prev = out.get(key)
        if prev is None:
            out[key] = row
            continue
        prev_ts = parse_ts(prev.get("timestamp"))
        if prev_ts is None or ts > prev_ts:
            out[key] = row
    return out


def benchmark_deltas(rows: list[dict[str, Any]], cutoff: datetime | None) -> list[dict[str, Any]]:
    before = latest_by_group(rows, cutoff=cutoff, older_equal=True)
    after = latest_by_group(rows, cutoff=cutoff, older_equal=False)
    deltas: list[dict[str, Any]] = []
    for key, latest in after.items():
        baseline = before.get(key)
        if baseline is None:
            continue
        latest_value = as_float(latest.get("metric_value"))
        baseline_value = as_float(baseline.get("metric_value"))
        delta = latest_value - baseline_value
        pct = 0.0 if abs(baseline_value) <= 1e-9 else delta / abs(baseline_value)
        deltas.append(
            {
                "metric_name": key[0],
                "gate_name": key[1],
                "strategy": key[2],
                "baseline_value": baseline_value,
                "latest_value": latest_value,
                "delta": delta,
                "pct_change": pct,
                "baseline_run_id": str(baseline.get("run_id") or ""),
                "latest_run_id": str(latest.get("run_id") or ""),
            }
        )
    deltas.sort(key=lambda r: abs(float(r.get("delta") or 0.0)), reverse=True)
    return deltas


def gate_snapshot(rows: list[dict[str, Any]], cutoff: datetime | None) -> dict[str, Any]:
    gate_rows = [r for r in rows if str(r.get("metric_name") or "") in {"gate_overall_pass", "gate_pass"}]
    before = latest_by_group(gate_rows, cutoff=cutoff, older_equal=True)
    after = latest_by_group(gate_rows, cutoff=cutoff, older_equal=False)
    changes: list[dict[str, Any]] = []
    for key, latest in after.items():
        baseline = before.get(key)
        if baseline is None:
            continue
        before_value = as_float(baseline.get("metric_value"))
        after_value = as_float(latest.get("metric_value"))
        if abs(before_value - after_value) <= 1e-9:
            continue
        changes.append(
            {
                "metric_name": key[0],
                "gate_name": key[1],
                "strategy": key[2],
                "before": before_value,
                "after": after_value,
                "baseline_run_id": str(baseline.get("run_id") or ""),
                "latest_run_id": str(latest.get("run_id") or ""),
            }
        )
    latest_status_rows = sorted(
        gate_rows,
        key=lambda r: (str(r.get("timestamp") or ""), str(r.get("run_id") or "")),
        reverse=True,
    )
    latest_entry = latest_status_rows[0] if latest_status_rows else {}
    return {
        "schema": "p27_gate_snapshot_v1",
        "generated_at": now_iso(),
        "latest_gate_name": str(latest_entry.get("gate_name") or ""),
        "latest_metric_name": str(latest_entry.get("metric_name") or ""),
        "latest_run_id": str(latest_entry.get("run_id") or ""),
        "latest_timestamp": str(latest_entry.get("timestamp") or ""),
        "latest_status": "PASS" if as_float(latest_entry.get("metric_value")) >= 0.5 else "FAIL",
        "changes": changes,
        "change_count": len(changes),
    }


def latest_named_file(artifacts_root: Path, file_name: str) -> Path | None:
    roots = [artifacts_root / "p26", artifacts_root / "p24", artifacts_root / "p23", artifacts_root / "p20"]
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            candidates.extend(root.rglob(file_name))
    if not candidates:
        candidates.extend(artifacts_root.rglob(file_name))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def latest_alert_summary(artifacts_root: Path) -> tuple[dict[str, Any], str]:
    path = latest_named_file(artifacts_root, "regression_alert_report.json")
    if path is None:
        return {}, ""
    payload = read_json(path) or {}
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        summary = {}
    return summary, str(path.resolve())


def load_status_publish(artifacts_root: Path) -> dict[str, Any]:
    status_path = artifacts_root / "status" / "latest_status.json"
    payload = read_json(status_path)
    return payload if payload is not None else {}


def call_release_notes(
    *,
    used_since_tag: str,
    since_run: str,
    trends_root: Path,
    out_md: Path,
) -> tuple[bool, str, str]:
    args = [
        sys.executable,
        "-m",
        "trainer.experiments.release_notes",
        "--out",
        str(out_md),
        "--include-commits",
        "--include-benchmarks",
        "--include-risks",
        "--trends-root",
        str(trends_root),
    ]
    if used_since_tag:
        args.extend(["--since-tag", used_since_tag])
    if since_run:
        args.extend(["--since-run", since_run])
    proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    ok = proc.returncode == 0
    return ok, proc.stdout.strip(), proc.stderr.strip()


def recommendation(
    *,
    gate_snapshot_obj: dict[str, Any],
    alert_summary: dict[str, Any],
    deltas: list[dict[str, Any]],
    candidate_decision: str,
    release_action: str,
) -> tuple[str, list[str]]:
    hard = int(alert_summary.get("hard_regression", 0) or 0)
    soft = int(alert_summary.get("soft_regression", 0) or 0)
    improve = int(alert_summary.get("improvement", 0) or 0)
    gate_latest = str(gate_snapshot_obj.get("latest_status") or "UNKNOWN")
    gate_regressed = any(float(c.get("before", 1.0)) > float(c.get("after", 1.0)) for c in gate_snapshot_obj.get("changes", []))

    perf_rows = [d for d in deltas if d.get("metric_name") in {"avg_ante_reached", "median_ante_reached", "win_rate"}]
    perf_positive = sum(1 for d in perf_rows if float(d.get("delta") or 0.0) > 0)
    perf_negative = sum(1 for d in perf_rows if float(d.get("delta") or 0.0) < 0)
    runtime_rows = [d for d in deltas if d.get("metric_name") == "runtime_seconds"]
    runtime_delta_sum = sum(float(d.get("delta") or 0.0) for d in runtime_rows)

    reasons = [
        f"performance: positives={perf_positive}, negatives={perf_negative}, improvements={improve}",
        f"stability: latest_gate={gate_latest}, gate_change_count={gate_snapshot_obj.get('change_count', 0)}",
        f"risk: hard={hard}, soft={soft}, candidate_decision={candidate_decision or 'n/a'}, release_action={release_action or 'n/a'}",
        f"cost: runtime_delta_total={runtime_delta_sum:.6f}s across {len(runtime_rows)} runtime series",
    ]

    if hard > 0 or gate_latest == "FAIL" or gate_regressed:
        return "Investigate", reasons
    if soft > 0 or str(candidate_decision).lower() == "hold" or str(release_action).lower() == "hold":
        return "Hold", reasons
    if perf_positive > perf_negative and improve > 0:
        return "Promote candidate", reasons
    return "Hold", reasons


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "metric_name",
        "gate_name",
        "strategy",
        "baseline_value",
        "latest_value",
        "delta",
        "pct_change",
        "baseline_run_id",
        "latest_run_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P27 release train candidate summary generator")
    parser.add_argument("--since-tag", default="")
    parser.add_argument("--since-run", default="")
    parser.add_argument("--candidate", default="")
    parser.add_argument("--out-dir", default="docs/artifacts/p27/release_train/latest")
    parser.add_argument("--trends-root", default="docs/artifacts/trends")
    parser.add_argument("--artifacts-root", default="docs/artifacts")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trends_root = Path(args.trends_root).resolve()
    artifacts_root = Path(args.artifacts_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    trend_rows = read_jsonl(trends_root / "trend_rows.jsonl")
    requested_since_tag = str(args.since_tag or "").strip()
    used_since_tag = requested_since_tag if tag_exists(requested_since_tag) else ""
    if not used_since_tag:
        used_since_tag = latest_tag()

    cutoff = cutoff_from_run(trend_rows, str(args.since_run or "").strip()) if args.since_run else None
    if cutoff is None:
        cutoff = tag_commit_time(used_since_tag)

    deltas = benchmark_deltas(trend_rows, cutoff)
    gate_snapshot_obj = gate_snapshot(trend_rows, cutoff)
    alert_summary, alert_report_path = latest_alert_summary(artifacts_root)
    status_publish_obj = load_status_publish(artifacts_root)

    candidate_obj = status_publish_obj.get("candidate", {}) if isinstance(status_publish_obj, dict) else {}
    champion_obj = status_publish_obj.get("champion", {}) if isinstance(status_publish_obj, dict) else {}
    release_state_obj = status_publish_obj.get("release_state", {}) if isinstance(status_publish_obj, dict) else {}
    if not isinstance(candidate_obj, dict):
        candidate_obj = {}
    if not isinstance(champion_obj, dict):
        champion_obj = {}
    if not isinstance(release_state_obj, dict):
        release_state_obj = {}

    release_bridge_md = out_dir / "release_notes_bridge.md"
    release_bridge_ok, release_bridge_stdout, release_bridge_stderr = call_release_notes(
        used_since_tag=used_since_tag,
        since_run=str(args.since_run or "").strip(),
        trends_root=trends_root,
        out_md=release_bridge_md,
    )
    release_bridge_json = release_bridge_md.with_suffix(".json")
    release_bridge_payload = read_json(release_bridge_json) if release_bridge_json.exists() else None

    rc_action, rc_reasons = recommendation(
        gate_snapshot_obj=gate_snapshot_obj,
        alert_summary=alert_summary,
        deltas=deltas,
        candidate_decision=str(candidate_obj.get("decision") or ""),
        release_action=str(release_state_obj.get("action") or ""),
    )

    benchmark_csv = out_dir / "benchmark_delta.csv"
    gate_snapshot_path = out_dir / "gate_snapshot.json"
    risk_snapshot_path = out_dir / "risk_snapshot.json"
    rc_json_path = out_dir / "rc_summary.json"
    rc_md_path = out_dir / "rc_summary.md"

    write_csv(benchmark_csv, deltas)
    gate_snapshot_path.write_text(json.dumps(gate_snapshot_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    risk_snapshot_obj = {
        "schema": "p27_risk_snapshot_v1",
        "generated_at": now_iso(),
        "alert_summary": alert_summary,
        "alert_report_path": alert_report_path,
        "candidate_decision": str(candidate_obj.get("decision") or ""),
        "candidate_reason": str(candidate_obj.get("reason") or ""),
        "release_action": str(release_state_obj.get("action") or ""),
        "release_reason": str(release_state_obj.get("reason") or ""),
    }
    risk_snapshot_path.write_text(json.dumps(risk_snapshot_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rc_summary = {
        "schema": "p27_rc_summary_v1",
        "generated_at": now_iso(),
        "requested_since_tag": requested_since_tag,
        "used_since_tag": used_since_tag,
        "used_since_run": str(args.since_run or ""),
        "cutoff_time": cutoff.isoformat() if cutoff else "",
        "candidate": str(args.candidate or candidate_obj.get("top_candidate_exp_id") or ""),
        "dry_run": bool(args.dry_run),
        "benchmarks": {
            "delta_count": len(deltas),
            "top_deltas": deltas[:25],
        },
        "gate_snapshot": gate_snapshot_obj,
        "risk_snapshot": risk_snapshot_obj,
        "champion": champion_obj,
        "candidate_state": candidate_obj,
        "release_state": release_state_obj,
        "release_notes_bridge": {
            "ok": release_bridge_ok,
            "md_path": str(release_bridge_md),
            "json_path": str(release_bridge_json) if release_bridge_json.exists() else "",
            "stdout": release_bridge_stdout,
            "stderr": release_bridge_stderr,
            "summary": release_bridge_payload.get("executive_summary", {}) if isinstance(release_bridge_payload, dict) else {},
        },
        "recommendation": {
            "action": rc_action,
            "reasons": rc_reasons,
        },
    }
    rc_json_path.write_text(json.dumps(rc_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# P27 Release Candidate Summary",
        "",
        f"- generated_at: {rc_summary['generated_at']}",
        f"- requested_since_tag: {requested_since_tag or 'N/A'}",
        f"- used_since_tag: {used_since_tag or 'N/A'}",
        f"- used_since_run: {rc_summary['used_since_run'] or 'N/A'}",
        f"- cutoff_time: {rc_summary['cutoff_time'] or 'N/A'}",
        f"- candidate: {rc_summary['candidate'] or 'N/A'}",
        f"- recommendation: {rc_action}",
        "",
        "## Decision Rationale",
    ]
    for reason in rc_reasons:
        md_lines.append(f"- {reason}")
    md_lines.extend(
        [
            "",
            "## Gate Snapshot",
            f"- latest_gate: {gate_snapshot_obj.get('latest_gate_name', '')}",
            f"- latest_status: {gate_snapshot_obj.get('latest_status', '')}",
            f"- gate_change_count: {gate_snapshot_obj.get('change_count', 0)}",
            "",
            "## Risk Snapshot",
            f"- hard_regression: {int(alert_summary.get('hard_regression', 0) or 0)}",
            f"- soft_regression: {int(alert_summary.get('soft_regression', 0) or 0)}",
            f"- noisy_needs_more_data: {int(alert_summary.get('noisy_needs_more_data', 0) or 0)}",
            f"- candidate_decision: {candidate_obj.get('decision', '')}",
            f"- release_action: {release_state_obj.get('action', '')}",
            "",
            "## Benchmark Deltas (Top 20)",
        ]
    )
    if deltas:
        for row in deltas[:20]:
            md_lines.append(
                f"- {row['metric_name']} | {row['gate_name']} | {row['strategy']}: "
                f"{row['baseline_value']:.6f} -> {row['latest_value']:.6f} (delta {row['delta']:.6f})"
            )
    else:
        md_lines.append("- no deltas available for selected window")
    rc_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "PASS",
                "used_since_tag": used_since_tag,
                "used_since_run": str(args.since_run or ""),
                "recommendation": rc_action,
                "rc_summary_md": str(rc_md_path),
                "rc_summary_json": str(rc_json_path),
                "benchmark_delta_csv": str(benchmark_csv),
                "gate_snapshot_json": str(gate_snapshot_path),
                "risk_snapshot_json": str(risk_snapshot_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
