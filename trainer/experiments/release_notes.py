from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_git(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, output.strip()


def _tag_exists(tag: str) -> bool:
    if not tag:
        return False
    code, _ = _run_git(["rev-parse", "--verify", f"refs/tags/{tag}"])
    return code == 0


def _latest_tag() -> str:
    code, out = _run_git(["describe", "--tags", "--abbrev=0"])
    if code == 0 and out:
        return out.splitlines()[-1].strip()
    code, out = _run_git(["tag", "--sort=-creatordate"])
    if code != 0 or not out:
        return ""
    return out.splitlines()[0].strip()


def _tag_commit_time(tag: str) -> str:
    if not tag:
        return ""
    code, out = _run_git(["show", "-s", "--format=%cI", tag])
    if code == 0 and out:
        return out.splitlines()[-1].strip()
    return ""


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _parse_ts(value: str) -> datetime | None:
    token = (value or "").strip()
    if not token:
        return None
    try:
        return datetime.fromisoformat(token.replace("Z", "+00:00"))
    except Exception:
        return None


def _cutoff_from_run(rows: list[dict[str, Any]], run_id: str) -> datetime | None:
    candidates = [r for r in rows if str(r.get("run_id") or "") == run_id]
    times = [_parse_ts(str(r.get("timestamp") or "")) for r in candidates]
    times = [t for t in times if t is not None]
    if not times:
        return None
    return sorted(times)[-1]


def _commit_rows(since_tag: str) -> list[dict[str, str]]:
    if since_tag:
        range_expr = f"{since_tag}..HEAD"
        args = ["log", range_expr, "--pretty=format:%H|%cI|%s"]
    else:
        args = ["log", "-n", "50", "--pretty=format:%H|%cI|%s"]
    code, out = _run_git(args)
    if code != 0 or not out:
        return []
    rows: list[dict[str, str]] = []
    for line in out.splitlines():
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        rows.append({"sha": parts[0], "time": parts[1], "subject": parts[2]})
    return rows


def _key_metric(metric_name: str) -> bool:
    return metric_name in {
        "avg_ante_reached",
        "median_ante_reached",
        "win_rate",
        "runtime_seconds",
        "flake_score",
        "gate_overall_pass",
        "gate_pass",
    }


def _latest_by_group(rows: list[dict[str, Any]], *, newer_than: datetime | None, older_equal: bool) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if not _key_metric(str(row.get("metric_name") or "")):
            continue
        ts = _parse_ts(str(row.get("timestamp") or ""))
        if ts is None:
            continue
        if newer_than is not None:
            if older_equal and ts > newer_than:
                continue
            if not older_equal and ts <= newer_than:
                continue
        key = (
            str(row.get("metric_name") or ""),
            str(row.get("gate_name") or ""),
            str(row.get("strategy") or "__all__"),
        )
        prev = out.get(key)
        if prev is None:
            out[key] = row
            continue
        prev_ts = _parse_ts(str(prev.get("timestamp") or ""))
        if prev_ts is None or ts > prev_ts:
            out[key] = row
    return out


def _benchmark_deltas(rows: list[dict[str, Any]], cutoff: datetime | None) -> list[dict[str, Any]]:
    before = _latest_by_group(rows, newer_than=cutoff, older_equal=True)
    after = _latest_by_group(rows, newer_than=cutoff, older_equal=False)
    deltas: list[dict[str, Any]] = []
    for key, latest in after.items():
        baseline = before.get(key)
        if baseline is None:
            continue
        latest_v = float(latest.get("metric_value") or 0.0)
        base_v = float(baseline.get("metric_value") or 0.0)
        delta = latest_v - base_v
        pct = 0.0 if abs(base_v) <= 1e-9 else delta / abs(base_v)
        deltas.append(
            {
                "metric_name": key[0],
                "gate_name": key[1],
                "strategy": key[2],
                "baseline_value": base_v,
                "latest_value": latest_v,
                "delta": delta,
                "pct_change": pct,
                "baseline_run_id": baseline.get("run_id"),
                "latest_run_id": latest.get("run_id"),
            }
        )
    deltas.sort(key=lambda r: abs(float(r.get("delta") or 0.0)), reverse=True)
    return deltas


def _gate_changes(rows: list[dict[str, Any]], cutoff: datetime | None) -> list[dict[str, Any]]:
    gate_rows = [r for r in rows if str(r.get("metric_name") or "") in {"gate_overall_pass", "gate_pass"}]
    before = _latest_by_group(gate_rows, newer_than=cutoff, older_equal=True)
    after = _latest_by_group(gate_rows, newer_than=cutoff, older_equal=False)
    changes: list[dict[str, Any]] = []
    for key, latest in after.items():
        baseline = before.get(key)
        if baseline is None:
            continue
        b = float(baseline.get("metric_value") or 0.0)
        a = float(latest.get("metric_value") or 0.0)
        if abs(a - b) <= 1e-9:
            continue
        changes.append(
            {
                "metric_name": key[0],
                "gate_name": key[1],
                "strategy": key[2],
                "before": b,
                "after": a,
                "baseline_run_id": baseline.get("run_id"),
                "latest_run_id": latest.get("run_id"),
            }
        )
    return sorted(changes, key=lambda r: (str(r.get("gate_name")), str(r.get("strategy"))))


def _failure_buckets(rows: list[dict[str, Any]], cutoff: datetime | None) -> list[dict[str, Any]]:
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        gate = str(row.get("gate_name") or "unknown")
        status = str(row.get("status") or "unknown").lower()
        ts = _parse_ts(str(row.get("timestamp") or ""))
        if ts is None:
            continue
        bucket = out.setdefault(gate, {"before_fail": 0, "after_fail": 0})
        if cutoff is None or ts > cutoff:
            if status != "pass":
                bucket["after_fail"] += 1
        else:
            if status != "pass":
                bucket["before_fail"] += 1
    rows_out = []
    for gate, values in out.items():
        rows_out.append(
            {
                "gate_name": gate,
                "before_fail": values["before_fail"],
                "after_fail": values["after_fail"],
                "delta_fail": values["after_fail"] - values["before_fail"],
            }
        )
    rows_out.sort(key=lambda r: abs(int(r["delta_fail"])), reverse=True)
    return rows_out


def _capabilities_from_commits(commits: list[dict[str, str]]) -> list[str]:
    hints: list[str] = []
    for c in commits:
        subject = str(c.get("subject") or "").strip()
        s = subject.lower()
        if not subject:
            continue
        if any(k in s for k in ("trend", "warehouse", "benchmark")):
            hints.append(subject)
        elif any(k in s for k in ("regression", "alert", "flake", "risk")):
            hints.append(subject)
        elif any(k in s for k in ("release", "notes", "summary")):
            hints.append(subject)
        elif any(k in s for k in ("nightly", "scheduler", "run_p26")):
            hints.append(subject)
        elif any(k in s for k in ("spec", "readme", "docs")):
            hints.append(subject)
    # keep order, unique
    dedup: list[str] = []
    seen: set[str] = set()
    for h in hints:
        if h in seen:
            continue
        seen.add(h)
        dedup.append(h)
    return dedup[:20]


def _latest_alert_summary() -> dict[str, Any]:
    path = Path("docs/artifacts/p26/alerts_latest/regression_alert_report.json")
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    except Exception:
        return {}
    return {}


def _recommended_action(alert_summary: dict[str, Any], gate_changes: list[dict[str, Any]]) -> str:
    hard = int(alert_summary.get("hard_regression", 0) or 0)
    soft = int(alert_summary.get("soft_regression", 0) or 0)
    improve = int(alert_summary.get("improvement", 0) or 0)
    gate_regressed = any(float(x.get("before", 1.0)) > float(x.get("after", 1.0)) for x in gate_changes)
    if hard > 0 or gate_regressed:
        return "investigate"
    if soft > 0:
        return "hold"
    if improve > 0:
        return "promote"
    return "hold"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate P26 release/benchmark summary notes")
    parser.add_argument("--since-tag", default="")
    parser.add_argument("--since-run", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--include-commits", action="store_true")
    parser.add_argument("--include-benchmarks", action="store_true")
    parser.add_argument("--include-risks", action="store_true")
    parser.add_argument("--trends-root", default="docs/artifacts/trends")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_md = Path(args.out).resolve()
    out_json = out_md.with_suffix(".json")
    out_md.parent.mkdir(parents=True, exist_ok=True)

    requested_tag = str(args.since_tag or "").strip()
    used_tag = requested_tag if _tag_exists(requested_tag) else ""
    if not used_tag:
        used_tag = _latest_tag()
    since_tag_commit_time = _tag_commit_time(used_tag)

    trend_rows = _load_jsonl(Path(args.trends_root).resolve() / "trend_rows.jsonl")
    cutoff: datetime | None = None
    if args.since_run:
        cutoff = _cutoff_from_run(trend_rows, args.since_run)
    if cutoff is None and since_tag_commit_time:
        cutoff = _parse_ts(since_tag_commit_time)

    commits = _commit_rows(used_tag) if args.include_commits else []
    benchmark_deltas = _benchmark_deltas(trend_rows, cutoff) if args.include_benchmarks else []
    gate_changes = _gate_changes(trend_rows, cutoff) if args.include_benchmarks else []
    failure_buckets = _failure_buckets(trend_rows, cutoff) if args.include_benchmarks else []
    capabilities = _capabilities_from_commits(commits)
    alert_summary = _latest_alert_summary() if args.include_risks else {}
    recommendation = _recommended_action(alert_summary, gate_changes)

    report = {
        "schema": "p26_release_summary_v1",
        "generated_at": _now_iso(),
        "requested_since_tag": requested_tag,
        "used_since_tag": used_tag,
        "used_since_run": args.since_run,
        "cutoff_time": cutoff.isoformat() if cutoff else "",
        "executive_summary": {
            "commit_count": len(commits),
            "benchmark_delta_count": len(benchmark_deltas),
            "gate_change_count": len(gate_changes),
            "recommendation": recommendation,
        },
        "what_changed": capabilities,
        "benchmark_deltas": benchmark_deltas[:50],
        "gate_changes": gate_changes,
        "failure_bucket_changes": failure_buckets[:30],
        "reliability_notes": alert_summary,
        "recommended_next_action": recommendation,
    }

    # Markdown summary
    lines = [
        "# P26 Release / Benchmark Summary",
        "",
        "## Executive Summary",
        f"- generated_at: {report['generated_at']}",
        f"- requested_since_tag: {requested_tag or 'N/A'}",
        f"- used_since_tag: {used_tag or 'N/A'}",
        f"- used_since_run: {args.since_run or 'N/A'}",
        f"- cutoff_time: {report['cutoff_time'] or 'N/A'}",
        f"- commit_count: {len(commits)}",
        f"- benchmark_delta_count: {len(benchmark_deltas)}",
        f"- gate_change_count: {len(gate_changes)}",
        f"- recommended_next_action: {recommendation}",
        "",
        "## What Changed",
    ]
    if capabilities:
        for item in capabilities:
            lines.append(f"- {item}")
    else:
        lines.append("- no explicit capability hints from commits")

    lines.extend(["", "## Benchmark Deltas"])
    if benchmark_deltas:
        for item in benchmark_deltas[:25]:
            lines.append(
                f"- {item['metric_name']} | {item['gate_name']} | {item['strategy']}: {item['baseline_value']:.6f} -> {item['latest_value']:.6f} (delta {item['delta']:.6f})"
            )
    else:
        lines.append("- no benchmark deltas found for the selected window")

    lines.extend(["", "## Gate Status Changes"])
    if gate_changes:
        for item in gate_changes:
            lines.append(f"- {item['gate_name']} ({item['strategy']}): {item['before']} -> {item['after']}")
    else:
        lines.append("- no gate pass-state change detected")

    lines.extend(["", "## Reliability Notes"])
    if alert_summary:
        for k in ("hard_regression", "soft_regression", "noisy_needs_more_data", "improvement", "no_signal"):
            lines.append(f"- {k}: {int(alert_summary.get(k, 0) or 0)}")
    else:
        lines.append("- no regression alert summary found")

    lines.extend(["", "## Recommended Next Action", f"- {recommendation}"])

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "requested_since_tag": requested_tag,
                "used_since_tag": used_tag,
                "used_since_run": args.since_run,
                "out_md": str(out_md),
                "out_json": str(out_json),
                "recommendation": recommendation,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
