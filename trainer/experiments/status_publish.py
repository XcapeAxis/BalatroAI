from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dashboard_data import build_dashboard_data, infer_trend_signal, metric_snapshot


RUN_GATE_RE = re.compile(r"report_p(\d+)_gate\.json$", re.IGNORECASE)
RUN_SWITCH_RE = re.compile(r"RunP(\d+)")
DOCS_COVERAGE_RE = re.compile(r"^COVERAGE_P(\d+)_STATUS\.md$", re.IGNORECASE)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)
    except Exception:
        try:
            payload = json.loads(text.lstrip("\ufeff"))
        except Exception:
            return None
    return payload if isinstance(payload, dict) else None


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def git_text(args: list[str], default: str = "") -> str:
    code, out = run_git(args)
    if code != 0 or not out:
        return default
    return out.splitlines()[-1].strip()


def detected_main_branch() -> str:
    token = git_text(["symbolic-ref", "refs/remotes/origin/HEAD"], default="")
    if token.startswith("refs/remotes/origin/"):
        return token.split("/")[-1]
    if run_git(["show-ref", "--verify", "--quiet", "refs/heads/main"])[0] == 0:
        return "main"
    if run_git(["show-ref", "--verify", "--quiet", "refs/heads/master"])[0] == 0:
        return "master"
    return "main"


def highest_supported_gate(run_regressions_path: Path) -> int:
    if not run_regressions_path.exists():
        return 0
    text = run_regressions_path.read_text(encoding="utf-8")
    matches = [int(m.group(1)) for m in RUN_SWITCH_RE.finditer(text)]
    return max(matches) if matches else 0


def latest_gate_report(artifacts_root: Path) -> dict[str, Any]:
    candidates: list[tuple[int, float, Path]] = []
    for milestone_dir in artifacts_root.glob("p*"):
        if not milestone_dir.is_dir():
            continue
        for path in milestone_dir.rglob("report_p*_gate.json"):
            m = RUN_GATE_RE.search(path.name)
            if not m:
                continue
            try:
                gate_num = int(m.group(1))
            except Exception:
                continue
            try:
                mtime = path.stat().st_mtime
            except Exception:
                mtime = 0.0
            candidates.append((gate_num, mtime, path))
    if not candidates:
        return {
            "gate_name": "unknown",
            "gate_number": 0,
            "status": "UNKNOWN",
            "pass": False,
            "report_path": "",
            "generated_at": "",
        }
    highest_gate = max(item[0] for item in candidates)
    latest = max([item for item in candidates if item[0] == highest_gate], key=lambda item: item[1])
    payload = read_json(latest[2]) or {}
    status = str(payload.get("status") or "UNKNOWN").strip().upper()
    return {
        "gate_name": f"RunP{highest_gate}",
        "gate_number": highest_gate,
        "status": status,
        "pass": status == "PASS",
        "report_path": str(latest[2].resolve()),
        "generated_at": str(payload.get("generated_at") or ""),
    }


def latest_alert_summary(artifacts_root: Path) -> tuple[dict[str, Any], str]:
    candidates: list[Path] = []
    p26_root = artifacts_root / "p26"
    if p26_root.exists():
        candidates.extend(p26_root.rglob("regression_alert_report.json"))
    if not candidates:
        candidates.extend(artifacts_root.rglob("regression_alert_report.json"))
    if not candidates:
        return {}, ""
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    payload = read_json(latest) or {}
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        summary = {}
    return summary, str(latest.resolve())


def latest_named_file(artifacts_root: Path, file_name: str) -> Path | None:
    roots = [artifacts_root / "p24", artifacts_root / "p23", artifacts_root / "p22", artifacts_root / "p20"]
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            candidates.extend(root.rglob(file_name))
    if not candidates:
        candidates.extend(artifacts_root.rglob(file_name))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def summarize_champion(artifacts_root: Path) -> dict[str, Any]:
    path = latest_named_file(artifacts_root, "champion.json")
    if path is None:
        return {"exists": False}
    payload = read_json(path) or {}
    return {
        "exists": True,
        "path": str(path.resolve()),
        "exp_id": str(payload.get("exp_id") or ""),
        "status": str(payload.get("status") or ""),
        "reason": str(payload.get("reason") or ""),
        "updated_at": str(payload.get("updated_at") or payload.get("generated_at") or ""),
        "run_id": str(payload.get("updated_by_run") or payload.get("run_id") or ""),
        "avg_ante_reached": maybe_float(payload.get("avg_ante_reached")),
        "median_ante_reached": maybe_float(payload.get("median_ante_reached") or payload.get("median_ante")),
        "win_rate": maybe_float(payload.get("win_rate")),
        "weighted_score": maybe_float(payload.get("weighted_score")),
    }


def summarize_candidate(artifacts_root: Path) -> dict[str, Any]:
    path = latest_named_file(artifacts_root, "candidate.json")
    if path is None:
        return {"exists": False}
    payload = read_json(path) or {}
    top = payload.get("top_candidate") if isinstance(payload.get("top_candidate"), dict) else {}
    return {
        "exists": True,
        "path": str(path.resolve()),
        "decision": str(payload.get("decision") or ""),
        "reason": str(payload.get("reason") or ""),
        "generated_at": str(payload.get("generated_at") or payload.get("updated_at") or ""),
        "run_id": str(payload.get("run_id") or ""),
        "top_candidate_exp_id": str(top.get("exp_id") or ""),
        "top_candidate_status": str(top.get("status") or ""),
        "top_candidate_avg_ante_reached": maybe_float(top.get("avg_ante_reached")),
        "top_candidate_median_ante_reached": maybe_float(top.get("median_ante_reached") or top.get("median_ante")),
        "top_candidate_win_rate": maybe_float(top.get("win_rate")),
        "top_candidate_weighted_score": maybe_float(top.get("weighted_score")),
    }


def summarize_release_state(artifacts_root: Path) -> dict[str, Any]:
    path = latest_named_file(artifacts_root, "release_state.json")
    if path is None:
        return {"exists": False}
    payload = read_json(path) or {}
    return {
        "exists": True,
        "path": str(path.resolve()),
        "channel": str(payload.get("channel") or ""),
        "action": str(payload.get("action") or ""),
        "reason": str(payload.get("reason") or ""),
        "generated_at": str(payload.get("generated_at") or ""),
        "perf_gate_pass": bool(payload.get("perf_gate_pass", False)),
        "risk_guard_pass": bool(payload.get("risk_guard_pass", False)),
        "canary_pass": bool(payload.get("canary_pass", False)),
    }


def docs_coverage_status(docs_root: Path, highest_gate: int) -> dict[str, Any]:
    coverage_ids: list[int] = []
    for item in docs_root.glob("COVERAGE_P*_STATUS.md"):
        m = DOCS_COVERAGE_RE.match(item.name)
        if m:
            coverage_ids.append(int(m.group(1)))
    coverage_ids = sorted(set(coverage_ids))
    low = 15
    high = max(highest_gate, coverage_ids[-1] if coverage_ids else 0)
    if high < low:
        high = low
    return {
        "range": f"P{low}-P{high}",
        "min": low,
        "max": high,
        "files_present": [f"P{n}" for n in coverage_ids],
    }


def repository_meta(repo_root: Path) -> dict[str, Any]:
    branch = git_text(["rev-parse", "--abbrev-ref", "HEAD"], default="unknown")
    main_branch = detected_main_branch()
    status_short = git_text(["status", "--porcelain"], default="")
    return {
        "repo_root": str(repo_root.resolve()),
        "branch": branch,
        "detected_main_branch": main_branch,
        "on_mainline": branch == main_branch,
        "working_tree_clean": len(status_short.strip()) == 0,
        "mainline_only_workflow": True,
    }


def build_badges(
    *,
    latest_gate: dict[str, Any],
    repo_meta_obj: dict[str, Any],
    seed_enabled: bool,
    orchestrator_enabled: bool,
    trend_enabled: bool,
    docs_coverage: dict[str, Any],
    python_badge: str,
    license_badge: str,
) -> dict[str, Any]:
    gate_status = str(latest_gate.get("status") or "UNKNOWN").upper()
    gate_color = "2EA44F" if gate_status == "PASS" else ("D73A49" if gate_status == "FAIL" else "6E7781")
    badges = [
        {
            "id": "latest_gate",
            "label": "Latest Gate",
            "message": f"{latest_gate.get('gate_name', 'unknown')} {gate_status}",
            "color": gate_color,
            "link": "scripts/run_regressions.ps1",
        },
        {
            "id": "mainline_workflow",
            "label": "Workflow",
            "message": "mainline-only",
            "color": "2EA44F" if bool(repo_meta_obj.get("on_mainline")) else "6E7781",
            "link": "scripts/git_sync.ps1",
        },
        {
            "id": "seed_governance",
            "label": "Seed Governance",
            "message": "P23+ enabled" if seed_enabled else "missing",
            "color": "0E8A16" if seed_enabled else "D73A49",
            "link": "configs/experiments/seeds_p23.yaml",
        },
        {
            "id": "orchestrator",
            "label": "Experiment Orchestrator",
            "message": "P22+ enabled" if orchestrator_enabled else "missing",
            "color": "1F6FEB" if orchestrator_enabled else "D73A49",
            "link": "scripts/run_p22.ps1",
        },
        {
            "id": "trend_warehouse",
            "label": "Trend Warehouse",
            "message": "P26+ enabled" if trend_enabled else "missing",
            "color": "0E8A16" if trend_enabled else "D73A49",
            "link": "docs/TREND_WAREHOUSE_P26.md",
        },
        {
            "id": "docs_coverage",
            "label": "Docs Coverage",
            "message": str(docs_coverage.get("range") or "unknown"),
            "color": "6E7781",
            "link": "docs/",
        },
        {
            "id": "platform",
            "label": "Platform",
            "message": "Windows",
            "color": "0078D6",
            "link": "USAGE_GUIDE.md",
        },
        {
            "id": "python",
            "label": "Python",
            "message": python_badge,
            "color": "3776AB",
            "link": "trainer/requirements.txt",
        },
        {
            "id": "license",
            "label": "License",
            "message": license_badge,
            "color": "6E7781",
            "link": "#license-and-contributing",
        },
    ]
    return {
        "schema": "p27_badges_v1",
        "generated_at": now_iso(),
        "badges": badges,
    }


def status_markdown(status: dict[str, Any]) -> str:
    latest_gate_obj = status.get("latest_gate", {})
    trend_obj = status.get("benchmark_snapshot", {})
    champion_obj = status.get("champion", {})
    candidate_obj = status.get("candidate", {})
    trend_warehouse_obj = status.get("trend_warehouse", {})
    lines = [
        "### Repository Status (Auto-generated, P27)",
        "",
        f"- branch: {status.get('repo', {}).get('branch', 'unknown')}",
        f"- latest_gate: {latest_gate_obj.get('gate_name', 'unknown')} ({latest_gate_obj.get('status', 'UNKNOWN')})",
        f"- recent_trend_signal: {trend_obj.get('trend_signal', 'unknown')}",
        f"- trend_warehouse_last_updated: {trend_warehouse_obj.get('last_updated', '')}",
        f"- trend_rows_count: {trend_warehouse_obj.get('rows_count', 0)}",
        f"- champion: {champion_obj.get('exp_id', 'n/a')} ({champion_obj.get('status', 'n/a')})",
        f"- candidate: {candidate_obj.get('top_candidate_exp_id', 'n/a')} (decision: {candidate_obj.get('decision', 'n/a')})",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P27 status publishing generator")
    parser.add_argument("--trends-root", default="docs/artifacts/trends")
    parser.add_argument("--artifacts-root", default="docs/artifacts")
    parser.add_argument("--out-root", default="docs/artifacts/status")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(".").resolve()
    artifacts_root = Path(args.artifacts_root).resolve()
    trends_root = Path(args.trends_root).resolve()
    out_root = Path(args.out_root).resolve()
    docs_root = (repo_root / "docs").resolve()

    trend_rows_path = trends_root / "trend_rows.jsonl"
    trend_summary_path = trends_root / "trend_index_summary.json"
    trend_summary = read_json(trend_summary_path) or {}
    trend_rows = read_jsonl(trend_rows_path)

    latest_gate_obj = latest_gate_report(artifacts_root)
    alert_summary, alert_path = latest_alert_summary(artifacts_root)

    champion = summarize_champion(artifacts_root)
    candidate = summarize_candidate(artifacts_root)
    release_state = summarize_release_state(artifacts_root)

    avg_snapshot = metric_snapshot(trend_rows, "avg_ante_reached")
    median_snapshot = metric_snapshot(trend_rows, "median_ante_reached")
    trend_signal = infer_trend_signal(alert_summary, [avg_snapshot, median_snapshot])

    script_gate = highest_supported_gate(repo_root / "scripts" / "run_regressions.ps1")
    docs_coverage = docs_coverage_status(docs_root, script_gate)

    repo_meta_obj = repository_meta(repo_root)
    seed_enabled = (repo_root / "configs" / "experiments" / "seeds_p23.yaml").exists()
    orchestrator_enabled = (repo_root / "scripts" / "run_p22.ps1").exists() and (
        repo_root / "trainer" / "experiments" / "orchestrator.py"
    ).exists()
    trend_enabled = trend_rows_path.exists() and trend_summary_path.exists()
    campaign_enabled = (repo_root / "scripts" / "run_p24.ps1").exists()
    python_badge = "3.12+"
    license_badge = "Not Specified"
    if (repo_root / "LICENSE").exists() or (repo_root / "LICENSE.md").exists():
        license_badge = "See License"

    trend_updated = ""
    if trend_rows_path.exists():
        trend_updated = datetime.fromtimestamp(trend_rows_path.stat().st_mtime, tz=timezone.utc).isoformat()

    status_obj = {
        "schema": "p27_status_publish_v1",
        "generated_at": now_iso(),
        "repo": repo_meta_obj,
        "latest_gate": latest_gate_obj,
        "seed_governance": {
            "enabled": seed_enabled,
            "source": "configs/experiments/seeds_p23.yaml",
            "label": "P23+",
        },
        "experiment_platform": {
            "orchestrator_enabled": orchestrator_enabled,
            "campaign_manager_enabled": campaign_enabled,
            "seed_governance_enabled": seed_enabled,
            "status": "ready" if (orchestrator_enabled and campaign_enabled and seed_enabled) else "partial",
        },
        "trend_warehouse": {
            "enabled": trend_enabled,
            "rows_count": int(trend_summary.get("rows_total", len(trend_rows)) or 0),
            "last_updated": trend_updated,
            "trend_rows_path": str(trend_rows_path),
            "trend_index_summary_path": str(trend_summary_path),
        },
        "regression_alerts": {
            "summary": alert_summary,
            "report_path": alert_path,
        },
        "benchmark_snapshot": {
            "trend_signal": trend_signal,
            "avg_ante_reached": avg_snapshot,
            "median_ante_reached": median_snapshot,
        },
        "champion": champion,
        "candidate": candidate,
        "release_state": release_state,
        "docs_coverage": docs_coverage,
        "highest_supported_gate": f"RunP{script_gate}" if script_gate > 0 else "unknown",
        "sources": {
            "trends_root": str(trends_root),
            "artifacts_root": str(artifacts_root),
            "generated_readme_status": str((repo_root / "docs" / "generated" / "README_STATUS.md").resolve()),
        },
    }

    badges_obj = build_badges(
        latest_gate=latest_gate_obj,
        repo_meta_obj=repo_meta_obj,
        seed_enabled=seed_enabled,
        orchestrator_enabled=orchestrator_enabled,
        trend_enabled=trend_enabled,
        docs_coverage=docs_coverage,
        python_badge=python_badge,
        license_badge=license_badge,
    )
    dashboard_data_obj = build_dashboard_data(
        sources={
            "trend_rows_jsonl": str(trend_rows_path),
            "trend_index_summary": str(trend_summary_path),
            "regression_alert_report": alert_path,
            "latest_gate_report": str(latest_gate_obj.get("report_path") or ""),
        },
        latest_gate=latest_gate_obj,
        alert_summary=alert_summary,
        trend_rows=trend_rows,
        champion_summary=champion,
        candidate_summary=candidate,
        release_summary=release_state,
    )
    dashboard_data_obj["repo"] = repo_meta_obj

    out_root.mkdir(parents=True, exist_ok=True)
    status_json_path = out_root / "latest_status.json"
    status_md_path = out_root / "latest_status.md"
    badges_json_path = out_root / "latest_badges.json"
    dashboard_json_path = out_root / "latest_dashboard_data.json"
    summary_json_path = out_root / "status_publish_summary.json"

    write_json(status_json_path, status_obj)
    status_md_path.write_text(status_markdown(status_obj), encoding="utf-8")
    write_json(badges_json_path, badges_obj)
    write_json(dashboard_json_path, dashboard_data_obj)
    write_json(
        summary_json_path,
        {
            "schema": "p27_status_publish_summary_v1",
            "generated_at": now_iso(),
            "status": "PASS",
            "out_files": {
                "latest_status_json": str(status_json_path),
                "latest_status_md": str(status_md_path),
                "latest_badges_json": str(badges_json_path),
                "latest_dashboard_data_json": str(dashboard_json_path),
            },
            "latest_gate": latest_gate_obj,
            "trend_signal": trend_signal,
            "rows_count": len(trend_rows),
        },
    )

    print(
        json.dumps(
            {
                "status": "PASS",
                "latest_status_json": str(status_json_path),
                "latest_status_md": str(status_md_path),
                "latest_badges_json": str(badges_json_path),
                "latest_dashboard_data_json": str(dashboard_json_path),
                "trend_signal": trend_signal,
                "latest_gate": latest_gate_obj.get("gate_name", "unknown"),
                "latest_gate_status": latest_gate_obj.get("status", "UNKNOWN"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
