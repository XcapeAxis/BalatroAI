from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRIC_RE = re.compile(r"([A-Za-z0-9_]+)\s*=\s*([-+]?[0-9]*\.?[0-9]+)")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_metric_snapshot(text: str) -> tuple[str, float | None]:
    if not text:
        return "", None
    m = METRIC_RE.search(text)
    if not m:
        return text, None
    key = m.group(1)
    try:
        value = float(m.group(2))
    except Exception:
        value = None
    return key, value


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                out.append(payload)
        except Exception:
            continue
    return out


def build_snapshot(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_exp: dict[str, dict[str, Any]] = {}
    failure_categories: dict[str, int] = {}
    top_metric = None
    top_exp = ""
    metric_key = ""

    campaign_id = ""
    stage_id = ""
    run_id = ""
    for ev in events:
        exp_id = str(ev.get("exp_id") or "")
        if not exp_id:
            continue
        run_id = str(ev.get("run_id") or run_id)
        campaign_id = str(ev.get("campaign_id") or campaign_id)
        stage_id = str(ev.get("campaign_stage") or stage_id)
        by_exp[exp_id] = ev
        status = str(ev.get("status") or "").lower()
        if status in {"failed", "timed_out", "budget_cut"}:
            failure_categories[status] = failure_categories.get(status, 0) + 1
        snap_text = str(ev.get("metric_snapshot") or "")
        k, v = _parse_metric_snapshot(snap_text)
        if v is not None and (top_metric is None or v > top_metric):
            top_metric = v
            top_exp = exp_id
            metric_key = k

    total = len(by_exp)
    completed = 0
    failed = 0
    skipped = 0
    current_exp = ""
    current_stage = ""
    current_seed = "-"
    elapsed = 0.0
    eta = None
    metric_snapshot = ""
    for exp_id, ev in by_exp.items():
        status = str(ev.get("status") or "").lower()
        if status in {"passed", "failed", "timed_out", "budget_cut", "skipped"}:
            completed += 1
        if status in {"failed", "timed_out", "budget_cut"}:
            failed += 1
        if status == "skipped":
            skipped += 1
        e = float(ev.get("elapsed_sec") or 0.0)
        if e >= elapsed:
            elapsed = e
            eta = ev.get("eta_sec")
            current_exp = exp_id
            current_stage = str(ev.get("stage") or "")
            seed_idx = ev.get("seed_index")
            seed_total = ev.get("seed_total")
            if seed_idx and seed_total:
                current_seed = f"{seed_idx}/{seed_total}"
            metric_snapshot = str(ev.get("metric_snapshot") or "")

    return {
        "generated_at": now_iso(),
        "run_id": run_id,
        "campaign_id": campaign_id or "N/A",
        "campaign_stage": stage_id or current_stage or "N/A",
        "total_experiments": total,
        "completed_experiments": completed,
        "failed_experiments": failed,
        "skipped_experiments": skipped,
        "budget_usage": {
            "experiments_done_pct": (completed / total) if total > 0 else 0.0,
            "elapsed_sec": elapsed,
            "eta_sec": eta,
        },
        "current": {
            "exp_id": current_exp or "N/A",
            "stage": current_stage or "N/A",
            "seed_progress": current_seed,
            "metric_snapshot": metric_snapshot,
        },
        "failure_categories": failure_categories,
        "top_candidate": {
            "exp_id": top_exp or "N/A",
            "metric_key": metric_key or "N/A",
            "metric_value": top_metric,
        },
    }


def format_snapshot(snapshot: dict[str, Any]) -> str:
    lines = [
        f"[dashboard] ts={snapshot.get('generated_at')}",
        f"campaign={snapshot.get('campaign_id')} stage={snapshot.get('campaign_stage')} run_id={snapshot.get('run_id')}",
        (
            "progress total={total} completed={done} failed={failed} skipped={skipped}".format(
                total=snapshot.get("total_experiments"),
                done=snapshot.get("completed_experiments"),
                failed=snapshot.get("failed_experiments"),
                skipped=snapshot.get("skipped_experiments"),
            )
        ),
        (
            "current exp={exp} stage={stage} seed={seed} elapsed={elapsed} eta={eta} metric={metric}".format(
                exp=(snapshot.get("current") or {}).get("exp_id"),
                stage=(snapshot.get("current") or {}).get("stage"),
                seed=(snapshot.get("current") or {}).get("seed_progress"),
                elapsed=(snapshot.get("budget_usage") or {}).get("elapsed_sec"),
                eta=(snapshot.get("budget_usage") or {}).get("eta_sec"),
                metric=(snapshot.get("current") or {}).get("metric_snapshot"),
            )
        ),
        f"failures={snapshot.get('failure_categories')}",
        f"top_candidate={snapshot.get('top_candidate')}",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P24 telemetry dashboard (TUI/headless)")
    p.add_argument("--watch", required=True, help="telemetry.jsonl path")
    p.add_argument("--headless-log", action="store_true")
    p.add_argument("--out", default="")
    p.add_argument("--interval-sec", type=float, default=1.0)
    p.add_argument("--duration-sec", type=float, default=5.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    watch_path = Path(args.watch).resolve()
    if not watch_path.exists():
        raise SystemExit(f"telemetry not found: {watch_path}")

    duration = max(0.5, float(args.duration_sec))
    interval = max(0.2, float(args.interval_sec))
    out_lines: list[str] = []
    started = time.time()
    while True:
        events = read_events(watch_path)
        snapshot = build_snapshot(events)
        text = format_snapshot(snapshot)
        if args.headless_log:
            out_lines.append(text)
        else:
            print("\x1b[2J\x1b[H" + text)
        if time.time() - started >= duration:
            break
        time.sleep(interval)

    if args.headless_log:
        out_path = Path(args.out).resolve() if args.out else (watch_path.parent / "dashboard_headless_log.txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n\n".join(out_lines).rstrip() + "\n", encoding="utf-8")
        print(json.dumps({"status": "PASS", "out": str(out_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

