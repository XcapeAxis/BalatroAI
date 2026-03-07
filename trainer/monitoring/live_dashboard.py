from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as fp:
        for line in fp:
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def collect_latest_events(root: Path) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in root.glob("**/*progress*.jsonl"):
        for row in _read_jsonl(path):
            if str(row.get("schema") or "") != "p49_progress_event_v1":
                continue
            key = (
                str(row.get("run_id") or ""),
                str(row.get("component") or ""),
                str(row.get("seed") or ""),
            )
            latest[key] = row
    return sorted(
        latest.values(),
        key=lambda row: (
            str(row.get("run_id") or ""),
            str(row.get("component") or ""),
            str(row.get("seed") or ""),
        ),
    )


def collect_campaign_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/campaign_state.json"), key=lambda item: str(item), reverse=True)[:8]:
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        stages = [dict(item) for item in (payload.get("stages") or []) if isinstance(item, dict)]
        active = next((item for item in stages if str(item.get("status") or "") == "running"), None)
        failed = next((item for item in stages if str(item.get("status") or "") == "failed"), None)
        stage = active or failed or (stages[-1] if stages else {})
        rows.append(
            {
                "campaign_id": str(payload.get("campaign_id") or ""),
                "experiment_id": str(payload.get("experiment_id") or ""),
                "seed": str(payload.get("seed") or ""),
                "stage_id": str((stage or {}).get("stage_id") or ""),
                "status": str((stage or {}).get("status") or ""),
            }
        )
    return rows


def collect_registry_summary(root: Path) -> dict[str, Any]:
    payload = _read_json(root / "registry" / "checkpoints_registry.json")
    items = payload.get("items") if isinstance(payload, dict) and isinstance(payload.get("items"), list) else []
    counts: dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        token = str(item.get("status") or "draft")
        counts[token] = int(counts.get(token, 0)) + 1
    return {"count": len(items), "status_counts": counts}


def render_text(rows: list[dict[str, Any]], campaign_rows: list[dict[str, Any]], registry_summary: dict[str, Any]) -> str:
    lines = [
        "[dashboard] P49/P51 live progress",
        "run_id            component             phase       status    learner      rollout      throughput   gpu_mb   warning",
        "-" * 112,
    ]
    for row in rows:
        lines.append(
            "{run:<16} {component:<20} {phase:<11} {status:<9} {learner:<12} {rollout:<12} {throughput:<11} {gpu:<8} {warning}".format(
                run=str(row.get("run_id") or "")[:16],
                component=str(row.get("component") or "")[:20],
                phase=str(row.get("phase") or "")[:11],
                status=str(row.get("status") or "")[:9],
                learner=str(row.get("learner_device") or "")[:12],
                rollout=str(row.get("rollout_device") or "")[:12],
                throughput=("{:.2f}".format(float(row.get("throughput"))) if row.get("throughput") is not None else "-"),
                gpu=("{:.1f}".format(float(row.get("gpu_mem_mb"))) if row.get("gpu_mem_mb") is not None else "-"),
                warning=str(row.get("warning") or "")[:60],
            )
        )
    lines.extend(
        [
            "",
            "[campaigns]",
            "campaign_id                 experiment              seed      stage                status",
            "-" * 96,
        ]
    )
    for row in campaign_rows:
        lines.append(
            "{campaign:<26} {exp:<22} {seed:<9} {stage:<20} {status}".format(
                campaign=str(row.get("campaign_id") or "")[:26],
                exp=str(row.get("experiment_id") or "")[:22],
                seed=str(row.get("seed") or "")[:9],
                stage=str(row.get("stage_id") or "")[:20],
                status=str(row.get("status") or "")[:12],
            )
        )
    lines.extend(
        [
            "",
            "[registry]",
            "count={count} status_counts={counts}".format(
                count=int(registry_summary.get("count") or 0),
                counts=json.dumps(registry_summary.get("status_counts") or {}, ensure_ascii=False),
            ),
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch unified P49 progress events in the terminal.")
    parser.add_argument("--watch", default="docs/artifacts")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    watch_root = Path(args.watch)
    if not watch_root.is_absolute():
        watch_root = (Path(__file__).resolve().parents[2] / watch_root).resolve()
    iteration = 0
    while True:
        rows = collect_latest_events(watch_root)
        campaign_rows = collect_campaign_rows(watch_root)
        registry_summary = collect_registry_summary(watch_root)
        os.system("cls" if os.name == "nt" else "clear")
        print(render_text(rows, campaign_rows, registry_summary))
        iteration += 1
        if bool(args.once) or (int(args.iterations) > 0 and iteration >= int(args.iterations)):
            break
        time.sleep(max(0.5, float(args.interval)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
