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


def render_text(rows: list[dict[str, Any]]) -> str:
    lines = [
        "[dashboard] P49 live progress",
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
        os.system("cls" if os.name == "nt" else "clear")
        print(render_text(rows))
        iteration += 1
        if bool(args.once) or (int(args.iterations) > 0 and iteration >= int(args.iterations)):
            break
        time.sleep(max(0.5, float(args.interval)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
