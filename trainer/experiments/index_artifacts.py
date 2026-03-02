from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .trends import (
    build_index_summary,
    dedupe_rows,
    index_artifacts,
    load_trend_rows,
    query_rows,
    write_trend_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P26 trend warehouse artifact indexer")
    parser.add_argument("--scan-root", default="docs/artifacts")
    parser.add_argument("--out-root", default="docs/artifacts/trends")
    parser.add_argument("--append", action="store_true", help="Append new rows onto existing trend rows")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild trend rows from scratch")
    parser.add_argument("--latest-only", action="store_true", help="Index only latest run folders for smoke")
    parser.add_argument("--query-milestone", default="")
    parser.add_argument("--query-strategy", default="")
    parser.add_argument("--query-gate", default="")
    parser.add_argument("--query-run-id", default="")
    parser.add_argument("--query-out", default="")
    return parser.parse_args()


def _write_query(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.append and args.rebuild:
        raise ValueError("choose either --append or --rebuild, not both")

    scan_root = Path(args.scan_root).resolve()
    out_root = Path(args.out_root).resolve()

    existing_rows: list[dict[str, Any]] = []
    mode = "rebuild"
    if args.append and not args.rebuild:
        mode = "append"
        existing_rows = load_trend_rows(out_root)
    elif not args.rebuild:
        # default to append semantics for iterative local runs.
        mode = "append"
        existing_rows = load_trend_rows(out_root)

    indexed_rows, stats = index_artifacts(scan_root=scan_root, latest_only=args.latest_only)
    merged_rows = dedupe_rows(existing_rows + indexed_rows)
    summary = build_index_summary(
        scan_root=scan_root,
        out_root=out_root,
        mode=mode,
        latest_only=bool(args.latest_only),
        stats=stats,
        rows=merged_rows,
    )
    summary["existing_rows"] = len(existing_rows)
    summary["new_rows"] = len(indexed_rows)
    summary["merged_rows"] = len(merged_rows)

    query_filters = any([args.query_milestone, args.query_strategy, args.query_gate, args.query_run_id])
    query_result_count = 0
    if query_filters:
        queried = query_rows(
            merged_rows,
            milestone=args.query_milestone,
            strategy=args.query_strategy,
            gate_name=args.query_gate,
            run_id=args.query_run_id,
        )
        query_result_count = len(queried)
        summary["query"] = {
            "milestone": args.query_milestone,
            "strategy": args.query_strategy,
            "gate": args.query_gate,
            "run_id": args.query_run_id,
            "rows": query_result_count,
        }
        if args.query_out:
            _write_query(Path(args.query_out).resolve(), queried)

    paths = write_trend_outputs(out_root=out_root, rows=merged_rows, summary=summary)
    print(
        json.dumps(
            {
                "status": "PASS",
                "mode": mode,
                "latest_only": bool(args.latest_only),
                "source_files_scanned": stats.source_files_scanned,
                "source_files_indexed": stats.source_files_indexed,
                "source_files_skipped": stats.source_files_skipped,
                "new_rows": len(indexed_rows),
                "total_rows": len(merged_rows),
                "query_rows": query_result_count,
                "paths": paths,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
