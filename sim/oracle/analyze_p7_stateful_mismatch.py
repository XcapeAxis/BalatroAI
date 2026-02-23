from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _classify_row(row: dict[str, Any]) -> str:
    status = str(row.get("status") or "")
    if status == "pass":
        return "aligned"
    if status == "gen_fail":
        return "gen_fail"
    if status == "oracle_fail":
        return "oracle_fail"
    path = str(row.get("first_diff_path") or "")
    if "jokers_state_projection" in path:
        return "joker_state_mismatch"
    if "score_observed" in path:
        return "score_delta_mismatch"
    if "round.hands_left" in path or "round.discards_left" in path:
        return "resource_mismatch"
    if "phase" in path:
        return "phase_mismatch"
    if path:
        return "other_diff"
    return "unknown"


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for row in rows:
            out = {k: row.get(k) for k in fields}
            w.writerow(out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze P7 stateful mismatch report and emit csv/md tables.")
    p.add_argument("--fixtures-dir", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    fixtures_dir = Path(args.fixtures_dir)
    report_path = fixtures_dir / "report_p7.json"
    if not report_path.exists():
        print(f"ERROR: report not found: {report_path}")
        return 2

    report = json.loads(report_path.read_text(encoding="utf-8"))
    results = report.get("results") if isinstance(report.get("results"), list) else []

    rows: list[dict[str, Any]] = []
    counter: Counter[str] = Counter()
    for item in results:
        if not isinstance(item, dict):
            continue
        cls = _classify_row(item)
        counter[cls] += 1
        rows.append(
            {
                "target": item.get("target"),
                "status": item.get("status"),
                "classification": cls,
                "template": item.get("template"),
                "steps_used": item.get("steps_used"),
                "first_diff_step": item.get("first_diff_step"),
                "first_diff_path": item.get("first_diff_path"),
                "oracle_hash": item.get("oracle_hash"),
                "sim_hash": item.get("sim_hash"),
                "failure_reason": item.get("failure_reason"),
                "dumped_oracle": item.get("dumped_oracle"),
                "dumped_sim": item.get("dumped_sim"),
            }
        )

    csv_path = fixtures_dir / "stateful_mismatch_table_p7.csv"
    md_path = fixtures_dir / "stateful_mismatch_table_p7.md"

    fields = [
        "target",
        "status",
        "classification",
        "template",
        "steps_used",
        "first_diff_step",
        "first_diff_path",
        "oracle_hash",
        "sim_hash",
        "failure_reason",
        "dumped_oracle",
        "dumped_sim",
    ]
    _write_csv(csv_path, rows, fields)

    lines: list[str] = []
    lines.append("# P7 Stateful Mismatch Analysis")
    lines.append("")
    lines.append(f"- total: **{len(rows)}**")
    for key, count in counter.most_common():
        lines.append(f"- {key}: **{count}**")
    lines.append("")
    lines.append("## Rows")
    lines.append("")
    lines.append("|target|status|classification|template|first_diff_step|first_diff_path|")
    lines.append("|---|---|---|---|---:|---|")
    for row in rows:
        lines.append(
            f"|{row.get('target','')}|{row.get('status','')}|{row.get('classification','')}|{row.get('template','')}|"
            f"{row.get('first_diff_step','')}|{row.get('first_diff_path','')}|"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"rows={len(rows)}")
    for key, count in counter.most_common():
        print(f"{key}={count}")
    print(f"csv={csv_path}")
    print(f"md={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
