from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _safe_float(v: Any) -> str:
    if isinstance(v, (int, float)):
        return f"{float(v):.6f}"
    return ""


def write_summary_tables(
    out_dir: Path,
    rows: list[dict[str, Any]],
    primary_metric: str,
    run_id: str,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary_table.csv"
    json_path = out_dir / "summary_table.json"
    md_path = out_dir / "summary_table.md"

    fieldnames = [
        "exp_id",
        "status",
        "primary_metric",
        "mean",
        "std",
        "seed_count",
        "catastrophic_failure_count",
        "elapsed_sec",
        "run_dir",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "exp_id": row.get("exp_id"),
                    "status": row.get("status"),
                    "primary_metric": primary_metric,
                    "mean": _safe_float(row.get("mean")),
                    "std": _safe_float(row.get("std")),
                    "seed_count": row.get("seed_count"),
                    "catastrophic_failure_count": row.get("catastrophic_failure_count"),
                    "elapsed_sec": _safe_float(row.get("elapsed_sec")),
                    "run_dir": row.get("run_dir"),
                }
            )

    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        f"# P22 Summary ({run_id})",
        "",
        "| exp_id | status | mean | std | seeds | failures | elapsed_sec |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| {exp} | {status} | {mean} | {std} | {seeds} | {fails} | {elapsed} |".format(
                exp=row.get("exp_id"),
                status=row.get("status"),
                mean=_safe_float(row.get("mean")),
                std=_safe_float(row.get("std")),
                seeds=row.get("seed_count"),
                fails=row.get("catastrophic_failure_count"),
                elapsed=_safe_float(row.get("elapsed_sec")),
            )
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "md": str(md_path),
    }


def write_comparison_report(
    path: Path,
    rows: list[dict[str, Any]],
    primary_metric: str,
    champion_update: dict[str, Any],
) -> None:
    lines = [
        "# P22 Experiment Comparison Report",
        "",
        f"- primary_metric: `{primary_metric}`",
        f"- experiments: `{len(rows)}`",
        "",
        "## Ranking",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            f"{idx}. `{row.get('exp_id')}` status={row.get('status')} "
            f"mean={_safe_float(row.get('mean'))} std={_safe_float(row.get('std'))} "
            f"failures={row.get('catastrophic_failure_count')}"
        )
    lines += [
        "",
        "## Champion Update",
        f"- decision: `{champion_update.get('decision')}`",
        f"- reason: {champion_update.get('reason')}",
        f"- champion_path: `{champion_update.get('champion_path')}`",
        f"- candidate_path: `{champion_update.get('candidate_path')}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

