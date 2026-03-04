from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _safe_float(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except Exception:
        return ""


def write_episode_records(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_table(out_dir: Path, rows: list[dict[str, Any]]) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary_table.json"
    csv_path = out_dir / "summary_table.csv"
    md_path = out_dir / "summary_table.md"

    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fieldnames = [
        "policy_id",
        "status",
        "episodes",
        "seed_count",
        "seeds",
        "mean_total_score",
        "std_total_score",
        "mean_chips",
        "mean_rounds_survived",
        "mean_episode_length",
        "win_rate",
        "p10_total_score",
        "p50_total_score",
        "p90_total_score",
        "invalid_action_rate",
        "timeout_rate",
        "mean_money_earned",
        "mean_rerolls_count",
        "mean_packs_opened",
        "mean_consumables_used",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "policy_id": row.get("policy_id"),
                    "status": row.get("status"),
                    "episodes": row.get("episodes"),
                    "seed_count": row.get("seed_count"),
                    "seeds": ",".join([str(x) for x in (row.get("seeds") or [])]),
                    "mean_total_score": _safe_float(row.get("mean_total_score")),
                    "std_total_score": _safe_float(row.get("std_total_score")),
                    "mean_chips": _safe_float(row.get("mean_chips")),
                    "mean_rounds_survived": _safe_float(row.get("mean_rounds_survived")),
                    "mean_episode_length": _safe_float(row.get("mean_episode_length")),
                    "win_rate": _safe_float(row.get("win_rate")),
                    "p10_total_score": _safe_float(row.get("p10_total_score")),
                    "p50_total_score": _safe_float(row.get("p50_total_score")),
                    "p90_total_score": _safe_float(row.get("p90_total_score")),
                    "invalid_action_rate": _safe_float(row.get("invalid_action_rate")),
                    "timeout_rate": _safe_float(row.get("timeout_rate")),
                    "mean_money_earned": _safe_float(row.get("mean_money_earned")),
                    "mean_rerolls_count": _safe_float(row.get("mean_rerolls_count")),
                    "mean_packs_opened": _safe_float(row.get("mean_packs_opened")),
                    "mean_consumables_used": _safe_float(row.get("mean_consumables_used")),
                }
            )

    md_lines = [
        "# P39 Policy Arena Summary",
        "",
        "| policy | status | episodes | seeds | mean_score | std_score | mean_rounds | win_rate | p10 | p50 | p90 | invalid_rate | timeout_rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| {policy} | {status} | {episodes} | {seed_count} | {mean_score} | {std_score} | {mean_rounds} | {win_rate} | {p10} | {p50} | {p90} | {invalid} | {timeout} |".format(
                policy=row.get("policy_id"),
                status=row.get("status"),
                episodes=row.get("episodes"),
                seed_count=row.get("seed_count"),
                mean_score=_safe_float(row.get("mean_total_score")),
                std_score=_safe_float(row.get("std_total_score")),
                mean_rounds=_safe_float(row.get("mean_rounds_survived")),
                win_rate=_safe_float(row.get("win_rate")),
                p10=_safe_float(row.get("p10_total_score")),
                p50=_safe_float(row.get("p50_total_score")),
                p90=_safe_float(row.get("p90_total_score")),
                invalid=_safe_float(row.get("invalid_action_rate")),
                timeout=_safe_float(row.get("timeout_rate")),
            )
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {"json": str(json_path), "csv": str(csv_path), "md": str(md_path)}


def write_bucket_metrics(out_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "bucket_metrics.json"
    md_path = out_dir / "bucket_metrics.md"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# Policy Arena Bucket and Slice Metrics", ""]
    for policy_row in payload.get("policies") if isinstance(payload.get("policies"), list) else []:
        if not isinstance(policy_row, dict):
            continue
        policy_id = str(policy_row.get("policy_id") or "unknown")
        lines.append(f"## {policy_id}")
        buckets = policy_row.get("buckets") if isinstance(policy_row.get("buckets"), dict) else {}
        for bucket_name, rows in buckets.items():
            lines.append(f"### {bucket_name}")
            lines.append("")
            lines.append("| bucket | count | ratio |")
            lines.append("|---|---:|---:|")
            for row in rows if isinstance(rows, list) else []:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    "| {bucket} | {count} | {ratio} |".format(
                        bucket=row.get("bucket"),
                        count=int(row.get("count") or 0),
                        ratio=_safe_float(row.get("ratio")),
                    )
                )
            lines.append("")
        slice_metrics = policy_row.get("slice_metrics") if isinstance(policy_row.get("slice_metrics"), dict) else {}
        if slice_metrics:
            lines.append("### slice_metrics")
            lines.append("")
            for slice_key, rows in slice_metrics.items():
                lines.append(f"#### {slice_key}")
                lines.append("")
                lines.append("| slice_label | count | mean_total_score | std_total_score | mean_rounds_survived | win_rate |")
                lines.append("|---|---:|---:|---:|---:|---:|")
                for row in rows if isinstance(rows, list) else []:
                    if not isinstance(row, dict):
                        continue
                    lines.append(
                        "| {slice_label} | {count} | {mean_total_score} | {std_total_score} | {mean_rounds_survived} | {win_rate} |".format(
                            slice_label=row.get("slice_label"),
                            count=int(row.get("count") or 0),
                            mean_total_score=_safe_float(row.get("mean_total_score")),
                            std_total_score=_safe_float(row.get("std_total_score")),
                            mean_rounds_survived=_safe_float(row.get("mean_rounds_survived")),
                            win_rate=_safe_float(row.get("win_rate")),
                        )
                    )
                lines.append("")
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path)}
