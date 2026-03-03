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
        "seed_set_name",
        "seed_hash",
        "seeds_used",
        "primary_metric",
        "mean",
        "std",
        "avg_ante_reached",
        "median_ante",
        "win_rate",
        "final_win_rate",
        "final_loss",
        "hand_top1",
        "hand_top3",
        "shop_top1",
        "illegal_action_rate",
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
                    "seed_set_name": row.get("seed_set_name"),
                    "seed_hash": row.get("seed_hash"),
                    "seeds_used": ",".join([str(s) for s in (row.get("seeds_used") or [])]),
                    "primary_metric": primary_metric,
                    "mean": _safe_float(row.get("mean")),
                    "std": _safe_float(row.get("std")),
                    "avg_ante_reached": _safe_float(row.get("avg_ante_reached")),
                    "median_ante": _safe_float(row.get("median_ante")),
                    "win_rate": _safe_float(row.get("win_rate")),
                    "final_win_rate": _safe_float(row.get("final_win_rate")),
                    "final_loss": _safe_float(row.get("final_loss")),
                    "hand_top1": _safe_float(row.get("hand_top1")),
                    "hand_top3": _safe_float(row.get("hand_top3")),
                    "shop_top1": _safe_float(row.get("shop_top1")),
                    "illegal_action_rate": _safe_float(row.get("illegal_action_rate")),
                    "seed_count": row.get("seed_count"),
                    "catastrophic_failure_count": row.get("catastrophic_failure_count"),
                    "elapsed_sec": _safe_float(row.get("elapsed_sec")),
                    "run_dir": row.get("run_dir"),
                }
            )

    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        f"# P23 Summary ({run_id})",
        "",
        "| exp_id | status | seed_set | mean | std | avg_ante | median_ante | win_rate | final_win_rate | final_loss | hand_top1 | hand_top3 | shop_top1 | illegal_rate | seeds | failures | elapsed_sec |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| {exp} | {status} | {seed_set} | {mean} | {std} | {avg_ante} | {median_ante} | {win_rate} | {final_win_rate} | {final_loss} | {hand_top1} | {hand_top3} | {shop_top1} | {illegal_rate} | {seeds} | {fails} | {elapsed} |".format(
                exp=row.get("exp_id"),
                status=row.get("status"),
                seed_set=row.get("seed_set_name"),
                mean=_safe_float(row.get("mean")),
                std=_safe_float(row.get("std")),
                avg_ante=_safe_float(row.get("avg_ante_reached")),
                median_ante=_safe_float(row.get("median_ante")),
                win_rate=_safe_float(row.get("win_rate")),
                final_win_rate=_safe_float(row.get("final_win_rate")),
                final_loss=_safe_float(row.get("final_loss")),
                hand_top1=_safe_float(row.get("hand_top1")),
                hand_top3=_safe_float(row.get("hand_top3")),
                shop_top1=_safe_float(row.get("shop_top1")),
                illegal_rate=_safe_float(row.get("illegal_action_rate")),
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
        "# P23 Experiment Comparison Report",
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
            f"avg_ante={_safe_float(row.get('avg_ante_reached'))} "
            f"median_ante={_safe_float(row.get('median_ante'))} "
            f"win_rate={_safe_float(row.get('win_rate'))} "
            f"hand_top1={_safe_float(row.get('hand_top1'))} "
            f"hand_top3={_safe_float(row.get('hand_top3'))} "
            f"shop_top1={_safe_float(row.get('shop_top1'))} "
            f"illegal_rate={_safe_float(row.get('illegal_action_rate'))} "
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
