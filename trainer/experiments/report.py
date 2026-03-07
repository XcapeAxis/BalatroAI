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
    config_provenance: dict[str, Any] | None = None,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary_table.csv"
    json_path = out_dir / "summary_table.json"
    md_path = out_dir / "summary_table.md"

    fieldnames = [
        "exp_id",
        "category",
        "default_enabled",
        "status",
        "seed_set_name",
        "seed_hash",
        "seeds_used",
        "seeds",
        "primary_metric",
        "mean",
        "std",
        "avg_reward",
        "reward_std",
        "best_episode_reward",
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
        "device_profile",
        "learner_device",
        "training_python",
        "dashboard_path",
        "readiness_report_path",
        "window_mode",
        "background_validation_ref",
        "ops_ui_path",
        "campaign_state_path",
        "registry_snapshot_path",
        "promotion_queue_path",
        "resume_report_path",
        "produced_checkpoint_ids",
        "run_dir",
        # P55 config provenance fields
        "config_source_path",
        "config_source_type",
        "config_hash",
        "sidecar_used",
        "sidecar_in_sync",
        "config_sync_report_path",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "exp_id": row.get("exp_id"),
                    "category": row.get("category"),
                    "default_enabled": row.get("default_enabled"),
                    "status": row.get("status"),
                    "seed_set_name": row.get("seed_set_name"),
                    "seed_hash": row.get("seed_hash"),
                    "seeds_used": ",".join([str(s) for s in (row.get("seeds_used") or [])]),
                    "seeds": ",".join([str(s) for s in (row.get("seeds_used") or [])]),
                    "primary_metric": primary_metric,
                    "mean": _safe_float(row.get("mean")),
                    "std": _safe_float(row.get("std")),
                    "avg_reward": _safe_float(row.get("avg_reward", row.get("mean"))),
                    "reward_std": _safe_float(row.get("reward_std", row.get("std"))),
                    "best_episode_reward": _safe_float(row.get("best_episode_reward")),
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
                    "device_profile": row.get("device_profile"),
                    "learner_device": row.get("learner_device"),
                    "training_python": row.get("training_python"),
                    "dashboard_path": row.get("dashboard_path"),
                    "readiness_report_path": row.get("readiness_report_path"),
                    "window_mode": row.get("window_mode"),
                    "background_validation_ref": row.get("background_validation_ref"),
                    "ops_ui_path": row.get("ops_ui_path"),
                    "campaign_state_path": row.get("campaign_state_path"),
                    "registry_snapshot_path": row.get("registry_snapshot_path"),
                    "promotion_queue_path": row.get("promotion_queue_path"),
                    "resume_report_path": row.get("resume_report_path"),
                    "produced_checkpoint_ids": ",".join([str(item) for item in (row.get("produced_checkpoint_ids") or [])]),
                    "run_dir": row.get("run_dir"),
                    # P55 provenance (run-level, not per-row; same value for all rows)
                    "config_source_path": (config_provenance or {}).get("config_source_path", ""),
                    "config_source_type": (config_provenance or {}).get("config_source_type", ""),
                    "config_hash": (config_provenance or {}).get("config_hash", ""),
                    "sidecar_used": str((config_provenance or {}).get("sidecar_used", "")),
                    "sidecar_in_sync": str((config_provenance or {}).get("sidecar_in_sync", "")),
                    "config_sync_report_path": (config_provenance or {}).get("config_sync_report_path", ""),
                }
            )

    # Embed provenance at the top of the JSON summary
    summary_data: dict[str, Any] = {
        "run_id": run_id,
        "primary_metric": primary_metric,
    }
    if config_provenance:
        summary_data["config_provenance"] = config_provenance
    summary_data["rows"] = rows
    json_path.write_text(json.dumps(summary_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    prov = config_provenance or {}
    md_lines = [
        f"# P23 Summary ({run_id})",
        "",
    ]
    if prov:
        sidecar_flag = "sidecar_fallback" if prov.get("sidecar_used") else "yaml_direct"
        sync_flag = "in_sync" if prov.get("sidecar_in_sync") else "DRIFT"
        md_lines += [
            "## Config Provenance (P55)",
            "",
            f"| Field | Value |",
            f"|---|---|",
            f"| source_path | `{prov.get('config_source_path', '')}` |",
            f"| source_type | `{prov.get('config_source_type', '')}` ({sidecar_flag}) |",
            f"| config_hash | `{str(prov.get('config_hash', ''))[:16]}` |",
            f"| sidecar_in_sync | **{sync_flag}** |",
            f"| sync_report | `{prov.get('config_sync_report_path', '')}` |",
            "",
        ]
    md_lines += [
        "## Results",
        "",
        "| exp_id | category | default_enabled | status | seed_set | mean | std | avg_reward | reward_std | best_episode_reward | avg_ante | median_ante | win_rate | final_win_rate | final_loss | hand_top1 | hand_top3 | shop_top1 | illegal_rate | seeds | failures | elapsed_sec | device_profile | learner_device |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        md_lines.append(
            "| {exp} | {category} | {default_enabled} | {status} | {seed_set} | {mean} | {std} | {avg_reward} | {reward_std} | {best_episode_reward} | {avg_ante} | {median_ante} | {win_rate} | {final_win_rate} | {final_loss} | {hand_top1} | {hand_top3} | {shop_top1} | {illegal_rate} | {seeds} | {fails} | {elapsed} | {device_profile} | {learner_device} |".format(
                exp=row.get("exp_id"),
                category=row.get("category"),
                default_enabled=str(bool(row.get("default_enabled"))).lower(),
                status=row.get("status"),
                seed_set=row.get("seed_set_name"),
                mean=_safe_float(row.get("mean")),
                std=_safe_float(row.get("std")),
                avg_reward=_safe_float(row.get("avg_reward", row.get("mean"))),
                reward_std=_safe_float(row.get("reward_std", row.get("std"))),
                best_episode_reward=_safe_float(row.get("best_episode_reward")),
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
                device_profile=row.get("device_profile") or "",
                learner_device=row.get("learner_device") or "",
            )
        )
    runtime_rows = [
        row
        for row in rows
        if any(
            row.get(key)
            for key in (
                "training_python",
                "dashboard_path",
                "readiness_report_path",
                "window_mode",
                "background_validation_ref",
                "ops_ui_path",
                "campaign_state_path",
                "registry_snapshot_path",
                "promotion_queue_path",
                "resume_report_path",
                "produced_checkpoint_ids",
            )
        )
    ]
    if runtime_rows:
        md_lines += [
            "",
            "## Runtime Details",
        ]
        for row in runtime_rows:
            md_lines.append(f"- `{row.get('exp_id')}`")
            if row.get("training_python"):
                md_lines.append(f"  training_python: `{row.get('training_python')}`")
            if row.get("dashboard_path"):
                md_lines.append(f"  dashboard_path: `{row.get('dashboard_path')}`")
            if row.get("readiness_report_path"):
                md_lines.append(f"  readiness_report_path: `{row.get('readiness_report_path')}`")
            if row.get("window_mode"):
                md_lines.append(f"  window_mode: `{row.get('window_mode')}`")
            if row.get("background_validation_ref"):
                md_lines.append(f"  background_validation_ref: `{row.get('background_validation_ref')}`")
            if row.get("ops_ui_path"):
                md_lines.append(f"  ops_ui_path: `{row.get('ops_ui_path')}`")
            if row.get("campaign_state_path"):
                md_lines.append(f"  campaign_state_path: `{row.get('campaign_state_path')}`")
            if row.get("registry_snapshot_path"):
                md_lines.append(f"  registry_snapshot_path: `{row.get('registry_snapshot_path')}`")
            if row.get("promotion_queue_path"):
                md_lines.append(f"  promotion_queue_path: `{row.get('promotion_queue_path')}`")
            if row.get("resume_report_path"):
                md_lines.append(f"  resume_report_path: `{row.get('resume_report_path')}`")
            if row.get("produced_checkpoint_ids"):
                md_lines.append(
                    "  produced_checkpoint_ids: `{}`".format(
                        ", ".join([str(item) for item in (row.get("produced_checkpoint_ids") or [])])
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
