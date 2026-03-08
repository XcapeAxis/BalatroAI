from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


def _safe_float(v: Any) -> str:
    if isinstance(v, (int, float)):
        return f"{float(v):.6f}"
    return ""


def _env_str(name: str) -> str:
    return str(os.environ.get(name) or "").strip()


def _env_bool(name: str) -> bool:
    return _env_str(name).lower() in {"1", "true", "yes", "on"}


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
        "doctor_report_path",
        "bootstrap_state_path",
        "setup_mode",
        "doctor_recommended_mode",
        "training_env_source",
        "training_env_name",
        "campaign_state_path",
        "registry_snapshot_path",
        "promotion_queue_path",
        "resume_report_path",
        "autonomy_mode",
        "agents_root_present",
        "autonomy_entry_ref",
        "decision_policy_path",
        "attention_queue_path",
        "morning_summary_path",
        "human_gate_triggered",
        "produced_checkpoint_ids",
        "calibration_ref",
        "guard_tuning_ref",
        "canary_eval_ref",
        "deployment_mode_recommendation",
        "run_dir",
        # P55 config provenance fields
        "config_source_path",
        "config_source_type",
        "config_hash",
        "sidecar_used",
        "sidecar_in_sync",
        "config_sync_report_path",
    ]

    resolved_autonomy_mode = _env_str("BALATRO_AUTONOMY_MODE")
    resolved_agents_root_present = _env_bool("BALATRO_AGENTS_ROOT_PRESENT")
    resolved_autonomy_entry_ref = _env_str("BALATRO_AUTONOMY_ENTRY_REF")
    resolved_decision_policy_path = _env_str("BALATRO_DECISION_POLICY_PATH")
    resolved_attention_queue_path = _env_str("BALATRO_ATTENTION_QUEUE_PATH")
    resolved_morning_summary_path = _env_str("BALATRO_MORNING_SUMMARY_PATH")
    normalized_rows: list[dict[str, Any]] = []

    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized_row = dict(row)
            normalized_row["autonomy_mode"] = normalized_row.get("autonomy_mode") or resolved_autonomy_mode
            if normalized_row.get("agents_root_present") in (None, ""):
                normalized_row["agents_root_present"] = resolved_agents_root_present
            normalized_row["autonomy_entry_ref"] = normalized_row.get("autonomy_entry_ref") or resolved_autonomy_entry_ref
            normalized_row["decision_policy_path"] = normalized_row.get("decision_policy_path") or resolved_decision_policy_path
            normalized_row["attention_queue_path"] = normalized_row.get("attention_queue_path") or resolved_attention_queue_path
            normalized_row["morning_summary_path"] = normalized_row.get("morning_summary_path") or resolved_morning_summary_path
            normalized_rows.append(normalized_row)
            writer.writerow(
                {
                    "exp_id": normalized_row.get("exp_id"),
                    "category": normalized_row.get("category"),
                    "default_enabled": normalized_row.get("default_enabled"),
                    "status": normalized_row.get("status"),
                    "seed_set_name": normalized_row.get("seed_set_name"),
                    "seed_hash": normalized_row.get("seed_hash"),
                    "seeds_used": ",".join([str(s) for s in (normalized_row.get("seeds_used") or [])]),
                    "seeds": ",".join([str(s) for s in (normalized_row.get("seeds_used") or [])]),
                    "primary_metric": primary_metric,
                    "mean": _safe_float(normalized_row.get("mean")),
                    "std": _safe_float(normalized_row.get("std")),
                    "avg_reward": _safe_float(normalized_row.get("avg_reward", normalized_row.get("mean"))),
                    "reward_std": _safe_float(normalized_row.get("reward_std", normalized_row.get("std"))),
                    "best_episode_reward": _safe_float(normalized_row.get("best_episode_reward")),
                    "avg_ante_reached": _safe_float(normalized_row.get("avg_ante_reached")),
                    "median_ante": _safe_float(normalized_row.get("median_ante")),
                    "win_rate": _safe_float(normalized_row.get("win_rate")),
                    "final_win_rate": _safe_float(normalized_row.get("final_win_rate")),
                    "final_loss": _safe_float(normalized_row.get("final_loss")),
                    "hand_top1": _safe_float(normalized_row.get("hand_top1")),
                    "hand_top3": _safe_float(normalized_row.get("hand_top3")),
                    "shop_top1": _safe_float(normalized_row.get("shop_top1")),
                    "illegal_action_rate": _safe_float(normalized_row.get("illegal_action_rate")),
                    "seed_count": normalized_row.get("seed_count"),
                    "catastrophic_failure_count": normalized_row.get("catastrophic_failure_count"),
                    "elapsed_sec": _safe_float(normalized_row.get("elapsed_sec")),
                    "device_profile": normalized_row.get("device_profile"),
                    "learner_device": normalized_row.get("learner_device"),
                    "training_python": normalized_row.get("training_python"),
                    "dashboard_path": normalized_row.get("dashboard_path"),
                    "readiness_report_path": normalized_row.get("readiness_report_path"),
                    "window_mode": normalized_row.get("window_mode"),
                    "background_validation_ref": normalized_row.get("background_validation_ref"),
                    "ops_ui_path": normalized_row.get("ops_ui_path"),
                    "doctor_report_path": normalized_row.get("doctor_report_path"),
                    "bootstrap_state_path": normalized_row.get("bootstrap_state_path"),
                    "setup_mode": normalized_row.get("setup_mode"),
                    "doctor_recommended_mode": normalized_row.get("doctor_recommended_mode"),
                    "training_env_source": normalized_row.get("training_env_source"),
                    "training_env_name": normalized_row.get("training_env_name"),
                    "campaign_state_path": normalized_row.get("campaign_state_path"),
                    "registry_snapshot_path": normalized_row.get("registry_snapshot_path"),
                    "promotion_queue_path": normalized_row.get("promotion_queue_path"),
                    "resume_report_path": normalized_row.get("resume_report_path"),
                    "autonomy_mode": normalized_row.get("autonomy_mode"),
                    "agents_root_present": normalized_row.get("agents_root_present"),
                    "autonomy_entry_ref": normalized_row.get("autonomy_entry_ref"),
                    "decision_policy_path": normalized_row.get("decision_policy_path"),
                    "attention_queue_path": normalized_row.get("attention_queue_path"),
                    "morning_summary_path": normalized_row.get("morning_summary_path"),
                    "human_gate_triggered": normalized_row.get("human_gate_triggered"),
                    "produced_checkpoint_ids": ",".join([str(item) for item in (normalized_row.get("produced_checkpoint_ids") or [])]),
                    "calibration_ref": normalized_row.get("calibration_ref"),
                    "guard_tuning_ref": normalized_row.get("guard_tuning_ref"),
                    "canary_eval_ref": normalized_row.get("canary_eval_ref"),
                    "deployment_mode_recommendation": normalized_row.get("deployment_mode_recommendation"),
                    "run_dir": normalized_row.get("run_dir"),
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
    summary_data["rows"] = normalized_rows
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
                "doctor_report_path",
                "bootstrap_state_path",
                "setup_mode",
                "doctor_recommended_mode",
                "training_env_source",
                "training_env_name",
                "campaign_state_path",
                "registry_snapshot_path",
                "promotion_queue_path",
                "resume_report_path",
                "autonomy_mode",
                "agents_root_present",
                "autonomy_entry_ref",
                "decision_policy_path",
                "attention_queue_path",
                "morning_summary_path",
                "human_gate_triggered",
                "produced_checkpoint_ids",
                "calibration_ref",
                "guard_tuning_ref",
                "canary_eval_ref",
                "deployment_mode_recommendation",
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
            if row.get("doctor_report_path"):
                md_lines.append(f"  doctor_report_path: `{row.get('doctor_report_path')}`")
            if row.get("bootstrap_state_path"):
                md_lines.append(f"  bootstrap_state_path: `{row.get('bootstrap_state_path')}`")
            if row.get("setup_mode"):
                md_lines.append(f"  setup_mode: `{row.get('setup_mode')}`")
            if row.get("doctor_recommended_mode"):
                md_lines.append(f"  doctor_recommended_mode: `{row.get('doctor_recommended_mode')}`")
            if row.get("training_env_source"):
                md_lines.append(f"  training_env_source: `{row.get('training_env_source')}`")
            if row.get("training_env_name"):
                md_lines.append(f"  training_env_name: `{row.get('training_env_name')}`")
            if row.get("campaign_state_path"):
                md_lines.append(f"  campaign_state_path: `{row.get('campaign_state_path')}`")
            if row.get("registry_snapshot_path"):
                md_lines.append(f"  registry_snapshot_path: `{row.get('registry_snapshot_path')}`")
            if row.get("promotion_queue_path"):
                md_lines.append(f"  promotion_queue_path: `{row.get('promotion_queue_path')}`")
            if row.get("resume_report_path"):
                md_lines.append(f"  resume_report_path: `{row.get('resume_report_path')}`")
            if row.get("autonomy_mode"):
                md_lines.append(f"  autonomy_mode: `{row.get('autonomy_mode')}`")
            if row.get("agents_root_present") not in (None, ""):
                md_lines.append(f"  agents_root_present: `{row.get('agents_root_present')}`")
            if row.get("autonomy_entry_ref"):
                md_lines.append(f"  autonomy_entry_ref: `{row.get('autonomy_entry_ref')}`")
            if row.get("decision_policy_path"):
                md_lines.append(f"  decision_policy_path: `{row.get('decision_policy_path')}`")
            if row.get("attention_queue_path"):
                md_lines.append(f"  attention_queue_path: `{row.get('attention_queue_path')}`")
            if row.get("morning_summary_path"):
                md_lines.append(f"  morning_summary_path: `{row.get('morning_summary_path')}`")
            if row.get("human_gate_triggered") not in (None, ""):
                md_lines.append(f"  human_gate_triggered: `{row.get('human_gate_triggered')}`")
            if row.get("produced_checkpoint_ids"):
                md_lines.append(
                    "  produced_checkpoint_ids: `{}`".format(
                        ", ".join([str(item) for item in (row.get("produced_checkpoint_ids") or [])])
                    )
                )
            if row.get("calibration_ref"):
                md_lines.append(f"  calibration_ref: `{row.get('calibration_ref')}`")
            if row.get("guard_tuning_ref"):
                md_lines.append(f"  guard_tuning_ref: `{row.get('guard_tuning_ref')}`")
            if row.get("canary_eval_ref"):
                md_lines.append(f"  canary_eval_ref: `{row.get('canary_eval_ref')}`")
            if row.get("deployment_mode_recommendation"):
                md_lines.append(f"  deployment_mode_recommendation: `{row.get('deployment_mode_recommendation')}`")
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
