from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


CAMPAIGN_STAGE_IDS = (
    "service_readiness",
    "train_rl",
    "train_world_model",
    "build_imagination_data",
    "arena_eval",
    "triage",
    "promotion_decision",
    "dashboard_build",
    "cleanup_finalize",
)

P52_ROUTER_CAMPAIGN_STAGE_IDS = (
    "build_router_dataset",
    "train_learned_router",
    "eval_learned_router",
    "arena_ablation",
    "triage",
    "promotion_queue_update",
    "dashboard_build",
)

P53_OPS_CAMPAIGN_STAGE_IDS = (
    "background_mode_validation",
    "promotion_queue_update",
    "dashboard_build",
    "ops_ui_metadata",
)

P56_ROUTER_CAMPAIGN_STAGE_IDS = (
    "build_router_dataset",
    "train_learned_router",
    "eval_calibration",
    "tune_guard_thresholds",
    "arena_ablation",
    "canary_eval",
    "triage",
    "promotion_queue_update",
    "dashboard_build",
)

P57_OVERNIGHT_STAGE_IDS = (
    "readiness_guard",
    "decision_policy_audit",
    "config_provenance_scan",
    "promotion_queue_scan",
    "campaign_health_scan",
    "dashboard_build",
    "morning_summary",
)

CAMPAIGN_STAGE_STATUSES = ("pending", "running", "completed", "failed", "blocked", "skipped")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_stage_status(value: Any) -> str:
    token = str(value or "pending").strip().lower()
    return token if token in CAMPAIGN_STAGE_STATUSES else "pending"


def build_stage_entry(stage_id: str, *, resume_safe: bool = True) -> dict[str, Any]:
    return {
        "stage_id": str(stage_id),
        "status": "pending",
        "started_at": "",
        "ended_at": "",
        "attempt_count": 0,
        "artifacts": {},
        "resume_safe": bool(resume_safe),
        "error_summary": "",
        "autonomy_decision": "",
        "autonomy_reason": "",
        "attention_item_ref": "",
        "continue_allowed": True,
        "human_gate_triggered": False,
    }


def build_campaign_state(
    *,
    campaign_id: str,
    run_id: str,
    experiment_id: str,
    seed: str,
    stage_ids: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ids = stage_ids or list(CAMPAIGN_STAGE_IDS)
    return {
        "schema": "p51_campaign_state_v1",
        "campaign_id": str(campaign_id),
        "run_id": str(run_id),
        "experiment_id": str(experiment_id),
        "seed": str(seed),
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "stages": [build_stage_entry(stage_id) for stage_id in ids],
        "metadata": dict(metadata or {}),
    }
