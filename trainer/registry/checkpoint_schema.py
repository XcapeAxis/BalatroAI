from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


CHECKPOINT_STATUSES = (
    "draft",
    "smoke_passed",
    "arena_passed",
    "promotion_review",
    "promoted",
    "archived",
    "rejected",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_status(value: Any) -> str:
    token = str(value or "draft").strip().lower()
    return token if token in CHECKPOINT_STATUSES else "draft"


def make_checkpoint_id(
    *,
    family: str,
    training_mode: str,
    source_run_id: str,
    seed_or_seed_group: str,
    artifact_path: str,
) -> str:
    family_token = str(family or "other").strip().lower() or "other"
    mode_token = str(training_mode or "unknown").strip().lower() or "unknown"
    run_token = str(source_run_id or "run").strip().lower() or "run"
    seed_token = str(seed_or_seed_group or "seed").strip().lower() or "seed"
    digest = hashlib.sha1(str(artifact_path or "").encode("utf-8")).hexdigest()[:10]
    return f"{family_token}:{mode_token}:{run_token}:{seed_token}:{digest}"


@dataclass
class CheckpointTransition:
    from_status: str
    to_status: str
    transition_time: str = field(default_factory=now_iso)
    reason: str = ""
    operator: str = "system"
    refs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["from_status"] = normalize_status(payload.get("from_status"))
        payload["to_status"] = normalize_status(payload.get("to_status"))
        payload["refs"] = dict(payload.get("refs") or {})
        return payload


@dataclass
class CheckpointEntry:
    checkpoint_id: str
    family: str
    training_mode: str = ""
    training_mode_category: str = ""
    source_run_id: str = ""
    source_experiment_id: str = ""
    seed_or_seed_group: str = ""
    device_profile: str = ""
    training_python: str = ""
    artifact_path: str = ""
    created_at: str = field(default_factory=now_iso)
    status: str = "draft"
    lineage_refs: dict[str, Any] = field(default_factory=dict)
    metrics_ref: str = ""
    arena_ref: str = ""
    triage_ref: str = ""
    calibration_ref: str = ""
    guard_tuning_ref: str = ""
    canary_eval_ref: str = ""
    deployment_mode_recommendation: str = ""
    notes: str = ""
    parent_checkpoint_id: str = ""
    wm_checkpoint_ref: str = ""
    imagined_data_used: bool = False
    curriculum_profile: str = ""
    git_commit: str = ""
    transitions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = normalize_status(payload.get("status"))
        payload["lineage_refs"] = dict(payload.get("lineage_refs") or {})
        payload["transitions"] = [
            item.to_dict() if isinstance(item, CheckpointTransition) else dict(item)
            for item in (payload.get("transitions") or [])
            if isinstance(item, (dict, CheckpointTransition))
        ]
        return payload


def normalize_entry(payload: dict[str, Any]) -> dict[str, Any]:
    item = dict(payload or {})
    family = str(item.get("family") or "other").strip().lower() or "other"
    checkpoint_id = str(item.get("checkpoint_id") or "").strip()
    if not checkpoint_id:
        checkpoint_id = make_checkpoint_id(
            family=family,
            training_mode=str(item.get("training_mode") or ""),
            source_run_id=str(item.get("source_run_id") or ""),
            seed_or_seed_group=str(item.get("seed_or_seed_group") or ""),
            artifact_path=str(item.get("artifact_path") or ""),
        )
    entry = CheckpointEntry(
        checkpoint_id=checkpoint_id,
        family=family,
        training_mode=str(item.get("training_mode") or ""),
        training_mode_category=str(item.get("training_mode_category") or ""),
        source_run_id=str(item.get("source_run_id") or ""),
        source_experiment_id=str(item.get("source_experiment_id") or ""),
        seed_or_seed_group=str(item.get("seed_or_seed_group") or ""),
        device_profile=str(item.get("device_profile") or ""),
        training_python=str(item.get("training_python") or ""),
        artifact_path=str(item.get("artifact_path") or ""),
        created_at=str(item.get("created_at") or now_iso()),
        status=normalize_status(item.get("status")),
        lineage_refs=(dict(item.get("lineage_refs") or {}) if isinstance(item.get("lineage_refs"), dict) else {}),
        metrics_ref=str(item.get("metrics_ref") or ""),
        arena_ref=str(item.get("arena_ref") or ""),
        triage_ref=str(item.get("triage_ref") or ""),
        calibration_ref=str(item.get("calibration_ref") or ""),
        guard_tuning_ref=str(item.get("guard_tuning_ref") or ""),
        canary_eval_ref=str(item.get("canary_eval_ref") or ""),
        deployment_mode_recommendation=str(item.get("deployment_mode_recommendation") or ""),
        notes=str(item.get("notes") or ""),
        parent_checkpoint_id=str(item.get("parent_checkpoint_id") or ""),
        wm_checkpoint_ref=str(item.get("wm_checkpoint_ref") or ""),
        imagined_data_used=bool(item.get("imagined_data_used", False)),
        curriculum_profile=str(item.get("curriculum_profile") or ""),
        git_commit=str(item.get("git_commit") or ""),
        transitions=[
            CheckpointTransition(
                from_status=str((transition or {}).get("from_status") or "draft"),
                to_status=str((transition or {}).get("to_status") or item.get("status") or "draft"),
                transition_time=str((transition or {}).get("transition_time") or now_iso()),
                reason=str((transition or {}).get("reason") or ""),
                operator=str((transition or {}).get("operator") or "system"),
                refs=dict((transition or {}).get("refs") or {}),
            ).to_dict()
            for transition in (item.get("transitions") or [])
            if isinstance(transition, dict)
        ],
    )
    return entry.to_dict()
