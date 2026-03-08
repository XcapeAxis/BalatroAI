from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trainer.campaigns.campaign_schema import build_campaign_state, normalize_stage_status, now_iso


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def load_campaign_state(
    *,
    state_path: str | Path,
    campaign_id: str,
    run_id: str,
    experiment_id: str,
    seed: str,
    stage_ids: list[str],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(state_path).resolve()
    payload = _read_json(path)
    if not isinstance(payload, dict):
        payload = build_campaign_state(
            campaign_id=campaign_id,
            run_id=run_id,
            experiment_id=experiment_id,
            seed=seed,
            stage_ids=stage_ids,
            metadata=metadata,
        )
        save_campaign_state(path, payload)
        return payload

    existing_ids = {str(item.get("stage_id") or "") for item in (payload.get("stages") or []) if isinstance(item, dict)}
    for stage_id in stage_ids:
        if stage_id in existing_ids:
            continue
        (payload.setdefault("stages", [])).append(
            {
                "stage_id": str(stage_id),
                "status": "pending",
                "started_at": "",
                "ended_at": "",
                "attempt_count": 0,
                "artifacts": {},
                "resume_safe": True,
                "error_summary": "",
                "autonomy_decision": "",
                "autonomy_reason": "",
                "attention_item_ref": "",
                "continue_allowed": True,
                "human_gate_triggered": False,
            }
        )
    for stage in (payload.get("stages") or []):
        if not isinstance(stage, dict):
            continue
        stage.setdefault("autonomy_decision", "")
        stage.setdefault("autonomy_reason", "")
        stage.setdefault("attention_item_ref", "")
        stage.setdefault("continue_allowed", True)
        stage.setdefault("human_gate_triggered", False)
    payload["updated_at"] = now_iso()
    if metadata:
        payload["metadata"] = {**(payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}), **metadata}
    save_campaign_state(path, payload)
    return payload


def save_campaign_state(state_path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(state_path).resolve()
    payload["updated_at"] = now_iso()
    _write_json(path, payload)
    return path


def get_stage(payload: dict[str, Any], stage_id: str) -> dict[str, Any]:
    for item in (payload.get("stages") or []):
        if isinstance(item, dict) and str(item.get("stage_id") or "") == str(stage_id):
            return item
    raise KeyError(f"unknown campaign stage: {stage_id}")


def should_skip_stage(payload: dict[str, Any], stage_id: str, *, force_rerun: bool = False) -> bool:
    if force_rerun:
        return False
    stage = get_stage(payload, stage_id)
    return normalize_stage_status(stage.get("status")) == "completed"


def mark_stage_started(payload: dict[str, Any], stage_id: str, *, allow_block_override: bool = False) -> dict[str, Any]:
    stage = get_stage(payload, stage_id)
    if normalize_stage_status(stage.get("status")) == "blocked" and not allow_block_override:
        blocked = unresolved_human_gate(payload)
        if blocked is not None and str(blocked.get("stage_id") or "") == str(stage_id):
            raise RuntimeError(
                "stage blocked by unresolved human gate: "
                + str(blocked.get("attention_item_ref") or blocked.get("autonomy_reason") or stage_id)
            )
    stage["status"] = "running"
    stage["started_at"] = stage.get("started_at") or now_iso()
    stage["attempt_count"] = int(stage.get("attempt_count") or 0) + 1
    stage["error_summary"] = ""
    payload["updated_at"] = now_iso()
    return payload


def mark_stage_completed(
    payload: dict[str, Any],
    stage_id: str,
    *,
    artifacts: dict[str, Any] | None = None,
    resume_safe: bool | None = None,
    skipped: bool = False,
) -> dict[str, Any]:
    stage = get_stage(payload, stage_id)
    stage["status"] = "skipped" if skipped else "completed"
    stage["ended_at"] = now_iso()
    stage["artifacts"] = dict(artifacts or {})
    if resume_safe is not None:
        stage["resume_safe"] = bool(resume_safe)
    payload["updated_at"] = now_iso()
    return payload


def mark_stage_failed(
    payload: dict[str, Any],
    stage_id: str,
    *,
    error_summary: str,
    artifacts: dict[str, Any] | None = None,
    resume_safe: bool | None = None,
) -> dict[str, Any]:
    stage = get_stage(payload, stage_id)
    stage["status"] = "failed"
    stage["ended_at"] = now_iso()
    stage["error_summary"] = str(error_summary or "")
    if artifacts is not None:
        stage["artifacts"] = dict(artifacts or {})
    if resume_safe is not None:
        stage["resume_safe"] = bool(resume_safe)
    payload["updated_at"] = now_iso()
    return payload


def set_stage_autonomy(
    payload: dict[str, Any],
    stage_id: str,
    *,
    decision: str,
    reason: str,
    continue_allowed: bool,
    human_gate_triggered: bool = False,
    attention_item_ref: str = "",
) -> dict[str, Any]:
    stage = get_stage(payload, stage_id)
    stage["autonomy_decision"] = str(decision or "")
    stage["autonomy_reason"] = str(reason or "")
    stage["continue_allowed"] = bool(continue_allowed)
    stage["human_gate_triggered"] = bool(human_gate_triggered)
    if attention_item_ref:
        stage["attention_item_ref"] = str(attention_item_ref)
    payload["updated_at"] = now_iso()
    return payload


def unresolved_human_gate(payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        from trainer.autonomy.attention_queue import get_attention_item_status
    except Exception:
        get_attention_item_status = None
    for stage in payload.get("stages") or []:
        if not isinstance(stage, dict):
            continue
        if not bool(stage.get("human_gate_triggered")):
            continue
        attention_ref = str(stage.get("attention_item_ref") or "").strip()
        if not attention_ref:
            return stage
        status = ""
        if get_attention_item_status is not None:
            try:
                status = str(get_attention_item_status(attention_ref) or "")
            except Exception:
                status = ""
        if status.lower() not in {"resolved", "ignored"}:
            return stage
    return None


def mark_stage_blocked(
    payload: dict[str, Any],
    stage_id: str,
    *,
    reason: str,
    attention_item_ref: str = "",
    artifacts: dict[str, Any] | None = None,
    resume_safe: bool | None = None,
) -> dict[str, Any]:
    stage = get_stage(payload, stage_id)
    stage["status"] = "blocked"
    stage["ended_at"] = now_iso()
    stage["error_summary"] = str(reason or "")
    if artifacts is not None:
        stage["artifacts"] = dict(artifacts or {})
    if resume_safe is not None:
        stage["resume_safe"] = bool(resume_safe)
    stage["autonomy_decision"] = "stop_and_queue_attention"
    stage["autonomy_reason"] = str(reason or "")
    stage["attention_item_ref"] = str(attention_item_ref or "")
    stage["continue_allowed"] = False
    stage["human_gate_triggered"] = True
    payload["updated_at"] = now_iso()
    return payload
