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
            }
        )
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


def mark_stage_started(payload: dict[str, Any], stage_id: str) -> dict[str, Any]:
    stage = get_stage(payload, stage_id)
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
