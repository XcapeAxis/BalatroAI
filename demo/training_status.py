from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def training_status_root() -> Path:
    return Path(__file__).resolve().parent.parent / "docs" / "artifacts" / "mvp" / "training_status"


def latest_status_path() -> Path:
    return training_status_root() / "latest.json"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)


def default_status() -> dict[str, Any]:
    now = now_iso()
    return {
        "schema": "mvp_training_status_v1",
        "job_id": "",
        "process_id": 0,
        "profile": "",
        "status": "idle",
        "status_label": "空闲",
        "message": "当前没有训练任务。",
        "started_at": "",
        "updated_at": now,
        "finished_at": "",
        "budget_minutes": 120,
        "device": "",
        "run_dir": "",
        "final_run_dir": "",
        "dataset": {},
        "training": {},
        "evaluation": {},
        "sweep": {},
        "artifacts": {},
    }


def infer_profile(payload: dict[str, Any] | None) -> str:
    status = dict(payload or {})
    profile = str(status.get("profile") or "").strip().lower()
    if profile:
        return profile

    budget_minutes = int(status.get("budget_minutes") or 0)
    if budget_minutes >= 90:
        return "standard"
    if budget_minutes >= 20:
        return "fast"
    if budget_minutes > 0:
        return "smoke"
    return ""


def read_status(path: Path | None = None) -> dict[str, Any]:
    actual = path or latest_status_path()
    if not actual.exists():
        return default_status()
    try:
        return json.loads(actual.read_text(encoding="utf-8"))
    except Exception:
        return default_status()


def write_status(payload: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    actual = path or latest_status_path()
    merged = default_status()
    merged.update(payload)
    merged["updated_at"] = now_iso()
    _atomic_write_json(actual, merged)
    return merged


def patch_status(path: Path | None = None, **fields: Any) -> dict[str, Any]:
    actual = path or latest_status_path()
    payload = read_status(actual)
    for key, value in fields.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            merged = dict(payload.get(key) or {})
            merged.update(value)
            payload[key] = merged
        else:
            payload[key] = value
    payload["updated_at"] = now_iso()
    _atomic_write_json(actual, payload)
    return payload
