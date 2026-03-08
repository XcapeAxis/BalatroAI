from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    preferred = Path("D:/MYFILES/BalatroAI")
    if preferred.exists():
        return preferred.resolve()
    return Path(__file__).resolve().parents[2]


def artifacts_root() -> Path:
    return repo_root() / "docs" / "artifacts"


def _latest_autonomy_artifact_root() -> Path:
    root = artifacts_root()
    for family in ("p60", "p59"):
        candidate = root / family
        if candidate.exists():
            return candidate
    return root / "p60"


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as fp:
        for line in fp:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def path_in_repo(path: str | Path) -> Path | None:
    target = Path(path)
    if not target.is_absolute():
        target = (repo_root() / target).resolve()
    else:
        target = target.resolve()
    try:
        target.relative_to(repo_root())
    except Exception:
        return None
    return target


def _sort_key(path: Path) -> tuple[float, str]:
    try:
        return (path.stat().st_mtime, str(path))
    except Exception:
        return (0.0, str(path))


def latest_matching_path(pattern: str, *, required_tokens: tuple[str, ...] = ()) -> Path | None:
    candidates: list[Path] = []
    for path in artifacts_root().glob(pattern):
        token = str(path).lower().replace("\\", "/")
        if any(required not in token for required in required_tokens):
            continue
        candidates.append(path.resolve())
    candidates.sort(key=_sort_key, reverse=True)
    return candidates[0] if candidates else None


def latest_p22_run() -> dict[str, Any]:
    runs_root = artifacts_root() / "p22" / "runs"
    if not runs_root.exists():
        return {}
    runs = sorted([path for path in runs_root.iterdir() if path.is_dir()], key=_sort_key, reverse=True)
    if not runs:
        return {}
    run_dir = runs[0]
    summary_path = run_dir / "summary_table.json"
    summary_rows = read_json(summary_path)
    rows = (
        [row for row in summary_rows if isinstance(row, dict)]
        if isinstance(summary_rows, list)
        else (
            [row for row in (summary_rows.get("rows") or []) if isinstance(row, dict)]
            if isinstance(summary_rows, dict)
            else []
        )
    )
    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "summary_path": str(summary_path.resolve()),
        "summary_rows": rows,
        "config_provenance": summary_rows.get("config_provenance") if isinstance(summary_rows, dict) and isinstance(summary_rows.get("config_provenance"), dict) else {},
    }


def latest_progress_events(limit: int = 30) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in artifacts_root().glob("**/*progress*.jsonl"):
        for row in read_jsonl(path):
            if str(row.get("schema") or "") != "p49_progress_event_v1":
                continue
            key = (
                str(row.get("run_id") or ""),
                str(row.get("component") or ""),
                str(row.get("seed") or ""),
            )
            latest[key] = row
    rows = sorted(
        latest.values(),
        key=lambda row: (
            str(row.get("timestamp") or ""),
            str(row.get("run_id") or ""),
            str(row.get("component") or ""),
            str(row.get("seed") or ""),
        ),
        reverse=True,
    )
    return rows[: max(1, int(limit))]


def latest_readiness_report() -> dict[str, Any]:
    path = latest_matching_path("p49/readiness/**/service_readiness_report.json")
    payload = read_json(path) if isinstance(path, Path) else {}
    return {
        "path": str(path.resolve()) if isinstance(path, Path) else "",
        "payload": payload if isinstance(payload, dict) else {},
    }


def latest_window_state() -> dict[str, Any]:
    path = artifacts_root() / "p53" / "window_supervisor" / "latest" / "window_state.json"
    payload = read_json(path)
    dominant_mode = ""
    rows = payload.get("window_mode_after") if isinstance(payload, dict) and isinstance(payload.get("window_mode_after"), list) else payload.get("windows")
    if isinstance(rows, list):
        for role in ("game_main", "other_balatro", "diagnostic_console"):
            for row in rows:
                if isinstance(row, dict) and str(row.get("role") or "") == role and str(row.get("mode") or "").strip():
                    dominant_mode = str(row.get("mode") or "")
                    break
            if dominant_mode:
                break
    return {
        "path": str(path.resolve()),
        "payload": payload if isinstance(payload, dict) else {},
        "dominant_mode": dominant_mode,
    }


def latest_background_validation() -> dict[str, Any]:
    path = artifacts_root() / "p53" / "background_mode_validation" / "latest" / "background_mode_validation.json"
    payload = read_json(path)
    return {
        "path": str(path.resolve()),
        "payload": payload if isinstance(payload, dict) else {},
    }


def latest_ops_ui_state() -> dict[str, Any]:
    path = artifacts_root() / "p53" / "ops_ui" / "latest" / "ops_ui_state.json"
    payload = read_json(path)
    return {
        "path": str(path.resolve()),
        "payload": payload if isinstance(payload, dict) else {},
    }


def campaign_rows(limit: int = 48) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(artifacts_root().glob("**/campaign_state.json"), key=_sort_key, reverse=True)[: max(4, int(limit))]:
        payload = read_json(path)
        if not isinstance(payload, dict):
            continue
        stages = [dict(item) for item in (payload.get("stages") or []) if isinstance(item, dict)]
        blocked = next((item for item in stages if str(item.get("status") or "") == "blocked" or bool(item.get("human_gate_triggered"))), None)
        active = next((item for item in stages if str(item.get("status") or "") == "running"), None)
        failed = next((item for item in stages if str(item.get("status") or "") == "failed"), None)
        latest_stage = blocked or active or failed or (stages[-1] if stages else {})
        exp_id = str(payload.get("experiment_id") or "")
        rows.append(
            {
                "campaign_id": str(payload.get("campaign_id") or ""),
                "run_id": str(payload.get("run_id") or ""),
                "experiment_id": exp_id,
                "seed": str(payload.get("seed") or ""),
                "stage_id": str((latest_stage or {}).get("stage_id") or ""),
                "status": str((latest_stage or {}).get("status") or ""),
                "autonomy_decision": str((latest_stage or {}).get("autonomy_decision") or ""),
                "autonomy_reason": str((latest_stage or {}).get("autonomy_reason") or ""),
                "attention_item_ref": str((latest_stage or {}).get("attention_item_ref") or ""),
                "human_gate_triggered": bool((latest_stage or {}).get("human_gate_triggered") or False),
                "state_path": str(path.resolve()),
                "resume_command": (
                    f"powershell -ExecutionPolicy Bypass -File scripts\\run_p22.ps1 -Only {exp_id} -ResumeLatestCampaign"
                    if exp_id
                    else ""
                ),
            }
        )
    return rows


def registry_entries(*, family: str = "", status: str = "", latest: bool = False, promoted: bool = False) -> dict[str, Any]:
    path = artifacts_root() / "registry" / "checkpoints_registry.json"
    payload = read_json(path)
    all_rows = payload.get("items") if isinstance(payload, dict) and isinstance(payload.get("items"), list) else []
    rows = [row for row in all_rows if isinstance(row, dict)]
    if family:
        rows = [row for row in rows if str(row.get("family") or "") == family]
    if status:
        rows = [row for row in rows if str(row.get("status") or "") == status]
    rows.sort(key=lambda row: (str(row.get("created_at") or ""), str(row.get("checkpoint_id") or "")), reverse=True)
    if latest:
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            key = str(row.get("family") or "")
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        rows = deduped
    if promoted:
        rows = [row for row in rows if str(row.get("status") or "") == "promoted"]
    counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    for row in all_rows:
        if not isinstance(row, dict):
            continue
        counts[str(row.get("status") or "draft")] = int(counts.get(str(row.get("status") or "draft"), 0)) + 1
        family_counts[str(row.get("family") or "other")] = int(family_counts.get(str(row.get("family") or "other"), 0)) + 1
    return {
        "path": str(path.resolve()),
        "counts": counts,
        "family_counts": family_counts,
        "entries": rows,
    }


def latest_promotion_queue() -> dict[str, Any]:
    path = latest_matching_path("**/promotion_queue.json")
    payload = read_json(path) if isinstance(path, Path) else {}
    return {
        "path": str(path.resolve()) if isinstance(path, Path) else "",
        "payload": payload if isinstance(payload, dict) else {},
    }


def latest_attention_queue() -> dict[str, Any]:
    path = artifacts_root() / "attention_required" / "attention_queue.json"
    payload = read_json(path)
    items = payload.get("items") if isinstance(payload, dict) and isinstance(payload.get("items"), list) else []
    open_items = [item for item in items if isinstance(item, dict) and str(item.get("status") or "") == "open"]
    return {
        "path": str(path.resolve()),
        "payload": payload if isinstance(payload, dict) else {},
        "open_items": open_items,
    }


def latest_morning_summary() -> dict[str, Any]:
    root = artifacts_root() / "morning_summary"
    latest_json = root / "latest.json"
    latest_md = root / "latest.md"
    payload = read_json(latest_json)
    excerpt = "\n".join(_read_text(latest_md).splitlines()[:24]) if latest_md.exists() else ""
    return {
        "json_path": str(latest_json.resolve()),
        "md_path": str(latest_md.resolve()),
        "payload": payload if isinstance(payload, dict) else {},
        "excerpt": excerpt,
    }


def latest_bootstrap_state() -> dict[str, Any]:
    path = artifacts_root() / "p58" / "bootstrap" / "latest_bootstrap_state.json"
    payload = read_json(path)
    return {
        "path": str(path.resolve()),
        "payload": payload if isinstance(payload, dict) else {},
    }


def latest_doctor_report() -> dict[str, Any]:
    json_path = artifacts_root() / "p58" / "latest_doctor.json"
    md_path = artifacts_root() / "p58" / "latest_doctor.md"
    payload = read_json(json_path)
    return {
        "json_path": str(json_path.resolve()),
        "md_path": str(md_path.resolve()),
        "payload": payload if isinstance(payload, dict) else {},
    }


def agents_status() -> dict[str, Any]:
    root = repo_root()
    paths = {
        "root": root / "AGENTS.md",
        "trainer": root / "trainer" / "AGENTS.md",
        "sim": root / "sim" / "AGENTS.md",
        "scripts": root / "scripts" / "AGENTS.md",
        "docs": root / "docs" / "AGENTS.md",
        "configs": root / "configs" / "AGENTS.md",
    }
    return {
        "paths": {name: str(path.resolve()) for name, path in paths.items()},
        "present": {name: path.exists() for name, path in paths.items()},
        "root_present": paths["root"].exists(),
        "subdir_present_count": sum(1 for name, path in paths.items() if name != "root" and path.exists()),
    }


def latest_agents_consistency() -> dict[str, Any]:
    root = _latest_autonomy_artifact_root()
    latest_json = root / "latest_agents_consistency.json"
    latest_md = root / "latest_agents_consistency.md"
    payload = read_json(latest_json)
    return {
        "artifact_family": root.name,
        "json_path": str(latest_json.resolve()),
        "md_path": str(latest_md.resolve()),
        "payload": payload if isinstance(payload, dict) else {},
    }


def latest_autonomy_entry() -> dict[str, Any]:
    root = _latest_autonomy_artifact_root()
    latest_json = root / "latest_autonomy_entry.json"
    latest_md = root / "latest_autonomy_entry.md"
    payload = read_json(latest_json)
    return {
        "artifact_family": root.name,
        "json_path": str(latest_json.resolve()),
        "md_path": str(latest_md.resolve()),
        "payload": payload if isinstance(payload, dict) else {},
    }


def environment_status() -> dict[str, Any]:
    doctor = latest_doctor_report()
    bootstrap = latest_bootstrap_state()
    doctor_payload = doctor.get("payload") if isinstance(doctor.get("payload"), dict) else {}
    bootstrap_payload = bootstrap.get("payload") if isinstance(bootstrap.get("payload"), dict) else {}
    return {
        "doctor": doctor,
        "bootstrap": bootstrap,
        "status": str(doctor_payload.get("status") or ""),
        "recommended_mode": str(doctor_payload.get("recommended_mode") or bootstrap_payload.get("recommended_mode") or ""),
        "ready_for_continuation": bool(doctor_payload.get("ready_for_continuation")) if doctor_payload else bool(bootstrap_payload.get("bootstrap_complete")),
        "selected_training_python": str((((doctor_payload.get("resolver") or {}).get("selected")) or {}).get("python") or bootstrap_payload.get("selected_training_python") or ""),
        "training_env_name": str((((doctor_payload.get("resolver") or {}).get("selected")) or {}).get("env_name") or ""),
        "training_env_source": str(((doctor_payload.get("resolver") or {}).get("selection_reason")) or ""),
        "next_steps": [str(item) for item in (doctor_payload.get("next_steps") or []) if str(item).strip()],
        "blocking_reasons": [str(item) for item in (doctor_payload.get("blocking_reasons") or []) if str(item).strip()],
        "warnings": [str(item) for item in (doctor_payload.get("warnings") or []) if str(item).strip()],
    }


def blocked_campaigns(limit: int = 24) -> list[dict[str, Any]]:
    return [
        row
        for row in campaign_rows(limit=limit)
        if bool(row.get("human_gate_triggered")) or str(row.get("status") or "") == "blocked"
    ]


def _pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        proc = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except Exception:
        return False
    text = str(proc.stdout or "")
    return str(pid) in text and "No tasks are running" not in text


def ops_jobs(limit: int = 20) -> list[dict[str, Any]]:
    jobs_root = artifacts_root() / "p53" / "ops_ui" / "jobs"
    if not jobs_root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(jobs_root.glob("*.json"), key=_sort_key, reverse=True)[: max(1, int(limit))]:
        payload = read_json(path)
        if not isinstance(payload, dict):
            continue
        job = dict(payload)
        pid = int(job.get("pid") or 0)
        if str(job.get("status") or "") == "running" and not _pid_running(pid):
            job["status"] = "completed_or_exited"
        rows.append(job)
    return rows


def ops_audit_tail(limit: int = 20) -> list[dict[str, Any]]:
    path = artifacts_root() / "p53" / "ops_audit" / "ops_audit.jsonl"
    rows = read_jsonl(path)
    return rows[-max(1, int(limit)) :]


def dashboard_summary() -> dict[str, Any]:
    root = artifacts_root() / "dashboard" / "latest"
    data_path = root / "dashboard_data.json"
    payload = read_json(data_path)
    return {
        "index_path": str((root / "index.html").resolve()),
        "data_path": str(data_path.resolve()),
        "payload": payload if isinstance(payload, dict) else {},
    }


def latest_resume_target() -> dict[str, Any]:
    rows = campaign_rows(limit=24)
    if not rows:
        return {}
    preferred = next(
        (
            row
            for row in rows
            if str(row.get("status") or "") in {"blocked", "failed", "running"}
            and str(row.get("resume_command") or "").strip()
        ),
        None,
    )
    return preferred or rows[0]


def build_ops_state(
    *,
    registry_family: str = "",
    registry_status: str = "",
    registry_latest: bool = False,
    registry_promoted: bool = False,
) -> dict[str, Any]:
    return {
        "schema": "p53_ops_ui_state_v1",
        "repo_root": str(repo_root()),
        "p22": latest_p22_run(),
        "progress": latest_progress_events(),
        "campaigns": campaign_rows(),
        "blocked_campaigns": blocked_campaigns(),
        "latest_resume_target": latest_resume_target(),
        "registry": registry_entries(
            family=registry_family,
            status=registry_status,
            latest=registry_latest,
            promoted=registry_promoted,
        ),
        "promotion_queue": latest_promotion_queue(),
        "attention_queue": latest_attention_queue(),
        "morning_summary": latest_morning_summary(),
        "bootstrap": latest_bootstrap_state(),
        "doctor": latest_doctor_report(),
        "agents": agents_status(),
        "agents_consistency": latest_agents_consistency(),
        "autonomy_entry": latest_autonomy_entry(),
        "environment": environment_status(),
        "readiness": latest_readiness_report(),
        "window_state": latest_window_state(),
        "background_validation": latest_background_validation(),
        "dashboard": dashboard_summary(),
        "ops_ui": latest_ops_ui_state(),
        "jobs": ops_jobs(),
        "audit_tail": ops_audit_tail(),
    }
