from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_repo_root(repo_root: str | Path | None = None) -> Path:
    if repo_root:
        return Path(repo_root).resolve()
    preferred = Path("D:/MYFILES/BalatroAI")
    if preferred.exists():
        return preferred.resolve()
    return Path(__file__).resolve().parents[2]


def certification_root(repo_root: str | Path | None = None) -> Path:
    return (resolve_repo_root(repo_root) / "docs" / "artifacts" / "certification_queue").resolve()


def queue_json_path(repo_root: str | Path | None = None) -> Path:
    return certification_root(repo_root) / "certification_queue.json"


def queue_md_path(repo_root: str | Path | None = None) -> Path:
    return certification_root(repo_root) / "certification_queue.md"


def state_json_path(repo_root: str | Path | None = None) -> Path:
    return certification_root(repo_root) / "certification_state.json"


def summary_md_path(repo_root: str | Path | None = None) -> Path:
    return certification_root(repo_root) / "certification_summary.md"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _read_safe_run_summary(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    return _read_json(Path(path).resolve())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _normalize_item(item: dict[str, Any]) -> dict[str, Any]:
    pending_tasks = [dict(task) for task in (item.get("pending_tasks") or []) if isinstance(task, dict)]
    return {
        "certification_id": str(item.get("certification_id") or ""),
        "created_at": str(item.get("created_at") or _now_iso()),
        "updated_at": str(item.get("updated_at") or _now_iso()),
        "status": str(item.get("status") or "pending"),
        "source_run_id": str(item.get("source_run_id") or ""),
        "source_kind": str(item.get("source_kind") or "fast_check"),
        "source_ref": str(item.get("source_ref") or ""),
        "trigger_reason": str(item.get("trigger_reason") or ""),
        "validation_tiers_completed": [str(item_) for item_ in (item.get("validation_tiers_completed") or []) if str(item_).strip()],
        "pending_tasks": pending_tasks,
        "recommended_command": str(item.get("recommended_command") or ""),
        "recommended_command_argv": [str(item_) for item_ in (item.get("recommended_command_argv") or []) if str(item_).strip()],
        "fast_check_report_path": str(item.get("fast_check_report_path") or ""),
        "related_artifacts": [str(item_) for item_ in (item.get("related_artifacts") or []) if str(item_).strip()],
        "result_summary_path": str(item.get("result_summary_path") or ""),
        "last_safe_run_summary_path": str(item.get("last_safe_run_summary_path") or ""),
        "last_exit_code": item.get("last_exit_code"),
    }


def _reconcile_item(item: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_item(item)
    if normalized.get("status") != "running":
        return normalized
    summary = (
        _read_safe_run_summary(normalized.get("last_safe_run_summary_path"))
        or _read_safe_run_summary(normalized.get("result_summary_path"))
    )
    if not isinstance(summary, dict):
        return normalized
    if not summary.get("end_at_utc"):
        return normalized
    exit_code = int(summary.get("exit_code") or 0)
    normalized["last_exit_code"] = exit_code
    normalized["updated_at"] = _now_iso()
    normalized["status"] = "passed" if exit_code == 0 and not bool(summary.get("timed_out")) else "failed"
    normalized["result_summary_path"] = str(normalized.get("result_summary_path") or normalized.get("last_safe_run_summary_path") or "")
    return normalized


def load_queue(repo_root: str | Path | None = None) -> dict[str, Any]:
    path = queue_json_path(repo_root)
    payload = _read_json(path)
    if not isinstance(payload, dict):
        payload = {
            "schema": "p61_certification_queue_v1",
            "generated_at": _now_iso(),
            "queue_path": str(path.resolve()),
            "items": [],
        }
    payload["queue_path"] = str(path.resolve())
    payload["items"] = [_reconcile_item(item) for item in (payload.get("items") or []) if isinstance(item, dict)]
    return payload


def _summary_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"pending": 0, "running": 0, "passed": 0, "failed": 0, "not_required": 0}
    for item in items:
        status = str(item.get("status") or "pending")
        counts[status] = int(counts.get(status, 0)) + 1
    return counts


def render_queue_md(payload: dict[str, Any]) -> str:
    items = [item for item in (payload.get("items") or []) if isinstance(item, dict)]
    counts = _summary_counts(items)
    lines = [
        "# Certification Queue",
        "",
        f"- generated_at: `{payload.get('generated_at')}`",
        f"- queue_path: `{payload.get('queue_path')}`",
        f"- pending: `{counts.get('pending', 0)}`",
        f"- running: `{counts.get('running', 0)}`",
        f"- passed: `{counts.get('passed', 0)}`",
        f"- failed: `{counts.get('failed', 0)}`",
        "",
        "## Items",
        "",
    ]
    if not items:
        lines.append("- No certification items.")
    for item in items:
        lines.append(
            "- `{}` status=`{}` source_run_id=`{}` reason=`{}` command=`{}`".format(
                item.get("certification_id") or "",
                item.get("status") or "",
                item.get("source_run_id") or "",
                item.get("trigger_reason") or "",
                item.get("recommended_command") or "",
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def _write_summary_files(repo_root: str | Path | None, payload: dict[str, Any], state_payload: dict[str, Any] | None = None) -> None:
    queue_md_path(repo_root).write_text(render_queue_md(payload), encoding="utf-8")
    state = state_payload or {
        "schema": "p61_certification_state_v1",
        "generated_at": _now_iso(),
        "queue_path": str(queue_json_path(repo_root).resolve()),
        "counts": _summary_counts(payload.get("items") or []),
        "latest_pending": next((item for item in (payload.get("items") or []) if isinstance(item, dict) and str(item.get("status") or "") == "pending"), {}),
    }
    _write_json(state_json_path(repo_root), state)
    summary_lines = [
        "# Certification Summary",
        "",
        f"- generated_at: `{state.get('generated_at')}`",
        f"- queue_path: `{state.get('queue_path')}`",
        f"- counts: `{json.dumps(state.get('counts') or {}, ensure_ascii=False)}`",
        f"- latest_pending: `{((state.get('latest_pending') or {}).get('certification_id') if isinstance(state.get('latest_pending'), dict) else '')}`",
    ]
    summary_md_path(repo_root).write_text("\n".join(summary_lines).rstrip() + "\n", encoding="utf-8")


def save_queue(payload: dict[str, Any], repo_root: str | Path | None = None, *, state_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    path = queue_json_path(repo_root)
    payload["generated_at"] = _now_iso()
    payload["queue_path"] = str(path.resolve())
    payload["items"] = [_normalize_item(item) for item in (payload.get("items") or []) if isinstance(item, dict)]
    _write_json(path, payload)
    _write_summary_files(repo_root, payload, state_payload=state_payload)
    return payload


def upsert_from_fast_check(report: dict[str, Any], *, repo_root: str | Path | None = None) -> dict[str, Any]:
    queue = load_queue(repo_root)
    deferred = [item for item in (report.get("deferred_certification") or []) if isinstance(item, dict)]
    if not deferred:
        state = {
            "schema": "p61_certification_state_v1",
            "generated_at": _now_iso(),
            "queue_path": str(queue_json_path(repo_root).resolve()),
            "counts": _summary_counts(queue.get("items") or []),
            "latest_pending": {},
            "status": "not_required",
            "source_run_id": str(report.get("run_id") or ""),
        }
        save_queue(queue, repo_root, state_payload=state)
        return state

    source_run_id = str(report.get("run_id") or "")
    source_ref = str(report.get("json_path") or "")
    item = next(
        (
            row
            for row in (queue.get("items") or [])
            if isinstance(row, dict)
            and str(row.get("source_run_id") or "") == source_run_id
            and str(row.get("source_ref") or "") == source_ref
        ),
        {},
    )
    item = _normalize_item(item)
    if not item.get("certification_id"):
        item["certification_id"] = "cert-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        item["created_at"] = _now_iso()
    first = deferred[0]
    item.update(
        {
            "updated_at": _now_iso(),
            "status": "pending",
            "source_run_id": source_run_id,
            "source_kind": "fast_check",
            "source_ref": source_ref,
            "trigger_reason": str(report.get("required_next_step") or "pending_certification"),
            "validation_tiers_completed": [str(item_) for item_ in (report.get("validation_tiers_completed") or []) if str(item_).strip()],
            "pending_tasks": deferred,
            "recommended_command": str(first.get("command_template") or ""),
            "recommended_command_argv": [str(item_) for item_ in (first.get("command_argv") or []) if str(item_).strip()],
            "fast_check_report_path": source_ref,
            "related_artifacts": [str(item_) for item_ in (report.get("artifact_refs") or []) if str(item_).strip()],
        }
    )
    items = [
        row
        for row in (queue.get("items") or [])
        if isinstance(row, dict)
        and not (
            str(row.get("source_run_id") or "") == source_run_id
            and str(row.get("source_ref") or "") == source_ref
        )
    ]
    items.insert(0, item)
    queue["items"] = items
    state = {
        "schema": "p61_certification_state_v1",
        "generated_at": _now_iso(),
        "queue_path": str(queue_json_path(repo_root).resolve()),
        "counts": _summary_counts(queue.get("items") or []),
        "latest_pending": item,
        "status": "pending",
        "source_run_id": source_run_id,
    }
    save_queue(queue, repo_root, state_payload=state)
    return state


def run_latest_pending(*, repo_root: str | Path | None = None, dry_run: bool = False, timeout_sec: int = 0) -> dict[str, Any]:
    root = resolve_repo_root(repo_root)
    queue = load_queue(root)
    items = [item for item in (queue.get("items") or []) if isinstance(item, dict)]
    target = next((item for item in items if str(item.get("status") or "") == "pending"), None)
    if not isinstance(target, dict):
        state = {
            "schema": "p61_certification_state_v1",
            "generated_at": _now_iso(),
            "queue_path": str(queue_json_path(root).resolve()),
            "counts": _summary_counts(items),
            "latest_pending": {},
            "status": "not_required",
        }
        save_queue(queue, root, state_payload=state)
        return state

    target["status"] = "running"
    target["updated_at"] = _now_iso()
    state = {
        "schema": "p61_certification_state_v1",
        "generated_at": _now_iso(),
        "queue_path": str(queue_json_path(root).resolve()),
        "counts": _summary_counts(items),
        "latest_pending": target,
        "status": "running",
    }
    save_queue(queue, root, state_payload=state)
    if dry_run:
        return state

    safe_run_script = root / "scripts" / "safe_run.ps1"
    timeout_value = int(timeout_sec or 0) if int(timeout_sec or 0) > 0 else 14400
    summary_path = certification_root(root) / f"safe_run_{target.get('certification_id')}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    command_argv = [str(item) for item in (target.get("recommended_command_argv") or []) if str(item).strip()]
    if not command_argv:
        target["status"] = "failed"
        target["updated_at"] = _now_iso()
        target["last_exit_code"] = 2
        save_queue(queue, root)
        return {
            "schema": "p61_certification_state_v1",
            "generated_at": _now_iso(),
            "queue_path": str(queue_json_path(root).resolve()),
            "counts": _summary_counts(items),
            "latest_pending": target,
            "status": "failed",
            "reason": "missing_command_argv",
        }

    env = os.environ.copy()
    env["BALATRO_CERTIFICATION_QUEUE_REF"] = str(queue_json_path(root).resolve())
    env["BALATRO_CERTIFICATION_STATUS"] = "running"
    cmd = [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(safe_run_script),
        "-TimeoutSec",
        str(timeout_value),
        "-SummaryJson",
        str(summary_path.resolve()),
        "--",
        *command_argv,
    ]
    proc = subprocess.run(cmd, cwd=str(root), env=env, check=False)
    target["updated_at"] = _now_iso()
    target["last_safe_run_summary_path"] = str(summary_path.resolve())
    target["last_exit_code"] = int(proc.returncode or 0)
    target["status"] = "passed" if int(proc.returncode or 0) == 0 else "failed"
    target["result_summary_path"] = str(summary_path.resolve())
    state = {
        "schema": "p61_certification_state_v1",
        "generated_at": _now_iso(),
        "queue_path": str(queue_json_path(root).resolve()),
        "counts": _summary_counts(items),
        "latest_pending": target,
        "status": target["status"],
    }
    save_queue(queue, root, state_payload=state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description="P61 certification queue")
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--enqueue-fast-report", default="")
    parser.add_argument("--run-latest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=0)
    args = parser.parse_args()
    root = resolve_repo_root(args.repo_root or None)
    if args.enqueue_fast_report:
        payload = _read_json(Path(args.enqueue_fast_report).resolve()) or {}
        result = upsert_from_fast_check(payload, repo_root=root)
    elif args.run_latest:
        result = run_latest_pending(repo_root=root, dry_run=bool(args.dry_run), timeout_sec=int(args.timeout_sec or 0))
    else:
        result = load_queue(root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
