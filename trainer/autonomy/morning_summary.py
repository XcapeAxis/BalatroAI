from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_repo_root() -> Path:
    preferred = Path("D:/MYFILES/BalatroAI")
    if preferred.exists():
        return preferred.resolve()
    return Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _latest_matching_path(root: Path, pattern: str) -> Path | None:
    paths = sorted(root.glob(pattern), key=lambda item: (item.stat().st_mtime if item.exists() else 0.0, str(item)), reverse=True)
    return paths[0].resolve() if paths else None


def _campaign_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/campaign_state.json"), key=lambda item: (item.stat().st_mtime, str(item)), reverse=True)[:24]:
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        stages = [dict(item) for item in (payload.get("stages") or []) if isinstance(item, dict)]
        current = next((stage for stage in stages if str(stage.get("status") or "") in {"running", "failed", "blocked"}), stages[-1] if stages else {})
        rows.append(
            {
                "campaign_id": str(payload.get("campaign_id") or ""),
                "run_id": str(payload.get("run_id") or ""),
                "experiment_id": str(payload.get("experiment_id") or ""),
                "seed": str(payload.get("seed") or ""),
                "stage_id": str((current or {}).get("stage_id") or ""),
                "status": str((current or {}).get("status") or ""),
                "autonomy_decision": str((current or {}).get("autonomy_decision") or ""),
                "autonomy_reason": str((current or {}).get("autonomy_reason") or ""),
                "attention_item_ref": str((current or {}).get("attention_item_ref") or ""),
                "state_path": str(path.resolve()),
            }
        )
    return rows


def _registry_summary(root: Path) -> dict[str, Any]:
    path = root / "registry" / "checkpoints_registry.json"
    payload = _read_json(path)
    items = payload.get("items") if isinstance(payload, dict) and isinstance(payload.get("items"), list) else []
    latest = [item for item in items if isinstance(item, dict)][:12]
    new_checkpoints = [
        {
            "checkpoint_id": str(item.get("checkpoint_id") or ""),
            "family": str(item.get("family") or ""),
            "status": str(item.get("status") or ""),
            "created_at": str(item.get("created_at") or ""),
            "deployment_mode_recommendation": str(item.get("deployment_mode_recommendation") or ""),
        }
        for item in latest
    ]
    return {"path": str(path.resolve()), "new_checkpoints": new_checkpoints}


def _promotion_queue(root: Path) -> dict[str, Any]:
    path = _latest_matching_path(root, "**/promotion_queue.json")
    payload = _read_json(path) if isinstance(path, Path) else {}
    review = payload.get("promotion_review") if isinstance(payload, dict) and isinstance(payload.get("promotion_review"), list) else []
    return {
        "path": str(path) if isinstance(path, Path) else "",
        "review_items": [item for item in review if isinstance(item, dict)][:12],
        "counts": payload.get("counts") if isinstance(payload, dict) and isinstance(payload.get("counts"), dict) else {},
    }


def _attention_queue(root: Path) -> dict[str, Any]:
    path = root / "attention_required" / "attention_queue.json"
    payload = _read_json(path)
    items = payload.get("items") if isinstance(payload, dict) and isinstance(payload.get("items"), list) else []
    open_items = [item for item in items if isinstance(item, dict) and str(item.get("status") or "") == "open"]
    return {
        "path": str(path.resolve()),
        "items": [item for item in items if isinstance(item, dict)],
        "open_items": open_items,
    }


def _doctor_status(root: Path) -> dict[str, Any]:
    json_path = root / "p58" / "latest_doctor.json"
    md_path = root / "p58" / "latest_doctor.md"
    payload = _read_json(json_path)
    return {
        "json_path": str(json_path.resolve()),
        "md_path": str(md_path.resolve()),
        "payload": payload if isinstance(payload, dict) else {},
    }


def _dashboard_excerpt(root: Path) -> dict[str, Any]:
    md_path = _latest_matching_path(root / "dashboard" / "latest", "*.html")
    data_path = root / "dashboard" / "latest" / "dashboard_data.json"
    payload = _read_json(data_path)
    return {
        "index_path": str(md_path) if isinstance(md_path, Path) else "",
        "data_path": str(data_path.resolve()),
        "payload": payload if isinstance(payload, dict) else {},
    }


def _latest_p22(root: Path) -> dict[str, Any]:
    runs_root = root / "p22" / "runs"
    runs = sorted([path for path in runs_root.iterdir() if path.is_dir()], key=lambda item: (item.stat().st_mtime, str(item)), reverse=True) if runs_root.exists() else []
    if not runs:
        return {}
    run_dir = runs[0]
    summary = _read_json(run_dir / "summary_table.json")
    rows = summary.get("rows") if isinstance(summary, dict) and isinstance(summary.get("rows"), list) else summary if isinstance(summary, list) else []
    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "summary_rows": [row for row in rows if isinstance(row, dict)],
    }


def _recommended_first_action(
    campaigns: list[dict[str, Any]],
    attention_open: list[dict[str, Any]],
    promotion_review: list[dict[str, Any]],
    doctor: dict[str, Any],
) -> str:
    doctor_payload = doctor.get("payload") if isinstance(doctor.get("payload"), dict) else {}
    if str(doctor_payload.get("status") or "") == "blocked":
        return "Repair the machine environment from the latest doctor report before resuming campaigns"
    blocking = next((item for item in attention_open if str(item.get("severity") or "") == "block"), None)
    if isinstance(blocking, dict):
        return str(blocking.get("title") or blocking.get("summary") or "Review blocking attention item")
    blocked_campaign = next((row for row in campaigns if str(row.get("status") or "") == "blocked"), None)
    if isinstance(blocked_campaign, dict):
        return "Resolve blocked campaign `{}` at stage `{}`".format(
            blocked_campaign.get("campaign_id") or "",
            blocked_campaign.get("stage_id") or "",
        )
    if promotion_review:
        first = promotion_review[0]
        return "Review promotion candidate `{}` before applying any deployment change".format(first.get("checkpoint_id") or "")
    failed = next((row for row in campaigns if str(row.get("status") or "") == "failed"), None)
    if isinstance(failed, dict):
        return "Inspect failed campaign `{}` at stage `{}`".format(failed.get("campaign_id") or "", failed.get("stage_id") or "")
    return "No blocking action. Start with the latest morning summary and dashboard."


def build_morning_summary(*, artifacts_root: str | Path | None = None, out_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(artifacts_root).resolve() if artifacts_root else (resolve_repo_root() / "docs" / "artifacts").resolve()
    out_dir = Path(out_root).resolve() if out_root else (root / "morning_summary").resolve()
    campaigns = _campaign_rows(root)
    registry = _registry_summary(root)
    promotion = _promotion_queue(root)
    attention = _attention_queue(root)
    doctor = _doctor_status(root)
    dashboard = _dashboard_excerpt(root)
    p22 = _latest_p22(root)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    latest_md = out_dir / "latest.md"
    latest_json = out_dir / "latest.json"
    stamped_md = out_dir / f"{timestamp}.md"
    stamped_json = out_dir / f"{timestamp}.json"
    first_action = _recommended_first_action(campaigns, attention.get("open_items") or [], promotion.get("review_items") or [], doctor)

    lines = [
        "# Morning Summary",
        "",
        f"- generated_at: `{_now_iso()}`",
        f"- latest_p22_run: `{p22.get('run_id') or ''}`",
        f"- attention_queue: `{attention.get('path') or ''}`",
        f"- promotion_queue: `{promotion.get('path') or ''}`",
        f"- doctor_report: `{doctor.get('json_path') or ''}`",
        "",
        "## Completed Campaigns / Stages",
        "",
    ]
    for row in campaigns[:12]:
        lines.append(
            "- `{campaign_id}` exp=`{experiment_id}` seed=`{seed}` stage=`{stage_id}` status=`{status}` decision=`{autonomy_decision}`".format(**row)
        )
    if not campaigns:
        lines.append("- No campaign states found.")
    lines += ["", "## New Checkpoints", ""]
    for row in registry.get("new_checkpoints") or []:
        lines.append(
            "- `{checkpoint_id}` family=`{family}` status=`{status}` deploy=`{deployment_mode_recommendation}`".format(**row)
        )
    if not (registry.get("new_checkpoints") or []):
        lines.append("- No recent checkpoints found.")
    lines += ["", "## Promotion Review", ""]
    for row in promotion.get("review_items") or []:
        lines.append(
            "- `{}` family=`{}` status=`{}` deploy=`{}`".format(
                row.get("checkpoint_id") or "",
                row.get("family") or "",
                row.get("status") or "",
                row.get("deployment_mode_recommendation") or "",
            )
        )
    if not (promotion.get("review_items") or []):
        lines.append("- No items in promotion_review.")
    lines += ["", "## Environment Doctor", ""]
    doctor_payload = doctor.get("payload") if isinstance(doctor.get("payload"), dict) else {}
    if doctor_payload:
        lines.append(
            "- status=`{}` recommended_mode=`{}` ready=`{}` report=`{}`".format(
                doctor_payload.get("status") or "",
                doctor_payload.get("recommended_mode") or "",
                doctor_payload.get("ready_for_continuation"),
                doctor.get("json_path") or "",
            )
        )
        for item in (doctor_payload.get("blocking_reasons") or [])[:6]:
            lines.append(f"  - block: {item}")
        for item in (doctor_payload.get("warnings") or [])[:6]:
            lines.append(f"  - warn: {item}")
    else:
        lines.append("- No doctor report found.")
    lines += ["", "## Attention Queue", ""]
    for item in attention.get("open_items") or []:
        lines.append(
            "- `{}` severity=`{}` category=`{}` title={}".format(
                item.get("attention_id") or "",
                item.get("severity") or "",
                item.get("category") or "",
                item.get("title") or "",
            )
        )
    if not (attention.get("open_items") or []):
        lines.append("- No open attention items.")
    lines += ["", "## Recommended First Action", "", f"- {first_action}", ""]
    payload = {
        "schema": "p57_morning_summary_v1",
        "generated_at": _now_iso(),
        "artifacts_root": str(root),
        "campaigns": campaigns,
        "registry": registry,
        "promotion_queue": promotion,
        "attention_queue": attention,
        "doctor": doctor,
        "dashboard": {"index_path": dashboard.get("index_path") or "", "data_path": dashboard.get("data_path") or ""},
        "latest_p22": p22,
        "recommended_first_action": first_action,
        "latest_md": str(latest_md.resolve()),
        "latest_json": str(latest_json.resolve()),
        "timestamped_md": str(stamped_md.resolve()),
        "timestamped_json": str(stamped_json.resolve()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines).rstrip() + "\n"
    latest_md.write_text(text, encoding="utf-8")
    stamped_md.write_text(text, encoding="utf-8")
    latest_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stamped_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def write_morning_summary(*, repo: str | Path | None = None, artifacts_root: str | Path | None = None, out_root: str | Path | None = None) -> dict[str, Any]:
    resolved_repo = Path(repo).resolve() if repo else resolve_repo_root()
    root = Path(artifacts_root).resolve() if artifacts_root else (resolved_repo / "docs" / "artifacts").resolve()
    return build_morning_summary(artifacts_root=root, out_root=out_root)


def main() -> int:
    parser = argparse.ArgumentParser(description="P57 morning summary builder")
    parser.add_argument("--artifacts-root", default="")
    parser.add_argument("--out-root", default="")
    args = parser.parse_args()
    payload = build_morning_summary(artifacts_root=args.artifacts_root or None, out_root=args.out_root or None)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
