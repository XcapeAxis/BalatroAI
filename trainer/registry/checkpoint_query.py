from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _parse_time(value: Any) -> datetime:
    token = str(value or "").strip()
    if not token:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    try:
        parsed = datetime.fromisoformat(token.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.fromtimestamp(0, tz=timezone.utc)


def filter_entries(
    entries: list[dict[str, Any]],
    *,
    family: str = "",
    status: str = "",
    source_run_id: str = "",
    source_experiment_id: str = "",
    checkpoint_id: str = "",
) -> list[dict[str, Any]]:
    out = list(entries)
    if checkpoint_id:
        out = [item for item in out if str(item.get("checkpoint_id") or "") == checkpoint_id]
    if family:
        token = str(family).strip().lower()
        out = [item for item in out if str(item.get("family") or "").strip().lower() == token]
    if status:
        token = str(status).strip().lower()
        out = [item for item in out if str(item.get("status") or "").strip().lower() == token]
    if source_run_id:
        token = str(source_run_id).strip()
        out = [item for item in out if str(item.get("source_run_id") or "") == token]
    if source_experiment_id:
        token = str(source_experiment_id).strip()
        out = [item for item in out if str(item.get("source_experiment_id") or "") == token]
    return out


def sort_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(entries, key=lambda item: (_parse_time(item.get("created_at")), str(item.get("checkpoint_id") or "")), reverse=True)


def latest_by_family(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for item in sort_entries(entries):
        family = str(item.get("family") or "other").strip().lower() or "other"
        latest.setdefault(family, item)
    return list(latest.values())


def promoted_by_family(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    promoted: dict[str, dict[str, Any]] = {}
    for item in sort_entries(entries):
        if str(item.get("status") or "").strip().lower() != "promoted":
            continue
        family = str(item.get("family") or "other").strip().lower() or "other"
        promoted.setdefault(family, item)
    return list(promoted.values())


def waiting_on_arena(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sort_entries([item for item in entries if str(item.get("status") or "") == "smoke_passed"])


def promotion_review_queue(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sort_entries([item for item in entries if str(item.get("status") or "") == "promotion_review"])


def stale_drafts(entries: list[dict[str, Any]], *, limit: int = 20) -> list[dict[str, Any]]:
    drafts = [item for item in sort_entries(entries) if str(item.get("status") or "") == "draft"]
    return drafts[: max(1, int(limit))]
