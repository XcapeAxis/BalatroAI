from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    token = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(value or ""))
    parts = [part for part in token.split("-") if part]
    return "-".join(parts)[:80] or "attention-item"


@dataclass
class AttentionItem:
    attention_id: str
    created_at: str
    severity: str = "info"
    category: str = "general"
    title: str = ""
    summary: str = ""
    summary_for_human: str = ""
    blocking_stage: str = ""
    blocking_scope: str = ""
    attempted_actions: list[str] | None = None
    recommended_options: list[dict[str, str]] | None = None
    recommended_default: str = ""
    required_human_input: list[str] | None = None
    artifact_refs: list[str] | None = None
    suggested_commands: list[str] | None = None
    related_campaign: str = ""
    related_checkpoint_ids: list[str] | None = None
    decision_deadline_hint: str = ""
    status: str = "open"
    campaign_id: str = ""
    run_id: str = ""
    experiment_id: str = ""
    seed: str = ""
    dedupe_key: str = ""
    resolution_note: str = ""
    resolved_at: str = ""
    item_md_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["attempted_actions"] = list(self.attempted_actions or [])
        payload["recommended_options"] = [dict(item) for item in (self.recommended_options or [])]
        payload["required_human_input"] = list(self.required_human_input or [])
        payload["artifact_refs"] = list(self.artifact_refs or [])
        payload["suggested_commands"] = list(self.suggested_commands or [])
        payload["related_checkpoint_ids"] = list(self.related_checkpoint_ids or [])
        return payload


def _normalize_option(option: Any) -> dict[str, str] | None:
    if isinstance(option, dict):
        label = str(option.get("label") or option.get("name") or "").strip()
        description = str(option.get("description") or option.get("summary") or "").strip()
        if not label and not description:
            return None
        return {
            "label": label or description,
            "description": description,
        }
    token = str(option or "").strip()
    if not token:
        return None
    return {"label": token, "description": ""}


def _normalize_required_input(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    token = str(value or "").strip()
    return [token] if token else []


def _normalize_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    token = str(value or "").strip()
    return [token] if token else []


def normalize_item(payload: dict[str, Any]) -> dict[str, Any]:
    item = AttentionItem(
        attention_id=str(payload.get("attention_id") or ""),
        created_at=str(payload.get("created_at") or now_iso()),
        severity=str(payload.get("severity") or "info"),
        category=str(payload.get("category") or "general"),
        title=str(payload.get("title") or ""),
        summary=str(payload.get("summary") or ""),
        summary_for_human=str(payload.get("summary_for_human") or payload.get("summary") or ""),
        blocking_stage=str(payload.get("blocking_stage") or ""),
        blocking_scope=str(payload.get("blocking_scope") or ""),
        attempted_actions=[str(item) for item in (payload.get("attempted_actions") or []) if str(item).strip()],
        recommended_options=[
            option
            for option in (_normalize_option(item) for item in (payload.get("recommended_options") or []))
            if option
        ],
        recommended_default=str(payload.get("recommended_default") or ""),
        required_human_input=_normalize_required_input(payload.get("required_human_input")),
        artifact_refs=[str(item) for item in (payload.get("artifact_refs") or []) if str(item).strip()],
        suggested_commands=_normalize_text_list(payload.get("suggested_commands")),
        related_campaign=str(payload.get("related_campaign") or payload.get("campaign_id") or ""),
        related_checkpoint_ids=_normalize_text_list(payload.get("related_checkpoint_ids")),
        decision_deadline_hint=str(payload.get("decision_deadline_hint") or ""),
        status=str(payload.get("status") or "open"),
        campaign_id=str(payload.get("campaign_id") or ""),
        run_id=str(payload.get("run_id") or ""),
        experiment_id=str(payload.get("experiment_id") or ""),
        seed=str(payload.get("seed") or ""),
        dedupe_key=str(payload.get("dedupe_key") or ""),
        resolution_note=str(payload.get("resolution_note") or ""),
        resolved_at=str(payload.get("resolved_at") or ""),
        item_md_path=str(payload.get("item_md_path") or ""),
    )
    if not item.attention_id:
        basis = item.dedupe_key or item.title or item.summary or item.category
        item.attention_id = f"attention-{slugify(basis)}"
    return item.to_dict()
