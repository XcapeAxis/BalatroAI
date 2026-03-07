from __future__ import annotations

from typing import Any

from trainer.registry.checkpoint_schema import CheckpointTransition, normalize_entry, normalize_status


ALLOWED_TRANSITIONS = {
    "draft": {"smoke_passed", "archived"},
    "smoke_passed": {"arena_passed", "archived"},
    "arena_passed": {"promotion_review", "archived"},
    "promotion_review": {"promoted", "rejected", "archived"},
    "promoted": {"archived"},
    "rejected": {"archived"},
    "archived": set(),
}


def can_transition(from_status: str, to_status: str) -> bool:
    source = normalize_status(from_status)
    target = normalize_status(to_status)
    if source == target:
        return True
    return target in ALLOWED_TRANSITIONS.get(source, set())


def apply_transition(
    entry: dict[str, Any],
    *,
    to_status: str,
    reason: str = "",
    operator: str = "system",
    refs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = normalize_entry(entry)
    current_status = normalize_status(normalized.get("status"))
    target_status = normalize_status(to_status)
    if not can_transition(current_status, target_status):
        raise ValueError(f"invalid checkpoint transition: {current_status} -> {target_status}")
    transition = CheckpointTransition(
        from_status=current_status,
        to_status=target_status,
        reason=str(reason or ""),
        operator=str(operator or "system"),
        refs=dict(refs or {}),
    ).to_dict()
    transitions = list(normalized.get("transitions") or [])
    transitions.append(transition)
    normalized["status"] = target_status
    normalized["transitions"] = transitions
    if refs:
        if "arena_ref" in refs and not normalized.get("arena_ref"):
            normalized["arena_ref"] = str(refs.get("arena_ref") or "")
        if "triage_ref" in refs and not normalized.get("triage_ref"):
            normalized["triage_ref"] = str(refs.get("triage_ref") or "")
    return normalize_entry(normalized)
