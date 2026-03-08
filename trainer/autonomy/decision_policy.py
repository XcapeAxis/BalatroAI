from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def resolve_repo_root() -> Path:
    preferred = Path("D:/MYFILES/BalatroAI")
    if preferred.exists():
        return preferred.resolve()
    return Path(__file__).resolve().parents[2]


def default_policy_path(repo_root: Path | None = None) -> Path:
    root = repo_root.resolve() if isinstance(repo_root, Path) else resolve_repo_root()
    yaml_path = root / "configs" / "runtime" / "decision_policy.yaml"
    if yaml_path.exists():
        return yaml_path
    return root / "configs" / "runtime" / "decision_policy.json"


def _read_payload(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is not None:
            payload = yaml.safe_load(text)
        else:
            payload = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"decision policy must be a mapping: {path}")
    return payload


def _normalize_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def normalize_policy(payload: dict[str, Any], *, source_path: str = "") -> dict[str, Any]:
    return {
        "schema": str(payload.get("schema") or "p57_decision_policy_v1"),
        "source_path": str(source_path or payload.get("source_path") or ""),
        "auto_allow_actions": _normalize_text_list(payload.get("auto_allow_actions")),
        "auto_suggest_actions": _normalize_text_list(payload.get("auto_suggest_actions")),
        "human_required_actions": _normalize_text_list(payload.get("human_required_actions")),
        "stop_conditions": _normalize_text_list(payload.get("stop_conditions")),
        "continue_on_warning_conditions": _normalize_text_list(payload.get("continue_on_warning_conditions")),
    }


def load_decision_policy(path: str | Path | None = None) -> dict[str, Any]:
    policy_path = Path(path).resolve() if path else default_policy_path()
    payload = _read_payload(policy_path)
    return normalize_policy(payload, source_path=str(policy_path))


def classify_action(action: str, policy: dict[str, Any]) -> dict[str, Any]:
    token = str(action or "").strip()
    norm = token.lower()
    if norm in {item.lower() for item in policy.get("auto_allow_actions") or []}:
        category = "auto-approve"
    elif norm in {item.lower() for item in policy.get("auto_suggest_actions") or []}:
        category = "auto-suggest-but-do-not-apply"
    else:
        category = "must-block-for-human"
    return {
        "action": token,
        "classification": category,
        "continue_allowed": category != "must-block-for-human",
        "requires_human": category == "must-block-for-human",
        "requires_attention": category != "auto-approve",
    }


def classify_condition(condition: str, policy: dict[str, Any]) -> dict[str, Any]:
    token = str(condition or "").strip()
    norm = token.lower()
    if norm in {item.lower() for item in policy.get("stop_conditions") or []}:
        decision = "stop_and_queue_attention"
    elif norm in {item.lower() for item in policy.get("continue_on_warning_conditions") or []}:
        decision = "continue_with_warning"
    else:
        decision = "continue"
    return {
        "condition": token,
        "decision": decision,
        "continue_allowed": decision != "stop_and_queue_attention",
        "requires_attention": decision != "continue",
    }


def evaluate_autonomy(
    *,
    policy: dict[str, Any],
    actions: list[str] | None = None,
    conditions: list[str] | None = None,
    summary: str = "",
) -> dict[str, Any]:
    action_rows = [classify_action(item, policy) for item in (actions or []) if str(item).strip()]
    condition_rows = [classify_condition(item, policy) for item in (conditions or []) if str(item).strip()]
    blocking_reasons = [
        f"action:{row['action']}"
        for row in action_rows
        if row.get("classification") == "must-block-for-human"
    ] + [
        f"condition:{row['condition']}"
        for row in condition_rows
        if row.get("decision") == "stop_and_queue_attention"
    ]
    warning_reasons = [
        f"action:{row['action']}"
        for row in action_rows
        if row.get("classification") == "auto-suggest-but-do-not-apply"
    ] + [
        f"condition:{row['condition']}"
        for row in condition_rows
        if row.get("decision") == "continue_with_warning"
    ]
    if blocking_reasons:
        decision = "stop_and_queue_attention"
    elif warning_reasons:
        decision = "continue_with_warning"
    else:
        decision = "continue"
    reasons = blocking_reasons or warning_reasons
    return {
        "decision": decision,
        "reason": "; ".join(reasons) if reasons else str(summary or "policy_clear"),
        "continue_allowed": decision != "stop_and_queue_attention",
        "human_gate_triggered": decision == "stop_and_queue_attention",
        "attention_needed": bool(reasons),
        "actions": action_rows,
        "conditions": condition_rows,
    }

