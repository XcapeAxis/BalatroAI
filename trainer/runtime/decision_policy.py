from __future__ import annotations

from pathlib import Path
from typing import Any

from trainer.autonomy.decision_policy import (
    classify_action as _classify_action,
    classify_condition as _classify_condition,
    evaluate_autonomy,
    load_decision_policy,
)


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


def load_policy(path: str | Path | None = None) -> dict[str, Any]:
    payload = load_decision_policy(Path(path).resolve() if path else default_policy_path())
    payload["policy_path"] = str(payload.get("source_path") or payload.get("policy_path") or "")
    return payload


def classify_action(action: str, *, policy: dict[str, Any]) -> dict[str, Any]:
    payload = _classify_action(action, policy)
    return {
        "action": str(payload.get("action") or action or ""),
        "category": str(payload.get("classification") or ""),
        "continue_allowed": bool(payload.get("continue_allowed", False)),
        "human_required": bool(payload.get("requires_human", False)),
        "policy_path": str(policy.get("policy_path") or policy.get("source_path") or ""),
    }


def condition_policy(condition: str, *, policy: dict[str, Any]) -> dict[str, Any]:
    payload = _classify_condition(condition, policy)
    return {
        "condition": str(payload.get("condition") or condition or ""),
        "decision": str(payload.get("decision") or ""),
        "continue_allowed": bool(payload.get("continue_allowed", False)),
        "requires_attention": bool(payload.get("requires_attention", False)),
        "policy_path": str(policy.get("policy_path") or policy.get("source_path") or ""),
    }


__all__ = [
    "classify_action",
    "condition_policy",
    "default_policy_path",
    "evaluate_autonomy",
    "load_policy",
    "resolve_repo_root",
]
