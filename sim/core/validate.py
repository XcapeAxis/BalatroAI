
import json
from pathlib import Path
from typing import Any

STATE_SCHEMA = Path(__file__).resolve().parent.parent / "spec" / "state_v1.json"
ACTION_SCHEMA = Path(__file__).resolve().parent.parent / "spec" / "action_v1.json"
TRACE_SCHEMA = Path(__file__).resolve().parent.parent / "spec" / "trace_v1.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _fallback_validate(data: dict[str, Any], required: list[str], label: str) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be object")
    miss = [k for k in required if k not in data]
    if miss:
        raise ValueError(f"{label} missing required keys: {miss}")


def _jsonschema_validate(data: dict[str, Any], schema: dict[str, Any]) -> bool:
    try:
        import jsonschema  # type: ignore
    except Exception:
        return False
    jsonschema.validate(data, schema)
    return True


def validate_state(data: dict[str, Any]) -> None:
    schema = _load(STATE_SCHEMA)
    if _jsonschema_validate(data, schema):
        return
    _fallback_validate(data, schema.get("required", []), "state_v1")


def validate_action(data: dict[str, Any]) -> None:
    schema = _load(ACTION_SCHEMA)
    if _jsonschema_validate(data, schema):
        return
    _fallback_validate(data, schema.get("required", []), "action_v1")


def validate_trace_line(data: dict[str, Any]) -> None:
    schema = _load(TRACE_SCHEMA)
    if _jsonschema_validate(data, schema):
        return
    _fallback_validate(data, schema.get("required", []), "trace_v1")


