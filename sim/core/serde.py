
import dataclasses
import json
from typing import Any


def to_builtin(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return to_builtin(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    return value


def canonical_dumps(value: Any) -> str:
    built = to_builtin(value)
    return json.dumps(built, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

