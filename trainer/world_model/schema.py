from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


PHASE_TOKENS = [
    "BLIND_SELECT",
    "SELECTING_HAND",
    "ROUND_EVAL",
    "SHOP",
    "SMODS_BOOSTER_OPENED",
    "MENU",
    "GAME_OVER",
    "OTHER",
]

ACTION_BUCKETS = [
    "PLAY",
    "DISCARD",
    "SHOP",
    "CONSUMABLE",
    "SELECT",
    "NEXT_ROUND",
    "CASH_OUT",
    "POSITION",
    "WAIT",
    "OTHER",
]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def fit_vector(values: list[Any], target_dim: int) -> list[float]:
    vector = [safe_float(v, 0.0) for v in list(values or [])]
    if len(vector) < int(target_dim):
        vector.extend([0.0] * (int(target_dim) - len(vector)))
    return vector[: int(target_dim)]


def _hash_bytes(text: str) -> bytes:
    return hashlib.sha256(str(text).encode("utf-8")).digest()


def stable_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def stable_hash_int(text: str, modulo: int) -> int:
    mod = max(1, int(modulo))
    return int(hashlib.sha256(str(text).encode("utf-8")).hexdigest(), 16) % mod


def hash_signal(text: Any, *, default: float = 0.0) -> float:
    token = str(text or "").strip()
    if not token:
        return float(default)
    try:
        digest = _hash_bytes(token)
        return float(int.from_bytes(digest[:4], byteorder="big", signed=False) / float(2**32 - 1))
    except Exception:
        return float(default)


def normalize_phase(value: Any) -> str:
    token = str(value or "OTHER").strip().upper()
    return token if token in PHASE_TOKENS else "OTHER"


def phase_one_hot(phase: str) -> list[float]:
    normalized = normalize_phase(phase)
    return [1.0 if normalized == token else 0.0 for token in PHASE_TOKENS]


def normalize_action_bucket(action_type: Any) -> str:
    token = str(action_type or "OTHER").strip().upper()
    if token.startswith("SHOP_") or token in {"PACK_OPEN", "BUY", "REROLL"}:
        return "SHOP"
    if token in {"USE_CONSUMABLE", "CONSUMABLE_USE"}:
        return "CONSUMABLE"
    if token in {"REORDER_HAND", "SWAP_HAND_CARD", "SWAP_HAND_CARDS", "SWAP_JOKER", "MOVE_JOKER"}:
        return "POSITION"
    if token in ACTION_BUCKETS:
        return token
    return "OTHER"


def action_bucket_one_hot(action_type: Any) -> list[float]:
    bucket = normalize_action_bucket(action_type)
    return [1.0 if bucket == token else 0.0 for token in ACTION_BUCKETS]


def canonical_action_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    normalized: dict[str, Any] = {}
    for key in sorted(payload.keys()):
        if key in {"rng_replay", "schema_version"}:
            continue
        value = payload.get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            normalized[str(key)] = value
        elif isinstance(value, list):
            normalized[str(key)] = [
                item
                for item in value
                if isinstance(item, (str, int, float, bool)) or item is None
            ]
    return normalized


def action_token_from_parts(
    *,
    phase: str,
    action_type: str,
    action_payload: dict[str, Any] | None = None,
    numeric_action: int | None = None,
) -> str:
    if numeric_action is not None and int(numeric_action) >= 0:
        return f"id:{int(numeric_action)}"
    payload = canonical_action_payload(action_payload)
    if payload:
        return "{phase}|{action_type}|{payload}".format(
            phase=normalize_phase(phase),
            action_type=str(action_type or "OTHER").strip().upper(),
            payload=json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )
    return f"{normalize_phase(phase)}|{str(action_type or 'OTHER').strip().upper()}"


def make_sample_id(parts: list[Any]) -> str:
    packed = json.dumps([str(part) for part in parts], ensure_ascii=False, separators=(",", ":"))
    return stable_hash(packed)[:24]


def resource_delta_vector(delta: dict[str, Any] | None) -> list[float]:
    payload = delta if isinstance(delta, dict) else {}
    return [
        safe_float(payload.get("chips_delta"), 0.0),
        safe_float(payload.get("money_delta"), 0.0),
        safe_float(payload.get("mult_delta"), 0.0),
        safe_float(payload.get("hands_left_delta"), 0.0),
        safe_float(payload.get("discards_left_delta"), 0.0),
    ]


@dataclass(frozen=True)
class WorldModelSample:
    sample_id: str
    source_id: str
    source_type: str
    source_path: str
    source_run_id: str
    seed: str
    episode_id: str
    step_id: int
    split: str
    valid_for_training: bool
    phase_t: str
    action_token: str
    action_id: int
    action_numeric: int
    obs_t: list[float]
    obs_t1: list[float]
    reward_t: float
    score_delta_t: float
    resource_delta_t: list[float]
    done_t: bool
    source_category: str
    feature_mode: str = "raw"
    latent_t: list[float] = field(default_factory=list)
    latent_t1: list[float] = field(default_factory=list)
    slice_labels: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
