from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReplayStep:
    step_id: int
    phase: str
    action_type: str
    action_payload: dict[str, Any]
    state_hashes: dict[str, str]
    score_delta: float
    reward: float
    resources_delta: dict[str, float]
    valid_for_training: bool
    invalid_reason: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReplayEpisode:
    replay_id: str
    episode_id: str
    source: str
    run_id: str
    seed: str
    stake: str
    steps: list[ReplayStep]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["steps"] = [s.to_dict() for s in self.steps]
        return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            text = line.strip().lstrip("\ufeff")
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid json line {line_no} in {path}: {exc}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def extract_state_hashes(row: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in row.items():
        if key.startswith("state_hash_") and isinstance(value, str) and value:
            out[key] = value
    return out


def normalize_action_payload(action: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    if not isinstance(action, dict):
        return "UNKNOWN", {}
    action_type = str(action.get("action_type") or "UNKNOWN").strip().upper() or "UNKNOWN"
    payload = {
        str(k): v
        for k, v in action.items()
        if str(k) not in {"schema_version", "phase", "action_type"}
    }
    return action_type, payload


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _snapshot_dict(row: dict[str, Any]) -> dict[str, Any]:
    snap = row.get("canonical_state_snapshot")
    if isinstance(snap, dict):
        return snap
    return {}


def snapshot_scalar_features(snapshot: dict[str, Any]) -> dict[str, float]:
    round_obj = snapshot.get("round") if isinstance(snapshot.get("round"), dict) else {}
    score_obj = snapshot.get("score") if isinstance(snapshot.get("score"), dict) else {}
    economy_obj = snapshot.get("economy") if isinstance(snapshot.get("economy"), dict) else {}
    return {
        "chips": _safe_float(score_obj.get("chips"), 0.0),
        "mult": _safe_float(score_obj.get("mult"), 0.0),
        "money": _safe_float(economy_obj.get("money"), 0.0),
        "hands_left": _safe_float(round_obj.get("hands_left"), 0.0),
        "discards_left": _safe_float(round_obj.get("discards_left"), 0.0),
    }


def derive_resource_delta(prev_snapshot: dict[str, Any], cur_snapshot: dict[str, Any]) -> dict[str, float]:
    prev = snapshot_scalar_features(prev_snapshot)
    cur = snapshot_scalar_features(cur_snapshot)
    return {f"{k}_delta": float(cur.get(k, 0.0) - prev.get(k, 0.0)) for k in cur.keys()}


def _read_seed(snapshot: dict[str, Any]) -> str:
    rng = snapshot.get("rng") if isinstance(snapshot.get("rng"), dict) else {}
    seed = rng.get("seed")
    if isinstance(seed, str) and seed.strip():
        return seed.strip()
    return ""


def _read_stake(snapshot: dict[str, Any]) -> str:
    rules = snapshot.get("rules") if isinstance(snapshot.get("rules"), dict) else {}
    stake = rules.get("stake")
    if isinstance(stake, str) and stake.strip():
        return stake.strip()
    return ""


def _detect_contract_hash(state_hashes: dict[str, str]) -> bool:
    if not state_hashes:
        return False
    for key in state_hashes.keys():
        if key in {
            "state_hash_hand_core",
            "state_hash_full",
            "state_hash_p14_real_action_observed_core",
            "state_hash_p32_real_action_position_observed_core",
        }:
            return True
        if key.startswith("state_hash_p"):
            return True
    return False


def evaluate_step_validity(
    *,
    action_type: str,
    state_hashes: dict[str, str],
    row: dict[str, Any],
    strict_real_contract: bool,
) -> tuple[bool, str]:
    if action_type in {"", "UNKNOWN", "WAIT", "NOOP"}:
        return False, "unsupported_or_empty_action"
    if not _detect_contract_hash(state_hashes):
        return False, "missing_contract_hash_scope"

    info = row.get("info") if isinstance(row.get("info"), dict) else {}
    if bool(info.get("mismatch") is True):
        return False, "trace_mismatch_flag"

    if strict_real_contract:
        action = row.get("action") if isinstance(row.get("action"), dict) else {}
        if not isinstance(action.get("rng_replay"), dict):
            return False, "missing_rng_replay_payload"
        if not (
            "state_hash_p14_real_action_observed_core" in state_hashes
            or "state_hash_p32_real_action_position_observed_core" in state_hashes
        ):
            return False, "missing_real_action_scope_hash"

    return True, ""


def row_to_replay_step(
    *,
    row: dict[str, Any],
    prev_snapshot: dict[str, Any],
    strict_real_contract: bool,
    default_step_id: int,
    source_path: str,
) -> ReplayStep:
    action = row.get("action") if isinstance(row.get("action"), dict) else {}
    action_type, action_payload = normalize_action_payload(action)
    step_id = int(row.get("step_id") if isinstance(row.get("step_id"), int) else default_step_id)
    phase = str(row.get("phase") or "UNKNOWN").strip().upper() or "UNKNOWN"
    state_hashes = extract_state_hashes(row)
    score_observed = row.get("score_observed") if isinstance(row.get("score_observed"), dict) else {}
    score_delta = _safe_float(score_observed.get("delta"), 0.0)
    reward = _safe_float(row.get("reward"), 0.0)
    current_snapshot = _snapshot_dict(row)
    resources_delta = derive_resource_delta(prev_snapshot, current_snapshot)
    valid, invalid_reason = evaluate_step_validity(
        action_type=action_type,
        state_hashes=state_hashes,
        row=row,
        strict_real_contract=strict_real_contract,
    )
    return ReplayStep(
        step_id=step_id,
        phase=phase,
        action_type=action_type,
        action_payload=action_payload,
        state_hashes=state_hashes,
        score_delta=score_delta,
        reward=reward,
        resources_delta=resources_delta,
        valid_for_training=bool(valid),
        invalid_reason=str(invalid_reason or ""),
        meta={
            "trace_path": source_path,
            "done": bool(row.get("done", False)),
            "score_source_field": score_observed.get("source_field"),
        },
    )


def build_episode_from_rows(
    *,
    replay_id: str,
    episode_id: str,
    source: str,
    run_id: str,
    rows: list[dict[str, Any]],
    strict_real_contract: bool,
    source_path: str,
    meta: dict[str, Any] | None = None,
) -> ReplayEpisode:
    prev_snapshot: dict[str, Any] = {}
    steps: list[ReplayStep] = []
    seed = ""
    stake = ""

    for idx, row in enumerate(rows):
        snapshot = _snapshot_dict(row)
        if not seed:
            seed = _read_seed(snapshot)
        if not stake:
            stake = _read_stake(snapshot)
        step = row_to_replay_step(
            row=row,
            prev_snapshot=prev_snapshot,
            strict_real_contract=strict_real_contract,
            default_step_id=idx,
            source_path=source_path,
        )
        steps.append(step)
        prev_snapshot = snapshot

    return ReplayEpisode(
        replay_id=replay_id,
        episode_id=episode_id,
        source=source,
        run_id=run_id,
        seed=seed,
        stake=stake,
        steps=steps,
        meta=meta or {},
    )


__all__ = [
    "ReplayEpisode",
    "ReplayStep",
    "build_episode_from_rows",
    "derive_resource_delta",
    "evaluate_step_validity",
    "extract_state_hashes",
    "normalize_action_payload",
    "read_jsonl",
]
