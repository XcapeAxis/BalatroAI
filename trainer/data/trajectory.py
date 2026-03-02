from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip().lstrip("\ufeff")
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid json in {path} line {line_no}: {exc}") from exc
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _list_trace_files(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() == ".jsonl":
        return [root]
    if not root.exists():
        return []
    files = sorted(root.rglob("oracle_trace*.jsonl"))
    # Prefer real-trace names first when both exist for the same run.
    files = sorted(files, key=lambda p: (0 if "oracle_trace_real" in p.name else 1, str(p)))
    return files


@dataclass(frozen=True)
class DecisionStep:
    step_id: int
    phase: str
    action_type: str
    action_args: dict[str, Any]
    state_hashes: dict[str, str]
    score_delta: float | None
    reward: float | None
    hand_type: str | None
    done: bool
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: str
    source: str
    seed: str | None
    stake: str | None
    deck: str | None
    run_id: str | None
    steps: list[DecisionStep]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["steps"] = [step.to_dict() for step in self.steps]
        return payload


def _extract_state_hashes(row: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in row.items():
        if key.startswith("state_hash_") and isinstance(value, str) and value:
            out[key] = value
    return out


def _extract_seed(row: dict[str, Any]) -> str | None:
    snap = row.get("canonical_state_snapshot")
    if isinstance(snap, dict):
        rng = snap.get("rng")
        if isinstance(rng, dict):
            seed = rng.get("seed")
            if isinstance(seed, str) and seed:
                return seed
    info = row.get("info")
    if isinstance(info, dict):
        action = info.get("input_action")
        if isinstance(action, dict):
            seed = action.get("seed")
            if isinstance(seed, str) and seed:
                return seed
    return None


def _extract_stake(row: dict[str, Any]) -> str | None:
    snap = row.get("canonical_state_snapshot")
    if isinstance(snap, dict):
        rules = snap.get("rules")
        if isinstance(rules, dict):
            stake = rules.get("stake")
            if isinstance(stake, str) and stake:
                return stake
    return None


def _extract_deck(row: dict[str, Any]) -> str | None:
    snap = row.get("canonical_state_snapshot")
    if isinstance(snap, dict):
        deck = snap.get("deck")
        if isinstance(deck, str) and deck:
            return deck
        tags = snap.get("tags")
        if isinstance(tags, list) and tags:
            return "unknown_tags"
    return None


def _row_to_step(row: dict[str, Any], default_step_id: int, source_file: str) -> DecisionStep:
    action = row.get("action") if isinstance(row.get("action"), dict) else {}
    action_type = str(action.get("action_type") or "UNKNOWN").strip().upper() or "UNKNOWN"
    action_args = {
        k: v
        for k, v in action.items()
        if k not in {"schema_version", "phase", "action_type"}
    }
    score_observed = row.get("score_observed") if isinstance(row.get("score_observed"), dict) else {}
    score_delta = _safe_float(score_observed.get("delta"), None)
    reward = _safe_float(row.get("reward"), None)
    computed_expected = row.get("computed_expected") if isinstance(row.get("computed_expected"), dict) else {}
    hand_type_raw = computed_expected.get("hand_type")
    hand_type = str(hand_type_raw).strip().upper() if isinstance(hand_type_raw, str) and hand_type_raw else None
    done = bool(row.get("done") or False)
    step_id = _safe_int(row.get("step_id"), default_step_id)
    if step_id is None:
        step_id = default_step_id

    return DecisionStep(
        step_id=step_id,
        phase=str(row.get("phase") or "UNKNOWN").strip().upper(),
        action_type=action_type,
        action_args=action_args,
        state_hashes=_extract_state_hashes(row),
        score_delta=score_delta,
        reward=reward,
        hand_type=hand_type,
        done=done,
        meta={
            "trace_file": source_file,
            "score_source_field": score_observed.get("source_field"),
            "computed_expected_available": bool(computed_expected.get("available", False)),
            "computed_expected_partial": bool(computed_expected.get("partial", False)),
        },
    )


def _build_trajectory_from_trace_rows(
    *,
    trajectory_id: str,
    source: str,
    rows: list[dict[str, Any]],
    source_file: str,
    run_id: str | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> Trajectory:
    steps = [_row_to_step(row, idx, source_file=source_file) for idx, row in enumerate(rows)]
    seed = _extract_seed(rows[0]) if rows else None
    stake = _extract_stake(rows[0]) if rows else None
    deck = _extract_deck(rows[0]) if rows else None
    return Trajectory(
        trajectory_id=trajectory_id,
        source=source,
        seed=seed,
        stake=stake,
        deck=deck,
        run_id=run_id,
        steps=steps,
        meta=extra_meta or {},
    )


def load_trajectories_from_oracle_traces(
    traces_root: str | Path,
    *,
    max_trajectories: int | None = None,
    require_steps: bool = True,
) -> list[Trajectory]:
    root = Path(traces_root)
    trajectories: list[Trajectory] = []
    trace_files = _list_trace_files(root)
    for trace_file in trace_files:
        rows = _read_jsonl(trace_file)
        trajectory = _build_trajectory_from_trace_rows(
            trajectory_id=trace_file.stem,
            source="oracle_trace",
            rows=rows,
            source_file=str(trace_file),
            run_id=trace_file.parent.name,
            extra_meta={"trace_root": str(root)},
        )
        if require_steps and not trajectory.steps:
            continue
        trajectories.append(trajectory)
        if max_trajectories is not None and len(trajectories) >= int(max_trajectories):
            break
    return trajectories


def load_trajectories_from_p13_drift_fixture(
    p13_root: str | Path,
    *,
    max_trajectories: int | None = None,
    require_steps: bool = False,
) -> list[Trajectory]:
    root = Path(p13_root)
    if not root.exists():
        return []

    fixture_dirs: list[Path] = []
    if (root / "manifest.json").exists():
        fixture_dirs = [root]
    elif (root / "fixture").is_dir():
        fixture_dirs = [root / "fixture"]
    else:
        fixture_dirs = sorted(root.glob("*/fixture"))

    trajectories: list[Trajectory] = []
    for fixture_dir in fixture_dirs:
        trace_path = fixture_dir / "oracle_trace_real.jsonl"
        if not trace_path.exists():
            continue
        manifest_path = fixture_dir / "manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    manifest = loaded
            except Exception:
                manifest = {}
        rows = _read_jsonl(trace_path)
        run_id = fixture_dir.parent.name
        trajectory = _build_trajectory_from_trace_rows(
            trajectory_id=f"p13_{run_id}",
            source="p13_drift_fixture",
            rows=rows,
            source_file=str(trace_path),
            run_id=run_id,
            extra_meta={
                "manifest_path": str(manifest_path) if manifest_path.exists() else "",
                "session_input": str(manifest.get("input") or ""),
                "actions_count": _safe_int(manifest.get("actions_count"), 0),
                "phase_distribution": manifest.get("phase_distribution") if isinstance(manifest.get("phase_distribution"), dict) else {},
            },
        )
        if require_steps and not trajectory.steps:
            continue
        trajectories.append(trajectory)
        if max_trajectories is not None and len(trajectories) >= int(max_trajectories):
            break
    return trajectories


__all__ = [
    "DecisionStep",
    "Trajectory",
    "load_trajectories_from_oracle_traces",
    "load_trajectories_from_p13_drift_fixture",
]
