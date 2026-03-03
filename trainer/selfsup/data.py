from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from trainer.data.trajectory import (
    DecisionStep,
    Trajectory,
    load_trajectories_from_oracle_traces,
    load_trajectories_from_p13_drift_fixture,
)


def _state_hash_signal(state_hashes: dict[str, str]) -> float:
    if not state_hashes:
        return 0.0
    values: list[float] = []
    for token in state_hashes.values():
        text = str(token).strip().lower()
        if len(text) < 8:
            continue
        try:
            values.append((int(text[:8], 16) % 1000000) / 1000000.0)
        except Exception:
            continue
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _scaled(value: float, scale: float, lo: float = -2.0, hi: float = 2.0) -> float:
    if scale <= 0:
        return 0.0
    return max(lo, min(hi, float(value) / float(scale)))


def _step_state_vector(step: DecisionStep) -> list[float]:
    phase = str(step.phase or "UNKNOWN").upper()
    action_type = str(step.action_type or "UNKNOWN").upper()
    score_delta = float(step.score_delta if step.score_delta is not None else 0.0)
    reward = float(step.reward if step.reward is not None else 0.0)
    return [
        _state_hash_signal(step.state_hashes),
        min(1.0, len(step.state_hashes) / 24.0),
        min(1.0, len(step.action_args) / 8.0),
        _scaled(score_delta, 250.0),
        _scaled(reward, 250.0),
        1.0 if phase == "SELECTING_HAND" else 0.0,
        1.0 if phase == "SHOP" else 0.0,
        1.0 if action_type == "PLAY" else 0.0,
        1.0 if action_type == "DISCARD" else 0.0,
        1.0 if action_type == "SHOP_BUY" else 0.0,
        1.0 if action_type == "SHOP_REROLL" else 0.0,
        1.0 if action_type == "USE_CONSUMABLE" else 0.0,
        1.0 if bool(step.done) else 0.0,
    ]


def _future_delta(steps: list[DecisionStep], start_idx: int, lookahead_k: int) -> float:
    if start_idx + 1 >= len(steps):
        return 0.0
    end_idx = min(len(steps), start_idx + 1 + max(1, int(lookahead_k)))
    total = 0.0
    for j in range(start_idx + 1, end_idx):
        total += float(steps[j].score_delta if steps[j].score_delta is not None else 0.0)
    return total


def _future_terminal(steps: list[DecisionStep], start_idx: int, lookahead_k: int) -> int:
    if start_idx + 1 >= len(steps):
        return 0
    end_idx = min(len(steps), start_idx + 1 + max(1, int(lookahead_k)))
    for j in range(start_idx + 1, end_idx):
        if bool(steps[j].done):
            return 1
    return 0


@dataclass(frozen=True)
class SourceSpec:
    kind: str
    path: str
    label: str = ""


@dataclass(frozen=True)
class SelfSupSample:
    state: dict[str, Any]
    aux: dict[str, Any]
    future: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_source_tokens(tokens: list[str]) -> list[SourceSpec]:
    specs: list[SourceSpec] = []
    for raw in tokens:
        token = str(raw).strip()
        if not token:
            continue
        if ":" in token and token.split(":", 1)[0].lower() in {"auto", "oracle", "p13", "real"}:
            kind, path = token.split(":", 1)
            specs.append(SourceSpec(kind=str(kind).strip().lower(), path=str(path).strip()))
            continue
        specs.append(SourceSpec(kind="auto", path=token))
    return specs


def _load_source(
    spec: SourceSpec,
    *,
    repo_root: Path,
    max_trajectories_per_source: int,
    require_steps: bool,
) -> tuple[list[Trajectory], str]:
    src_path = (repo_root / spec.path).resolve() if not Path(spec.path).is_absolute() else Path(spec.path)
    kind = str(spec.kind or "auto").strip().lower()
    if kind == "oracle":
        return (
            load_trajectories_from_oracle_traces(
                src_path,
                max_trajectories=max_trajectories_per_source,
                require_steps=require_steps,
            ),
            "oracle",
        )
    if kind == "p13":
        return (
            load_trajectories_from_p13_drift_fixture(
                src_path,
                max_trajectories=max_trajectories_per_source,
                require_steps=require_steps,
            ),
            "p13",
        )
    if kind == "real":
        rows = load_trajectories_from_oracle_traces(
            src_path,
            max_trajectories=max_trajectories_per_source,
            require_steps=require_steps,
        )
        return rows, "real"

    # Auto detect with safe fallback.
    if "p13" in str(src_path).lower() or (src_path / "fixture").exists():
        rows = load_trajectories_from_p13_drift_fixture(
            src_path,
            max_trajectories=max_trajectories_per_source,
            require_steps=require_steps,
        )
        if rows:
            return rows, "p13"
    if list(src_path.rglob("oracle_trace_real.jsonl")):
        rows = load_trajectories_from_oracle_traces(
            src_path,
            max_trajectories=max_trajectories_per_source,
            require_steps=require_steps,
        )
        if rows:
            return rows, "real"
    rows = load_trajectories_from_oracle_traces(
        src_path,
        max_trajectories=max_trajectories_per_source,
        require_steps=require_steps,
    )
    return rows, "oracle"


def load_trajectories_from_sources(
    *,
    repo_root: Path,
    sources: list[SourceSpec],
    max_trajectories_per_source: int = 20,
    require_steps: bool = True,
) -> tuple[list[Trajectory], list[dict[str, Any]]]:
    all_trajectories: list[Trajectory] = []
    source_stats: list[dict[str, Any]] = []
    for spec in sources:
        src_path = (repo_root / spec.path).resolve() if not Path(spec.path).is_absolute() else Path(spec.path)
        loaded, resolved_kind = _load_source(
            spec,
            repo_root=repo_root,
            max_trajectories_per_source=max_trajectories_per_source,
            require_steps=require_steps,
        )
        if resolved_kind == "real":
            loaded = [replace(row, source="real_trace") for row in loaded]
        all_trajectories.extend(loaded)
        source_stats.append(
            {
                "requested_kind": spec.kind,
                "resolved_kind": resolved_kind,
                "path": str(src_path),
                "count": len(loaded),
            }
        )
    return all_trajectories, source_stats


def build_samples_from_trajectories(
    trajectories: list[Trajectory],
    *,
    lookahead_k: int = 3,
    max_samples: int = 0,
) -> list[SelfSupSample]:
    rows: list[SelfSupSample] = []
    for trajectory in trajectories:
        steps = list(trajectory.steps)
        for idx, step in enumerate(steps):
            next_step = steps[idx + 1] if idx + 1 < len(steps) else step
            row = SelfSupSample(
                state={
                    "phase": str(step.phase or "UNKNOWN").upper(),
                    "action_type": str(step.action_type or "UNKNOWN").upper(),
                    "vector": _step_state_vector(step),
                    "state_hashes": dict(step.state_hashes),
                },
                aux={
                    "stake": str(trajectory.stake or ""),
                    "deck": str(trajectory.deck or ""),
                    "seed": str(trajectory.seed or ""),
                    "source": str(trajectory.source or ""),
                    "score_delta_t": float(step.score_delta if step.score_delta is not None else 0.0),
                    "reward_t": float(step.reward if step.reward is not None else 0.0),
                },
                future={
                    "delta_chips_k": _future_delta(steps, idx, lookahead_k),
                    "terminal_within_k": _future_terminal(steps, idx, lookahead_k),
                    "next_state_vector": _step_state_vector(next_step),
                    "next_action_type": str(next_step.action_type or "UNKNOWN").upper(),
                    "next_phase": str(next_step.phase or "UNKNOWN").upper(),
                },
                meta={
                    "run_id": str(trajectory.run_id or ""),
                    "trajectory_id": str(trajectory.trajectory_id),
                    "step_idx": int(idx),
                    "step_id": int(step.step_id),
                    "lookahead_k": int(lookahead_k),
                    "source": str(trajectory.source or ""),
                },
            )
            rows.append(row)
            if max_samples > 0 and len(rows) >= int(max_samples):
                return rows
    return rows


def summarize_samples(
    samples: list[SelfSupSample],
    *,
    source_stats: list[dict[str, Any]],
    lookahead_k: int,
) -> dict[str, Any]:
    action_hist: Counter[str] = Counter()
    phase_hist: Counter[str] = Counter()
    stake_hist: Counter[str] = Counter()
    source_hist: Counter[str] = Counter()
    terminal_pos = 0
    delta_values: list[float] = []
    for row in samples:
        action_hist[str(row.state.get("action_type") or "UNKNOWN")] += 1
        phase_hist[str(row.state.get("phase") or "UNKNOWN")] += 1
        stake_hist[str(row.aux.get("stake") or "UNKNOWN")] += 1
        source_hist[str(row.aux.get("source") or "UNKNOWN")] += 1
        terminal_pos += int(row.future.get("terminal_within_k") or 0)
        delta_values.append(float(row.future.get("delta_chips_k") or 0.0))
    avg_delta = float(sum(delta_values) / max(1, len(delta_values)))
    return {
        "schema": "p36_selfsup_dataset_summary_v1",
        "sample_count": len(samples),
        "lookahead_k": int(lookahead_k),
        "source_stats": source_stats,
        "source_distribution": dict(source_hist),
        "action_distribution": dict(action_hist),
        "phase_distribution": dict(phase_hist),
        "stake_distribution": dict(stake_hist),
        "terminal_within_k_rate": float(terminal_pos / max(1, len(samples))),
        "future_delta_chips_avg": avg_delta,
        "future_delta_chips_min": float(min(delta_values) if delta_values else 0.0),
        "future_delta_chips_max": float(max(delta_values) if delta_values else 0.0),
    }


__all__ = [
    "SelfSupSample",
    "SourceSpec",
    "build_samples_from_trajectories",
    "load_trajectories_from_sources",
    "parse_source_tokens",
    "summarize_samples",
]
