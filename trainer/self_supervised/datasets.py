from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trainer.data.trajectory import (
    DecisionStep,
    Trajectory,
    load_trajectories_from_oracle_traces,
    load_trajectories_from_p13_drift_fixture,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _score_bucket(score_delta: float | None, thresholds: tuple[float, float]) -> int:
    v = float(score_delta or 0.0)
    lo, hi = thresholds
    if v <= lo:
        return 0
    if v <= hi:
        return 1
    return 2


def _future_terminal(steps: list[DecisionStep], idx: int, horizon_steps: int) -> int:
    max_idx = min(len(steps), idx + max(1, int(horizon_steps)) + 1)
    for j in range(idx + 1, max_idx):
        if bool(steps[j].done):
            return 1
    return 0


def _step_features(step: DecisionStep) -> list[float]:
    phase = str(step.phase or "UNKNOWN").upper()
    action_type = str(step.action_type or "UNKNOWN").upper()
    reward = float(step.reward if step.reward is not None else 0.0)
    score_delta = float(step.score_delta if step.score_delta is not None else 0.0)
    return [
        _state_hash_signal(step.state_hashes),
        min(1.0, len(step.state_hashes) / 24.0),
        min(1.0, len(step.action_args) / 8.0),
        max(-2.0, min(2.0, reward / 300.0)),
        max(-2.0, min(2.0, score_delta / 300.0)),
        1.0 if phase == "SELECTING_HAND" else 0.0,
        1.0 if phase == "SHOP" else 0.0,
        1.0 if action_type == "PLAY" else 0.0,
        1.0 if action_type == "DISCARD" else 0.0,
        1.0 if bool(step.done) else 0.0,
    ]


@dataclass(frozen=True)
class P33DatasetRow:
    trajectory_id: str
    source: str
    seed: str
    step_id: int
    phase: str
    action_type: str
    features: list[float]
    target_next_score_bucket: int
    target_terminal_within_horizon: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "source": self.source,
            "seed": self.seed,
            "step_id": self.step_id,
            "phase": self.phase,
            "action_type": self.action_type,
            "features": self.features,
            "target_next_score_bucket": int(self.target_next_score_bucket),
            "target_terminal_within_horizon": int(self.target_terminal_within_horizon),
        }


def collect_trajectories(
    *,
    repo_root: Path,
    data_cfg: dict[str, Any],
) -> tuple[list[Trajectory], list[dict[str, Any]]]:
    sources = data_cfg.get("sources") if isinstance(data_cfg.get("sources"), list) else []
    max_per_source = int(data_cfg.get("max_trajectories_per_source") or 12)
    loaded: list[Trajectory] = []
    source_summaries: list[dict[str, Any]] = []
    for item in sources:
        if not isinstance(item, dict):
            continue
        source_type = str(item.get("type") or "").strip().lower()
        rel_path = str(item.get("path") or "").strip()
        if not rel_path:
            continue
        src_path = (repo_root / rel_path).resolve()
        if source_type in {"p13_drift_fixture", "p13_fixture"}:
            rows = load_trajectories_from_p13_drift_fixture(src_path, max_trajectories=max_per_source, require_steps=True)
        elif source_type == "oracle_traces":
            rows = load_trajectories_from_oracle_traces(src_path, max_trajectories=max_per_source, require_steps=True)
        else:
            rows = []
        loaded.extend(rows)
        source_summaries.append(
            {
                "type": source_type,
                "path": str(src_path),
                "trajectories_loaded": len(rows),
            }
        )
    return loaded, source_summaries


def build_dataset_rows(
    trajectories: list[Trajectory],
    *,
    bucket_thresholds: tuple[float, float],
    horizon_steps: int,
    max_samples: int = 0,
) -> list[P33DatasetRow]:
    rows: list[P33DatasetRow] = []
    for traj in trajectories:
        steps = list(traj.steps)
        for idx, step in enumerate(steps):
            next_delta = None
            if idx + 1 < len(steps):
                next_delta = steps[idx + 1].score_delta
            row = P33DatasetRow(
                trajectory_id=str(traj.trajectory_id),
                source=str(traj.source),
                seed=str(traj.seed or ""),
                step_id=int(step.step_id),
                phase=str(step.phase or "UNKNOWN"),
                action_type=str(step.action_type or "UNKNOWN"),
                features=_step_features(step),
                target_next_score_bucket=_score_bucket(next_delta, bucket_thresholds),
                target_terminal_within_horizon=_future_terminal(steps, idx, horizon_steps),
            )
            rows.append(row)
            if max_samples > 0 and len(rows) >= int(max_samples):
                return rows
    return rows


def write_dataset_jsonl(path: Path, rows: list[P33DatasetRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fp:
        for row in rows:
            fp.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def write_dataset_stats(
    *,
    out_path: Path,
    rows: list[P33DatasetRow],
    source_summaries: list[dict[str, Any]],
    dataset_path: Path,
    horizon_steps: int,
    bucket_thresholds: tuple[float, float],
) -> dict[str, Any]:
    label_hist = {"0": 0, "1": 0, "2": 0}
    term_hist = {"0": 0, "1": 0}
    seeds = sorted({row.seed for row in rows if row.seed})
    for row in rows:
        label_hist[str(int(row.target_next_score_bucket))] = label_hist.get(str(int(row.target_next_score_bucket)), 0) + 1
        term_hist[str(int(row.target_terminal_within_horizon))] = term_hist.get(str(int(row.target_terminal_within_horizon)), 0) + 1
    payload = {
        "schema": "p33_selfsup_dataset_stats_v1",
        "generated_at": _now_iso(),
        "dataset_path": str(dataset_path),
        "num_samples": len(rows),
        "feature_dim": (len(rows[0].features) if rows else 0),
        "label_distribution_next_score_bucket": label_hist,
        "label_distribution_terminal_within_horizon": term_hist,
        "seeds": seeds,
        "source_summaries": source_summaries,
        "task": {
            "name": "next_score_delta_bucket",
            "bucket_thresholds": [float(bucket_thresholds[0]), float(bucket_thresholds[1])],
            "horizon_steps": int(horizon_steps),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


__all__ = [
    "P33DatasetRow",
    "build_dataset_rows",
    "collect_trajectories",
    "write_dataset_jsonl",
    "write_dataset_stats",
]

