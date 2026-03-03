from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import ReplayEpisode, build_episode_from_rows, read_jsonl


def _discover_real_trace_files(root: Path) -> list[Path]:
    if root.is_file() and root.name.lower() == "oracle_trace_real.jsonl":
        return [root]
    if not root.exists():
        return []
    return sorted(root.rglob("oracle_trace_real.jsonl"))


def _resolve_run_id(trace_file: Path) -> str:
    parent = trace_file.parent
    if parent.name.lower() in {"fixture", "fixtures", "latest"}:
        return parent.parent.name
    return parent.name


def _load_manifest(trace_file: Path) -> dict[str, Any]:
    manifest_path = trace_file.parent / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        return {}
    return {}


def ingest_real_replays(
    roots: list[str | Path],
    *,
    max_episodes: int = 0,
    require_non_empty: bool = True,
) -> list[ReplayEpisode]:
    episodes: list[ReplayEpisode] = []
    for root_raw in roots:
        root = Path(root_raw)
        files = _discover_real_trace_files(root)
        for trace_file in files:
            rows = read_jsonl(trace_file)
            if require_non_empty and not rows:
                continue

            run_id = _resolve_run_id(trace_file)
            manifest = _load_manifest(trace_file)
            episode = build_episode_from_rows(
                replay_id=f"real:{run_id}:{trace_file.stem}",
                episode_id=f"real-{run_id}",
                source="real_trace",
                run_id=run_id,
                rows=rows,
                strict_real_contract=True,
                source_path=str(trace_file),
                meta={
                    "root": str(root),
                    "trace_file": str(trace_file),
                    "manifest_path": str(trace_file.parent / "manifest.json"),
                    "actions_count": int(manifest.get("actions_count") or 0),
                    "explicit_actions_count": int(manifest.get("explicit_actions_count") or 0),
                    "inferred_actions_count": int(manifest.get("inferred_actions_count") or 0),
                },
            )

            actions_count = int(manifest.get("actions_count") or 0)
            if actions_count <= 0 and episode.steps:
                patched_steps = []
                for step in episode.steps:
                    patched_steps.append(
                        type(step)(
                            step_id=step.step_id,
                            phase=step.phase,
                            action_type=step.action_type,
                            action_payload=step.action_payload,
                            state_hashes=step.state_hashes,
                            score_delta=step.score_delta,
                            reward=step.reward,
                            resources_delta=step.resources_delta,
                            valid_for_training=False,
                            invalid_reason="manifest_actions_count_zero",
                            meta=dict(step.meta),
                        )
                    )
                episode = type(episode)(
                    replay_id=episode.replay_id,
                    episode_id=episode.episode_id,
                    source=episode.source,
                    run_id=episode.run_id,
                    seed=episode.seed,
                    stake=episode.stake,
                    steps=patched_steps,
                    meta=dict(episode.meta),
                )

            episodes.append(episode)
            if max_episodes > 0 and len(episodes) >= int(max_episodes):
                return episodes
    return episodes


__all__ = ["ingest_real_replays"]
