from __future__ import annotations

from pathlib import Path

from .schema import ReplayEpisode, build_episode_from_rows, read_jsonl


def _discover_sim_trace_files(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() == ".jsonl" and root.name.lower().startswith("oracle_trace"):
        return [root]
    if not root.exists():
        return []
    return sorted(root.rglob("oracle_trace*.jsonl"))


def _resolve_run_id(trace_file: Path) -> str:
    parent = trace_file.parent
    if parent.name.lower() in {"fixture", "fixtures", "latest"}:
        return parent.parent.name
    return parent.name


def _is_sim_like_trace(path: Path, rows: list[dict]) -> bool:
    path_text = str(path).lower()
    if "fixtures_runtime" in path_text:
        return True
    if "synthetic_session" in path_text or "position_contract" in path_text:
        return True
    if "p10" in path_text or "p8" in path_text or "p7" in path_text:
        return True

    if not rows:
        return False
    row0 = rows[0]
    info = row0.get("info") if isinstance(row0.get("info"), dict) else {}
    engine_info = info.get("engine_info") if isinstance(info.get("engine_info"), dict) else {}
    backend = str(engine_info.get("backend") or "").strip().lower()
    if backend == "sim":
        return True
    source = str(info.get("source") or "").strip().lower()
    return source.startswith("p32_") or source.startswith("sim")


def ingest_sim_replays(
    roots: list[str | Path],
    *,
    max_episodes: int = 0,
    require_non_empty: bool = True,
) -> list[ReplayEpisode]:
    episodes: list[ReplayEpisode] = []
    for root_raw in roots:
        root = Path(root_raw)
        files = _discover_sim_trace_files(root)
        for trace_file in files:
            rows = read_jsonl(trace_file)
            if require_non_empty and not rows:
                continue
            if not _is_sim_like_trace(trace_file, rows):
                continue
            run_id = _resolve_run_id(trace_file)
            episode = build_episode_from_rows(
                replay_id=f"sim:{run_id}:{trace_file.stem}",
                episode_id=f"sim-{run_id}",
                source="sim_trace",
                run_id=run_id,
                rows=rows,
                strict_real_contract=False,
                source_path=str(trace_file),
                meta={
                    "root": str(root),
                    "trace_file": str(trace_file),
                },
            )
            episodes.append(episode)
            if max_episodes > 0 and len(episodes) >= int(max_episodes):
                return episodes
    return episodes


__all__ = ["ingest_sim_replays"]
