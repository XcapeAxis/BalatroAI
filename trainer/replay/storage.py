from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .ingest_real import ingest_real_replays
from .ingest_sim import ingest_sim_replays
from .schema import ReplayEpisode, ReplayStep


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _flatten_steps(episodes: list[ReplayEpisode]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ep in episodes:
        for step in ep.steps:
            row = {
                "replay_id": ep.replay_id,
                "episode_id": ep.episode_id,
                "source": ep.source,
                "run_id": ep.run_id,
                "seed": ep.seed,
                "stake": ep.stake,
                **step.to_dict(),
            }
            rows.append(row)
    return rows


def _summarize(episodes: list[ReplayEpisode], steps: list[dict[str, Any]]) -> dict[str, Any]:
    source_hist = Counter(str(ep.source) for ep in episodes)
    action_hist = Counter(str(step.get("action_type") or "UNKNOWN") for step in steps)
    invalid_hist = Counter(str(step.get("invalid_reason") or "") for step in steps if not bool(step.get("valid_for_training")))
    valid_count = sum(1 for step in steps if bool(step.get("valid_for_training")))
    total = len(steps)
    return {
        "schema": "p36_replay_summary_v1",
        "generated_at": _now_iso(),
        "episode_count": len(episodes),
        "step_count": total,
        "valid_step_count": valid_count,
        "invalid_step_count": max(0, total - valid_count),
        "valid_fraction": float(valid_count / total) if total > 0 else 0.0,
        "source_distribution": dict(source_hist),
        "action_distribution": dict(action_hist),
        "invalid_reason_distribution": dict(invalid_hist),
    }


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# P36 Replay Dataset Summary",
        "",
        f"- generated_at: {summary.get('generated_at')}",
        f"- episodes: {summary.get('episode_count')}",
        f"- steps: {summary.get('step_count')}",
        f"- valid_step_count: {summary.get('valid_step_count')}",
        f"- invalid_step_count: {summary.get('invalid_step_count')}",
        f"- valid_fraction: {summary.get('valid_fraction')}",
        "",
        "## Source Distribution",
        "",
        "| source | episodes |",
        "|---|---:|",
    ]
    source_hist = summary.get("source_distribution") if isinstance(summary.get("source_distribution"), dict) else {}
    for key, value in sorted(source_hist.items(), key=lambda kv: str(kv[0])):
        lines.append(f"| {key} | {value} |")

    lines.extend([
        "",
        "## Action Distribution",
        "",
        "| action_type | count |",
        "|---|---:|",
    ])
    action_hist = summary.get("action_distribution") if isinstance(summary.get("action_distribution"), dict) else {}
    for key, value in sorted(action_hist.items(), key=lambda kv: str(kv[0])):
        lines.append(f"| {key} | {value} |")

    lines.extend([
        "",
        "## Invalid Reasons",
        "",
        "| reason | count |",
        "|---|---:|",
    ])
    invalid_hist = summary.get("invalid_reason_distribution") if isinstance(summary.get("invalid_reason_distribution"), dict) else {}
    for key, value in sorted(invalid_hist.items(), key=lambda kv: str(kv[0])):
        label = key if key else "<empty>"
        lines.append(f"| {label} | {value} |")
    return "\n".join(lines) + "\n"


def write_replay_dataset(
    episodes: list[ReplayEpisode],
    *,
    out_dir: str | Path,
) -> dict[str, Any]:
    output_dir = Path(out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = _flatten_steps(episodes)
    summary = _summarize(episodes, steps)

    steps_path = output_dir / "replay_steps.jsonl"
    index_path = output_dir / "replay_index.json"
    summary_path = output_dir / "replay_summary.json"
    summary_md_path = output_dir / "replay_summary.md"

    _write_jsonl(steps_path, steps)
    _write_json(
        index_path,
        {
            "schema": "p36_replay_index_v1",
            "generated_at": _now_iso(),
            "episodes": [ep.to_dict() for ep in episodes],
        },
    )
    _write_json(summary_path, summary)
    summary_md_path.write_text(_summary_markdown(summary), encoding="utf-8")

    result = {
        "status": "ok",
        "out_dir": str(output_dir),
        "steps_path": str(steps_path),
        "index_path": str(index_path),
        "summary_path": str(summary_path),
        "summary_md_path": str(summary_md_path),
        **summary,
    }
    return result


def build_replay_dataset(
    *,
    real_roots: list[str | Path],
    sim_roots: list[str | Path],
    out_dir: str | Path,
    max_episodes_per_source: int = 0,
) -> dict[str, Any]:
    episodes: list[ReplayEpisode] = []
    if real_roots:
        episodes.extend(
            ingest_real_replays(
                real_roots,
                max_episodes=max_episodes_per_source,
                require_non_empty=True,
            )
        )
    if sim_roots:
        episodes.extend(
            ingest_sim_replays(
                sim_roots,
                max_episodes=max_episodes_per_source,
                require_non_empty=True,
            )
        )
    return write_replay_dataset(episodes, out_dir=out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified replay dataset from real/sim trace artifacts.")
    parser.add_argument("--real-roots", nargs="*", default=[], help="Real trace roots (p13/p32 sessions/fixtures).")
    parser.add_argument("--sim-roots", nargs="*", default=[], help="Sim trace roots (fixtures_runtime/p10/p8/p7).")
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory. Default: docs/artifacts/p36/replay/<timestamp>",
    )
    parser.add_argument(
        "--max-episodes-per-source",
        type=int,
        default=0,
        help="Optional per-source cap. 0 means no cap.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = (
        (repo_root / "docs/artifacts/p36/replay" / _now_stamp()).resolve()
        if not str(args.out_dir).strip()
        else ((repo_root / str(args.out_dir)).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir))
    )

    real_roots = [str((repo_root / p).resolve()) if not Path(p).is_absolute() else str(Path(p)) for p in (args.real_roots or [])]
    sim_roots = [str((repo_root / p).resolve()) if not Path(p).is_absolute() else str(Path(p)) for p in (args.sim_roots or [])]

    result = build_replay_dataset(
        real_roots=real_roots,
        sim_roots=sim_roots,
        out_dir=out_dir,
        max_episodes_per_source=max(0, int(args.max_episodes_per_source)),
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
