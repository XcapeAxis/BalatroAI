from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trainer.selfsup.data import (
    build_samples_from_trajectories,
    load_trajectories_from_sources,
    parse_source_tokens,
    summarize_samples,
)


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


def _write_summary_md(path: Path, summary: dict[str, Any], dataset_jsonl: Path) -> None:
    lines = [
        "# P36 Self-Supervised Dataset Summary",
        "",
        f"- generated_at: {summary.get('generated_at')}",
        f"- dataset_jsonl: {dataset_jsonl}",
        f"- sample_count: {summary.get('sample_count')}",
        f"- lookahead_k: {summary.get('lookahead_k')}",
        f"- terminal_within_k_rate: {summary.get('terminal_within_k_rate')}",
        f"- future_delta_chips_avg: {summary.get('future_delta_chips_avg')}",
        "",
        "## Source Stats",
        "",
        "| requested_kind | resolved_kind | path | count |",
        "|---|---|---|---:|",
    ]
    for row in summary.get("source_stats", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {rk} | {ak} | {path} | {count} |".format(
                rk=row.get("requested_kind", ""),
                ak=row.get("resolved_kind", ""),
                path=row.get("path", ""),
                count=row.get("count", 0),
            )
        )

    def _add_hist_section(title: str, key: str) -> None:
        lines.extend(["", f"## {title}", "", "| key | count |", "|---|---:|"])
        hist = summary.get(key, {})
        if isinstance(hist, dict):
            for token, value in sorted(hist.items(), key=lambda kv: str(kv[0])):
                lines.append(f"| {token} | {value} |")

    _add_hist_section("Action Distribution", "action_distribution")
    _add_hist_section("Phase Distribution", "phase_distribution")
    _add_hist_section("Stake Distribution", "stake_distribution")
    _add_hist_section("Source Distribution", "source_distribution")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_index(index_path: Path, *, summary: dict[str, Any], out_dir: Path, dataset_jsonl: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if not index_path.exists():
        index_path.write_text(
            "# P36 Self-Supervised Datasets Index\n\n| generated_at | out_dir | sample_count | dataset_jsonl |\n|---|---|---:|---|\n",
            encoding="utf-8",
        )
    line = "| {generated_at} | {out_dir} | {sample_count} | {dataset} |\n".format(
        generated_at=summary.get("generated_at", _now_iso()),
        out_dir=str(out_dir).replace("\\", "/"),
        sample_count=int(summary.get("sample_count") or 0),
        dataset=str(dataset_jsonl).replace("\\", "/"),
    )
    with index_path.open("a", encoding="utf-8", newline="\n") as fp:
        fp.write(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified P36 self-supervised dataset from existing trace artifacts.")
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="One or more source spec tokens. Format: auto:<path> | oracle:<path> | p13:<path> or plain <path>.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory. If omitted, uses docs/artifacts/p36/selfsup_datasets/<timestamp>.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Optional max sample cap across all sources.")
    parser.add_argument("--lookahead-k", type=int, default=3, help="Future horizon k for delta/terminal targets.")
    parser.add_argument(
        "--max-trajectories-per-source",
        type=int,
        default=30,
        help="Cap trajectories loaded from each source root.",
    )
    parser.add_argument(
        "--index-path",
        default="docs/artifacts/p36/SELF_SUP_DATASETS_INDEX.md",
        help="Index markdown updated with this dataset run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = (
        (repo_root / "docs/artifacts/p36/selfsup_datasets" / _now_stamp()).resolve()
        if not str(args.out_dir).strip()
        else ((repo_root / str(args.out_dir)).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir))
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = parse_source_tokens([str(x) for x in (args.sources or [])])
    trajectories, source_stats = load_trajectories_from_sources(
        repo_root=repo_root,
        sources=sources,
        max_trajectories_per_source=max(1, int(args.max_trajectories_per_source)),
        require_steps=True,
    )
    rows = build_samples_from_trajectories(
        trajectories,
        lookahead_k=max(1, int(args.lookahead_k)),
        max_samples=max(0, int(args.max_samples)),
    )
    if not rows:
        raise SystemExit("no samples generated from provided sources")

    dataset_jsonl = out_dir / "dataset.jsonl"
    payload_rows = [r.to_dict() for r in rows]
    _write_jsonl(dataset_jsonl, payload_rows)

    summary = summarize_samples(
        rows,
        source_stats=source_stats,
        lookahead_k=max(1, int(args.lookahead_k)),
    )
    summary = {
        **summary,
        "generated_at": _now_iso(),
        "repo_root": str(repo_root),
        "out_dir": str(out_dir),
        "dataset_jsonl": str(dataset_jsonl),
        "sources": [str(s) for s in (args.sources or [])],
    }
    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"
    _write_json(summary_json, summary)
    _write_summary_md(summary_md, summary, dataset_jsonl)

    index_path = (repo_root / str(args.index_path)).resolve() if not Path(args.index_path).is_absolute() else Path(args.index_path)
    _append_index(index_path, summary=summary, out_dir=out_dir, dataset_jsonl=dataset_jsonl)

    print(
        json.dumps(
            {
                "status": "ok",
                "out_dir": str(out_dir),
                "summary_json": str(summary_json),
                "sample_count": int(summary.get("sample_count") or 0),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

