from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from trainer.common.slices import compute_slice_labels
from trainer.closed_loop.replay_manifest import (
    infer_generation_method,
    infer_source_run_id,
    make_sample_id,
    now_iso,
    write_json,
    write_markdown,
)


LINEAGE_VERSION = "p41_lineage_v1"
LINEAGE_REQUIRED_FIELDS = [
    "sample_id",
    "source_type",
    "source_run_id",
    "source_artifact_path",
    "source_seed",
    "episode_id",
    "step_id",
    "generation_method",
    "valid_for_training",
    "lineage_version",
]


def _normalize_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y"}:
        return True
    if token in {"0", "false", "no", "n"}:
        return False
    return default


def build_lineage_entry(
    *,
    entry: dict[str, Any],
    record: dict[str, Any] | None,
) -> dict[str, Any]:
    row = dict(entry)
    rec = record if isinstance(record, dict) else {}
    path = str(row.get("path") or rec.get("path") or "").strip()
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    preview = rec.get("preview_row") if isinstance(rec.get("preview_row"), dict) else {}

    source_type = str(row.get("source_type") or rec.get("source_type") or "unknown")
    source_run_id = str(rec.get("source_run_id") or infer_source_run_id(path))
    source_seed = str(rec.get("source_seed") or metadata.get("seed") or preview.get("seed") or "")
    episode_id = str(rec.get("episode_id") or preview.get("episode_id") or "")
    step_id = rec.get("step_id")
    if step_id in {None, ""}:
        step_id = preview.get("step_id")
    if step_id in {None, ""}:
        step_id = preview.get("step_index")
    if step_id in {None, ""}:
        step_id = preview.get("t")
    generation_method = str(
        rec.get("generation_method")
        or metadata.get("generation_method")
        or infer_generation_method(source_type, path, metadata)
    )
    valid_for_training = _normalize_bool(
        rec.get("valid_for_training", metadata.get("valid_for_training", preview.get("valid_for_training"))),
        default=True,
    )

    line_start = int(((row.get("sample_span") or {}) if isinstance(row.get("sample_span"), dict) else {}).get("line_start") or 1)
    line_end = int(((row.get("sample_span") or {}) if isinstance(row.get("sample_span"), dict) else {}).get("line_end") or row.get("sample_count") or 0)
    sample_id = make_sample_id(
        [
            source_type,
            path,
            source_seed,
            episode_id,
            str(step_id),
            str(line_start),
            str(line_end),
        ]
    )

    slice_labels = compute_slice_labels(
        {
            "state": preview if isinstance(preview, dict) else {},
            "action_type": preview.get("action_type") if isinstance(preview, dict) else "",
            "phase": preview.get("phase") if isinstance(preview, dict) else "",
        }
    )

    row.update(
        {
            "sample_id": sample_id,
            "source_type": source_type,
            "source_run_id": source_run_id,
            "source_artifact_path": path,
            "source_seed": source_seed,
            "episode_id": episode_id,
            "step_id": step_id if step_id not in {None} else "",
            "generation_method": generation_method,
            "valid_for_training": bool(valid_for_training),
            "lineage_version": LINEAGE_VERSION,
            "slice_labels": slice_labels,
        }
    )
    for key, val in slice_labels.items():
        row[key] = val
    return row


def summarize_lineage(
    *,
    run_id: str,
    entries: list[dict[str, Any]],
    source_stats: list[dict[str, Any]] | None = None,
    seeds: list[str] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    source_counts: Counter[str] = Counter()
    source_sample_counts: Counter[str] = Counter()
    source_run_counts: Counter[str] = Counter()
    seed_counts: Counter[str] = Counter()
    episode_counts: Counter[str] = Counter()
    valid_true = 0
    valid_false = 0
    missing_fields: Counter[str] = Counter()
    slice_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for entry in entries:
        st = str(entry.get("source_type") or "unknown")
        source_counts[st] += 1
        source_sample_counts[st] += int(entry.get("sample_count") or 0)
        sr = str(entry.get("source_run_id") or "")
        if sr:
            source_run_counts[sr] += 1
        seed = str(entry.get("source_seed") or "")
        if seed:
            seed_counts[seed] += 1
        episode_id = str(entry.get("episode_id") or "")
        if episode_id:
            episode_counts[episode_id] += 1

        if bool(entry.get("valid_for_training")):
            valid_true += 1
        else:
            valid_false += 1

        for req in LINEAGE_REQUIRED_FIELDS:
            val = entry.get(req)
            if val in {None, ""}:
                missing_fields[req] += 1

        for slice_key in (
            "slice_stage",
            "slice_resource_pressure",
            "slice_action_type",
            "slice_position_sensitive",
            "slice_stateful_joker_present",
        ):
            token = str(entry.get(slice_key, "unknown"))
            slice_counts[slice_key][token] += 1

    source_rows: list[dict[str, Any]] = []
    all_source_types = sorted(set(source_counts.keys()) | {str((s or {}).get("source_type") or "") for s in (source_stats or []) if isinstance(s, dict)})
    for st in all_source_types:
        if not st:
            continue
        source_rows.append(
            {
                "source_type": st,
                "entry_count": int(source_counts.get(st, 0)),
                "sample_count": int(source_sample_counts.get(st, 0)),
                "share_by_entries": (float(source_counts.get(st, 0)) / max(1, len(entries))),
                "share_by_samples": (float(source_sample_counts.get(st, 0)) / max(1, sum(source_sample_counts.values()))),
            }
        )

    summary = {
        "schema": "p41_replay_lineage_summary_v1",
        "generated_at": now_iso(),
        "run_id": run_id,
        "lineage_version": LINEAGE_VERSION,
        "entry_count": len(entries),
        "sample_count": int(sum(source_sample_counts.values())),
        "source_distribution": source_rows,
        "source_run_coverage": {
            "unique_source_run_count": len(source_run_counts),
            "source_run_ids": sorted(source_run_counts.keys()),
        },
        "seed_coverage": {
            "configured_seeds": [str(s) for s in (seeds or [])],
            "discovered_seed_count": len(seed_counts),
            "discovered_seeds": sorted(seed_counts.keys()),
        },
        "episode_coverage": {
            "unique_episode_count": len(episode_counts),
        },
        "validity": {
            "valid_for_training_true": int(valid_true),
            "valid_for_training_false": int(valid_false),
            "invalid_ratio": (float(valid_false) / max(1, len(entries))),
        },
        "slice_distribution": {
            key: [
                {"label": label, "count": int(count), "ratio": float(count) / max(1, sum(counter.values()))}
                for label, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
            ]
            for key, counter in slice_counts.items()
        },
        "lineage_missing_fields": dict(sorted(missing_fields.items())),
        "warnings": [str(w) for w in (warnings or [])],
    }
    return summary


def lineage_summary_markdown(summary: dict[str, Any]) -> list[str]:
    lines = [
        f"# P41 Replay Lineage Summary ({summary.get('run_id')})",
        "",
        f"- lineage_version: `{summary.get('lineage_version')}`",
        f"- entry_count: `{int(summary.get('entry_count') or 0)}`",
        f"- sample_count: `{int(summary.get('sample_count') or 0)}`",
        "",
        "## Source Distribution",
        "",
        "| source_type | entries | samples | entry_share | sample_share |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary.get("source_distribution") if isinstance(summary.get("source_distribution"), list) else []:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {source_type} | {entry_count} | {sample_count} | {entry_share:.3f} | {sample_share:.3f} |".format(
                source_type=row.get("source_type"),
                entry_count=int(row.get("entry_count") or 0),
                sample_count=int(row.get("sample_count") or 0),
                entry_share=float(row.get("share_by_entries") or 0.0),
                sample_share=float(row.get("share_by_samples") or 0.0),
            )
        )

    seed_cov = summary.get("seed_coverage") if isinstance(summary.get("seed_coverage"), dict) else {}
    ep_cov = summary.get("episode_coverage") if isinstance(summary.get("episode_coverage"), dict) else {}
    validity = summary.get("validity") if isinstance(summary.get("validity"), dict) else {}
    lines.extend(
        [
            "",
            "## Coverage",
            f"- discovered_seed_count: {int(seed_cov.get('discovered_seed_count') or 0)}",
            f"- unique_episode_count: {int(ep_cov.get('unique_episode_count') or 0)}",
            f"- invalid_ratio: {float(validity.get('invalid_ratio') or 0.0):.4f}",
        ]
    )
    missing = summary.get("lineage_missing_fields") if isinstance(summary.get("lineage_missing_fields"), dict) else {}
    if missing:
        lines.extend(["", "## Missing Required Fields"])
        for key, val in sorted(missing.items()):
            lines.append(f"- {key}: {int(val)}")
    warnings = summary.get("warnings") if isinstance(summary.get("warnings"), list) else []
    if warnings:
        lines.extend(["", "## Warnings"])
        for warn in warnings:
            lines.append(f"- {str(warn)}")
    return lines


def write_lineage_summary(
    *,
    out_dir: Path,
    run_id: str,
    entries: list[dict[str, Any]],
    source_stats: list[dict[str, Any]] | None = None,
    seeds: list[str] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    summary = summarize_lineage(
        run_id=run_id,
        entries=entries,
        source_stats=source_stats,
        seeds=seeds,
        warnings=warnings,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "replay_lineage_summary.json"
    md_path = out_dir / "replay_lineage_summary.md"
    write_json(json_path, summary)
    write_markdown(md_path, lineage_summary_markdown(summary))
    return {
        "summary_json": str(json_path),
        "summary_md": str(md_path),
        "entry_count": int(summary.get("entry_count") or 0),
    }

