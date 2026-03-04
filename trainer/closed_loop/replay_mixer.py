from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from trainer.closed_loop.replay_manifest import (
    build_seeds_payload,
    now_iso,
    now_stamp,
    to_abs_path,
    write_json,
    write_markdown,
)
from trainer.closed_loop.replay_sources import dump_source_resolution_json, resolve_replay_sources


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".yml", ".yaml"}:
        if yaml is not None:
            obj = yaml.safe_load(text)
        else:
            try:
                obj = json.loads(text)
            except Exception:
                sidecar = path.with_suffix(".json")
                if sidecar.exists():
                    obj = json.loads(sidecar.read_text(encoding="utf-8-sig"))
                else:
                    raise RuntimeError(f"PyYAML unavailable and no sidecar JSON for: {path}")
    else:
        obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError(f"config must be mapping: {path}")
    return obj


def _normalize_seeds(cfg: dict[str, Any]) -> list[str]:
    seeds_cfg = cfg.get("seeds")
    if isinstance(seeds_cfg, list):
        return [str(s).strip() for s in seeds_cfg if str(s).strip()]
    if isinstance(seeds_cfg, dict):
        raw = seeds_cfg.get("list") if isinstance(seeds_cfg.get("list"), list) else []
        return [str(s).strip() for s in raw if str(s).strip()]
    return ["AAAAAAA", "BBBBBBB"]


def _split_cfg(cfg: dict[str, Any]) -> dict[str, float]:
    split_cfg = cfg.get("split") if isinstance(cfg.get("split"), dict) else {}
    train = float(split_cfg.get("train") or 0.9)
    val = float(split_cfg.get("val") or (1.0 - train))
    if train <= 0.0:
        train = 0.8
    if val <= 0.0:
        val = max(0.01, 1.0 - train)
    norm = train + val
    return {"train": train / norm, "val": val / norm}


def _default_sources() -> list[dict[str, Any]]:
    return [
        {"id": "p10_long_episode", "type": "p10_long_episode", "weight": 0.25, "max_samples": 1200},
        {"id": "p13_dagger_or_real", "type": "p13_dagger_or_real", "weight": 0.35, "max_samples": 1600},
        {"id": "selfsup_replay", "type": "selfsup_replay", "weight": 0.30, "max_samples": 1600},
        {"id": "arena_failures", "type": "arena_failures", "weight": 0.10, "max_samples": 800},
    ]


def _pick_sources(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    raw = cfg.get("sources")
    if isinstance(raw, list) and raw:
        return [dict(item) for item in raw if isinstance(item, dict)]
    return _default_sources()


def _allocate_record_samples(
    *,
    records: list[dict[str, Any]],
    selected_total: int,
) -> list[tuple[dict[str, Any], int]]:
    if selected_total <= 0 or not records:
        return []
    available = max(1, sum(max(0, int(r.get("sample_count") or 0)) for r in records))
    pending = selected_total
    allocations: list[tuple[dict[str, Any], int]] = []
    for idx, record in enumerate(records):
        count = max(0, int(record.get("sample_count") or 0))
        if count <= 0:
            continue
        if idx == len(records) - 1:
            take = min(count, pending)
        else:
            ratio = float(count) / float(available)
            take = min(count, max(0, int(round(selected_total * ratio))))
            take = min(take, pending)
        allocations.append((record, take))
        pending -= take
        if pending <= 0:
            break
    if pending > 0:
        for idx in range(len(allocations)):
            record, take = allocations[idx]
            room = max(0, int(record.get("sample_count") or 0) - take)
            if room <= 0:
                continue
            bump = min(room, pending)
            allocations[idx] = (record, take + bump)
            pending -= bump
            if pending <= 0:
                break
    return [(rec, take) for rec, take in allocations if take > 0]


def _build_selected_entries(
    *,
    source_rows: list[dict[str, Any]],
    split: dict[str, float],
    quick: bool,
    quick_cap_per_source: int,
    dry_run: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    selected_entries: list[dict[str, Any]] = []
    source_stats: list[dict[str, Any]] = []
    total_train = 0
    total_val = 0
    total_selected = 0

    for source in source_rows:
        source_id = str(source.get("source_id") or "")
        source_type = str(source.get("source_type") or "")
        status = str(source.get("status") or "stub")
        available = int(source.get("available_samples") or 0)
        records = source.get("records") if isinstance(source.get("records"), list) else []
        max_samples = int(source.get("max_samples") or 0)
        if quick:
            max_samples = min(max_samples if max_samples > 0 else quick_cap_per_source, quick_cap_per_source)
        selected = 0
        train_count = 0
        val_count = 0

        if status == "ok" and available > 0 and records:
            if max_samples <= 0:
                max_samples = available
            planned = min(available, max_samples)
            allocations = _allocate_record_samples(records=records, selected_total=planned)
            for record, take in allocations:
                if take <= 0:
                    continue
                split_train = int(round(float(take) * float(split["train"])))
                split_train = max(0, min(split_train, take))
                split_val = take - split_train
                train_count += split_train
                val_count += split_val
                selected += take
                entry = {
                    "source_id": source_id,
                    "source_type": source_type,
                    "path": str(record.get("path") or ""),
                    "format_hint": str(record.get("format_hint") or ""),
                    "sample_count": int(take),
                    "sampled_subset": bool(int(record.get("sample_count") or 0) > int(take)),
                    "sample_span": {
                        "line_start": 1,
                        "line_end": int(take),
                    },
                    "split_counts": {"train": int(split_train), "val": int(split_val)},
                    "estimated_count": bool(record.get("estimated_count")),
                    "metadata": record.get("metadata") if isinstance(record.get("metadata"), dict) else {},
                }
                if not dry_run:
                    selected_entries.append(entry)
                else:
                    # Dry-run keeps planning information but marks no materialized sample index.
                    entry["dry_run_only"] = True
                    selected_entries.append(entry)

        source_stats.append(
            {
                "source_id": source_id,
                "source_type": source_type,
                "status": status,
                "reason": str(source.get("reason") or ""),
                "weight": float(source.get("weight") or 1.0),
                "available_samples": int(available),
                "selected_samples": int(selected),
                "train_samples": int(train_count),
                "val_samples": int(val_count),
                "record_count": int(len(records)),
            }
        )
        total_selected += selected
        total_train += train_count
        total_val += val_count

    totals = {
        "selected_samples": int(total_selected),
        "train_samples": int(total_train),
        "val_samples": int(total_val),
        "selected_entries": int(len(selected_entries)),
    }
    return selected_entries, source_stats, totals


def _stats_markdown(
    *,
    run_id: str,
    source_stats: list[dict[str, Any]],
    totals: dict[str, int],
    warnings: list[str],
) -> list[str]:
    lines = [
        f"# P40 Replay Mix Stats ({run_id})",
        "",
        "| source_id | source_type | status | available | selected | train | val | reason |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in source_stats:
        lines.append(
            "| {source_id} | {source_type} | {status} | {available} | {selected} | {train} | {val} | {reason} |".format(
                source_id=row.get("source_id"),
                source_type=row.get("source_type"),
                status=row.get("status"),
                available=row.get("available_samples"),
                selected=row.get("selected_samples"),
                train=row.get("train_samples"),
                val=row.get("val_samples"),
                reason=str(row.get("reason") or "").replace("|", "/"),
            )
        )
    lines.extend(
        [
            "",
            "## Totals",
            f"- selected_samples: {int(totals.get('selected_samples') or 0)}",
            f"- train_samples: {int(totals.get('train_samples') or 0)}",
            f"- val_samples: {int(totals.get('val_samples') or 0)}",
            f"- selected_entries: {int(totals.get('selected_entries') or 0)}",
        ]
    )
    if warnings:
        lines.extend(["", "## Warnings"])
        lines.extend([f"- {w}" for w in warnings])
    return lines


def run_replay_mixer(
    *,
    config_path: str | Path,
    out_dir: str | Path | None = None,
    run_id: str = "",
    quick: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = to_abs_path(repo_root, config_path)
    cfg = _read_yaml_or_json(cfg_path)

    output_cfg = cfg.get("output") if isinstance(cfg.get("output"), dict) else {}
    artifacts_root = str(output_cfg.get("artifacts_root") or "docs/artifacts/p40/replay_mixer")
    chosen_run_id = str(run_id or output_cfg.get("run_id") or now_stamp())
    if out_dir:
        run_dir = to_abs_path(repo_root, out_dir)
    else:
        run_dir = to_abs_path(repo_root, artifacts_root) / chosen_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    seeds = _normalize_seeds(cfg)
    seeds_payload = build_seeds_payload(seeds, seed_policy_version="p40.replay_mixer")
    split = _split_cfg(cfg)
    sources_cfg = _pick_sources(cfg)

    quick_cfg = cfg.get("quick") if isinstance(cfg.get("quick"), dict) else {}
    quick_cap_per_source = int(quick_cfg.get("max_samples_per_source") or 240)

    resolved_sources, warnings = resolve_replay_sources(
        repo_root=repo_root,
        source_cfgs=sources_cfg,
        quick=quick,
        dry_run=dry_run,
    )
    dump_source_resolution_json(run_dir / "source_resolution.json", resolved_sources, warnings)

    selected_entries, source_stats, totals = _build_selected_entries(
        source_rows=resolved_sources,
        split=split,
        quick=quick,
        quick_cap_per_source=quick_cap_per_source,
        dry_run=dry_run,
    )

    status = "ok" if int(totals.get("selected_samples") or 0) > 0 else "stub"
    manifest = {
        "schema": "p40_replay_mix_manifest_v1",
        "generated_at": now_iso(),
        "run_id": chosen_run_id,
        "status": status,
        "config_path": str(cfg_path),
        "run_dir": str(run_dir),
        "quick": bool(quick),
        "dry_run": bool(dry_run),
        "split": split,
        "seeds": list(seeds),
        "seed_hash": seeds_payload.get("seed_hash"),
        "sources": source_stats,
        "selected_entries": selected_entries,
        "totals": totals,
        "warnings": warnings,
    }
    stats = {
        "schema": "p40_replay_mix_stats_v1",
        "generated_at": now_iso(),
        "run_id": chosen_run_id,
        "status": status,
        "sources": source_stats,
        "totals": totals,
        "warnings": warnings,
    }
    stats_md = _stats_markdown(run_id=chosen_run_id, source_stats=source_stats, totals=totals, warnings=warnings)

    write_json(run_dir / "replay_mix_manifest.json", manifest)
    write_json(run_dir / "replay_mix_stats.json", stats)
    write_markdown(run_dir / "replay_mix_stats.md", stats_md)
    write_json(run_dir / "seeds_used.json", seeds_payload)

    return {
        "status": status,
        "run_id": chosen_run_id,
        "run_dir": str(run_dir),
        "replay_mix_manifest": str(run_dir / "replay_mix_manifest.json"),
        "replay_mix_stats": str(run_dir / "replay_mix_stats.json"),
        "seeds_used": str(run_dir / "seeds_used.json"),
        "selected_samples": int(totals.get("selected_samples") or 0),
        "warnings_count": len(warnings),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P40 replay mixer: unify replay sources into one manifest.")
    parser.add_argument("--config", default="configs/experiments/p40_replay_mix_smoke.yaml")
    parser.add_argument("--out-dir", default="", help="Optional explicit run output directory")
    parser.add_argument("--run-id", default="", help="Optional explicit run_id")
    parser.add_argument("--quick", action="store_true", help="Use small source caps for smoke runs")
    parser.add_argument("--dry-run", action="store_true", help="Resolve sources and write plan without heavy work")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_replay_mixer(
        config_path=args.config,
        out_dir=(args.out_dir if str(args.out_dir).strip() else None),
        run_id=str(args.run_id or ""),
        quick=bool(args.quick),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if str(summary.get("status")) in {"ok", "stub"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
