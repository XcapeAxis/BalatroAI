from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from pathlib import Path
from typing import Any

from trainer.closed_loop.replay_lineage import LINEAGE_REQUIRED_FIELDS
from trainer.closed_loop.replay_manifest import now_iso, now_stamp, read_json, to_abs_path, write_json, write_markdown


def _pick_latest_manifest(repo_root: Path) -> Path:
    roots = [
        repo_root / "docs" / "artifacts" / "p41" / "replay_mixer",
        repo_root / "docs" / "artifacts" / "p40" / "replay_mixer",
    ]
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for run_dir in root.iterdir():
            if not run_dir.is_dir():
                continue
            manifest = run_dir / "replay_mix_manifest.json"
            if manifest.exists():
                candidates.append(manifest.resolve())
    if not candidates:
        raise FileNotFoundError("no replay_mix_manifest.json found under p41/p40 replay_mixer roots")
    return sorted(candidates, key=lambda p: str(p))[-1]


def _health_markdown(payload: dict[str, Any]) -> list[str]:
    lines = [
        f"# Replay Lineage Health ({payload.get('run_id')})",
        "",
        f"- status: `{payload.get('status')}`",
        f"- entry_count: `{int(payload.get('entry_count') or 0)}`",
        f"- required_field_missing_ratio: `{float(payload.get('required_field_missing_ratio') or 0.0):.4f}`",
        f"- missing_source_paths: `{int(payload.get('missing_source_paths') or 0)}`",
        "",
        "## Required Field Missing Count",
    ]
    missing_counts = payload.get("required_field_missing_count")
    if isinstance(missing_counts, dict) and missing_counts:
        for key, val in sorted(missing_counts.items()):
            lines.append(f"- {key}: {int(val)}")
    else:
        lines.append("- none")
    warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
    if warnings:
        lines.extend(["", "## Warnings"])
        for warn in warnings:
            lines.append(f"- {str(warn)}")
    return lines


def run_lineage_check(
    *,
    manifest_path: str | Path | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    if manifest_path:
        mp = to_abs_path(repo_root, manifest_path)
    else:
        mp = _pick_latest_manifest(repo_root)

    manifest = read_json(mp)
    if not isinstance(manifest, dict):
        raise RuntimeError(f"invalid manifest json: {mp}")
    entries = manifest.get("selected_entries") if isinstance(manifest.get("selected_entries"), list) else []
    typed_entries = [e for e in entries if isinstance(e, dict)]

    missing_count = {key: 0 for key in LINEAGE_REQUIRED_FIELDS}
    missing_source_paths = 0
    warnings: list[str] = []
    for row in typed_entries:
        for key in LINEAGE_REQUIRED_FIELDS:
            if row.get(key) in {None, ""}:
                missing_count[key] += 1
        src_path = str(row.get("source_artifact_path") or row.get("path") or "").strip()
        if src_path:
            p = Path(src_path)
            if not p.is_absolute():
                p = (repo_root / p).resolve()
            if not p.exists():
                missing_source_paths += 1
        else:
            missing_source_paths += 1

    total_required_slots = max(1, len(typed_entries) * len(LINEAGE_REQUIRED_FIELDS))
    total_missing_slots = int(sum(missing_count.values()))
    required_field_missing_ratio = float(total_missing_slots) / float(total_required_slots)

    status = "ok"
    if required_field_missing_ratio > 0.10:
        status = "warn"
        warnings.append("required lineage fields missing ratio > 10%")
    if missing_source_paths > 0:
        warnings.append("some source artifact paths are missing (may be cleanup side-effect)")

    run_id = str(manifest.get("run_id") or now_stamp())
    if out_dir:
        out_root = to_abs_path(repo_root, out_dir)
    else:
        out_root = mp.parent
    out_root.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema": "p41_lineage_health_v1",
        "generated_at": now_iso(),
        "manifest_path": str(mp),
        "run_id": run_id,
        "status": status,
        "entry_count": int(len(typed_entries)),
        "required_field_missing_count": {k: int(v) for k, v in sorted(missing_count.items())},
        "required_field_missing_ratio": required_field_missing_ratio,
        "missing_source_paths": int(missing_source_paths),
        "warnings": warnings,
    }
    json_path = out_root / "lineage_health.json"
    md_path = out_root / "lineage_health.md"
    write_json(json_path, payload)
    write_markdown(md_path, _health_markdown(payload))
    return {
        "status": status,
        "run_id": run_id,
        "manifest_path": str(mp),
        "lineage_health_json": str(json_path),
        "lineage_health_md": str(md_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check replay lineage completeness and source path health.")
    parser.add_argument("--manifest", default="", help="Optional replay_mix_manifest.json path")
    parser.add_argument("--out-dir", default="", help="Optional output directory for health report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_lineage_check(
        manifest_path=(args.manifest if str(args.manifest).strip() else None),
        out_dir=(args.out_dir if str(args.out_dir).strip() else None),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if str(summary.get("status")) in {"ok", "warn"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

