from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trainer.closed_loop.replay_manifest import (
    count_jsonl_rows,
    infer_generation_method,
    infer_source_run_id,
    read_json,
    read_jsonl,
)


SUPPORTED_SOURCE_TYPES = {
    "p10_long_episode",
    "p13_dagger_or_real",
    "selfsup_replay",
    "arena_failures",
}


def _normalize_path_list(repo_root: Path, roots: list[str] | None) -> list[Path]:
    out: list[Path] = []
    raw = roots if isinstance(roots, list) else []
    for item in raw:
        token = str(item).strip()
        if not token:
            continue
        p = Path(token)
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        out.append(p)
    return out


def _gather_glob_paths(root: Path, patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    if not root.exists():
        return out
    for pattern in patterns:
        for hit in root.glob(pattern):
            if hit.is_file():
                out.append(hit.resolve())
    unique = sorted(set(out), key=lambda p: str(p))
    return unique


def _detect_format_hint(path: Path) -> str:
    low_name = path.name.lower()
    low_full = str(path).lower()
    if "dagger_dataset" in low_name or ("datasets" in low_full and "dagger" in low_full):
        return "bc_record_v1"
    if low_name.startswith("session_"):
        return "p13_session_trace"
    if "action_trace" in low_name:
        return "action_trace_v1"
    if "replay_steps" in low_name:
        return "replay_steps_v1"
    if "failure" in low_name:
        return "failure_replay_v1"

    rows = read_jsonl(path, max_rows=1)
    if rows:
        row = rows[0]
        if isinstance(row, dict) and row.get("expert_action_id") is not None:
            return "bc_record_v1"
        if isinstance(row, dict) and row.get("policy_id") is not None and row.get("episode_index") is not None:
            return "arena_episode_record_v1"
    return "jsonl_unknown"


def _peek_jsonl_row(path: Path) -> dict[str, Any] | None:
    rows = read_jsonl(path, max_rows=1)
    if rows and isinstance(rows[0], dict):
        return rows[0]
    return None


def _derive_lineage_fields(
    *,
    source_type: str,
    path: Path,
    preview_row: dict[str, Any] | None,
    extra_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    row = preview_row if isinstance(preview_row, dict) else {}
    meta = extra_meta if isinstance(extra_meta, dict) else {}

    source_seed = str(row.get("seed") or meta.get("seed") or "").strip()
    episode_id = str(row.get("episode_id") or "").strip()
    if not episode_id:
        policy_id = str(row.get("policy_id") or "").strip()
        episode_index = row.get("episode_index")
        if policy_id and source_seed and episode_index is not None:
            episode_id = f"{policy_id}|{source_seed}|{episode_index}"
    step_id = row.get("step_id")
    if step_id is None:
        step_id = row.get("step_index")
    if step_id is None:
        step_id = row.get("t")

    valid_for_training = row.get("valid_for_training")
    if valid_for_training is None:
        valid_for_training = meta.get("valid_for_training")
    if valid_for_training is None:
        valid_for_training = True
    valid_for_training = bool(valid_for_training)

    return {
        "source_run_id": infer_source_run_id(path),
        "source_seed": source_seed,
        "episode_id": episode_id,
        "step_id": step_id if step_id is not None else "",
        "generation_method": infer_generation_method(source_type, str(path), meta),
        "valid_for_training": valid_for_training,
        "preview_row": row if row else {},
    }


def _jsonl_record(
    path: Path,
    *,
    scan_limit: int,
    source_type: str,
    source_id: str,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    preview = _peek_jsonl_row(path)
    line_count, truncated = count_jsonl_rows(path, max_scan_rows=max(0, int(scan_limit)))
    lineage = _derive_lineage_fields(
        source_type=source_type,
        path=path,
        preview_row=preview,
        extra_meta=extra_meta,
    )
    return {
        "source_type": source_type,
        "source_id": source_id,
        "path": str(path),
        "sample_count": int(line_count),
        "estimated_count": bool(truncated),
        "format_hint": _detect_format_hint(path),
        "metadata": extra_meta or {},
        "source_run_id": str(lineage.get("source_run_id") or ""),
        "source_seed": str(lineage.get("source_seed") or ""),
        "episode_id": str(lineage.get("episode_id") or ""),
        "step_id": lineage.get("step_id") if lineage.get("step_id") is not None else "",
        "generation_method": str(lineage.get("generation_method") or "unknown"),
        "valid_for_training": bool(lineage.get("valid_for_training", True)),
        "preview_row": lineage.get("preview_row") if isinstance(lineage.get("preview_row"), dict) else {},
    }


def _resolve_p10_source(
    *,
    repo_root: Path,
    source_cfg: dict[str, Any],
    scan_limit: int,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], list[str], str]:
    roots = _normalize_path_list(repo_root, source_cfg.get("roots"))
    if not roots:
        roots = [(repo_root / "docs/artifacts/p10").resolve(), (repo_root / "sim/tests/fixtures_runtime").resolve()]

    report_patterns = source_cfg.get("report_patterns")
    if not isinstance(report_patterns, list) or not report_patterns:
        report_patterns = ["report_p10_*.json"]

    report_paths: list[Path] = []
    for root in roots:
        if root.name.lower() == "fixtures_runtime":
            continue
        report_paths.extend(_gather_glob_paths(root, [str(p) for p in report_patterns]))
    report_paths = sorted(set(report_paths), key=lambda p: str(p), reverse=True)

    records: list[dict[str, Any]] = []
    consumed_paths: list[str] = []
    warned_missing: set[str] = set()
    for report_path in report_paths:
        payload = read_json(report_path)
        if not isinstance(payload, dict):
            continue
        consumed_paths.append(str(report_path))
        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            continue
        for row in raw_results:
            if not isinstance(row, dict):
                continue
            artifacts = row.get("artifacts") if isinstance(row.get("artifacts"), dict) else {}
            action_trace_raw = str(artifacts.get("action_trace") or "").strip()
            if not action_trace_raw:
                continue
            action_trace_path = Path(action_trace_raw)
            if not action_trace_path.is_absolute():
                action_trace_path = (repo_root / action_trace_path).resolve()
            if not action_trace_path.exists():
                key = str(action_trace_path)
                if key not in warned_missing:
                    warnings.append(f"p10 action trace missing: {action_trace_path}")
                    warned_missing.add(key)
                continue
            records.append(
                _jsonl_record(
                    action_trace_path,
                    scan_limit=scan_limit,
                    source_type="p10_long_episode",
                    source_id=str(source_cfg.get("id") or "p10_long_episode"),
                    extra_meta={
                        "target": str(row.get("target") or ""),
                        "stake": str(row.get("stake") or ""),
                        "template": str(row.get("template") or ""),
                    },
                )
            )

    if not records:
        fallback_patterns = source_cfg.get("fallback_patterns")
        if not isinstance(fallback_patterns, list) or not fallback_patterns:
            fallback_patterns = ["**/action_trace_p10_*.jsonl", "**/action_trace_*p10*.jsonl"]
        for root in roots:
            for path in _gather_glob_paths(root, [str(p) for p in fallback_patterns]):
                records.append(
                    _jsonl_record(
                        path,
                        scan_limit=scan_limit,
                        source_type="p10_long_episode",
                        source_id=str(source_cfg.get("id") or "p10_long_episode"),
                    )
                )
                consumed_paths.append(str(path))

    reason = "ok" if records else "p10 traces unavailable on this machine"
    return records, consumed_paths, reason


def _resolve_p13_source(
    *,
    repo_root: Path,
    source_cfg: dict[str, Any],
    scan_limit: int,
) -> tuple[list[dict[str, Any]], list[str], str]:
    roots = _normalize_path_list(repo_root, source_cfg.get("roots"))
    if not roots:
        roots = [
            (repo_root / "docs/artifacts/p13").resolve(),
            (repo_root / "docs/artifacts/p16").resolve(),
        ]

    patterns = source_cfg.get("patterns")
    if not isinstance(patterns, list) or not patterns:
        patterns = [
            "**/session_*.jsonl",
            "**/fixture/action_trace_real.jsonl",
            "**/datasets/*dagger*.jsonl",
        ]

    records: list[dict[str, Any]] = []
    consumed_paths: list[str] = []
    source_id = str(source_cfg.get("id") or "p13_dagger_or_real")
    for root in roots:
        for path in _gather_glob_paths(root, [str(p) for p in patterns]):
            consumed_paths.append(str(path))
            records.append(
                _jsonl_record(
                    path,
                    scan_limit=scan_limit,
                    source_type="p13_dagger_or_real",
                    source_id=source_id,
                )
            )
    reason = "ok" if records else "p13/p16 dagger or session traces not found"
    return records, consumed_paths, reason


def _resolve_selfsup_source(
    *,
    repo_root: Path,
    source_cfg: dict[str, Any],
    scan_limit: int,
) -> tuple[list[dict[str, Any]], list[str], str]:
    roots = _normalize_path_list(repo_root, source_cfg.get("roots"))
    if not roots:
        roots = [(repo_root / "docs/artifacts/p36").resolve()]

    patterns = source_cfg.get("patterns")
    if not isinstance(patterns, list) or not patterns:
        patterns = ["**/replay_steps.jsonl", "**/dataset.jsonl"]

    records: list[dict[str, Any]] = []
    consumed_paths: list[str] = []
    source_id = str(source_cfg.get("id") or "selfsup_replay")
    for root in roots:
        for path in _gather_glob_paths(root, [str(p) for p in patterns]):
            consumed_paths.append(str(path))
            records.append(
                _jsonl_record(
                    path,
                    scan_limit=scan_limit,
                    source_type="selfsup_replay",
                    source_id=source_id,
                )
            )
    reason = "ok" if records else "self-supervised replay traces not found"
    return records, consumed_paths, reason


def _find_latest_manifest(repo_root: Path, relative_glob: str) -> Path | None:
    paths = sorted((repo_root).glob(relative_glob), key=lambda p: str(p))
    if not paths:
        return None
    return paths[-1].resolve()


def _resolve_failure_source(
    *,
    repo_root: Path,
    source_cfg: dict[str, Any],
    scan_limit: int,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], list[str], str]:
    source_id = str(source_cfg.get("id") or "arena_failures")
    manifest_raw = str(source_cfg.get("failure_pack_manifest") or "").strip()
    manifest_path: Path | None = None
    if manifest_raw:
        manifest_path = Path(manifest_raw)
        if not manifest_path.is_absolute():
            manifest_path = (repo_root / manifest_path).resolve()
    if manifest_path is None:
        manifest_path = _find_latest_manifest(repo_root, "docs/artifacts/p40/failure_mining/*/failure_pack_manifest.json")

    if manifest_path is None or not manifest_path.exists():
        return [], [], "failure pack manifest not found"

    payload = read_json(manifest_path)
    if not isinstance(payload, dict):
        warnings.append(f"failure pack manifest invalid json: {manifest_path}")
        return [], [str(manifest_path)], "failure pack manifest unreadable"

    status = str(payload.get("status") or "").strip().lower()
    replay_path_raw = (
        str(payload.get("replay_jsonl_path") or "").strip()
        or str(((payload.get("paths") or {}).get("replay_jsonl") if isinstance(payload.get("paths"), dict) else "") or "")
    )
    replay_path = Path(replay_path_raw) if replay_path_raw else None
    if replay_path is not None and not replay_path.is_absolute():
        replay_path = (repo_root / replay_path).resolve()

    records: list[dict[str, Any]] = []
    if replay_path is not None and replay_path.exists():
        records.append(
            _jsonl_record(
                replay_path,
                scan_limit=scan_limit,
                source_type="arena_failures",
                source_id=source_id,
                extra_meta={
                    "failure_manifest": str(manifest_path),
                    "failure_status": status or "unknown",
                },
            )
        )
    else:
        warnings.append(f"failure replay jsonl missing for manifest: {manifest_path}")

    reason = "ok" if records else ("failure pack status=" + (status or "unknown"))
    return records, [str(manifest_path)], reason


def resolve_replay_sources(
    *,
    repo_root: Path,
    source_cfgs: list[dict[str, Any]],
    quick: bool = False,
    dry_run: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    resolved: list[dict[str, Any]] = []

    scan_limit_default = 5000 if quick else 0
    for idx, src in enumerate(source_cfgs):
        if not isinstance(src, dict):
            warnings.append(f"source[{idx}] is not a mapping and is skipped")
            continue
        source_type = str(src.get("type") or "").strip().lower()
        source_id = str(src.get("id") or f"{source_type}_{idx+1}")
        weight = float(src.get("weight") or 1.0)
        enabled = bool(src.get("enabled", True))
        max_samples = int(src.get("max_samples") or 0)
        max_episodes = int(src.get("max_episodes") or 0)
        scan_limit = int(src.get("scan_limit_rows") or scan_limit_default)

        if source_type not in SUPPORTED_SOURCE_TYPES:
            resolved.append(
                {
                    "source_id": source_id,
                    "source_type": source_type,
                    "weight": weight,
                    "enabled": enabled,
                    "status": "stub",
                    "reason": f"unsupported_source_type:{source_type}",
                    "max_samples": max_samples,
                    "max_episodes": max_episodes,
                    "records": [],
                    "available_samples": 0,
                    "source_paths": [],
                }
            )
            continue

        if not enabled:
            resolved.append(
                {
                    "source_id": source_id,
                    "source_type": source_type,
                    "weight": weight,
                    "enabled": False,
                    "status": "skipped",
                    "reason": "disabled",
                    "max_samples": max_samples,
                    "max_episodes": max_episodes,
                    "records": [],
                    "available_samples": 0,
                    "source_paths": [],
                }
            )
            continue

        if source_type == "p10_long_episode":
            records, paths, reason = _resolve_p10_source(
                repo_root=repo_root,
                source_cfg=src,
                scan_limit=scan_limit,
                warnings=warnings,
            )
        elif source_type == "p13_dagger_or_real":
            records, paths, reason = _resolve_p13_source(
                repo_root=repo_root,
                source_cfg=src,
                scan_limit=scan_limit,
            )
        elif source_type == "selfsup_replay":
            records, paths, reason = _resolve_selfsup_source(
                repo_root=repo_root,
                source_cfg=src,
                scan_limit=scan_limit,
            )
        else:
            records, paths, reason = _resolve_failure_source(
                repo_root=repo_root,
                source_cfg=src,
                scan_limit=scan_limit,
                warnings=warnings,
            )

        available_samples = int(sum(int(r.get("sample_count") or 0) for r in records))
        if dry_run:
            for record in records:
                record["sample_count"] = int(record.get("sample_count") or 0)

        status = "ok" if records else "stub"
        resolved.append(
            {
                "source_id": source_id,
                "source_type": source_type,
                "weight": weight,
                "enabled": True,
                "status": status,
                "reason": reason,
                "max_samples": max_samples,
                "max_episodes": max_episodes,
                "records": records,
                "available_samples": available_samples,
                "source_paths": paths,
            }
        )

    return resolved, warnings


def dump_source_resolution_json(path: Path, payload: list[dict[str, Any]], warnings: list[str]) -> None:
    doc = {
        "schema": "p40_replay_sources_resolution_v1",
        "sources": payload,
        "warnings": warnings,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
