from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from trainer.closed_loop.replay_manifest import build_seeds_payload, write_json, write_markdown
from trainer.hybrid.router_labels import build_knowledge_base, infer_router_label
from trainer.hybrid.router_schema import (
    DATASET_SAMPLE_SCHEMA,
    build_feature_encoder,
    canonicalize_controller_id,
    normalize_available_controllers,
    normalize_routing_features,
    supported_controller_ids,
)


TRACE_FILENAME = "routing_trace.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            try:
                payload = json.loads(text)
            except Exception:
                sidecar = path.with_suffix(".json")
                if sidecar.exists():
                    payload = json.loads(sidecar.read_text(encoding="utf-8"))
                else:
                    raise RuntimeError(f"PyYAML unavailable for {path}")
        else:
            payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"config must be mapping: {path}")
    return payload


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as fp:
        for line in fp:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _resolve_repo_root() -> Path:
    preferred = Path("D:/MYFILES/BalatroAI")
    if preferred.exists():
        return preferred.resolve()
    return Path(__file__).resolve().parents[2]


def _default_config() -> dict[str, Any]:
    return {
        "schema": "p52_router_dataset_config_v1",
        "artifacts": {
            "trace_roots": [
                "docs/artifacts/p48/router_traces",
                "docs/artifacts/p48/arena_ablation",
                "docs/artifacts/p22/runs",
            ],
            "knowledge_roots": [
                "docs/artifacts/p39",
                "docs/artifacts/p41",
                "docs/artifacts/p48",
                "docs/artifacts/p51",
                "docs/artifacts/p22/runs",
            ],
        },
        "dataset": {
            "allow_rule_labels": True,
            "min_label_confidence": 0.20,
            "max_trace_files": 0,
            "max_samples": 0,
        },
        "output": {
            "artifacts_root": "docs/artifacts/p52/router_dataset",
        },
    }


def _merged_config(path: str | Path | None) -> dict[str, Any]:
    payload = _default_config()
    if not path:
        return payload
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = (_resolve_repo_root() / cfg_path).resolve()
    override = _read_yaml_or_json(cfg_path)
    for top_key, value in override.items():
        if isinstance(value, dict) and isinstance(payload.get(top_key), dict):
            merged = dict(payload.get(top_key) or {})
            merged.update(value)
            payload[top_key] = merged
        else:
            payload[top_key] = value
    return payload


def _resolve_paths(repo_root: Path, values: list[str]) -> list[Path]:
    rows: list[Path] = []
    for value in values:
        path = Path(str(value))
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        if path.exists():
            rows.append(path)
    return rows


def _trace_paths(trace_roots: list[Path], *, max_trace_files: int) -> list[Path]:
    rows: list[Path] = []
    for root in trace_roots:
        rows.extend(root.glob(f"**/{TRACE_FILENAME}"))
    ordered = sorted({path.resolve() for path in rows}, key=lambda item: str(item), reverse=True)
    if max_trace_files > 0:
        ordered = ordered[:max_trace_files]
    return ordered


def _knowledge_paths(roots: list[Path]) -> tuple[list[Path], list[Path]]:
    summary_paths: list[Path] = []
    bucket_paths: list[Path] = []
    for root in roots:
        summary_paths.extend(root.glob("**/summary_table.json"))
        bucket_paths.extend(root.glob("**/bucket_metrics.json"))
    return (
        sorted({path.resolve() for path in summary_paths}, key=lambda item: str(item)),
        sorted({path.resolve() for path in bucket_paths}, key=lambda item: str(item)),
    )


def _ancestor_dirs(path: Path, repo_root: Path) -> list[Path]:
    rows: list[Path] = []
    current = path.parent
    while True:
        rows.append(current)
        if current == repo_root or current.parent == current:
            break
        current = current.parent
    return rows


def _find_nearest_file(trace_path: Path, repo_root: Path, filename: str) -> str:
    for directory in _ancestor_dirs(trace_path, repo_root):
        candidate = directory / filename
        if candidate.exists():
            return str(candidate.resolve())
    return ""


def _candidate_or_blank(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    return ""


def _trace_context(trace_rows: list[dict[str, Any]], trace_path: Path, repo_root: Path) -> dict[str, Any]:
    first_row = trace_rows[0] if trace_rows else {}
    promotion_decision_path = _find_nearest_file(trace_path, repo_root, "promotion_decision.json")
    promotion_payload = _read_json(Path(promotion_decision_path)) if promotion_decision_path else None
    promotion_payload = promotion_payload if isinstance(promotion_payload, dict) else {}
    run_manifest_path = _find_nearest_file(trace_path, repo_root, "run_manifest.json")
    run_manifest = _read_json(Path(run_manifest_path)) if run_manifest_path else None
    run_manifest = run_manifest if isinstance(run_manifest, dict) else {}
    summary_path = _find_nearest_file(trace_path, repo_root, "summary_table.json")
    bucket_path = _find_nearest_file(trace_path, repo_root, "bucket_metrics.json")
    triage_path = _find_nearest_file(trace_path, repo_root, "triage_report.json")

    campaign_run_id = ""
    source_run_id = ""
    parts = [part.lower() for part in trace_path.parts]
    if "campaign_runs" in parts:
        index = parts.index("campaign_runs")
        if index + 1 < len(trace_path.parts):
            campaign_run_id = trace_path.parts[index + 1]
    if campaign_run_id:
        source_run_id = campaign_run_id
    elif str(run_manifest.get("run_id") or "").strip():
        source_run_id = str(run_manifest.get("run_id") or "")
    elif isinstance(first_row.get("trace_context"), dict) and str((first_row.get("trace_context") or {}).get("run_id") or "").strip():
        source_run_id = str((first_row.get("trace_context") or {}).get("run_id") or "")
    else:
        source_run_id = trace_path.parent.name

    return {
        "trace_path": str(trace_path.resolve()),
        "summary_path": summary_path,
        "bucket_path": bucket_path,
        "triage_path": triage_path,
        "promotion_decision_path": promotion_decision_path,
        "run_manifest_path": run_manifest_path,
        "run_manifest": run_manifest,
        "promotion_decision": promotion_payload,
        "source_run_id": source_run_id,
        "campaign_run_id": campaign_run_id,
        "candidate_checkpoint_id": _candidate_or_blank(
            promotion_payload,
            "candidate_checkpoint_id",
        ),
        "champion_checkpoint_id": _candidate_or_blank(
            promotion_payload,
            "champion_checkpoint_id",
        ),
        "candidate_policy": _candidate_or_blank(
            promotion_payload,
            "candidate_policy",
            "candidate_policy_id",
        ),
        "champion_policy": _candidate_or_blank(
            promotion_payload,
            "champion_policy",
            "champion_policy_id",
        ),
    }


def _local_knowledge(context: dict[str, Any]) -> dict[str, Any]:
    summary_paths = [Path(context["summary_path"])] if str(context.get("summary_path") or "").strip() else []
    bucket_paths = [Path(context["bucket_path"])] if str(context.get("bucket_path") or "").strip() else []
    return build_knowledge_base(summary_paths=summary_paths, bucket_paths=bucket_paths)


def _sample_seed(row: dict[str, Any], context: dict[str, Any]) -> str:
    if str(row.get("seed") or "").strip():
        return str(row.get("seed") or "")
    trace_context = row.get("trace_context") if isinstance(row.get("trace_context"), dict) else {}
    if str(trace_context.get("seed") or "").strip():
        return str(trace_context.get("seed") or "")
    parts = Path(str(context.get("trace_path") or "")).parts
    for part in parts:
        if part.startswith("seed_"):
            return part
    return ""


def _available_controllers(row: dict[str, Any]) -> list[str]:
    if isinstance(row.get("available_controllers"), dict):
        return normalize_available_controllers(row.get("available_controllers"))
    if isinstance(row.get("available_controllers"), list):
        return normalize_available_controllers(row.get("available_controllers"))
    if isinstance(row.get("routing_score_breakdown"), dict):
        return normalize_available_controllers(row.get("routing_score_breakdown"))
    return supported_controller_ids()


def _chosen_controller_rule(row: dict[str, Any]) -> str:
    for key in ("chosen_controller_rule", "initial_selected_controller", "selected_controller"):
        token = canonicalize_controller_id(row.get(key))
        if token:
            return token
    return ""


def _flatten_sample(
    *,
    row: dict[str, Any],
    context: dict[str, Any],
    global_knowledge: dict[str, Any],
    local_knowledge: dict[str, Any],
    sample_index: int,
    min_label_confidence: float,
    allow_rule_labels: bool,
) -> dict[str, Any]:
    features = normalize_routing_features(row)
    available_controllers = _available_controllers(row)
    label_payload = infer_router_label(
        sample=row,
        global_knowledge=global_knowledge,
        local_knowledge=local_knowledge,
        allow_rule_labels=allow_rule_labels,
    )
    label_confidence = float(label_payload.get("label_confidence") or 0.0)
    valid_for_training = bool(label_payload.get("valid_for_training")) and label_confidence >= float(min_label_confidence)
    sample_id = str(row.get("sample_id") or f"{context.get('source_run_id')}::{sample_index:05d}")
    if "::" not in sample_id:
        sample_id = f"{context.get('source_run_id')}::{sample_id}"
    sample = {
        "schema": DATASET_SAMPLE_SCHEMA,
        "sample_id": sample_id,
        "routing_features": features,
        "available_controllers": available_controllers,
        "chosen_controller_rule": _chosen_controller_rule(row),
        "target_controller_label": str(label_payload.get("target_controller_label") or ""),
        "target_controller_scores": dict(label_payload.get("target_controller_scores") or {}),
        "label_source": str(label_payload.get("label_source") or "missing"),
        "label_confidence": label_confidence,
        "label_evidence": list(label_payload.get("label_evidence") or []),
        "slice_stage": str(features.get("slice_stage") or "unknown"),
        "slice_resource_pressure": str(features.get("slice_resource_pressure") or "unknown"),
        "slice_action_type": str(features.get("slice_action_type") or "unknown"),
        "slice_position_sensitive": str(features.get("slice_position_sensitive") or "unknown"),
        "slice_stateful_joker_present": str(features.get("slice_stateful_joker_present") or "unknown"),
        "phase": str(features.get("phase") or "unknown"),
        "seed": _sample_seed(row, context),
        "source_run_id": str(context.get("source_run_id") or ""),
        "campaign_run_id": str(context.get("campaign_run_id") or ""),
        "candidate_checkpoint_id": str(context.get("candidate_checkpoint_id") or ""),
        "champion_checkpoint_id": str(context.get("champion_checkpoint_id") or ""),
        "candidate_policy": str(context.get("candidate_policy") or ""),
        "champion_policy": str(context.get("champion_policy") or ""),
        "arena_ref": str(context.get("summary_path") or ""),
        "triage_ref": str(context.get("triage_path") or ""),
        "promotion_decision_ref": str(context.get("promotion_decision_path") or ""),
        "run_manifest_ref": str(context.get("run_manifest_path") or ""),
        "trace_ref": str(context.get("trace_path") or ""),
        "valid_for_training": valid_for_training,
        "trace_index": _safe_int(row.get("trace_index"), sample_index),
    }
    if isinstance(row.get("routing_score_breakdown"), dict):
        sample["routing_score_breakdown"] = dict(row.get("routing_score_breakdown") or {})
    return sample


def _stats_markdown(stats: dict[str, Any]) -> list[str]:
    lines = [
        "# P52 Router Dataset Stats",
        "",
        f"- sample_count: {int(stats.get('sample_count') or 0)}",
        f"- valid_for_training_count: {int(stats.get('valid_for_training_count') or 0)}",
        f"- trace_file_count: {int(stats.get('trace_file_count') or 0)}",
        f"- unique_source_runs: {int(stats.get('unique_source_run_count') or 0)}",
        "",
        "## Target Controller Distribution",
    ]
    target_rows = stats.get("target_controller_distribution") if isinstance(stats.get("target_controller_distribution"), list) else []
    if target_rows:
        for row in target_rows:
            if not isinstance(row, dict):
                continue
            lines.append(
                "- {controller}: count={count} ratio={ratio:.3f}".format(
                    controller=row.get("controller_id"),
                    count=int(row.get("count") or 0),
                    ratio=float(row.get("ratio") or 0.0),
                )
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Label Sources"])
    source_rows = stats.get("label_source_distribution") if isinstance(stats.get("label_source_distribution"), list) else []
    if source_rows:
        for row in source_rows:
            if not isinstance(row, dict):
                continue
            lines.append(
                "- {source}: count={count} ratio={ratio:.3f}".format(
                    source=row.get("label_source"),
                    count=int(row.get("count") or 0),
                    ratio=float(row.get("ratio") or 0.0),
                )
            )
    else:
        lines.append("- none")
    return lines


def _distribution(counter: Counter[str], *, key_name: str) -> list[dict[str, Any]]:
    total = sum(counter.values())
    return [
        {key_name: key, "count": int(count), "ratio": float(count) / max(1, total)}
        for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def build_router_dataset(
    *,
    config_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    run_id: str = "",
    quick: bool = False,
) -> dict[str, Any]:
    repo_root = _resolve_repo_root()
    cfg = _merged_config(config_path)
    artifacts_cfg = cfg.get("artifacts") if isinstance(cfg.get("artifacts"), dict) else {}
    dataset_cfg = cfg.get("dataset") if isinstance(cfg.get("dataset"), dict) else {}
    output_cfg = cfg.get("output") if isinstance(cfg.get("output"), dict) else {}

    trace_roots = _resolve_paths(repo_root, [str(item) for item in (artifacts_cfg.get("trace_roots") or [])])
    knowledge_roots = _resolve_paths(repo_root, [str(item) for item in (artifacts_cfg.get("knowledge_roots") or [])])
    max_trace_files = _safe_int(dataset_cfg.get("max_trace_files"), 0)
    max_samples = _safe_int(dataset_cfg.get("max_samples"), 0)
    if quick:
        max_trace_files = min(max_trace_files or 6, 6)
        max_samples = min(max_samples or 768, 768)
    trace_paths = _trace_paths(trace_roots, max_trace_files=max_trace_files)
    summary_paths, bucket_paths = _knowledge_paths(knowledge_roots)
    global_knowledge = build_knowledge_base(summary_paths=summary_paths, bucket_paths=bucket_paths)

    chosen_run_id = str(run_id or output_cfg.get("run_id") or _now_stamp())
    output_root = (
        (repo_root / str(output_cfg.get("artifacts_root") or "docs/artifacts/p52/router_dataset")).resolve()
        if out_dir is None
        else (Path(out_dir) if Path(out_dir).is_absolute() else (repo_root / out_dir).resolve())
    )
    run_dir = output_root / chosen_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    allow_rule_labels = bool(dataset_cfg.get("allow_rule_labels", True))
    min_label_confidence = float(dataset_cfg.get("min_label_confidence") or 0.20)

    samples: list[dict[str, Any]] = []
    seeds_seen: list[str] = []
    source_runs: set[str] = set()
    trace_sources: list[dict[str, Any]] = []
    for trace_index, trace_path in enumerate(trace_paths, start=1):
        trace_rows = _read_jsonl(trace_path)
        if not trace_rows:
            continue
        context = _trace_context(trace_rows, trace_path, repo_root)
        local_knowledge = _local_knowledge(context)
        trace_sources.append(
            {
                "trace_path": str(trace_path.resolve()),
                "sample_count": len(trace_rows),
                "summary_path": str(context.get("summary_path") or ""),
                "bucket_path": str(context.get("bucket_path") or ""),
                "triage_path": str(context.get("triage_path") or ""),
                "source_run_id": str(context.get("source_run_id") or ""),
            }
        )
        source_runs.add(str(context.get("source_run_id") or trace_path.parent.name))
        for row_index, row in enumerate(trace_rows):
            if max_samples > 0 and len(samples) >= max_samples:
                break
            sample = _flatten_sample(
                row=row,
                context=context,
                global_knowledge=global_knowledge,
                local_knowledge=local_knowledge,
                sample_index=row_index,
                min_label_confidence=min_label_confidence,
                allow_rule_labels=allow_rule_labels,
            )
            seed = str(sample.get("seed") or "").strip()
            if seed and seed not in seeds_seen:
                seeds_seen.append(seed)
            samples.append(sample)
        if max_samples > 0 and len(samples) >= max_samples:
            break

    if trace_paths and not samples:
        raise RuntimeError("router dataset builder found routing traces but produced no samples")

    dataset_jsonl = run_dir / "router_dataset.jsonl"
    with dataset_jsonl.open("w", encoding="utf-8", newline="\n") as fp:
        for sample in samples:
            fp.write(json.dumps(sample, ensure_ascii=False) + "\n")

    encoder = build_feature_encoder([sample for sample in samples if bool(sample.get("valid_for_training"))] or samples)
    write_json(run_dir / "feature_encoder.json", encoder)

    valid_samples = [sample for sample in samples if bool(sample.get("valid_for_training"))]
    label_source_counter: Counter[str] = Counter()
    target_controller_counter: Counter[str] = Counter()
    chosen_controller_counter: Counter[str] = Counter()
    source_run_counter: Counter[str] = Counter()
    slice_stage_counter: Counter[str] = Counter()
    confidence_values: list[float] = []
    for sample in samples:
        label_source_counter[str(sample.get("label_source") or "missing")] += 1
        target_controller_counter[str(sample.get("target_controller_label") or "unknown")] += 1
        chosen_controller_counter[str(sample.get("chosen_controller_rule") or "unknown")] += 1
        source_run_counter[str(sample.get("source_run_id") or "unknown")] += 1
        slice_stage_counter[str(sample.get("slice_stage") or "unknown")] += 1
        confidence_values.append(float(sample.get("label_confidence") or 0.0))

    stats = {
        "schema": "p52_router_dataset_stats_v1",
        "generated_at": _now_iso(),
        "run_id": chosen_run_id,
        "sample_count": len(samples),
        "valid_for_training_count": len(valid_samples),
        "trace_file_count": len(trace_sources),
        "unique_source_run_count": len(source_runs),
        "mean_label_confidence": (sum(confidence_values) / max(1, len(confidence_values))) if confidence_values else 0.0,
        "target_controller_distribution": _distribution(target_controller_counter, key_name="controller_id"),
        "chosen_controller_distribution": _distribution(chosen_controller_counter, key_name="controller_id"),
        "label_source_distribution": _distribution(label_source_counter, key_name="label_source"),
        "source_run_distribution": _distribution(source_run_counter, key_name="source_run_id"),
        "slice_stage_distribution": _distribution(slice_stage_counter, key_name="slice_stage"),
        "supported_controller_ids": supported_controller_ids(),
        "trace_sources": trace_sources,
    }
    write_json(run_dir / "router_dataset_stats.json", stats)
    write_markdown(run_dir / "router_dataset_stats.md", _stats_markdown(stats))
    write_json(run_dir / "router_samples_preview.json", samples[: min(12, len(samples))])
    write_json(run_dir / "seeds_used.json", build_seeds_payload(seeds_seen or ["NO_TRACE_SEED"], seed_policy_version="p52.router_dataset"))

    manifest = {
        "schema": "p52_router_dataset_manifest_v1",
        "generated_at": _now_iso(),
        "run_id": chosen_run_id,
        "run_dir": str(run_dir.resolve()),
        "dataset_jsonl": str(dataset_jsonl.resolve()),
        "feature_encoder_json": str((run_dir / "feature_encoder.json").resolve()),
        "stats_json": str((run_dir / "router_dataset_stats.json").resolve()),
        "stats_md": str((run_dir / "router_dataset_stats.md").resolve()),
        "preview_json": str((run_dir / "router_samples_preview.json").resolve()),
        "seeds_used_json": str((run_dir / "seeds_used.json").resolve()),
        "sample_count": len(samples),
        "valid_for_training_count": len(valid_samples),
        "trace_file_count": len(trace_sources),
        "knowledge_summary_path_count": len(summary_paths),
        "knowledge_bucket_path_count": len(bucket_paths),
        "controller_ids": supported_controller_ids(),
        "config": cfg,
    }
    write_json(run_dir / "router_dataset_manifest.json", manifest)

    return {
        "status": "ok" if samples else "empty",
        "run_id": chosen_run_id,
        "run_dir": str(run_dir),
        "dataset_manifest_json": str((run_dir / "router_dataset_manifest.json").resolve()),
        "dataset_jsonl": str(dataset_jsonl.resolve()),
        "stats_json": str((run_dir / "router_dataset_stats.json").resolve()),
        "stats_md": str((run_dir / "router_dataset_stats.md").resolve()),
        "sample_count": len(samples),
        "valid_for_training_count": len(valid_samples),
        "trace_file_count": len(trace_sources),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a P52 learned-router dataset from routing traces and arena outputs.")
    parser.add_argument("--config", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_router_dataset(
        config_path=(str(args.config).strip() or None),
        out_dir=(str(args.out_dir).strip() or None),
        run_id=str(args.run_id or ""),
        quick=bool(args.quick),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if str(summary.get("status") or "") in {"ok", "empty"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
