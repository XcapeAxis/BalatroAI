from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from trainer.closed_loop.replay_manifest import write_json, write_markdown
from trainer.hybrid.learned_router_eval import run_learned_router_ablation
from trainer.hybrid.router_benchmark_schema import (
    _safe_float,
    aggregate_seed_results,
    aggregate_slice_results,
    aggregate_trace_results,
    numeric_summary,
    summarize_episode_rows,
)
from trainer.registry.checkpoint_registry import list_entries


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _resolve_repo_root() -> Path:
    preferred = Path("D:/MYFILES/BalatroAI")
    if preferred.exists():
        return preferred.resolve()
    return Path(__file__).resolve().parents[2]


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is not None:
            payload = yaml.safe_load(text)
        else:
            sidecar = path.with_suffix(".json")
            if not sidecar.exists():
                raise RuntimeError(f"PyYAML unavailable and no sidecar JSON for: {path}")
            payload = json.loads(sidecar.read_text(encoding="utf-8-sig"))
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"config must be mapping: {path}")
    return payload


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged.get(key) or {}), value)
        else:
            merged[key] = value
    return merged


def _default_config() -> dict[str, Any]:
    return {
        "schema": "p56_router_benchmark_config_v1",
        "benchmark": {
            "artifacts_root": "docs/artifacts/p56/router_benchmark",
            "slice_keys": [
                "slice_stage",
                "slice_resource_pressure",
                "slice_action_type",
                "slice_position_sensitive",
                "slice_stateful_joker_present",
            ],
            "catastrophic": {
                "invalid_action_rate_min": 0.25,
                "total_score_max": 40.0,
            },
            "seed_pools": [
                {"name": "small", "seeds": ["AAAAAAA", "BBBBBBB"], "quick": True, "include_in_quick": True},
                {"name": "medium", "seeds": ["AAAAAAA", "BBBBBBB", "CCCCCCC"], "quick": True, "include_in_quick": True},
                {"name": "nightly", "seeds": ["AAAAAAA", "BBBBBBB", "CCCCCCC", "DDDDDDD"], "quick": False, "include_in_quick": False},
            ],
        },
    }


def _merged_config(path: str | Path | None) -> dict[str, Any]:
    payload = _default_config()
    if not path:
        return payload
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = (_resolve_repo_root() / cfg_path).resolve()
    return _deep_merge(payload, _read_yaml_or_json(cfg_path))


def _read_json(path: Path) -> Any:
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


def _resolve_checkpoint_refs(
    *,
    checkpoint_path: str | Path | None,
    checkpoint_id: str = "",
    train_manifest_path: str | Path | None = None,
    dataset_manifest_path: str | Path | None = None,
) -> tuple[str, str, str, str]:
    repo_root = _resolve_repo_root()
    checkpoint_token = str(checkpoint_path or "").strip()
    train_manifest_token = str(train_manifest_path or "").strip()
    dataset_manifest_token = str(dataset_manifest_path or "").strip()
    checkpoint_id_token = str(checkpoint_id or "").strip()
    if checkpoint_token:
        resolved_checkpoint = Path(checkpoint_token)
        if not resolved_checkpoint.is_absolute():
            resolved_checkpoint = (repo_root / resolved_checkpoint).resolve()
        return str(resolved_checkpoint), checkpoint_id_token, train_manifest_token, dataset_manifest_token

    entries = [item for item in list_entries() if str(item.get("family") or "") == "learned_router"]
    entries.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    for entry in entries:
        artifact_path = str(entry.get("artifact_path") or "").strip()
        if not artifact_path:
            continue
        resolved_checkpoint = Path(artifact_path)
        if not resolved_checkpoint.exists():
            continue
        checkpoint_id_token = checkpoint_id_token or str(entry.get("checkpoint_id") or "")
        lineage_refs = entry.get("lineage_refs") if isinstance(entry.get("lineage_refs"), dict) else {}
        train_manifest_token = train_manifest_token or str(lineage_refs.get("train_manifest_json") or "")
        dataset_manifest_token = dataset_manifest_token or str(lineage_refs.get("dataset_manifest_json") or "")
        return str(resolved_checkpoint.resolve()), checkpoint_id_token, train_manifest_token, dataset_manifest_token
    return "", checkpoint_id_token, train_manifest_token, dataset_manifest_token


def _annotated_trace_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    trace_root = run_dir / "router_traces"
    if not trace_root.exists():
        return rows
    for trace_path in sorted(trace_root.glob("*.jsonl")):
        policy_id = trace_path.stem
        for row in _read_jsonl(trace_path):
            item = dict(row)
            item["policy_id"] = str(item.get("policy_id") or policy_id)
            rows.append(item)
    return rows


def _policy_summary_rows(summary_rows: list[dict[str, Any]], trace_rows: list[dict[str, Any]], episode_rows: list[dict[str, Any]], catastrophic_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    trace_index = {
        str(item.get("policy_id") or ""): dict(item)
        for item in aggregate_trace_results(trace_rows)
        if isinstance(item, dict)
    }
    episode_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in episode_rows:
        episode_index[str(row.get("policy_id") or "unknown")].append(row)
    out: list[dict[str, Any]] = []
    for row in summary_rows:
        policy_id = str(row.get("policy_id") or "")
        trace_summary = trace_index.get(policy_id, {})
        episode_summary = summarize_episode_rows(episode_index.get(policy_id, []), catastrophic_cfg=catastrophic_cfg)
        out.append(
            {
                "policy_id": policy_id,
                "episode_count": int(episode_summary.get("episode_count") or int(row.get("episodes") or 0)),
                "seed_count": int(episode_summary.get("seed_count") or int(row.get("seed_count") or 0)),
                "mean_total_score": float(row.get("mean_total_score") or episode_summary.get("mean_total_score") or 0.0),
                "std_total_score": float(row.get("std_total_score") or episode_summary.get("std_total_score") or 0.0),
                "median_total_score": float(row.get("p50_total_score") or episode_summary.get("median_total_score") or 0.0),
                "win_rate": float(row.get("win_rate") or episode_summary.get("win_rate") or 0.0),
                "invalid_action_rate": float(row.get("invalid_action_rate") or episode_summary.get("mean_invalid_action_rate") or 0.0),
                "catastrophic_failure_count": int(episode_summary.get("catastrophic_failure_count") or 0),
                "checkpoint_id": str(row.get("checkpoint_id") or ""),
                "world_model_checkpoint_id": str(row.get("world_model_checkpoint_id") or ""),
                "controller_selection_distribution": trace_summary.get("controller_selection_distribution") if isinstance(trace_summary.get("controller_selection_distribution"), list) else [],
                "final_controller_distribution": trace_summary.get("final_controller_distribution") if isinstance(trace_summary.get("final_controller_distribution"), list) else [],
                "guard_trigger_rate": float(trace_summary.get("guard_trigger_rate") or 0.0),
                "fallback_rate": float(trace_summary.get("fallback_rate") or 0.0),
                "canary_eligible_rate": float(trace_summary.get("canary_eligible_rate") or 0.0),
                "canary_usage_rate": float(trace_summary.get("canary_usage_rate") or 0.0),
                "canary_reject_reason_distribution": trace_summary.get("canary_reject_reason_distribution") if isinstance(trace_summary.get("canary_reject_reason_distribution"), list) else [],
            }
        )
    return out


def _policy_markdown(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["# P56 Router Benchmark", "", "## Policies"]
    for row in rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            "- {policy}: mean={mean:.3f} std={std:.3f} median={median:.3f} cat_fail={cat} guard={guard:.3f} canary_usage={canary:.3f}".format(
                policy=str(row.get("policy_id") or ""),
                mean=_safe_float(row.get("mean_total_score"), 0.0),
                std=_safe_float(row.get("std_total_score"), 0.0),
                median=_safe_float(row.get("median_total_score"), 0.0),
                cat=int(row.get("catastrophic_failure_count") or 0),
                guard=_safe_float(row.get("guard_trigger_rate"), 0.0),
                canary=_safe_float(row.get("canary_usage_rate"), 0.0),
            )
        )
    return lines


def run_router_benchmark(
    *,
    config_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    run_id: str = "",
    quick: bool = False,
    checkpoint_path: str | Path | None = None,
    checkpoint_id: str = "",
    train_manifest_path: str | Path | None = None,
    dataset_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = _resolve_repo_root()
    cfg = _merged_config(config_path)
    benchmark_cfg = cfg.get("benchmark") if isinstance(cfg.get("benchmark"), dict) else {}
    catastrophic_cfg = benchmark_cfg.get("catastrophic") if isinstance(benchmark_cfg.get("catastrophic"), dict) else {}
    slice_keys = [str(item) for item in (benchmark_cfg.get("slice_keys") or []) if str(item).strip()]
    chosen_run_id = str(run_id or _now_stamp())
    output_root = (
        (repo_root / str(benchmark_cfg.get("artifacts_root") or "docs/artifacts/p56/router_benchmark")).resolve()
        if out_dir is None
        else (Path(out_dir) if Path(out_dir).is_absolute() else (repo_root / out_dir).resolve())
    )
    run_dir = output_root / chosen_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_checkpoint, resolved_checkpoint_id, resolved_train_manifest, resolved_dataset_manifest = _resolve_checkpoint_refs(
        checkpoint_path=checkpoint_path,
        checkpoint_id=checkpoint_id,
        train_manifest_path=train_manifest_path,
        dataset_manifest_path=dataset_manifest_path,
    )

    seed_pools = [dict(item) for item in (benchmark_cfg.get("seed_pools") or []) if isinstance(item, dict)]
    if not seed_pools:
        seed_pools = list(_default_config()["benchmark"]["seed_pools"])

    pool_results: list[dict[str, Any]] = []
    all_episode_rows: list[dict[str, Any]] = []
    all_trace_rows: list[dict[str, Any]] = []
    for pool in seed_pools:
        pool_name = str(pool.get("name") or f"pool_{len(pool_results)+1}")
        if quick and not bool(pool.get("include_in_quick", False)):
            continue
        seeds = [str(seed).strip() for seed in (pool.get("seeds") or []) if str(seed).strip()]
        if not seeds:
            continue
        result = run_learned_router_ablation(
            config_path=config_path,
            out_dir=run_dir / "pool_runs",
            run_id=f"{chosen_run_id}-{pool_name}",
            quick=bool(pool.get("quick", quick)),
            seeds_override=seeds,
            dataset_manifest_path=resolved_dataset_manifest or None,
            train_manifest_path=resolved_train_manifest or None,
            checkpoint_path=resolved_checkpoint or None,
            checkpoint_id_override=resolved_checkpoint_id,
        )
        ablation_run_dir = Path(str(result.get("run_dir") or "")).resolve()
        summary_rows_payload = _read_json(ablation_run_dir / "summary_table.json")
        summary_rows = [row for row in summary_rows_payload if isinstance(row, dict)] if isinstance(summary_rows_payload, list) else []
        episode_rows = _read_jsonl(ablation_run_dir / "episode_records.jsonl")
        trace_rows = _annotated_trace_rows(ablation_run_dir)
        all_episode_rows.extend(episode_rows)
        all_trace_rows.extend(trace_rows)
        pool_summary_rows = _policy_summary_rows(summary_rows, trace_rows, episode_rows, catastrophic_cfg)
        pool_results.append(
            {
                "pool_name": pool_name,
                "seeds": seeds,
                "run_dir": str(ablation_run_dir),
                "summary_rows": pool_summary_rows,
                "summary_table_json": str((ablation_run_dir / "summary_table.json").resolve()),
                "routing_summary_json": str((ablation_run_dir / "routing_summary.json").resolve()),
                "episode_records_jsonl": str((ablation_run_dir / "episode_records.jsonl").resolve()),
            }
        )

    overall_seed_results = aggregate_seed_results(all_episode_rows, catastrophic_cfg=catastrophic_cfg)
    overall_slice_results = aggregate_slice_results(all_episode_rows, slice_keys=slice_keys, catastrophic_cfg=catastrophic_cfg)
    trace_summary_index = {
        str(item.get("policy_id") or ""): dict(item)
        for item in aggregate_trace_results(all_trace_rows)
        if isinstance(item, dict)
    }
    episode_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    checkpoint_id_index: dict[str, set[str]] = defaultdict(set)
    wm_checkpoint_index: dict[str, set[str]] = defaultdict(set)
    for row in all_episode_rows:
        episode_index[str(row.get("policy_id") or "unknown")].append(row)
    for pool in pool_results:
        for row in pool.get("summary_rows") or []:
            if not isinstance(row, dict):
                continue
            policy_id = str(row.get("policy_id") or "")
            checkpoint_id_value = str(row.get("checkpoint_id") or "").strip()
            wm_checkpoint_id_value = str(row.get("world_model_checkpoint_id") or "").strip()
            if checkpoint_id_value:
                checkpoint_id_index[policy_id].add(checkpoint_id_value)
            if wm_checkpoint_id_value:
                wm_checkpoint_index[policy_id].add(wm_checkpoint_id_value)

    overall_rows: list[dict[str, Any]] = []
    for policy_id in sorted(episode_index.keys()):
        episode_summary = summarize_episode_rows(episode_index[policy_id], catastrophic_cfg=catastrophic_cfg)
        trace_summary = trace_summary_index.get(policy_id, {})
        overall_rows.append(
            {
                "policy_id": policy_id,
                **episode_summary,
                "controller_selection_distribution": trace_summary.get("controller_selection_distribution") if isinstance(trace_summary.get("controller_selection_distribution"), list) else [],
                "final_controller_distribution": trace_summary.get("final_controller_distribution") if isinstance(trace_summary.get("final_controller_distribution"), list) else [],
                "guard_trigger_rate": float(trace_summary.get("guard_trigger_rate") or 0.0),
                "fallback_rate": float(trace_summary.get("fallback_rate") or 0.0),
                "canary_eligible_rate": float(trace_summary.get("canary_eligible_rate") or 0.0),
                "canary_usage_rate": float(trace_summary.get("canary_usage_rate") or 0.0),
                "canary_reject_reason_distribution": trace_summary.get("canary_reject_reason_distribution") if isinstance(trace_summary.get("canary_reject_reason_distribution"), list) else [],
                "checkpoint_ids": sorted(checkpoint_id_index.get(policy_id) or []),
                "world_model_checkpoint_ids": sorted(wm_checkpoint_index.get(policy_id) or []),
            }
        )

    rule_row = next((row for row in overall_rows if str(row.get("policy_id") or "") == "hybrid_controller_rule"), {})
    rule_mean = _safe_float(rule_row.get("mean_total_score"), 0.0)
    for row in overall_rows:
        row["score_delta_vs_rule"] = _safe_float(row.get("mean_total_score"), 0.0) - rule_mean

    manifest = {
        "schema": "p56_router_benchmark_manifest_v1",
        "generated_at": _now_iso(),
        "run_id": chosen_run_id,
        "config_path": str(Path(config_path).resolve()) if config_path else "",
        "run_dir": str(run_dir.resolve()),
        "checkpoint_path": resolved_checkpoint,
        "checkpoint_id": resolved_checkpoint_id,
        "train_manifest_path": resolved_train_manifest,
        "dataset_manifest_path": resolved_dataset_manifest,
        "seed_pools": seed_pools,
        "pool_results": pool_results,
    }
    summary = {
        "schema": "p56_router_benchmark_summary_v1",
        "generated_at": _now_iso(),
        "run_id": chosen_run_id,
        "run_dir": str(run_dir.resolve()),
        "checkpoint_path": resolved_checkpoint,
        "checkpoint_id": resolved_checkpoint_id,
        "policy_rows": overall_rows,
        "pool_count": len(pool_results),
        "seed_result_count": len(overall_seed_results),
        "slice_result_count": len(overall_slice_results),
        "evaluation_budget": {
            "episode_count": len(all_episode_rows),
            "trace_count": len(all_trace_rows),
            "seed_count": len({str(row.get("seed") or "") for row in all_episode_rows}),
        },
    }
    write_json(run_dir / "benchmark_manifest.json", manifest)
    write_json(run_dir / "benchmark_summary.json", summary)
    write_json(run_dir / "seed_results.json", overall_seed_results)
    write_json(run_dir / "slice_results.json", overall_slice_results)
    write_markdown(run_dir / "benchmark_summary.md", _policy_markdown(overall_rows))
    return {
        "status": "ok" if overall_rows else "empty",
        "run_id": chosen_run_id,
        "run_dir": str(run_dir.resolve()),
        "benchmark_manifest_json": str((run_dir / "benchmark_manifest.json").resolve()),
        "benchmark_summary_json": str((run_dir / "benchmark_summary.json").resolve()),
        "benchmark_summary_md": str((run_dir / "benchmark_summary.md").resolve()),
        "seed_results_json": str((run_dir / "seed_results.json").resolve()),
        "slice_results_json": str((run_dir / "slice_results.json").resolve()),
        "policy_count": len(overall_rows),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P56 multi-seed router benchmark harness")
    parser.add_argument("--config", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--checkpoint-id", default="")
    parser.add_argument("--train-manifest", default="")
    parser.add_argument("--dataset-manifest", default="")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    summary = run_router_benchmark(
        config_path=(str(args.config).strip() or None),
        out_dir=(str(args.out_dir).strip() or None),
        run_id=str(args.run_id or ""),
        quick=bool(args.quick),
        checkpoint_path=(str(args.checkpoint_path).strip() or None),
        checkpoint_id=str(args.checkpoint_id or ""),
        train_manifest_path=(str(args.train_manifest).strip() or None),
        dataset_manifest_path=(str(args.dataset_manifest).strip() or None),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if str(summary.get("status") or "") in {"ok", "empty"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
