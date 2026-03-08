from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import copy
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from trainer.closed_loop.replay_manifest import write_json, write_markdown
from trainer.hybrid.router_benchmark import run_router_benchmark
from trainer.registry.checkpoint_registry import list_entries, update_checkpoint


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


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
        "schema": "p56_guard_tuning_config_v1",
        "guard_tuning": {
            "artifacts_root": "docs/artifacts/p56/guard_tuning",
            "max_candidates": 6,
            "seed_pools": [
                {"name": "tuning", "seeds": ["AAAAAAA", "BBBBBBB"], "quick": True, "include_in_quick": True}
            ],
            "objective": {
                "catastrophic_penalty": 50.0,
                "guard_trigger_penalty": 5.0,
                "score_tolerance_vs_rule": 0.0,
            },
            "sweep": {
                "router_confidence_min": [0.45, 0.60, 0.70],
                "wm_uncertainty_max": [1.0, 0.85],
                "feature_completeness_min": [0.80, 0.90],
                "ood_score_max": [6.0, 4.5],
                "high_risk_slice_force_rule": [False, True],
            },
        }
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


def _resolve_checkpoint_refs(
    *,
    checkpoint_path: str | Path | None,
    checkpoint_id: str = "",
    train_manifest_path: str | Path | None = None,
    dataset_manifest_path: str | Path | None = None,
) -> tuple[str, str, str, str]:
    repo_root = _resolve_repo_root()
    checkpoint_token = str(checkpoint_path or "").strip()
    train_token = str(train_manifest_path or "").strip()
    dataset_token = str(dataset_manifest_path or "").strip()
    checkpoint_id_token = str(checkpoint_id or "").strip()
    if checkpoint_token:
        resolved = Path(checkpoint_token)
        if not resolved.is_absolute():
            resolved = (repo_root / resolved).resolve()
        return str(resolved), checkpoint_id_token, train_token, dataset_token
    entries = [item for item in list_entries() if str(item.get("family") or "") == "learned_router"]
    entries.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    for entry in entries:
        artifact_path = str(entry.get("artifact_path") or "").strip()
        if not artifact_path:
            continue
        resolved = Path(artifact_path)
        if not resolved.exists():
            continue
        checkpoint_id_token = checkpoint_id_token or str(entry.get("checkpoint_id") or "")
        lineage_refs = entry.get("lineage_refs") if isinstance(entry.get("lineage_refs"), dict) else {}
        train_token = train_token or str(lineage_refs.get("train_manifest_json") or "")
        dataset_token = dataset_token or str(lineage_refs.get("dataset_manifest_json") or "")
        return str(resolved.resolve()), checkpoint_id_token, train_token, dataset_token
    return "", checkpoint_id_token, train_token, dataset_token


def _policy_row(summary_payload: dict[str, Any], policy_id: str) -> dict[str, Any]:
    for row in summary_payload.get("policy_rows") or []:
        if isinstance(row, dict) and str(row.get("policy_id") or "") == policy_id:
            return row
    return {}


def _candidate_score(summary_payload: dict[str, Any], objective_cfg: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    rule_row = _policy_row(summary_payload, "hybrid_controller_rule")
    learned_row = _policy_row(summary_payload, "hybrid_controller_learned")
    guarded_row = _policy_row(summary_payload, "hybrid_controller_learned_with_rule_guard")
    rule_score = _safe_float(rule_row.get("mean_total_score"), 0.0)
    learned_score = _safe_float(learned_row.get("mean_total_score"), 0.0)
    guarded_score = _safe_float(guarded_row.get("mean_total_score"), 0.0)
    guarded_cat = _safe_float(guarded_row.get("catastrophic_failure_count"), 0.0)
    rule_cat = _safe_float(rule_row.get("catastrophic_failure_count"), 0.0)
    guard_trigger_rate = _safe_float(guarded_row.get("guard_trigger_rate"), 0.0)
    catastrophic_penalty = _safe_float(objective_cfg.get("catastrophic_penalty"), 50.0)
    guard_trigger_penalty = _safe_float(objective_cfg.get("guard_trigger_penalty"), 5.0)
    score_tolerance_vs_rule = _safe_float(objective_cfg.get("score_tolerance_vs_rule"), 0.0)
    objective = guarded_score - catastrophic_penalty * guarded_cat - guard_trigger_penalty * guard_trigger_rate
    preferred = guarded_cat <= rule_cat and guarded_score >= (rule_score - score_tolerance_vs_rule)
    return objective, {
        "preferred": bool(preferred),
        "rule_score": rule_score,
        "learned_score": learned_score,
        "guarded_score": guarded_score,
        "rule_catastrophic_failures": int(rule_cat),
        "guarded_catastrophic_failures": int(guarded_cat),
        "guard_trigger_rate": guard_trigger_rate,
        "score_delta_vs_rule": guarded_score - rule_score,
        "stability_delta_vs_learned": guarded_score - learned_score,
    }


def run_router_guard_tuning(
    *,
    config_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    run_id: str = "",
    checkpoint_path: str | Path | None = None,
    checkpoint_id: str = "",
    train_manifest_path: str | Path | None = None,
    dataset_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = _resolve_repo_root()
    cfg = _merged_config(config_path)
    tuning_cfg = cfg.get("guard_tuning") if isinstance(cfg.get("guard_tuning"), dict) else {}
    output_root = (
        (repo_root / str(tuning_cfg.get("artifacts_root") or "docs/artifacts/p56/guard_tuning")).resolve()
        if out_dir is None
        else (Path(out_dir) if Path(out_dir).is_absolute() else (repo_root / out_dir).resolve())
    )
    chosen_run_id = str(run_id or _now_stamp())
    run_dir = output_root / chosen_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_checkpoint, resolved_checkpoint_id, resolved_train_manifest, resolved_dataset_manifest = _resolve_checkpoint_refs(
        checkpoint_path=checkpoint_path,
        checkpoint_id=checkpoint_id,
        train_manifest_path=train_manifest_path,
        dataset_manifest_path=dataset_manifest_path,
    )
    if not resolved_checkpoint:
        raise FileNotFoundError("no learned-router checkpoint available for guard tuning")

    sweep_cfg = tuning_cfg.get("sweep") if isinstance(tuning_cfg.get("sweep"), dict) else {}
    keys = [str(key) for key in sweep_cfg.keys()]
    value_lists = [list(value) if isinstance(value, list) else [value] for value in sweep_cfg.values()]
    candidates = [dict(zip(keys, values, strict=False)) for values in itertools.product(*value_lists)]
    max_candidates = max(1, _safe_int(tuning_cfg.get("max_candidates"), 6))
    candidates = candidates[:max_candidates]
    seed_pools = [dict(item) for item in (tuning_cfg.get("seed_pools") or []) if isinstance(item, dict)]
    objective_cfg = tuning_cfg.get("objective") if isinstance(tuning_cfg.get("objective"), dict) else {}

    base_cfg = _read_yaml_or_json((repo_root / config_path).resolve()) if config_path and not Path(config_path).is_absolute() else (_read_yaml_or_json(Path(config_path)) if config_path else {})
    if not base_cfg:
        base_cfg = {}

    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        candidate_cfg = copy.deepcopy(base_cfg)
        candidate_cfg.setdefault("routing", {})
        candidate_cfg["routing"]["rule_guard"] = dict(candidate)
        if seed_pools:
            candidate_cfg["benchmark"] = dict(candidate_cfg.get("benchmark") or {})
            candidate_cfg["benchmark"]["seed_pools"] = seed_pools
        candidate_config_path = run_dir / "candidate_configs" / f"candidate_{index:02d}.json"
        candidate_config_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_config_path.write_text(json.dumps(candidate_cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        benchmark_summary = run_router_benchmark(
            config_path=candidate_config_path,
            out_dir=run_dir / "benchmark_runs",
            run_id=f"{chosen_run_id}-candidate-{index:02d}",
            quick=True,
            checkpoint_path=resolved_checkpoint,
            checkpoint_id=resolved_checkpoint_id,
            train_manifest_path=resolved_train_manifest or None,
            dataset_manifest_path=resolved_dataset_manifest or None,
        )
        benchmark_payload = _read_json(Path(str(benchmark_summary.get("benchmark_summary_json") or "")))
        benchmark_payload = benchmark_payload if isinstance(benchmark_payload, dict) else {}
        objective_score, metrics = _candidate_score(benchmark_payload, objective_cfg)
        rows.append(
            {
                "candidate_index": index,
                "guard_config": candidate,
                "objective_score": objective_score,
                "metrics": metrics,
                "benchmark_summary_json": str(benchmark_summary.get("benchmark_summary_json") or ""),
            }
        )

    preferred_rows = [row for row in rows if bool(((row.get("metrics") or {}) if isinstance(row.get("metrics"), dict) else {}).get("preferred"))]
    ranking = sorted(preferred_rows or rows, key=lambda row: (-_safe_float(row.get("objective_score"), 0.0), int(row.get("candidate_index") or 0)))
    recommended = ranking[0] if ranking else {}
    recommended_guard_config = dict(recommended.get("guard_config") or {}) if isinstance(recommended, dict) else {}
    write_json(
        run_dir / "recommended_guard_config.json",
        {
            "schema": "p56_recommended_guard_config_v1",
            "generated_at": _now_iso(),
            "checkpoint_id": resolved_checkpoint_id,
            "guard_config": recommended_guard_config,
            "selection_metrics": recommended.get("metrics") if isinstance(recommended.get("metrics"), dict) else {},
            "objective_score": _safe_float(recommended.get("objective_score"), 0.0),
        },
    )
    payload = {
        "schema": "p56_guard_tuning_results_v1",
        "generated_at": _now_iso(),
        "run_id": chosen_run_id,
        "checkpoint_path": resolved_checkpoint,
        "checkpoint_id": resolved_checkpoint_id,
        "results": rows,
        "recommended_guard_config": recommended_guard_config,
        "recommended_benchmark_summary_json": str(recommended.get("benchmark_summary_json") or ""),
    }
    write_json(run_dir / "guard_tuning_results.json", payload)
    write_markdown(
        run_dir / "guard_tuning_results.md",
        [
            "# P56 Guard Threshold Tuning",
            "",
            f"- checkpoint_id: `{resolved_checkpoint_id}`",
            f"- candidate_count: {len(rows)}",
            f"- recommended_guard_config: `{json.dumps(recommended_guard_config, ensure_ascii=False)}`",
            "",
            "## Candidates",
            *[
                "- idx={idx} objective={objective:.4f} preferred={preferred} guarded_score={guarded:.3f} score_delta_vs_rule={delta:.3f} guard_trigger_rate={guard:.3f} config={config}".format(
                    idx=int(row.get("candidate_index") or 0),
                    objective=_safe_float(row.get("objective_score"), 0.0),
                    preferred=str(bool(((row.get("metrics") or {}) if isinstance(row.get("metrics"), dict) else {}).get("preferred"))).lower(),
                    guarded=_safe_float((((row.get("metrics") or {}) if isinstance(row.get("metrics"), dict) else {}).get("guarded_score")), 0.0),
                    delta=_safe_float((((row.get("metrics") or {}) if isinstance(row.get("metrics"), dict) else {}).get("score_delta_vs_rule")), 0.0),
                    guard=_safe_float((((row.get("metrics") or {}) if isinstance(row.get("metrics"), dict) else {}).get("guard_trigger_rate")), 0.0),
                    config=json.dumps(row.get("guard_config") or {}, ensure_ascii=False),
                )
                for row in rows
            ],
        ],
    )
    if resolved_checkpoint_id:
        update_checkpoint(
            resolved_checkpoint_id,
            {
                "guard_tuning_ref": str((run_dir / "guard_tuning_results.json").resolve()),
            },
        )
    return {
        "status": "ok",
        "run_id": chosen_run_id,
        "run_dir": str(run_dir.resolve()),
        "checkpoint_id": resolved_checkpoint_id,
        "guard_tuning_results_json": str((run_dir / "guard_tuning_results.json").resolve()),
        "guard_tuning_results_md": str((run_dir / "guard_tuning_results.md").resolve()),
        "recommended_guard_config_json": str((run_dir / "recommended_guard_config.json").resolve()),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P56 guard threshold tuning")
    parser.add_argument("--config", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--checkpoint-id", default="")
    parser.add_argument("--train-manifest", default="")
    parser.add_argument("--dataset-manifest", default="")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    summary = run_router_guard_tuning(
        config_path=(str(args.config).strip() or None),
        out_dir=(str(args.out_dir).strip() or None),
        run_id=str(args.run_id or ""),
        checkpoint_path=(str(args.checkpoint_path).strip() or None),
        checkpoint_id=str(args.checkpoint_id or ""),
        train_manifest_path=(str(args.train_manifest).strip() or None),
        dataset_manifest_path=(str(args.dataset_manifest).strip() or None),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if str(summary.get("status") or "") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
