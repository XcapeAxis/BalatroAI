from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from trainer.closed_loop.regression_triage import run_regression_triage
from trainer.closed_loop.replay_manifest import write_json, write_markdown
from trainer.hybrid.controller_registry import build_controller_registry, discover_policy_model_path, discover_world_model_checkpoint
from trainer.hybrid.hybrid_controller import AdaptiveHybridController
from trainer.hybrid.learned_router_train import run_learned_router_train
from trainer.hybrid.router_dataset import build_router_dataset
from trainer.hybrid.routing_features import collect_sample_states
from trainer.registry.checkpoint_registry import list_entries, snapshot_registry, update_checkpoint_status
from trainer.registry.promotion_queue import build_promotion_queue_summary


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


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _resolve_repo_root() -> Path:
    preferred = Path("D:/MYFILES/BalatroAI")
    if preferred.exists():
        return preferred.resolve()
    return Path(__file__).resolve().parents[2]


def _default_eval_config() -> dict[str, Any]:
    return {
        "schema": "p52_learned_router_eval_config_v1",
        "controllers": {
            "policy_model_path": "",
            "world_model_checkpoint": "",
            "search": {"max_branch": 80, "max_depth": 2, "time_budget_ms": 15.0},
            "wm_rerank": {"top_k": 4, "horizon": 1, "uncertainty_penalty": 0.55},
        },
        "routing": {
            "router": {
                "policy_margin_high": 0.20,
                "policy_margin_low": 0.08,
                "policy_entropy_high": 1.25,
                "wm_uncertainty_prefer_max": 0.80,
                "wm_uncertainty_disable": 1.20,
                "search_budget_min_level": "medium",
                "high_risk_search_bonus": 0.75,
                "late_stage_search_bonus": 0.50,
                "wm_reward_bonus_scale": 0.20,
            },
            "rule_guard": {
                "min_confidence": 0.45,
                "high_risk_min_confidence": 0.60,
                "min_feature_completeness": 0.80,
                "max_ood_score": 6.0,
                "max_wm_uncertainty": 1.0,
            },
            "learned_router": {"temperature": 1.0, "device": "auto"},
        },
        "dataset": {"quick": True},
        "train": {"quick": True},
        "inference_smoke": {"seed": "AAAAAAA", "max_states": 8},
        "arena_compare": {
            "backend": "sim",
            "mode": "long_episode",
            "episodes_per_seed": 1,
            "max_steps": 120,
            "timeout_sec": 3600,
            "policies": [
                "policy_baseline",
                "policy_plus_wm_rerank",
                "hybrid_controller_rule",
                "hybrid_controller_learned",
                "hybrid_controller_learned_with_rule_guard",
                "search_baseline",
                "heuristic_baseline",
            ],
            "champion_policy": "hybrid_controller_rule",
            "candidate_policy": "hybrid_controller_learned_with_rule_guard",
        },
        "triage": {"output_artifacts_root": "docs/artifacts/p52/triage"},
        "output": {
            "router_inference_root": "docs/artifacts/p52/router_inference",
            "arena_ablation_root": "docs/artifacts/p52/arena_ablation",
        },
    }


def _merged_config(path: str | Path | None) -> dict[str, Any]:
    payload = _default_eval_config()
    if not path:
        return payload
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = (_resolve_repo_root() / cfg_path).resolve()
    override = _read_yaml_or_json(cfg_path)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            merged = dict(payload.get(key) or {})
            merged.update(value)
            payload[key] = merged
        else:
            payload[key] = value
    return payload


def _dataset_artifacts_root(repo_root: Path, cfg: dict[str, Any]) -> Path:
    dataset_cfg = cfg.get("dataset") if isinstance(cfg.get("dataset"), dict) else {}
    token = str(dataset_cfg.get("artifacts_root") or "docs/artifacts/p52/router_dataset")
    return (repo_root / token).resolve()


def _train_artifacts_root(repo_root: Path, cfg: dict[str, Any]) -> Path:
    train_cfg = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
    token = str(train_cfg.get("artifacts_root") or "docs/artifacts/p52/router_train")
    return (repo_root / token).resolve()


def _routing_trace_summary(rows: list[dict[str, Any]], *, policy_id: str) -> dict[str, Any]:
    selection_counter: Counter[str] = Counter()
    invalid_incidents = 0
    guard_count = 0
    fallback_count = 0
    for row in rows:
        selection_counter[str(row.get("selected_controller") or "unknown")] += 1
        if bool(row.get("guard_triggered")):
            guard_count += 1
        if bool(row.get("fallback_used")):
            fallback_count += 1
        if str(row.get("selected_controller") or "") not in {
            "policy_baseline",
            "policy_plus_wm_rerank",
            "search_baseline",
            "heuristic_baseline",
        }:
            invalid_incidents += 1
    total = sum(selection_counter.values())
    return {
        "policy_id": policy_id,
        "decision_count": total,
        "controller_selection_distribution": [
            {"controller_id": controller_id, "count": int(count), "ratio": float(count) / max(1, total)}
            for controller_id, count in sorted(selection_counter.items(), key=lambda item: (-item[1], item[0]))
        ],
        "guard_trigger_rate": float(guard_count) / max(1, total),
        "fallback_rate": float(fallback_count) / max(1, total),
        "invalid_routing_incidents": int(invalid_incidents),
    }


def run_router_inference_smoke(
    *,
    checkpoint_path: str,
    config_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    run_id: str = "",
    seed: str = "AAAAAAA",
    max_states: int = 8,
) -> dict[str, Any]:
    repo_root = _resolve_repo_root()
    cfg = _merged_config(config_path)
    controllers_cfg = cfg.get("controllers") if isinstance(cfg.get("controllers"), dict) else {}
    routing_cfg = cfg.get("routing") if isinstance(cfg.get("routing"), dict) else {}
    router_cfg = routing_cfg.get("router") if isinstance(routing_cfg.get("router"), dict) else {}
    learned_cfg = routing_cfg.get("learned_router") if isinstance(routing_cfg.get("learned_router"), dict) else {}
    rule_guard_cfg = routing_cfg.get("rule_guard") if isinstance(routing_cfg.get("rule_guard"), dict) else {}
    output_cfg = cfg.get("output") if isinstance(cfg.get("output"), dict) else {}

    chosen_run_id = str(run_id or _now_stamp())
    output_root = (
        (repo_root / str(output_cfg.get("router_inference_root") or "docs/artifacts/p52/router_inference")).resolve()
        if out_dir is None
        else (Path(out_dir) if Path(out_dir).is_absolute() else (repo_root / out_dir).resolve())
    )
    run_dir = output_root / chosen_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    combined_trace_path = run_dir / "routing_trace.jsonl"

    model_path = str(controllers_cfg.get("policy_model_path") or discover_policy_model_path(repo_root))
    wm_checkpoint = str(controllers_cfg.get("world_model_checkpoint") or discover_world_model_checkpoint(repo_root))
    search_cfg = controllers_cfg.get("search") if isinstance(controllers_cfg.get("search"), dict) else {}
    wm_cfg = controllers_cfg.get("wm_rerank") if isinstance(controllers_cfg.get("wm_rerank"), dict) else {}
    states = collect_sample_states(seed=seed, max_states=max_states, max_steps=max_states * 2)

    mode_summaries: list[dict[str, Any]] = []
    for mode in ("rule", "learned", "learned_with_rule_guard"):
        trace_path = run_dir / f"routing_trace_{mode}.jsonl"
        controller = AdaptiveHybridController(
            name=f"hybrid_controller_{mode}",
            model_path=model_path,
            world_model_checkpoint=wm_checkpoint,
            top_k=max(2, _safe_int(wm_cfg.get("top_k"), 4)),
            router_config=router_cfg,
            router_mode=mode,
            learned_router_checkpoint=checkpoint_path,
            learned_router_config=learned_cfg,
            rule_guard_config=rule_guard_cfg,
            search_max_branch=_safe_int(search_cfg.get("max_branch"), 80),
            search_max_depth=_safe_int(search_cfg.get("max_depth"), 2),
            search_time_budget_ms=_safe_float(search_cfg.get("time_budget_ms"), 15.0),
            wm_horizon=_safe_int(wm_cfg.get("horizon"), 1),
            wm_uncertainty_penalty=_safe_float(wm_cfg.get("uncertainty_penalty"), 0.55),
            trace_path=str(trace_path),
            trace_context={"pipeline": "p52_router_inference_smoke", "run_id": chosen_run_id, "compare_mode": mode},
        )
        controller.reset(seed)
        try:
            for sample in states:
                obs = sample.get("state") if isinstance(sample.get("state"), dict) else {}
                controller.act(obs)
                trace_row = dict(controller.last_trace or {})
                trace_row["compare_mode"] = mode
                _append_jsonl(combined_trace_path, trace_row)
        finally:
            controller.close()
        mode_rows = _read_jsonl(trace_path)
        mode_summaries.append(
            {
                "policy_id": f"hybrid_controller_{mode}",
                "router_mode": mode,
                "trace_path": str(trace_path.resolve()),
                **_routing_trace_summary(mode_rows, policy_id=f"hybrid_controller_{mode}"),
            }
        )

    payload = {
        "schema": "p52_router_inference_smoke_v1",
        "generated_at": _now_iso(),
        "run_id": chosen_run_id,
        "seed": seed,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "trace_path": str(combined_trace_path.resolve()),
        "mode_summaries": mode_summaries,
    }
    write_json(run_dir / "routing_summary.json", payload)
    write_markdown(
        run_dir / "routing_summary.md",
        [
            "# P52 Router Inference Smoke",
            "",
            f"- run_id: `{chosen_run_id}`",
            f"- seed: `{seed}`",
            f"- checkpoint_path: `{Path(checkpoint_path).resolve()}`",
            "",
            "## Mode Summaries",
            *[
                "- {mode}: decisions={count} guard_trigger_rate={guard:.3f} invalid={invalid}".format(
                    mode=str(row.get("router_mode") or ""),
                    count=int(row.get("decision_count") or 0),
                    guard=float(row.get("guard_trigger_rate") or 0.0),
                    invalid=int(row.get("invalid_routing_incidents") or 0),
                )
                for row in mode_summaries
            ],
        ],
    )
    return {
        "status": "ok",
        "run_id": chosen_run_id,
        "run_dir": str(run_dir),
        "routing_trace_jsonl": str(combined_trace_path.resolve()),
        "routing_summary_json": str((run_dir / "routing_summary.json").resolve()),
        "routing_summary_md": str((run_dir / "routing_summary.md").resolve()),
        "mode_summaries": mode_summaries,
    }


def _run_process(command: list[str], *, cwd: Path, timeout_sec: int) -> dict[str, Any]:
    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, timeout=max(60, int(timeout_sec)))
    return {
        "returncode": int(proc.returncode),
        "stdout": str(proc.stdout or ""),
        "stderr": str(proc.stderr or ""),
        "command": command,
    }


def _pick_summary_row(rows: list[dict[str, Any]], policy_id: str) -> dict[str, Any]:
    for row in rows:
        if str(row.get("policy_id") or "") == str(policy_id):
            return row
    return {}


def _routing_variants_summary(run_dir: Path) -> dict[str, Any]:
    variant_rows: list[dict[str, Any]] = []
    trace_root = run_dir / "router_traces"
    for trace_path in sorted(trace_root.glob("*.jsonl")):
        policy_id = trace_path.stem
        rows = _read_jsonl(trace_path)
        variant_rows.append(
            {
                "policy_id": policy_id,
                "trace_path": str(trace_path.resolve()),
                **_routing_trace_summary(rows, policy_id=policy_id),
            }
        )
    return {
        "schema": "p52_routing_summary_v1",
        "generated_at": _now_iso(),
        "variants": variant_rows,
    }


def _slice_metric_index(bucket_payload: dict[str, Any], policy_id: str) -> dict[str, dict[str, dict[str, Any]]]:
    policies = bucket_payload.get("policies") if isinstance(bucket_payload.get("policies"), list) else []
    target = next(
        (
            row
            for row in policies
            if isinstance(row, dict) and str(row.get("policy_id") or "") == str(policy_id)
        ),
        {},
    )
    slice_metrics = target.get("slice_metrics") if isinstance(target.get("slice_metrics"), dict) else {}
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for slice_key, rows in slice_metrics.items():
        if not isinstance(rows, list):
            continue
        out[str(slice_key)] = {
            str(row.get("slice_label") or "unknown"): dict(row)
            for row in rows
            if isinstance(row, dict)
        }
    return out


def _build_slice_eval(bucket_payload: dict[str, Any], *, baseline_policy: str, target_policies: list[str]) -> dict[str, Any]:
    baseline_index = _slice_metric_index(bucket_payload, baseline_policy)
    comparisons: list[dict[str, Any]] = []
    for target_policy in target_policies:
        target_index = _slice_metric_index(bucket_payload, target_policy)
        for slice_key, labels in target_index.items():
            baseline_labels = baseline_index.get(slice_key) or {}
            for slice_label, metrics in labels.items():
                baseline_metrics = baseline_labels.get(slice_label) or {}
                if not baseline_metrics:
                    continue
                comparisons.append(
                    {
                        "baseline_policy": baseline_policy,
                        "target_policy": target_policy,
                        "slice_key": slice_key,
                        "slice_label": slice_label,
                        "count": min(_safe_int(metrics.get("count"), 0), _safe_int(baseline_metrics.get("count"), 0)),
                        "score_delta": _safe_float(metrics.get("mean_total_score"), 0.0)
                        - _safe_float(baseline_metrics.get("mean_total_score"), 0.0),
                        "win_rate_delta": _safe_float(metrics.get("win_rate"), 0.0)
                        - _safe_float(baseline_metrics.get("win_rate"), 0.0),
                        "target_mean_total_score": _safe_float(metrics.get("mean_total_score"), 0.0),
                        "baseline_mean_total_score": _safe_float(baseline_metrics.get("mean_total_score"), 0.0),
                    }
                )
    comparisons.sort(
        key=lambda row: (
            str(row.get("target_policy") or ""),
            float(row.get("score_delta") or 0.0),
            float(row.get("win_rate_delta") or 0.0),
            -int(row.get("count") or 0),
            str(row.get("slice_key") or ""),
            str(row.get("slice_label") or ""),
        )
    )
    return {
        "schema": "p52_learned_router_slice_eval_v1",
        "generated_at": _now_iso(),
        "baseline_policy": baseline_policy,
        "target_policies": list(target_policies),
        "comparisons": comparisons,
    }


def _promotion_payload(summary_rows: list[dict[str, Any]], *, checkpoint_id: str) -> dict[str, Any]:
    rule_row = _pick_summary_row(summary_rows, "hybrid_controller_rule")
    guarded_row = _pick_summary_row(summary_rows, "hybrid_controller_learned_with_rule_guard")
    learned_row = _pick_summary_row(summary_rows, "hybrid_controller_learned")
    rule_score = _safe_float(rule_row.get("mean_total_score"), 0.0)
    guarded_score = _safe_float(guarded_row.get("mean_total_score"), 0.0)
    learned_score = _safe_float(learned_row.get("mean_total_score"), 0.0)
    recommend_review = guarded_score >= (rule_score - 1e-6)
    reasons = []
    if guarded_score >= rule_score:
        reasons.append("guarded router matches or exceeds rule router mean_total_score")
    else:
        reasons.append("guarded router does not exceed rule router mean_total_score")
    if learned_score < rule_score:
        reasons.append("unguarded learned router underperforms rule router")
    return {
        "schema": "p52_learned_router_promotion_decision_v1",
        "generated_at": _now_iso(),
        "checkpoint_id": checkpoint_id,
        "candidate_policy": "hybrid_controller_learned_with_rule_guard",
        "champion_policy": "hybrid_controller_rule",
        "candidate_score": guarded_score,
        "champion_score": rule_score,
        "score_delta": guarded_score - rule_score,
        "recommendation": ("promotion_review" if recommend_review else "observe"),
        "recommend_promotion": bool(recommend_review),
        "reasons": reasons,
    }


def run_learned_router_ablation(
    *,
    config_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    run_id: str = "",
    quick: bool = False,
    seeds_override: list[str] | None = None,
    dataset_manifest_path: str | Path | None = None,
    train_manifest_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_id_override: str = "",
) -> dict[str, Any]:
    repo_root = _resolve_repo_root()
    cfg = _merged_config(config_path)
    controllers_cfg = cfg.get("controllers") if isinstance(cfg.get("controllers"), dict) else {}
    routing_cfg = cfg.get("routing") if isinstance(cfg.get("routing"), dict) else {}
    router_cfg = routing_cfg.get("router") if isinstance(routing_cfg.get("router"), dict) else {}
    learned_cfg = routing_cfg.get("learned_router") if isinstance(routing_cfg.get("learned_router"), dict) else {}
    rule_guard_cfg = routing_cfg.get("rule_guard") if isinstance(routing_cfg.get("rule_guard"), dict) else {}
    arena_cfg = cfg.get("arena_compare") if isinstance(cfg.get("arena_compare"), dict) else {}
    triage_cfg = cfg.get("triage") if isinstance(cfg.get("triage"), dict) else {}
    output_cfg = cfg.get("output") if isinstance(cfg.get("output"), dict) else {}

    chosen_run_id = str(run_id or _now_stamp())
    output_root = (
        (repo_root / str(output_cfg.get("arena_ablation_root") or "docs/artifacts/p52/arena_ablation")).resolve()
        if out_dir is None
        else (Path(out_dir) if Path(out_dir).is_absolute() else (repo_root / out_dir).resolve())
    )
    run_dir = output_root / chosen_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_summary: dict[str, Any]
    if dataset_manifest_path:
        dataset_manifest = Path(dataset_manifest_path)
        if not dataset_manifest.is_absolute():
            dataset_manifest = (repo_root / dataset_manifest).resolve()
        dataset_summary = {
            "status": "ok",
            "dataset_manifest_json": str(dataset_manifest.resolve()),
        }
    else:
        dataset_summary = build_router_dataset(
            config_path=config_path,
            out_dir=_dataset_artifacts_root(repo_root, cfg),
            quick=quick,
        )

    train_summary: dict[str, Any]
    if train_manifest_path or checkpoint_path:
        train_manifest = {}
        if train_manifest_path:
            manifest_path = Path(train_manifest_path)
            if not manifest_path.is_absolute():
                manifest_path = (repo_root / manifest_path).resolve()
            train_manifest = _read_json(manifest_path) if manifest_path.exists() else {}
            if not isinstance(train_manifest, dict):
                train_manifest = {}
        resolved_checkpoint = Path(str(checkpoint_path or train_manifest.get("best_checkpoint") or ""))
        if not resolved_checkpoint.is_absolute():
            resolved_checkpoint = (repo_root / resolved_checkpoint).resolve()
        if not resolved_checkpoint.exists():
            raise FileNotFoundError(f"learned router checkpoint not found: {resolved_checkpoint}")
        train_summary = {
            "status": "ok",
            "train_manifest_json": str(train_manifest_path or train_manifest.get("train_manifest_json") or ""),
            "best_checkpoint": str(resolved_checkpoint.resolve()),
            "checkpoint_id": str(checkpoint_id_override or train_manifest.get("checkpoint_id") or ""),
        }
    else:
        train_summary = run_learned_router_train(
            config_path=config_path,
            out_dir=_train_artifacts_root(repo_root, cfg),
            run_id=f"{chosen_run_id}-train",
            quick=quick,
            dataset_manifest_path=str(dataset_summary.get("dataset_manifest_json") or ""),
        )
    checkpoint_id = str(checkpoint_id_override or train_summary.get("checkpoint_id") or "")
    if checkpoint_id:
        update_checkpoint_status(
            checkpoint_id,
            to_status="smoke_passed",
            reason="p52_learned_router_train_completed",
            operator="p52_learned_router_eval",
            refs={"metrics_ref": str(train_summary.get("metrics_json") or "")},
        )

    inference_summary = run_router_inference_smoke(
        checkpoint_path=str(train_summary.get("best_checkpoint") or ""),
        config_path=config_path,
        out_dir=repo_root / "docs" / "artifacts" / "p52" / "router_inference",
        run_id=f"{chosen_run_id}-inference",
        seed=str((cfg.get("inference_smoke") or {}).get("seed") or "AAAAAAA"),
        max_states=max(4, _safe_int((cfg.get("inference_smoke") or {}).get("max_states"), 8)),
    )

    model_path = str(controllers_cfg.get("policy_model_path") or discover_policy_model_path(repo_root))
    wm_checkpoint = str(controllers_cfg.get("world_model_checkpoint") or discover_world_model_checkpoint(repo_root))
    search_cfg = controllers_cfg.get("search") if isinstance(controllers_cfg.get("search"), dict) else {}
    wm_cfg = controllers_cfg.get("wm_rerank") if isinstance(controllers_cfg.get("wm_rerank"), dict) else {}
    seeds = list(seeds_override or ["AAAAAAA", "BBBBBBB"])
    if quick and len(seeds) > 2:
        seeds = seeds[:2]

    router_trace_root = run_dir / "router_traces"
    router_trace_root.mkdir(parents=True, exist_ok=True)
    policy_assist_map = {
        "policy_plus_wm_rerank": {
            "base_policy": "policy_baseline",
            "candidate_source": "policy_topk",
            "model_path": model_path,
            "world_model_checkpoint": wm_checkpoint,
            "top_k": max(2, _safe_int(wm_cfg.get("top_k"), 4)),
            "horizon": _safe_int(wm_cfg.get("horizon"), 1),
            "uncertainty_penalty": _safe_float(wm_cfg.get("uncertainty_penalty"), 0.55),
            "search_max_branch": _safe_int(search_cfg.get("max_branch"), 80),
            "search_max_depth": _safe_int(search_cfg.get("max_depth"), 2),
            "search_time_budget_ms": _safe_float(search_cfg.get("time_budget_ms"), 15.0),
        },
        "hybrid_controller_rule": {
            "model_path": model_path,
            "world_model_checkpoint": wm_checkpoint,
            "router_mode": "rule",
            "router_config": router_cfg,
            "search_max_branch": _safe_int(search_cfg.get("max_branch"), 80),
            "search_max_depth": _safe_int(search_cfg.get("max_depth"), 2),
            "search_time_budget_ms": _safe_float(search_cfg.get("time_budget_ms"), 15.0),
            "wm_horizon": _safe_int(wm_cfg.get("horizon"), 1),
            "wm_uncertainty_penalty": _safe_float(wm_cfg.get("uncertainty_penalty"), 0.55),
            "trace_path": str((router_trace_root / "hybrid_controller_rule.jsonl").resolve()),
            "trace_context": {"pipeline": "p52", "run_id": chosen_run_id, "router_mode": "rule"},
        },
        "hybrid_controller_learned": {
            "model_path": model_path,
            "world_model_checkpoint": wm_checkpoint,
            "router_mode": "learned",
            "router_config": router_cfg,
            "learned_router_checkpoint": str(train_summary.get("best_checkpoint") or ""),
            "learned_router_config": learned_cfg,
            "rule_guard_config": rule_guard_cfg,
            "search_max_branch": _safe_int(search_cfg.get("max_branch"), 80),
            "search_max_depth": _safe_int(search_cfg.get("max_depth"), 2),
            "search_time_budget_ms": _safe_float(search_cfg.get("time_budget_ms"), 15.0),
            "wm_horizon": _safe_int(wm_cfg.get("horizon"), 1),
            "wm_uncertainty_penalty": _safe_float(wm_cfg.get("uncertainty_penalty"), 0.55),
            "trace_path": str((router_trace_root / "hybrid_controller_learned.jsonl").resolve()),
            "trace_context": {"pipeline": "p52", "run_id": chosen_run_id, "router_mode": "learned"},
        },
        "hybrid_controller_learned_with_rule_guard": {
            "model_path": model_path,
            "world_model_checkpoint": wm_checkpoint,
            "router_mode": "learned_with_rule_guard",
            "router_config": router_cfg,
            "learned_router_checkpoint": str(train_summary.get("best_checkpoint") or ""),
            "learned_router_config": learned_cfg,
            "rule_guard_config": rule_guard_cfg,
            "search_max_branch": _safe_int(search_cfg.get("max_branch"), 80),
            "search_max_depth": _safe_int(search_cfg.get("max_depth"), 2),
            "search_time_budget_ms": _safe_float(search_cfg.get("time_budget_ms"), 15.0),
            "wm_horizon": _safe_int(wm_cfg.get("horizon"), 1),
            "wm_uncertainty_penalty": _safe_float(wm_cfg.get("uncertainty_penalty"), 0.55),
            "trace_path": str((router_trace_root / "hybrid_controller_learned_with_rule_guard.jsonl").resolve()),
            "trace_context": {"pipeline": "p52", "run_id": chosen_run_id, "router_mode": "learned_with_rule_guard"},
        },
    }
    write_json(run_dir / "policy_assist_map.json", policy_assist_map)
    write_json(run_dir / "policy_model_map.json", {"policy_baseline": model_path, "policy_plus_wm_rerank": model_path})
    write_json(run_dir / "controller_registry.json", build_controller_registry(repo_root=repo_root, world_model_checkpoint=wm_checkpoint, model_path=model_path))

    policies = [str(item) for item in (arena_cfg.get("policies") or []) if str(item).strip()]
    arena_cmd = [
        sys.executable,
        "-B",
        "-m",
        "trainer.policy_arena.arena_runner",
        "--out-dir",
        str(output_root),
        "--run-id",
        chosen_run_id,
        "--backend",
        str(arena_cfg.get("backend") or "sim"),
        "--mode",
        str(arena_cfg.get("mode") or "long_episode"),
        "--policies",
        ",".join(policies),
        "--policy-assist-map-json",
        str((run_dir / "policy_assist_map.json").resolve()),
        "--policy-model-map-json",
        str((run_dir / "policy_model_map.json").resolve()),
        "--world-model-checkpoint",
        wm_checkpoint,
        "--seeds",
        ",".join(seeds),
        "--episodes-per-seed",
        str(max(1, _safe_int(arena_cfg.get("episodes_per_seed"), 1))),
        "--max-steps",
        str(max(1, _safe_int(arena_cfg.get("max_steps"), 120))),
        "--skip-unavailable",
    ]
    if quick:
        arena_cmd.append("--quick")
    arena_result = _run_process(arena_cmd, cwd=repo_root, timeout_sec=_safe_int(arena_cfg.get("timeout_sec"), 3600))
    if int(arena_result.get("returncode") or 0) != 0:
        raise RuntimeError(f"arena_runner failed: {arena_result.get('stderr') or arena_result.get('stdout')}")

    summary_rows_payload = _read_json(run_dir / "summary_table.json")
    summary_rows = [row for row in summary_rows_payload if isinstance(row, dict)] if isinstance(summary_rows_payload, list) else []
    routing_summary = _routing_variants_summary(run_dir)
    write_json(run_dir / "routing_summary.json", routing_summary)
    bucket_payload = _read_json(run_dir / "bucket_metrics.json")
    bucket_payload = bucket_payload if isinstance(bucket_payload, dict) else {}
    slice_eval_payload = _build_slice_eval(
        bucket_payload,
        baseline_policy="hybrid_controller_rule",
        target_policies=["hybrid_controller_learned", "hybrid_controller_learned_with_rule_guard"],
    )
    write_json(run_dir / "slice_eval.json", slice_eval_payload)

    promotion_payload = _promotion_payload(summary_rows, checkpoint_id=checkpoint_id)
    write_json(run_dir / "promotion_decision.json", promotion_payload)
    write_markdown(
        run_dir / "promotion_decision.md",
        [
            "# P52 Promotion Decision",
            "",
            f"- checkpoint_id: `{checkpoint_id}`",
            f"- candidate_policy: `{promotion_payload.get('candidate_policy')}`",
            f"- champion_policy: `{promotion_payload.get('champion_policy')}`",
            f"- candidate_score: {float(promotion_payload.get('candidate_score') or 0.0):.6f}",
            f"- champion_score: {float(promotion_payload.get('champion_score') or 0.0):.6f}",
            f"- score_delta: {float(promotion_payload.get('score_delta') or 0.0):.6f}",
            f"- recommendation: `{promotion_payload.get('recommendation')}`",
            "",
            "## Reasons",
            *[f"- {str(reason)}" for reason in (promotion_payload.get("reasons") or [])],
        ],
    )

    run_manifest = _read_json(run_dir / "run_manifest.json")
    run_manifest = run_manifest if isinstance(run_manifest, dict) else {}
    run_manifest["learned_router"] = {
        "checkpoint_path": str(train_summary.get("best_checkpoint") or ""),
        "checkpoint_id": checkpoint_id,
        "dataset_manifest_json": str(dataset_summary.get("dataset_manifest_json") or ""),
        "train_manifest_json": str(train_summary.get("train_manifest_json") or ""),
        "routing_summary_json": str((run_dir / "routing_summary.json").resolve()),
        "router_inference_summary_json": str(inference_summary.get("routing_summary_json") or ""),
        "controller_registry_json": str((run_dir / "controller_registry.json").resolve()),
        "guarded_policy": "hybrid_controller_learned_with_rule_guard",
        "rule_policy": "hybrid_controller_rule",
    }
    write_json(run_dir / "run_manifest.json", run_manifest)

    triage_dir = (repo_root / str(triage_cfg.get("output_artifacts_root") or "docs/artifacts/p52/triage")).resolve() / chosen_run_id
    triage_summary = run_regression_triage(current_run_dir=run_dir, out_dir=triage_dir)
    if checkpoint_id:
        update_checkpoint_status(
            checkpoint_id,
            to_status="arena_passed",
            reason="p52_arena_ablation_completed",
            operator="p52_learned_router_eval",
            refs={
                "arena_ref": str((run_dir / "summary_table.json").resolve()),
                "triage_ref": str(triage_summary.get("triage_report_json") or ""),
            },
        )
        if bool(promotion_payload.get("recommend_promotion")):
            update_checkpoint_status(
                checkpoint_id,
                to_status="promotion_review",
                reason="p52_promotion_decision_ready",
                operator="p52_learned_router_eval",
                refs={
                    "arena_ref": str((run_dir / "summary_table.json").resolve()),
                    "triage_ref": str(triage_summary.get("triage_report_json") or ""),
                    "promotion_decision": str((run_dir / "promotion_decision.json").resolve()),
                },
            )

    registry_snapshot_path = run_dir / "checkpoint_registry_snapshot.json"
    promotion_queue_path = run_dir / "promotion_queue.json"
    snapshot_registry(out_path=registry_snapshot_path)
    write_json(promotion_queue_path, build_promotion_queue_summary(list_entries()))

    return {
        "status": "ok",
        "run_id": chosen_run_id,
        "run_dir": str(run_dir),
        "dataset_manifest_json": str(dataset_summary.get("dataset_manifest_json") or ""),
        "train_manifest_json": str(train_summary.get("train_manifest_json") or ""),
        "router_checkpoint": str(train_summary.get("best_checkpoint") or ""),
        "router_checkpoint_id": checkpoint_id,
        "summary_table_json": str((run_dir / "summary_table.json").resolve()),
        "slice_eval_json": str((run_dir / "slice_eval.json").resolve()),
        "routing_summary_json": str((run_dir / "routing_summary.json").resolve()),
        "promotion_decision_json": str((run_dir / "promotion_decision.json").resolve()),
        "triage_report_json": str(triage_summary.get("triage_report_json") or ""),
        "registry_snapshot_path": str(registry_snapshot_path.resolve()),
        "promotion_queue_path": str(promotion_queue_path.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P52 learned-router inference smoke and arena ablation.")
    parser.add_argument("--config", default="")
    parser.add_argument("--mode", default="ablation", choices=["inference_smoke", "ablation"])
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--max-states", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "inference_smoke":
        if not str(args.checkpoint or "").strip():
            raise SystemExit("--checkpoint is required for inference_smoke")
        summary = run_router_inference_smoke(
            checkpoint_path=str(args.checkpoint),
            config_path=(str(args.config).strip() or None),
            out_dir=(str(args.out_dir).strip() or None),
            run_id=str(args.run_id or ""),
            seed=str(args.seed or "AAAAAAA"),
            max_states=max(1, int(args.max_states)),
        )
    else:
        summary = run_learned_router_ablation(
            config_path=(str(args.config).strip() or None),
            out_dir=(str(args.out_dir).strip() or None),
            run_id=str(args.run_id or ""),
            quick=bool(args.quick),
        )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if str(summary.get("status") or "") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
