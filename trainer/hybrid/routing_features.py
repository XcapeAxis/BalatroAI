from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sim.pybind.sim_env import SimEnvBackend
from trainer.closed_loop.replay_manifest import write_json
from trainer.common.slices import compute_slice_labels
from trainer.hybrid.controller_registry import build_controller_registry, discover_policy_model_path, discover_world_model_checkpoint
from trainer.policy_arena.adapters.heuristic_adapter import HeuristicAdapter
from trainer.policy_arena.policy_adapter import phase_default_action, phase_from_obs
from trainer.world_model.candidate_actions import generate_candidate_actions


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


def _softmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    finite = [score for score in scores if math.isfinite(score)]
    if not finite:
        return [1.0 / len(scores)] * len(scores)
    pivot = max(finite)
    exps = [math.exp(max(-50.0, min(50.0, score - pivot))) if math.isfinite(score) else 0.0 for score in scores]
    denom = sum(exps)
    if denom <= 0.0:
        return [1.0 / len(scores)] * len(scores)
    return [value / denom for value in exps]


def _entropy(probs: list[float]) -> float:
    total = 0.0
    for prob in probs:
        if prob > 0.0:
            total -= prob * math.log(prob)
    return total


def _budget_level(search_time_budget_ms: float, search_max_depth: int) -> str:
    if float(search_time_budget_ms) >= 20.0 or int(search_max_depth) >= 3:
        return "high"
    if float(search_time_budget_ms) >= 10.0 or int(search_max_depth) >= 2:
        return "medium"
    return "low"


def _candidate_action_type(candidate_rows: list[dict[str, Any]]) -> str:
    for row in candidate_rows:
        if not isinstance(row, dict):
            continue
        action = row.get("action") if isinstance(row.get("action"), dict) else {}
        token = str(action.get("action_type") or "").upper()
        if token:
            return token
    return "WAIT"


def collect_sample_states(
    *,
    seed: str = "AAAAAAA",
    max_states: int = 8,
    max_steps: int = 18,
) -> list[dict[str, Any]]:
    backend = SimEnvBackend(seed=seed)
    helper = HeuristicAdapter(name="p48_feature_sampler")
    helper.reset(seed)
    state = backend.reset(seed=seed)
    rows: list[dict[str, Any]] = []
    try:
        for step_idx in range(max(1, int(max_steps))):
            rows.append(
                {
                    "sample_id": f"{seed}-{step_idx:03d}",
                    "step_idx": int(step_idx),
                    "phase": phase_from_obs(state),
                    "state": state,
                }
            )
            if len(rows) >= max(1, int(max_states)):
                break
            try:
                action = helper.act(state)
            except Exception:
                action = phase_default_action(state, seed=seed)
            state, _reward, done, _info = backend.step(action)
            if bool(done):
                break
    finally:
        helper.close()
    return rows


def extract_routing_features(
    *,
    obs: dict[str, Any],
    registry_payload: dict[str, Any] | None = None,
    planner: Any = None,
    legal_actions: list[dict[str, Any]] | None = None,
    seed: str = "AAAAAAA",
    top_k: int = 4,
    model_path: str = "",
    search_max_branch: int = 80,
    search_max_depth: int = 2,
    search_time_budget_ms: float = 15.0,
) -> dict[str, Any]:
    registry = registry_payload or build_controller_registry(model_path=model_path)
    controller_rows = registry.get("controllers") if isinstance(registry.get("controllers"), list) else []
    controller_map = {
        str(row.get("controller_id") or ""): row for row in controller_rows if isinstance(row, dict) and str(row.get("controller_id") or "")
    }

    policy_candidates = generate_candidate_actions(
        obs=obs,
        source="policy_topk",
        top_k=max(2, int(top_k)),
        legal_actions=legal_actions,
        seed=seed,
        model_path=model_path,
        search_max_branch=search_max_branch,
        search_max_depth=search_max_depth,
        search_time_budget_ms=search_time_budget_ms,
    )
    policy_rows = policy_candidates.get("candidates") if isinstance(policy_candidates.get("candidates"), list) else []
    policy_source = str(policy_candidates.get("source") or "")
    policy_row_source = str(((policy_rows[0] if policy_rows else {}) or {}).get("source") or policy_source or "unknown")
    probabilities: list[float] = []
    for row in policy_rows:
        if not isinstance(row, dict):
            continue
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        value = metadata.get("probability")
        if value is not None:
            probabilities.append(max(0.0, _safe_float(value, 0.0)))
    if len(probabilities) != len(policy_rows):
        probabilities = _softmax([_safe_float((row or {}).get("source_score"), 0.0) for row in policy_rows])
    total_prob = sum(probabilities)
    if total_prob > 0.0:
        probabilities = [value / total_prob for value in probabilities]

    policy_margin = 0.0
    if len(probabilities) >= 2:
        policy_margin = float(probabilities[0] - probabilities[1])
    elif len(probabilities) == 1:
        policy_margin = float(probabilities[0])

    policy_entropy = _entropy(probabilities)
    policy_top1_prob = float(probabilities[0]) if probabilities else 0.0
    policy_top2_prob = float(probabilities[1]) if len(probabilities) > 1 else 0.0
    policy_concentration = float(sum(probabilities[: min(3, len(probabilities))]))

    action_type = _candidate_action_type(policy_rows)
    slice_labels = compute_slice_labels(
        {
            "state": obs,
            "phase": phase_from_obs(obs),
            "action_type": action_type,
        }
    )

    wm_uncertainty = 1.0
    wm_predicted_return = 0.0
    wm_score = 0.0
    wm_probe_candidate_id = ""
    wm_probe_count = 0
    if planner is not None and bool(getattr(planner, "available", False)) and policy_rows:
        probe = planner.rerank_candidates(obs=obs, candidates=policy_rows[: max(1, min(2, len(policy_rows)))])
        ranked = probe.get("ranked_candidates") if isinstance(probe.get("ranked_candidates"), list) else []
        stats = probe.get("planner_stats") if isinstance(probe.get("planner_stats"), dict) else {}
        wm_probe_count = len(ranked)
        if ranked:
            best = ranked[0]
            wm_uncertainty = _safe_float(best.get("uncertainty_score"), _safe_float(stats.get("average_uncertainty"), 1.0))
            wm_predicted_return = _safe_float(best.get("predicted_return"), 0.0)
            wm_score = _safe_float(best.get("wm_score"), 0.0)
            wm_probe_candidate_id = str(best.get("candidate_id") or best.get("action_token") or "")
        else:
            wm_uncertainty = _safe_float(stats.get("average_uncertainty"), 1.0)

    features = {
        "schema": "p48_routing_features_v1",
        "generated_at": _now_iso(),
        "seed": str(seed),
        "phase": phase_from_obs(obs),
        "policy_available": str(policy_row_source) == "policy_topk" and str((controller_map.get("policy_baseline") or {}).get("status") or "") == "active",
        "heuristic_available": str((controller_map.get("heuristic_baseline") or {}).get("status") or "") == "active",
        "search_available": str((controller_map.get("search_baseline") or {}).get("status") or "") == "active",
        "wm_available": bool(getattr(planner, "available", False)),
        "fallback_available": str((controller_map.get("heuristic_baseline") or {}).get("status") or "") == "active",
        "policy_margin": float(policy_margin),
        "policy_entropy": float(policy_entropy),
        "policy_top1_prob": float(policy_top1_prob),
        "policy_top2_prob": float(policy_top2_prob),
        "policy_topk_concentration": float(policy_concentration),
        "policy_candidate_count": len(policy_rows),
        "policy_primary_source": policy_row_source or "unknown",
        "wm_uncertainty": float(wm_uncertainty),
        "wm_predicted_return": float(wm_predicted_return),
        "wm_score": float(wm_score),
        "wm_probe_candidate_id": wm_probe_candidate_id,
        "wm_probe_count": int(wm_probe_count),
        "slice_stage": str(slice_labels.get("slice_stage") or "unknown"),
        "slice_resource_pressure": str(slice_labels.get("slice_resource_pressure") or "unknown"),
        "slice_action_type": str(slice_labels.get("slice_action_type") or "unknown"),
        "slice_position_sensitive": str(slice_labels.get("slice_position_sensitive") if slice_labels.get("slice_position_sensitive") is not None else "unknown"),
        "slice_stateful_joker_present": str(slice_labels.get("slice_stateful_joker_present") if slice_labels.get("slice_stateful_joker_present") is not None else "unknown"),
        "budget_level": _budget_level(search_time_budget_ms, search_max_depth),
        "search_time_budget_ms": float(search_time_budget_ms),
        "search_max_depth": int(search_max_depth),
        "search_max_branch": int(search_max_branch),
        "legal_action_count": int(len(legal_actions or [])),
        "round_num": _safe_int(obs.get("round_num"), _safe_int((obs.get("round") or {}).get("round_num"), 0) if isinstance(obs.get("round"), dict) else 0),
        "ante_num": _safe_int(obs.get("ante_num"), 0),
        "money": _safe_float(obs.get("money"), 0.0),
    }
    return features


def run_routing_feature_smoke(
    *,
    out_dir: str | Path | None = None,
    seed: str = "AAAAAAA",
    max_states: int = 6,
    top_k: int = 4,
    model_path: str = "",
    world_model_checkpoint: str = "",
    search_max_branch: int = 80,
    search_max_depth: int = 2,
    search_time_budget_ms: float = 15.0,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    output_root = (
        (repo_root / "docs/artifacts/p48").resolve()
        if out_dir is None
        else (Path(out_dir) if Path(out_dir).is_absolute() else (repo_root / out_dir).resolve())
    )
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_model_path = str(model_path or discover_policy_model_path(repo_root))
    resolved_wm_checkpoint = str(world_model_checkpoint or discover_world_model_checkpoint(repo_root))
    registry = build_controller_registry(
        repo_root=repo_root,
        world_model_checkpoint=resolved_wm_checkpoint,
        model_path=resolved_model_path,
    )
    planner = None
    if resolved_wm_checkpoint and Path(resolved_wm_checkpoint).exists():
        try:
            from trainer.world_model.lookahead_planner import WorldModelLookaheadPlanner

            planner = WorldModelLookaheadPlanner(
                checkpoint_path=resolved_wm_checkpoint,
                horizon=1,
                uncertainty_penalty=0.5,
                reward_weight=1.0,
                score_weight=0.5,
                value_weight=0.15,
            )
        except Exception:
            planner = None
    rows: list[dict[str, Any]] = []
    for sample in collect_sample_states(seed=seed, max_states=max_states):
        state = sample.get("state") if isinstance(sample.get("state"), dict) else {}
        features = extract_routing_features(
            obs=state,
            registry_payload=registry,
            planner=planner,
            seed=seed,
            top_k=top_k,
            model_path=resolved_model_path,
            search_max_branch=search_max_branch,
            search_max_depth=search_max_depth,
            search_time_budget_ms=search_time_budget_ms,
        )
        rows.append(
            {
                "sample_id": str(sample.get("sample_id") or ""),
                "step_idx": int(sample.get("step_idx") or 0),
                "phase": str(sample.get("phase") or features.get("phase") or "UNKNOWN"),
                "features": features,
            }
        )

    payload = {
        "schema": "p48_routing_features_smoke_v1",
        "generated_at": _now_iso(),
        "seed": seed,
        "world_model_checkpoint": resolved_wm_checkpoint,
        "policy_model_path": resolved_model_path,
        "sample_count": len(rows),
        "rows": rows,
    }
    out_path = output_root / f"routing_features_smoke_{_now_stamp()}.json"
    write_json(out_path, payload)
    return {"status": "ok", "out": str(out_path), "payload": payload}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P48 routing feature extraction smoke.")
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--max-states", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--world-model-checkpoint", default="")
    parser.add_argument("--search-max-branch", type=int, default=80)
    parser.add_argument("--search-max-depth", type=int, default=2)
    parser.add_argument("--search-time-budget-ms", type=float, default=15.0)
    parser.add_argument("--out-dir", default="docs/artifacts/p48")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_routing_feature_smoke(
        out_dir=args.out_dir,
        seed=str(args.seed or "AAAAAAA"),
        max_states=max(1, int(args.max_states)),
        top_k=max(1, int(args.top_k)),
        model_path=str(args.model_path or ""),
        world_model_checkpoint=str(args.world_model_checkpoint or ""),
        search_max_branch=int(args.search_max_branch),
        search_max_depth=int(args.search_max_depth),
        search_time_budget_ms=float(args.search_time_budget_ms),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
