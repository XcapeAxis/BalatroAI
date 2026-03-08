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

from sim.pybind.sim_env import SimEnvBackend
from trainer import action_space, action_space_shop
from trainer.common.slices import (
    as_legacy_action_bucket,
    as_legacy_ante_bucket,
    as_legacy_risk_bucket,
    compute_slice_labels,
)
from trainer.registry.checkpoint_registry import find_by_artifact_path
from trainer.policy_arena.adapters import (
    HeuristicAdapter,
    HybridAdapter,
    ModelAdapter,
    SearchAdapter,
    WMRerankAdapter,
    WorldModelAssistAdapter,
)
from trainer.policy_arena.arena_metrics import summarize_bucket_metrics, summarize_policy_rows
from trainer.policy_arena.arena_report import write_bucket_metrics, write_episode_records, write_summary_table
from trainer.policy_arena.policy_adapter import BasePolicyAdapter, normalize_action, phase_default_action, phase_from_obs


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _parse_csv(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for part in str(text or "").split(","):
        token = part.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _checkpoint_ref_for_path(raw_path: str) -> dict[str, Any]:
    token = str(raw_path or "").strip()
    if not token:
        return {}
    try:
        payload = find_by_artifact_path(Path(token).resolve())
    except Exception:
        payload = None
    return dict(payload or {}) if isinstance(payload, dict) else {}


def _legal_actions_hint(state: dict[str, Any]) -> list[dict[str, Any]] | None:
    phase = phase_from_obs(state)
    if phase == "SELECTING_HAND":
        hand_cards = (state.get("hand") or {}).get("cards") if isinstance(state.get("hand"), dict) else []
        hand_size = min(len(hand_cards or []), action_space.MAX_HAND)
        if hand_size <= 0:
            return []
        legal_ids = action_space.legal_action_ids(hand_size)
        out: list[dict[str, Any]] = []
        for aid in legal_ids[:64]:
            atype, mask = action_space.decode(hand_size, int(aid))
            out.append({"action_type": atype, "indices": action_space.mask_to_indices(mask, hand_size), "id": int(aid)})
        return out
    if phase in {"SHOP", "SMODS_BOOSTER_OPENED"} or "PACK" in phase or "BOOSTER" in phase:
        legal_ids = action_space_shop.legal_action_ids(state)
        out = []
        for aid in legal_ids[:32]:
            out.append({"id": int(aid), "action": action_space_shop.action_from_id(state, int(aid))})
        return out
    return None


def _dominant_bucket_value(counts: dict[str, int]) -> str:
    if not counts:
        return "unknown"
    ordered = sorted(counts.items(), key=lambda kv: (-int(kv[1]), kv[0]))
    return str(ordered[0][0])


def _build_policy_adapter(
    policy_id: str,
    *,
    model_path: str = "",
    model_paths: dict[str, str] | None = None,
    assist_config: dict[str, Any] | None = None,
    world_model_checkpoint: str = "",
    world_model_assist_mode: str = "one_step_heuristic",
    world_model_weight: float = 0.35,
    world_model_uncertainty_penalty: float = 0.5,
) -> BasePolicyAdapter:
    token = str(policy_id or "").strip().lower()
    model_map = model_paths if isinstance(model_paths, dict) else {}
    assist = assist_config if isinstance(assist_config, dict) else {}
    resolved_model_path = str(assist.get("model_path") or model_map.get(policy_id) or model_path or "")
    resolved_wm_checkpoint = str(assist.get("world_model_checkpoint") or world_model_checkpoint or "")
    if token in {"heuristic", "heuristic_baseline", "baseline", "rule"}:
        return HeuristicAdapter(name=policy_id)
    if token in {"policy", "policy_baseline", "model_policy", "model"}:
        return ModelAdapter(name=policy_id, model_path=resolved_model_path, strategy="bc")
    if token in {"heuristic_wm_assist", "baseline_wm_assist", "wm_assist", "world_model_assist"} or "wm_assist" in token:
        return WorldModelAssistAdapter(
            name=policy_id,
            base_policy="heuristic_baseline",
            world_model_checkpoint=resolved_wm_checkpoint,
            assist_mode=world_model_assist_mode,
            weight=float(world_model_weight),
            uncertainty_penalty=float(world_model_uncertainty_penalty),
        )
    if token in {"search", "search_expert", "search_baseline"}:
        return SearchAdapter(name=policy_id)
    if token in {"hybrid", "hybrid_search_heuristic"}:
        return HybridAdapter(name=policy_id)
    if token in {"hybrid_controller_v1", "adaptive_hybrid", "hybrid_router"} or "hybrid_controller" in token:
        from trainer.hybrid.hybrid_controller import AdaptiveHybridController

        return AdaptiveHybridController(
            name=policy_id,
            model_path=resolved_model_path,
            world_model_checkpoint=resolved_wm_checkpoint,
            top_k=_safe_int(assist.get("top_k"), 4),
            router_config=(assist.get("router_config") if isinstance(assist.get("router_config"), dict) else {}),
            router_mode=str(assist.get("router_mode") or "rule"),
            learned_router_checkpoint=str(assist.get("learned_router_checkpoint") or ""),
            learned_router_config=(assist.get("learned_router_config") if isinstance(assist.get("learned_router_config"), dict) else {}),
            rule_guard_config=(assist.get("rule_guard_config") if isinstance(assist.get("rule_guard_config"), dict) else {}),
            canary_config=(assist.get("canary_config") if isinstance(assist.get("canary_config"), dict) else {}),
            search_max_branch=_safe_int(assist.get("search_max_branch"), 80),
            search_max_depth=_safe_int(assist.get("search_max_depth"), 2),
            search_time_budget_ms=_safe_float(assist.get("search_time_budget_ms"), 15.0),
            wm_horizon=_safe_int(assist.get("wm_horizon"), _safe_int(assist.get("horizon"), 1)),
            wm_uncertainty_penalty=_safe_float(assist.get("wm_uncertainty_penalty"), _safe_float(assist.get("uncertainty_penalty"), world_model_uncertainty_penalty)),
            trace_path=str(assist.get("trace_path") or ""),
            trace_context=(assist.get("trace_context") if isinstance(assist.get("trace_context"), dict) else {}),
        )
    if token in {
        "heuristic_wm_rerank",
        "heuristic_plus_wm_rerank",
        "search_wm_rerank",
        "search_plus_wm_rerank",
        "policy_wm_rerank",
        "policy_plus_wm_rerank",
        "model_wm_rerank",
        "baseline_wm_rerank",
    } or "wm_rerank" in token:
        default_base = "heuristic_baseline"
        if token.startswith("search"):
            default_base = "search_expert"
        elif token.startswith("policy") or token.startswith("model"):
            default_base = "model_policy"
        return WMRerankAdapter(
            name=policy_id,
            base_policy=str(assist.get("base_policy") or default_base),
            candidate_source=str(assist.get("candidate_source") or assist.get("base_policy") or default_base),
            model_path=resolved_model_path,
            world_model_checkpoint=resolved_wm_checkpoint,
            top_k=_safe_int(assist.get("top_k"), 4),
            horizon=_safe_int(assist.get("horizon"), 1),
            gamma=_safe_float(assist.get("gamma"), 0.95),
            uncertainty_penalty=_safe_float(assist.get("uncertainty_penalty"), world_model_uncertainty_penalty),
            reward_weight=_safe_float(assist.get("reward_weight"), 1.0),
            score_weight=_safe_float(assist.get("score_weight"), 0.5),
            value_weight=_safe_float(assist.get("value_weight"), 0.15),
            terminal_bonus=_safe_float(assist.get("terminal_bonus"), 0.0),
            search_max_branch=_safe_int(assist.get("search_max_branch"), 80),
            search_max_depth=_safe_int(assist.get("search_max_depth"), 2),
            search_time_budget_ms=_safe_float(assist.get("search_time_budget_ms"), 15.0),
        )
    if policy_id in model_map or token in {"model", "model_policy", "bc", "pv", "dagger", "risk_aware", "deploy_student"}:
        return ModelAdapter(name=policy_id, model_path=resolved_model_path, strategy=token or "bc")
    raise ValueError(f"unsupported policy id: {policy_id}")


def _run_episode(
    *,
    policy_id: str,
    adapter: BasePolicyAdapter,
    seed: str,
    episode_idx: int,
    max_steps: int,
    mode: str,
    warnings: list[str],
) -> dict[str, Any]:
    backend = SimEnvBackend(seed=seed)
    status = "ok"
    error = ""
    timeout_count = 0
    invalid_action_count = 0
    action_counter: Counter[str] = Counter()
    bucket_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    try:
        state = backend.reset(seed=seed)
    except Exception as exc:
        return {
            "policy_id": policy_id,
            "seed": seed,
            "episode_index": int(episode_idx),
            "status": "failed",
            "error": f"reset_failed:{exc}",
            "total_score": 0.0,
            "chips": 0.0,
            "rounds_survived": 0,
            "money_earned": 0.0,
            "rerolls_count": 0,
            "packs_opened": 0,
            "consumables_used": 0,
            "episode_length": 0,
            "invalid_action_rate": 1.0,
            "timeout_rate": 0.0,
            "win_proxy": 0.0,
            "bucket_counts": {},
            "action_counts": {},
            "mode": mode,
        }

    start_money = _safe_float(state.get("money"), 0.0)
    steps = 0
    for step_idx in range(max(1, int(max_steps))):
        phase = phase_from_obs(state)
        pre_action_slices = compute_slice_labels({"state": state, "phase": phase, "action_type": "WAIT"})
        bucket_counts["ante"][as_legacy_ante_bucket(str(pre_action_slices.get("slice_stage") or "unknown"))] += 1
        bucket_counts["risk"][as_legacy_risk_bucket(str(pre_action_slices.get("slice_resource_pressure") or "unknown"))] += 1
        pos_token = pre_action_slices.get("slice_position_sensitive")
        if pos_token is True:
            bucket_counts["position_sensitive"]["yes"] += 1
        elif pos_token is False:
            bucket_counts["position_sensitive"]["no"] += 1
        else:
            bucket_counts["position_sensitive"]["unknown"] += 1
        # Unified P41 slices (kept in addition to legacy buckets for backward compatibility).
        for key in (
            "slice_stage",
            "slice_resource_pressure",
            "slice_position_sensitive",
            "slice_stateful_joker_present",
        ):
            token = str(pre_action_slices.get(key) if pre_action_slices.get(key) is not None else "unknown")
            bucket_counts[key][token] += 1

        legal_actions = _legal_actions_hint(state)
        try:
            action = adapter.act(state, legal_actions=legal_actions)
        except Exception as exc:
            invalid_action_count += 1
            msg = f"[warning] policy={policy_id} seed={seed} ep={episode_idx} step={step_idx} act_exception={exc}"
            warnings.append(msg)
            action = phase_default_action(state, seed=seed)

        action = normalize_action(action, phase=phase)
        action_type = str(action.get("action_type") or "WAIT").upper()
        action_counter[action_type] += 1
        step_slices = compute_slice_labels({"state": state, "phase": phase, "action_type": action_type})
        bucket_counts["action_type"][as_legacy_action_bucket(str(step_slices.get("slice_action_type") or "unknown"))] += 1
        bucket_counts["slice_action_type"][str(step_slices.get("slice_action_type") or "unknown")] += 1

        try:
            next_state, _reward, done, _info = backend.step(action)
        except Exception as exc:
            invalid_action_count += 1
            msg = f"[warning] policy={policy_id} seed={seed} ep={episode_idx} step={step_idx} step_exception={exc}"
            warnings.append(msg)
            fallback = phase_default_action(state, seed=seed)
            try:
                next_state, _reward, done, _info = backend.step(fallback)
                action_counter["FALLBACK"] += 1
            except Exception as exc_fallback:
                status = "failed"
                error = f"step_failed:{exc_fallback}"
                break

        state = next_state
        steps = step_idx + 1
        if bool(done):
            break

    score_block = state.get("score") if isinstance(state.get("score"), dict) else {}
    round_block = state.get("round") if isinstance(state.get("round"), dict) else {}
    total_score = _safe_float(score_block.get("chips"), _safe_float(round_block.get("chips"), 0.0))
    rounds_survived = _safe_int(state.get("round_num"), 0)
    end_money = _safe_float(state.get("money"), start_money)

    rerolls_count = int(action_counter.get("SHOP_REROLL", 0))
    packs_opened = int(action_counter.get("PACK_OPEN", 0))
    consumables_used = int(action_counter.get("CONSUMABLE_USE", 0))
    invalid_action_rate = float(invalid_action_count) / float(max(1, steps))
    timeout_rate = float(timeout_count) / float(max(1, steps))
    win_proxy = 1.0 if rounds_survived >= 3 else 0.0
    episode_slice_labels = {
        "slice_stage": _dominant_bucket_value(bucket_counts.get("slice_stage", {})),
        "slice_resource_pressure": _dominant_bucket_value(bucket_counts.get("slice_resource_pressure", {})),
        "slice_action_type": _dominant_bucket_value(bucket_counts.get("slice_action_type", {})),
        "slice_position_sensitive": _dominant_bucket_value(bucket_counts.get("slice_position_sensitive", {})),
        "slice_stateful_joker_present": _dominant_bucket_value(bucket_counts.get("slice_stateful_joker_present", {})),
    }

    return {
        "policy_id": policy_id,
        "seed": seed,
        "episode_index": int(episode_idx),
        "status": status,
        "error": error,
        "total_score": float(total_score),
        "chips": float(total_score),
        "rounds_survived": int(rounds_survived),
        "money_earned": float(end_money - start_money),
        "rerolls_count": int(rerolls_count),
        "packs_opened": int(packs_opened),
        "consumables_used": int(consumables_used),
        "episode_length": int(steps),
        "invalid_action_rate": float(invalid_action_rate),
        "timeout_rate": float(timeout_rate),
        "win_proxy": float(win_proxy),
        "slice_labels": episode_slice_labels,
        "bucket_counts": {k: dict(v) for k, v in bucket_counts.items()},
        "action_counts": {k: int(v) for k, v in action_counter.items()},
        "mode": mode,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P39 policy arena runner.")
    parser.add_argument("--out-dir", default="docs/artifacts/p39/arena_runs")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--backend", default="sim")
    parser.add_argument("--mode", choices=["long_episode", "state_benchmark", "mixed"], default="long_episode")
    parser.add_argument("--policies", default="heuristic_baseline,search_expert,model_policy")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--policy-model-map-json", default="")
    parser.add_argument("--policy-assist-map-json", default="")
    parser.add_argument("--world-model-checkpoint", default="")
    parser.add_argument("--world-model-assist-mode", default="one_step_heuristic")
    parser.add_argument("--world-model-weight", type=float, default=0.35)
    parser.add_argument("--world-model-uncertainty-penalty", type=float, default=0.5)
    parser.add_argument("--seeds", default="AAAAAAA,BBBBBBB,CCCCCCC")
    parser.add_argument("--episodes-per-seed", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--skip-unavailable", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = (repo_root / out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    run_id = str(args.run_id or _now_stamp())
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    mode = str(args.mode or "long_episode")
    policies = _parse_csv(args.policies)
    seeds = _parse_csv(args.seeds)
    explicit_policies = bool(str(args.policies or "").strip())
    explicit_seeds = bool(str(args.seeds or "").strip())
    episodes_per_seed = max(1, int(args.episodes_per_seed))
    max_steps = max(1, int(args.max_steps))

    if args.quick:
        mode = "long_episode"
        if not policies:
            policies = ["heuristic_baseline", "search_expert"]
        elif not explicit_policies and len(policies) > 2:
            policies = policies[:2]
        if not explicit_seeds and len(seeds) < 2:
            seeds = ["AAAAAAA", "BBBBBBB"]
        if len(seeds) > 2:
            seeds = seeds[:2]
        episodes_per_seed = min(2, episodes_per_seed)
        max_steps = min(max_steps, 120)

    if str(args.backend).lower() != "sim":
        raise SystemExit("P39 arena v1 currently supports only --backend sim")
    if not policies:
        raise SystemExit("empty policy list")
    if not seeds:
        raise SystemExit("empty seed list")

    warnings: list[str] = []
    adapters: dict[str, BasePolicyAdapter] = {}
    adapter_descriptions: dict[str, dict[str, Any]] = {}
    policy_model_map: dict[str, str] = {}
    policy_assist_map: dict[str, dict[str, Any]] = {}
    if str(args.policy_model_map_json or "").strip():
        policy_model_map = _read_json(Path(str(args.policy_model_map_json)).resolve())
        policy_model_map = {str(key): str(value) for key, value in policy_model_map.items() if str(value).strip()}
    if str(args.policy_assist_map_json or "").strip():
        loaded_assist = _read_json(Path(str(args.policy_assist_map_json)).resolve())
        if isinstance(loaded_assist, dict):
            for key, value in loaded_assist.items():
                if isinstance(value, dict):
                    policy_assist_map[str(key)] = dict(value)
    for policy_id in policies:
        try:
            adapter = _build_policy_adapter(
                policy_id,
                model_path=str(args.model_path or ""),
                model_paths=policy_model_map,
                assist_config=policy_assist_map.get(policy_id),
                world_model_checkpoint=str(args.world_model_checkpoint or ""),
                world_model_assist_mode=str(args.world_model_assist_mode or "one_step_heuristic"),
                world_model_weight=float(args.world_model_weight),
                world_model_uncertainty_penalty=float(args.world_model_uncertainty_penalty),
            )
            desc = adapter.describe()
            adapter_block = desc.get("adapter") if isinstance(desc.get("adapter"), dict) else {}
            model_entry = _checkpoint_ref_for_path(str(adapter_block.get("model_path") or ""))
            wm_entry = _checkpoint_ref_for_path(str(adapter_block.get("world_model_checkpoint") or ""))
            if model_entry:
                adapter_block["checkpoint_id"] = str(model_entry.get("checkpoint_id") or "")
            if wm_entry:
                adapter_block["world_model_checkpoint_id"] = str(wm_entry.get("checkpoint_id") or "")
            desc["adapter"] = adapter_block
            adapter_descriptions[policy_id] = desc
            if args.skip_unavailable and str((desc.get("adapter") or {}).get("status") or "") == "stub":
                warnings.append(f"[warning] skip policy={policy_id} because adapter status=stub and --skip-unavailable is enabled")
                continue
            adapters[policy_id] = adapter
        except Exception as exc:
            warnings.append(f"[warning] policy init failed policy={policy_id}: {exc}")

    episode_records: list[dict[str, Any]] = []
    for policy_id, adapter in adapters.items():
        for seed in seeds:
            adapter.reset(seed=seed)
            for ep in range(1, episodes_per_seed + 1):
                attempts = 0
                row = {}
                while attempts < 2:
                    attempts += 1
                    row = _run_episode(
                        policy_id=policy_id,
                        adapter=adapter,
                        seed=seed,
                        episode_idx=ep,
                        max_steps=max_steps,
                        mode=mode,
                        warnings=warnings,
                    )
                    row["retry_attempt"] = int(attempts)
                    if str(row.get("status") or "") == "ok":
                        break
                    if attempts < 2:
                        warnings.append(
                            f"[warning] retry episode policy={policy_id} seed={seed} episode={ep} after status={row.get('status')}"
                        )
                episode_records.append(row)

    for adapter in adapters.values():
        adapter.close()

    summary_rows = summarize_policy_rows(episode_records)
    for row in summary_rows:
        if not isinstance(row, dict):
            continue
        desc = adapter_descriptions.get(str(row.get("policy_id") or "")) if isinstance(adapter_descriptions, dict) else None
        adapter_block = desc.get("adapter") if isinstance(desc, dict) and isinstance(desc.get("adapter"), dict) else {}
        row["checkpoint_id"] = str(adapter_block.get("checkpoint_id") or "")
        row["world_model_checkpoint_id"] = str(adapter_block.get("world_model_checkpoint_id") or "")
    bucket_payload = summarize_bucket_metrics(episode_records)

    episode_records_path = run_dir / "episode_records.jsonl"
    write_episode_records(episode_records_path, episode_records)
    summary_paths = write_summary_table(run_dir, summary_rows)
    bucket_paths = write_bucket_metrics(run_dir, bucket_payload)

    warnings_path = run_dir / "warnings.log"
    warnings_path.write_text("\n".join(warnings).rstrip() + ("\n" if warnings else ""), encoding="utf-8")

    hard_fail_count = sum(1 for row in episode_records if str(row.get("status") or "") != "ok")
    manifest = {
        "schema": "p39_arena_run_manifest_v1",
        "generated_at": _now_iso(),
        "run_id": run_id,
        "mode": mode,
        "backend": "sim",
        "config": {
            "policies": policies,
            "seeds": seeds,
            "episodes_per_seed": episodes_per_seed,
            "max_steps": max_steps,
            "quick": bool(args.quick),
            "world_model_checkpoint": str(args.world_model_checkpoint or ""),
            "world_model_assist_mode": str(args.world_model_assist_mode or "one_step_heuristic"),
            "world_model_weight": float(args.world_model_weight),
            "world_model_uncertainty_penalty": float(args.world_model_uncertainty_penalty),
            "policy_model_map_json": str(args.policy_model_map_json or ""),
            "policy_model_map": policy_model_map,
            "policy_assist_map_json": str(args.policy_assist_map_json or ""),
            "policy_assist_map": policy_assist_map,
        },
        "adapters": adapter_descriptions,
        "paths": {
            "episode_records": str(episode_records_path),
            "summary_table_json": summary_paths.get("json"),
            "summary_table_csv": summary_paths.get("csv"),
            "summary_table_md": summary_paths.get("md"),
            "bucket_metrics_json": bucket_paths.get("json"),
            "bucket_metrics_md": bucket_paths.get("md"),
            "warnings_log": str(warnings_path),
            "run_dir": str(run_dir),
        },
        "episode_total": int(len(episode_records)),
        "hard_fail_count": int(hard_fail_count),
        "status": "PASS" if summary_rows else "FAIL",
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    output = {
        "status": manifest["status"],
        "run_id": run_id,
        "run_dir": str(run_dir),
        "episode_total": int(len(episode_records)),
        "hard_fail_count": int(hard_fail_count),
        "warnings_count": int(len(warnings)),
        "summary_table_json": summary_paths.get("json"),
    }
    print(json.dumps(output, ensure_ascii=False))
    return 0 if summary_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
