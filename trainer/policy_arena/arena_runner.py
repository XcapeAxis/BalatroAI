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
from trainer.policy_arena.adapters import HeuristicAdapter, HybridAdapter, ModelAdapter, SearchAdapter
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


def _ante_bucket(state: dict[str, Any]) -> str:
    ante = _safe_int(state.get("ante_num"), 0)
    if ante <= 0:
        round_num = _safe_int(state.get("round_num"), 1)
        ante = max(1, ((round_num - 1) // 3) + 1)
    if ante <= 2:
        return "ante_1_2"
    if ante <= 4:
        return "ante_3_4"
    return "ante_5_plus"


def _risk_bucket(state: dict[str, Any]) -> str:
    money = _safe_float(state.get("money"), 0.0)
    round_info = state.get("round") if isinstance(state.get("round"), dict) else {}
    hands_left = _safe_int(round_info.get("hands_left"), 0)
    discards_left = _safe_int(round_info.get("discards_left"), 0)
    if money <= 4.0 or hands_left <= 1 or discards_left <= 1:
        return "resource_tight"
    if money >= 15.0 and hands_left >= 2:
        return "resource_relaxed"
    return "resource_balanced"


def _is_position_sensitive_state(state: dict[str, Any]) -> bool:
    tags = state.get("tags") if isinstance(state.get("tags"), list) else []
    joined_tags = " ".join([str(x).lower() for x in tags])
    if any(k in joined_tags for k in ("left", "right", "position", "order")):
        return True

    jokers = state.get("jokers") if isinstance(state.get("jokers"), list) else []
    for joker in jokers:
        if not isinstance(joker, dict):
            continue
        key = str(joker.get("key") or joker.get("joker_id") or "").lower()
        if any(k in key for k in ("photograph", "hanging_chad", "blueprint", "brainstorm")):
            return True

    consumables = (state.get("consumables") or {}).get("cards") if isinstance(state.get("consumables"), dict) else []
    for card in consumables if isinstance(consumables, list) else []:
        if not isinstance(card, dict):
            continue
        key = str(card.get("key") or "").lower()
        if any(k in key for k in ("left", "right", "swap", "reorder")):
            return True
    return False


def _action_bucket(action_type: str, phase: str) -> str:
    at = str(action_type or "").upper()
    ph = str(phase or "").upper()
    if at in {"PLAY", "DISCARD"}:
        return at
    if at in {"SHOP_REROLL", "SHOP_BUY", "SELL", "NEXT_ROUND"} or ph == "SHOP":
        return "SHOP"
    if at in {"CONSUMABLE_USE"}:
        return "CONSUMABLE"
    if at in {"MOVE_HAND_CARD", "MOVE_JOKER", "SWAP_HAND_CARD", "SWAP_JOKER", "REORDER_HAND", "REORDER_JOKERS"}:
        return "POSITION"
    if at == "PACK_OPEN" or "PACK" in ph or "BOOSTER" in ph:
        return "PACK"
    return "OTHER"


def _build_policy_adapter(policy_id: str, *, model_path: str = "") -> BasePolicyAdapter:
    token = str(policy_id or "").strip().lower()
    if token in {"heuristic", "heuristic_baseline", "baseline", "rule"}:
        return HeuristicAdapter(name=policy_id)
    if token in {"search", "search_expert"}:
        return SearchAdapter(name=policy_id)
    if token in {"hybrid", "hybrid_search_heuristic"}:
        return HybridAdapter(name=policy_id)
    if token in {"model", "model_policy", "bc", "pv", "dagger", "risk_aware", "deploy_student"}:
        return ModelAdapter(name=policy_id, model_path=model_path, strategy=token)
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
        bucket_counts["ante"][_ante_bucket(state)] += 1
        bucket_counts["risk"][_risk_bucket(state)] += 1
        bucket_counts["position_sensitive"]["yes" if _is_position_sensitive_state(state) else "no"] += 1

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
        bucket_counts["action_type"][_action_bucket(action_type, phase)] += 1

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
    episodes_per_seed = max(1, int(args.episodes_per_seed))
    max_steps = max(1, int(args.max_steps))

    if args.quick:
        mode = "long_episode"
        if not policies:
            policies = ["heuristic_baseline", "search_expert"]
        policies = policies[:2]
        if len(seeds) < 2:
            seeds = ["AAAAAAA", "BBBBBBB"]
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
    for policy_id in policies:
        try:
            adapter = _build_policy_adapter(policy_id, model_path=str(args.model_path or ""))
            desc = adapter.describe()
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

