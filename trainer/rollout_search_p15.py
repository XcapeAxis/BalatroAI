from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

from trainer import action_space
from trainer import action_space_shop
from trainer.dataset import JsonlWriter
from trainer.env_client import create_backend
from trainer.expert_policy import choose_action as choose_macro_action
from trainer.features import extract_features
from trainer.features_shop import extract_shop_features
from trainer.search.beam_expert import choose_topk
from trainer.utils import set_global_seed, setup_logger, timestamp, warn_if_unstable_python


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate P15 search-labeled dataset on sim backend.")
    p.add_argument("--backend", choices=["sim"], default="sim")
    p.add_argument("--stake", default="gold")
    p.add_argument("--episodes", type=int, default=120)
    p.add_argument("--max-steps-per-episode", type=int, default=320)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--seed-prefix", default="AAAAAAA")
    p.add_argument("--out", required=True)
    p.add_argument("--hand-target-samples", type=int, default=20000)
    p.add_argument("--shop-target-samples", type=int, default=5000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--search", default="algo=beam")
    p.add_argument("--search-budget-ms", type=float, default=20.0)
    p.add_argument("--hand-max-candidates", type=int, default=40)
    p.add_argument("--shop-max-candidates", type=int, default=20)
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--timeout-sec", type=float, default=8.0)
    p.add_argument("--include-obs-raw", action="store_true")
    p.add_argument("--fail-fast", dest="fail_fast", action="store_true", default=True, help="Abort immediately on step/action errors.")
    p.add_argument("--no-fail-fast", dest="fail_fast", action="store_false", help="Continue with fallbacks on recoverable errors.")
    return p.parse_args()


def _calc_reward(prev_state: dict | None, cur_state: dict) -> tuple[float, str]:
    cur = float((cur_state.get("round") or {}).get("chips") or 0.0)
    if prev_state is None:
        return 0.0, "chips_delta_proxy"
    prev = float((prev_state.get("round") or {}).get("chips") or 0.0)
    return float(cur - prev), "chips_delta_proxy"


def _macro_action(state: dict, seed: str) -> dict:
    decision = choose_macro_action(state, start_seed=seed)
    macro_action = str(decision.macro_action or "WAIT").upper()
    action = {"action_type": macro_action}
    action.update(dict(decision.macro_params or {}))
    return action


def _parse_search_algo(raw: str) -> str:
    text = str(raw or "").strip().lower()
    if "=" in text:
        left, right = text.split("=", 1)
        if left.strip() == "algo":
            text = right.strip()
    return text or "beam"


def _synthetic_shop_state(idx: int) -> dict:
    # Minimal stable SHOP snapshot for shop-head supervision when env never reaches SHOP.
    base_money = 8.0 + float(idx % 7)
    return {
        "state": "SHOP",
        "money": base_money,
        "round": {
            "chips": float((idx % 5) * 10),
            "hands_left": max(0, 3 - (idx % 3)),
            "discards_left": max(0, 2 - (idx % 2)),
            "reroll_cost": 5.0,
            "ante": 1 + (idx % 4),
            "round_num": 1 + (idx % 3),
        },
        "score": {"target_chips": 300.0 + float((idx % 4) * 50)},
        "shop": {"cards": [{"key": f"shop_card_{idx}_0"}]},
        "vouchers": {"cards": [{"key": f"voucher_{idx}_0"}]},
        "packs": {"cards": [{"key": f"pack_{idx}_0"}]},
        "consumables": {"cards": [{"key": f"cons_{idx}_0"}]},
        "jokers": [{"key": f"joker_{idx}_0"}],
        "pack_choices": {"cards": []},
    }


def main() -> int:
    args = parse_args()
    logger = setup_logger("trainer.rollout_search_p15")
    warn_if_unstable_python(logger)
    set_global_seed(args.seed)

    if args.workers != 1:
        logger.warning("workers=%d requested; forcing serial worker=1 for determinism", args.workers)

    search_algo = _parse_search_algo(args.search)
    if search_algo not in {"beam"}:
        logger.warning("unsupported search algo=%s, fallback to beam", search_algo)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    backend = create_backend("sim", timeout_sec=float(args.timeout_sec), seed=args.seed_prefix, logger=logger)
    hand_records = 0
    shop_records = 0
    total_records = 0
    episodes_done = 0

    with JsonlWriter(out_path) as writer:
        for ep in range(args.episodes):
            ep_seed = f"{args.seed_prefix}-p15-{ep}"
            state = backend.reset(seed=ep_seed)
            prev_state = None

            if str(args.stake or "").strip().lower() not in {"", "white"}:
                try:
                    state, _, _, _ = backend.step({"action_type": "START", "seed": ep_seed, "stake": str(args.stake).upper()})
                except Exception:
                    if args.fail_fast:
                        raise

            for step in range(args.max_steps_per_episode):
                phase = str(state.get("state") or "UNKNOWN")
                done = phase == "GAME_OVER"
                reward, reward_info = _calc_reward(prev_state, state)
                cards = (state.get("hand") or {}).get("cards") or []
                hand_size = min(len(cards), action_space.MAX_HAND)

                rec = {
                    "schema_version": "record_v1",
                    "timestamp": timestamp(),
                    "episode_id": ep,
                    "step_id": step,
                    "instance_id": 0,
                    "base_url": "sim://search_p15",
                    "phase": phase,
                    "done": bool(done),
                    "hand_size": hand_size,
                    "legal_action_ids": [],
                    "expert_action_id": None,
                    "macro_action": "WAIT",
                    "reward": reward,
                    "reward_info": reward_info,
                    "features": extract_features(state),
                    "shop_legal_action_ids": [],
                    "shop_expert_action_id": None,
                    "shop_features": None,
                    "value_target": float(reward),
                }
                if args.include_obs_raw:
                    rec["obs_raw"] = state

                if done:
                    writer.write_record(rec)
                    total_records += 1
                    break

                if phase == "SELECTING_HAND" and hand_size > 0:
                    legal_ids = action_space.legal_action_ids(hand_size)
                    rec["legal_action_ids"] = legal_ids

                    decision = choose_topk(
                        state,
                        topk=max(1, int(args.topk)),
                        hand_max_candidates=int(args.hand_max_candidates),
                        shop_max_candidates=int(args.shop_max_candidates),
                        max_depth=int(args.max_depth),
                        time_budget_ms=float(args.search_budget_ms),
                        seed=ep_seed,
                        fail_fast=bool(args.fail_fast),
                    )
                    action = dict(decision.top1)
                    topk_payload = []
                    for item in decision.topk:
                        topk_payload.append({"action": item.action, "value": item.value, "score_breakdown": item.score_breakdown})
                    rec["teacher_topk"] = topk_payload
                    rec["teacher_score_breakdown"] = topk_payload[0]["score_breakdown"] if topk_payload else {}
                    rec["value_target"] = float(topk_payload[0]["value"] if topk_payload else reward)

                    if action.get("action_type") in {"PLAY", "DISCARD"}:
                        try:
                            aid = action_space.encode(
                                hand_size,
                                str(action.get("action_type")),
                                action_space.indices_to_mask(list(action.get("indices") or [])),
                            )
                        except Exception:
                            if args.fail_fast:
                                raise
                            aid = legal_ids[0] if legal_ids else None
                        rec["expert_action_id"] = int(aid) if aid is not None else None
                    else:
                        rec["expert_action_id"] = int(legal_ids[0]) if legal_ids else None
                    rec["macro_action"] = None
                    hand_records += 1

                elif phase in action_space_shop.SHOP_PHASES:
                    shop_feat = extract_shop_features(state)
                    rec["shop_features"] = shop_feat
                    shop_legal = action_space_shop.legal_action_ids(state)
                    rec["shop_legal_action_ids"] = shop_legal

                    decision = choose_topk(
                        state,
                        topk=max(1, int(args.topk)),
                        hand_max_candidates=int(args.hand_max_candidates),
                        shop_max_candidates=int(args.shop_max_candidates),
                        max_depth=int(args.max_depth),
                        time_budget_ms=float(args.search_budget_ms),
                        seed=ep_seed,
                        fail_fast=bool(args.fail_fast),
                    )
                    action = dict(decision.top1)
                    topk_payload = []
                    for item in decision.topk:
                        topk_payload.append({"action": item.action, "value": item.value, "score_breakdown": item.score_breakdown})
                    rec["teacher_topk"] = topk_payload
                    rec["teacher_score_breakdown"] = topk_payload[0]["score_breakdown"] if topk_payload else {}
                    rec["value_target"] = float(topk_payload[0]["value"] if topk_payload else reward)

                    try:
                        aid = action_space_shop.encode(
                            str(action.get("action_type") or "WAIT"),
                            action.get("params") if isinstance(action.get("params"), dict) else None,
                        )
                    except Exception:
                        if args.fail_fast:
                            raise
                        aid = shop_legal[0] if shop_legal else action_space_shop.encode("WAIT", {})
                    rec["shop_expert_action_id"] = int(aid)
                    rec["macro_action"] = str(action.get("action_type") or "WAIT")
                    action = action_space_shop.action_from_id(state, int(aid))
                    shop_records += 1
                else:
                    action = _macro_action(state, ep_seed)
                    rec["macro_action"] = str(action.get("action_type") or "WAIT")

                writer.write_record(rec)
                total_records += 1

                try:
                    next_state, _, done_flag, _ = backend.step(action)
                except Exception:
                    if args.fail_fast:
                        raise RuntimeError(f"step failed for phase={phase} action={action!r}")
                    fallback = _macro_action(state, ep_seed)
                    try:
                        next_state, _, done_flag, _ = backend.step(fallback)
                    except Exception:
                        if args.fail_fast:
                            raise RuntimeError(f"fallback step failed for phase={phase} fallback={fallback!r}")
                        break

                prev_state = state
                state = next_state
                if done_flag:
                    break
                if hand_records >= args.hand_target_samples and shop_records >= args.shop_target_samples:
                    break

            episodes_done += 1
            logger.info(
                "ep=%d/%d records=%d hand=%d shop=%d",
                ep + 1,
                args.episodes,
                total_records,
                hand_records,
                shop_records,
            )
            if hand_records >= args.hand_target_samples and shop_records >= args.shop_target_samples:
                break

        synth_idx = 0
        while shop_records < args.shop_target_samples:
            synth_state = _synthetic_shop_state(synth_idx)
            synth_idx += 1
            reward = 0.0
            rec = {
                "schema_version": "record_v1",
                "timestamp": timestamp(),
                "episode_id": -1,
                "step_id": synth_idx,
                "instance_id": 0,
                "base_url": "sim://search_p15_synth_shop",
                "phase": "SHOP",
                "done": False,
                "hand_size": 0,
                "legal_action_ids": [],
                "expert_action_id": None,
                "macro_action": "WAIT",
                "reward": reward,
                "reward_info": "synth_shop_proxy",
                "features": extract_features(synth_state),
                "shop_legal_action_ids": action_space_shop.legal_action_ids(synth_state),
                "shop_expert_action_id": None,
                "shop_features": extract_shop_features(synth_state),
                "value_target": 0.0,
            }
            decision = choose_topk(
                synth_state,
                topk=max(1, int(args.topk)),
                hand_max_candidates=int(args.hand_max_candidates),
                shop_max_candidates=int(args.shop_max_candidates),
                max_depth=int(args.max_depth),
                time_budget_ms=float(args.search_budget_ms),
                seed=f"synth-{synth_idx}",
                fail_fast=bool(args.fail_fast),
            )
            action = dict(decision.top1)
            topk_payload = []
            for item in decision.topk:
                topk_payload.append({"action": item.action, "value": item.value, "score_breakdown": item.score_breakdown})
            rec["teacher_topk"] = topk_payload
            rec["teacher_score_breakdown"] = topk_payload[0]["score_breakdown"] if topk_payload else {}
            rec["value_target"] = float(topk_payload[0]["value"] if topk_payload else 0.0)
            try:
                aid = action_space_shop.encode(
                    str(action.get("action_type") or "WAIT"),
                    action.get("params") if isinstance(action.get("params"), dict) else None,
                )
            except Exception:
                if args.fail_fast:
                    raise
                legal = rec["shop_legal_action_ids"]
                aid = legal[0] if legal else action_space_shop.encode("WAIT", {})
            rec["shop_expert_action_id"] = int(aid)
            rec["macro_action"] = str(action.get("action_type") or "WAIT")
            writer.write_record(rec)
            total_records += 1
            shop_records += 1

    backend.close()
    logger.info(
        "rollout_search_p15 done episodes=%d records=%d hand=%d shop=%d out=%s",
        episodes_done,
        total_records,
        hand_records,
        shop_records,
        out_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
