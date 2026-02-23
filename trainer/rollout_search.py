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
from trainer.expert_policy import choose_action
from trainer.expert_policy_shop import choose_shop_action
from trainer.features import extract_features
from trainer.features_shop import extract_shop_features
from trainer.search_expert import choose_action as choose_search_action
from trainer.utils import set_global_seed, setup_logger, timestamp, warn_if_unstable_python


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate search-labeled hand-decision dataset on sim backend.")
    parser.add_argument("--backend", choices=["sim"], default="sim", help="Only sim backend is supported for rollout_search.")
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--max-steps-per-episode", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seed-prefix", type=str, default="AAAAAAA")
    parser.add_argument("--out", type=str, default="trainer_data/p10_search_dataset.jsonl")
    parser.add_argument("--include-obs-raw", action="store_true")
    parser.add_argument("--max-branch", type=int, default=80)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--time-budget-ms", type=float, default=15.0)
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    return parser.parse_args()


def _calc_reward(prev_state: dict | None, cur_state: dict) -> tuple[float, str]:
    cur = float((cur_state.get("round") or {}).get("chips") or 0.0)
    if prev_state is None:
        return 0.0, "chips_delta_proxy"
    prev = float((prev_state.get("round") or {}).get("chips") or 0.0)
    return float(cur - prev), "chips_delta_proxy"


def _build_macro_action(phase_decision, idle_sleep: float) -> dict:
    macro_action = str(phase_decision.macro_action or "WAIT").upper()
    macro_params = dict(phase_decision.macro_params or {})
    action = {"action_type": macro_action}
    if macro_action == "WAIT":
        action["sleep"] = max(0.0, float(idle_sleep))
    action.update(macro_params)
    return action


def main() -> int:
    args = parse_args()
    logger = setup_logger("trainer.rollout_search")
    warn_if_unstable_python(logger)
    set_global_seed(args.seed)

    if args.episodes <= 0:
        logger.error("--episodes must be > 0")
        return 2
    if args.max_steps_per_episode <= 0:
        logger.error("--max-steps-per-episode must be > 0")
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    backend = create_backend("sim", timeout_sec=float(args.timeout_sec), seed=args.seed_prefix, logger=logger)

    episodes_done = 0
    records_written = 0
    hand_records = 0
    shop_records = 0

    with JsonlWriter(out_path) as writer:
        for episode_id in range(args.episodes):
            state = backend.reset(seed=f"{args.seed_prefix}-s{episode_id}")
            prev_state = None

            for step_id in range(args.max_steps_per_episode):
                phase = str(state.get("state") or "UNKNOWN")
                feat = extract_features(state)
                done = phase == "GAME_OVER"
                reward, reward_info = _calc_reward(prev_state, state)
                hand_cards = (state.get("hand") or {}).get("cards") or []
                hand_size = min(len(hand_cards), action_space.MAX_HAND)

                record = {
                    "schema_version": "record_v1",
                    "timestamp": timestamp(),
                    "episode_id": episode_id,
                    "step_id": step_id,
                    "instance_id": 0,
                    "base_url": "sim://search",
                    "phase": phase,
                    "done": bool(done),
                    "hand_size": hand_size,
                    "legal_action_ids": [],
                    "expert_action_id": None,
                    "macro_action": "WAIT",
                    "reward": reward,
                    "reward_info": reward_info,
                    "features": feat,
                    "shop_legal_action_ids": [],
                    "shop_expert_action_id": None,
                    "shop_features": None,
                }
                if args.include_obs_raw:
                    record["obs_raw"] = state

                if done:
                    writer.write_record(record)
                    records_written += 1
                    break

                if phase == "SELECTING_HAND" and hand_size > 0:
                    decision = choose_search_action(
                        state,
                        max_branch=int(args.max_branch),
                        max_depth=int(args.max_depth),
                        time_budget_ms=float(args.time_budget_ms),
                        seed=f"{args.seed_prefix}-search",
                    )

                    legal_ids = action_space.legal_action_ids(hand_size)
                    mask = action_space.indices_to_mask(decision.indices)
                    try:
                        expert_action_id = action_space.encode(hand_size, decision.action_type, mask)
                    except Exception:
                        fallback = choose_action(state, start_seed=f"{args.seed_prefix}-{episode_id}")
                        if fallback.action_type is None or fallback.mask_int is None:
                            expert_action_id = legal_ids[0]
                            atype, mask = action_space.decode(hand_size, legal_ids[0])
                            indices = action_space.mask_to_indices(mask, hand_size)
                            action = {"action_type": atype, "indices": indices}
                        else:
                            expert_action_id = action_space.encode(hand_size, fallback.action_type, fallback.mask_int)
                            indices = action_space.mask_to_indices(fallback.mask_int, hand_size)
                            action = {"action_type": fallback.action_type, "indices": indices}
                    else:
                        action = {"action_type": decision.action_type, "indices": list(decision.indices)}

                    record["legal_action_ids"] = legal_ids
                    record["expert_action_id"] = int(expert_action_id)
                    record["macro_action"] = None
                    hand_records += 1

                elif phase in action_space_shop.SHOP_PHASES:
                    shop_decision = choose_shop_action(state)
                    shop_feat = extract_shop_features(state)
                    shop_legal = action_space_shop.legal_action_ids(state)

                    record["shop_legal_action_ids"] = shop_legal
                    record["shop_expert_action_id"] = int(shop_decision.action_id)
                    record["shop_features"] = shop_feat
                    record["macro_action"] = str((shop_decision.action or {}).get("action_type") or "WAIT")

                    action = dict(shop_decision.action)
                    shop_records += 1

                else:
                    phase_decision = choose_action(state, start_seed=f"{args.seed_prefix}-{episode_id}")
                    action = _build_macro_action(phase_decision, idle_sleep=0.05)
                    record["macro_action"] = str(action.get("action_type") or "WAIT")

                writer.write_record(record)
                records_written += 1

                try:
                    next_state, _, _, _ = backend.step(action)
                except Exception:
                    # Recover from strict phase/resource errors by deferring to macro policy.
                    fallback = _build_macro_action(choose_action(state, start_seed=f"{args.seed_prefix}-{episode_id}"), idle_sleep=0.05)
                    try:
                        next_state, _, _, _ = backend.step(fallback)
                        record["macro_action"] = str(fallback.get("action_type") or "WAIT")
                    except Exception:
                        break
                prev_state = state
                state = next_state

            episodes_done += 1
            logger.info(
                "episode=%d/%d records_total=%d hand_records=%d shop_records=%d",
                episode_id + 1,
                args.episodes,
                records_written,
                hand_records,
                shop_records,
            )

    backend.close()

    logger.info(
        "rollout_search done: episodes=%d records=%d hand_records=%d shop_records=%d out=%s",
        episodes_done,
        records_written,
        hand_records,
        shop_records,
        out_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
