from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from demo.scenario_loader import load_scenarios
from sim.core.engine import SimEnv
from demo.model_inference import legal_action_ids_from_state
from trainer import action_space
from trainer.expert_policy import choose_action
from trainer.features import extract_features


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_run_dir() -> Path:
    run_id = now_stamp()
    return Path(__file__).resolve().parent.parent / "docs" / "artifacts" / "mvp" / "model_train" / run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small supervised dataset for the MVP demo model.")
    parser.add_argument("--episodes", type=int, default=240)
    parser.add_argument("--max-steps", type=int, default=36)
    parser.add_argument("--explore-prob", type=float, default=0.15)
    parser.add_argument("--scenario-copies", type=int, default=48)
    parser.add_argument("--seed-prefix", default="MVPDATA")
    parser.add_argument("--run-dir", default="", help="Output directory under docs/artifacts/mvp/model_train.")
    return parser.parse_args()


def _macro_to_action(decision, seed: str) -> dict[str, Any]:
    if decision.action_type and decision.mask_int is not None:
        return {
            "action_type": str(decision.action_type),
            "indices": action_space.mask_to_indices(int(decision.mask_int), action_space.MAX_HAND),
        }
    macro = str(decision.macro_action or "wait").lower()
    params = dict(decision.macro_params or {})
    if macro == "select":
        return {"action_type": "SELECT", "index": int(params.get("index", 0))}
    if macro == "cash_out":
        return {"action_type": "CASH_OUT"}
    if macro == "next_round":
        return {"action_type": "NEXT_ROUND"}
    if macro == "start":
        return {"action_type": "START", "seed": str(params.get("seed") or seed)}
    return {"action_type": "WAIT"}


def _record_from_state(state: dict[str, Any], *, episode_id: str, step_id: int, seed: str) -> dict[str, Any] | None:
    cards = (state.get("hand") or {}).get("cards") if isinstance(state.get("hand"), dict) else []
    cards = cards if isinstance(cards, list) else []
    hand_size = min(len(cards), action_space.MAX_HAND)
    if hand_size <= 0:
        return None

    decision = choose_action(state, start_seed=seed)
    if not decision.action_type or decision.mask_int is None:
        return None

    mask_int = int(decision.mask_int)
    action_id = action_space.encode(hand_size, str(decision.action_type), mask_int)
    indices = action_space.mask_to_indices(mask_int, hand_size)
    return {
        "schema": "mvp_demo_dataset_v1",
        "episode_id": episode_id,
        "step_id": int(step_id),
        "seed": seed,
        "phase": str(state.get("state") or "UNKNOWN"),
        "hand_size": hand_size,
        "legal_action_ids": legal_action_ids_from_state(state),
        "expert_action_id": int(action_id),
        "expert_action_type": str(decision.action_type),
        "expert_mask_int": mask_int,
        "expert_indices": indices,
        "expert_reason": str(decision.reason or ""),
        "features": extract_features(state),
        "resources": {
            "hands_left": int((state.get("round") or {}).get("hands_left") or 0),
            "discards_left": int((state.get("round") or {}).get("discards_left") or 0),
            "money": float(state.get("money") or 0.0),
            "chips": float((state.get("score") or {}).get("chips") or 0.0),
            "target": float((state.get("score") or {}).get("target_chips") or 0.0),
        },
    }


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = run_dir / "dataset.jsonl"

    rng = random.Random(7)
    action_counter: Counter[str] = Counter()
    hand_size_counter: Counter[int] = Counter()
    records = 0
    episode_summaries: list[dict[str, Any]] = []

    with dataset_path.open("w", encoding="utf-8", newline="\n") as fp:
        for episode_idx in range(max(1, int(args.episodes))):
            seed = f"{args.seed_prefix}_{episode_idx:04d}"
            env = SimEnv(seed=seed)
            env.reset(seed=seed)
            episode_id = f"episode_{episode_idx:04d}"
            hand_records = 0
            for step_idx in range(max(1, int(args.max_steps))):
                state = env.get_state()
                phase = str(state.get("state") or "UNKNOWN")
                if phase == "SELECTING_HAND":
                    round_info = state.get("round") if isinstance(state.get("round"), dict) else {}
                    if int(round_info.get("hands_left") or 0) <= 0 and int(round_info.get("discards_left") or 0) <= 0:
                        break
                    record = _record_from_state(state, episode_id=episode_id, step_id=step_idx, seed=seed)
                    if record is not None:
                        fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                        records += 1
                        hand_records += 1
                        action_counter[str(record["expert_action_type"])] += 1
                        hand_size_counter[int(record["hand_size"])] += 1

                    decision = choose_action(state, start_seed=seed)
                    hand_size = min(len((state.get("hand") or {}).get("cards") or []), action_space.MAX_HAND)
                    if hand_size <= 0 or decision.mask_int is None or not decision.action_type:
                        action = {"action_type": "WAIT"}
                    else:
                        if rng.random() < max(0.0, min(1.0, float(args.explore_prob))):
                            aid = rng.choice(action_space.legal_action_ids(hand_size))
                            atype, mask_int = action_space.decode(hand_size, aid)
                            action = {
                                "action_type": atype,
                                "indices": action_space.mask_to_indices(mask_int, hand_size),
                            }
                        else:
                            action = {
                                "action_type": str(decision.action_type),
                                "indices": action_space.mask_to_indices(int(decision.mask_int), hand_size),
                            }
                else:
                    decision = choose_action(state, start_seed=seed)
                    action = _macro_to_action(decision, seed=seed)
                try:
                    env.step(action)
                except ValueError as exc:
                    if str(exc) in {"no_hands_left", "no_discards_left"}:
                        break
                    raise

        scenarios = load_scenarios()
        for scenario in scenarios.values():
            for copy_idx in range(max(0, int(args.scenario_copies))):
                seed = f"{scenario.scenario_id}_{copy_idx:03d}"
                env = SimEnv(seed=seed)
                env.reset(from_snapshot=scenario.snapshot)
                episode_id = f"scenario_{scenario.scenario_id}_{copy_idx:03d}"
                hand_records = 0
                for step_idx in range(3):
                    state = env.get_state()
                    phase = str(state.get("state") or "UNKNOWN")
                    if phase != "SELECTING_HAND":
                        decision = choose_action(state, start_seed=seed)
                        action = _macro_to_action(decision, seed=seed)
                        try:
                            env.step(action)
                        except ValueError:
                            break
                        continue
                    record = _record_from_state(state, episode_id=episode_id, step_id=step_idx, seed=seed)
                    if record is None:
                        break
                    fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records += 1
                    hand_records += 1
                    action_counter[str(record["expert_action_type"])] += 1
                    hand_size_counter[int(record["hand_size"])] += 1
                    action = {
                        "action_type": str(record["expert_action_type"]),
                        "indices": list(record["expert_indices"]),
                    }
                    try:
                        env.step(action)
                    except ValueError:
                        break
                episode_summaries.append({"episode_id": episode_id, "seed": seed, "hand_records": hand_records})
            episode_summaries.append({"episode_id": episode_id, "seed": seed, "hand_records": hand_records})

    stats = {
        "schema": "mvp_demo_dataset_stats_v1",
        "dataset_path": str(dataset_path),
        "episodes": max(1, int(args.episodes)),
        "max_steps": max(1, int(args.max_steps)),
        "explore_prob": float(args.explore_prob),
        "total_records": records,
        "action_counts": dict(action_counter),
        "hand_size_counts": {str(key): value for key, value in sorted(hand_size_counter.items())},
        "episode_summaries": episode_summaries[:20],
    }
    (run_dir / "dataset_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "dataset_path.txt").write_text(str(dataset_path), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "dataset_path": str(dataset_path), "records": records}, ensure_ascii=False))
    return 0 if records > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
