from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from demo.model_inference import legal_action_ids_from_state
from demo.scenario_loader import load_scenarios
from sim.core.engine import SimEnv
from trainer import action_space
from trainer.expert_policy import choose_action
from trainer.features import extract_features


ProgressCallback = Callable[[dict[str, Any]], None]


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_run_dir() -> Path:
    run_id = now_stamp()
    return Path(__file__).resolve().parent.parent / "docs" / "artifacts" / "mvp" / "model_train" / run_id


@dataclass
class BuildDatasetConfig:
    episodes: int = 480
    max_steps: int = 40
    explore_prob: float = 0.12
    scenario_copies: int = 96
    seed_prefix: str = "MVPDATA"
    run_dir: Path | None = None
    progress_json: Path | None = None
    progress_every: int = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 MVP Demo 监督学习数据集。")
    parser.add_argument("--episodes", type=int, default=480)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--explore-prob", type=float, default=0.12)
    parser.add_argument("--scenario-copies", type=int, default=96)
    parser.add_argument("--seed-prefix", default="MVPDATA")
    parser.add_argument("--run-dir", default="", help="输出目录，默认写入 docs/artifacts/mvp/model_train/<run_id>")
    parser.add_argument("--progress-json", default="", help="可选。持续写入构建进度 JSON。")
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
        "schema": "mvp_demo_dataset_v2",
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


def _write_progress(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_dataset(config: BuildDatasetConfig, progress_callback: ProgressCallback | None = None) -> dict[str, Any]:
    run_dir = Path(config.run_dir).resolve() if config.run_dir else default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = run_dir / "dataset.jsonl"

    rng = random.Random(7)
    action_counter: Counter[str] = Counter()
    hand_size_counter: Counter[int] = Counter()
    records = 0
    processed_episodes = 0
    episode_summaries: list[dict[str, Any]] = []
    progress_json = Path(config.progress_json).resolve() if config.progress_json else None

    def emit(update: dict[str, Any]) -> None:
        payload = {
            "schema": "mvp_dataset_progress_v1",
            "stage": "building_dataset",
            "run_dir": str(run_dir),
            "dataset_path": str(dataset_path),
            "episodes_total": int(config.episodes),
            "episodes_done": processed_episodes,
            "records": records,
            **update,
        }
        _write_progress(progress_json, payload)
        if progress_callback is not None:
            progress_callback(payload)

    emit({"status": "running", "message": "开始生成监督学习数据集。"})

    with dataset_path.open("w", encoding="utf-8", newline="\n") as fp:
        for episode_idx in range(max(1, int(config.episodes))):
            seed = f"{config.seed_prefix}_{episode_idx:04d}"
            env = SimEnv(seed=seed)
            env.reset(seed=seed)
            episode_id = f"episode_{episode_idx:04d}"
            hand_records = 0
            for step_idx in range(max(1, int(config.max_steps))):
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
                        if rng.random() < max(0.0, min(1.0, float(config.explore_prob))):
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

            processed_episodes += 1
            episode_summaries.append({"episode_id": episode_id, "seed": seed, "hand_records": hand_records})
            if processed_episodes == 1 or processed_episodes % max(1, int(config.progress_every)) == 0:
                emit(
                    {
                        "status": "running",
                        "message": f"已完成 {processed_episodes}/{int(config.episodes)} 局数据采样。",
                        "progress": round(processed_episodes / max(1, int(config.episodes)), 4),
                    }
                )

        scenarios = load_scenarios()
        total_scenario_jobs = max(0, int(config.scenario_copies)) * max(1, len(scenarios))
        scenario_done = 0
        for scenario in scenarios.values():
            for copy_idx in range(max(0, int(config.scenario_copies))):
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
                scenario_done += 1
                if scenario_done == 1 or scenario_done % max(1, int(config.progress_every)) == 0:
                    emit(
                        {
                            "status": "running",
                            "message": f"正在强化演示场景样本：{scenario_done}/{total_scenario_jobs}",
                            "scenario_progress": round(scenario_done / max(1, total_scenario_jobs), 4),
                        }
                    )

    stats = {
        "schema": "mvp_demo_dataset_stats_v2",
        "dataset_path": str(dataset_path),
        "episodes": max(1, int(config.episodes)),
        "max_steps": max(1, int(config.max_steps)),
        "explore_prob": float(config.explore_prob),
        "scenario_copies": max(0, int(config.scenario_copies)),
        "total_records": records,
        "action_counts": dict(action_counter),
        "hand_size_counts": {str(key): value for key, value in sorted(hand_size_counter.items())},
        "episode_summaries": episode_summaries[:24],
        "config": asdict(config) | {"run_dir": str(run_dir), "progress_json": str(progress_json) if progress_json else ""},
    }
    (run_dir / "dataset_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "dataset_path.txt").write_text(str(dataset_path), encoding="utf-8")

    result = {"run_dir": str(run_dir), "dataset_path": str(dataset_path), "records": records, "dataset_stats": stats}
    emit({"status": "finished", "message": f"数据集构建完成，共 {records} 条样本。", "progress": 1.0, "result": result})
    return result


def main() -> int:
    args = parse_args()
    config = BuildDatasetConfig(
        episodes=int(args.episodes),
        max_steps=int(args.max_steps),
        explore_prob=float(args.explore_prob),
        scenario_copies=int(args.scenario_copies),
        seed_prefix=str(args.seed_prefix),
        run_dir=Path(args.run_dir).resolve() if args.run_dir else None,
        progress_json=Path(args.progress_json).resolve() if args.progress_json else None,
    )
    result = build_dataset(config)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if int(result["records"]) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
