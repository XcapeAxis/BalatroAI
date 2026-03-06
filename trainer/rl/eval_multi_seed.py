from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trainer.rl.distributed_rollout import SnapshotMaskedRolloutPolicy, load_policy_snapshot
from trainer.rl.env_adapter import RLEnvAdapter


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _evaluate_single_checkpoint(
    *,
    checkpoint_path: str | Path,
    seeds: list[str],
    episodes_per_seed: int,
    max_steps_per_episode: int,
    backend: str,
    reward_config: dict[str, Any],
    env_config: dict[str, Any],
    greedy: bool,
) -> dict[str, Any]:
    snapshot, snapshot_meta = load_policy_snapshot(checkpoint_path)
    actor = SnapshotMaskedRolloutPolicy(
        policy_snapshot=snapshot,
        seed=f"eval-{Path(str(checkpoint_path)).stem}",
        greedy=bool(greedy),
    )
    if snapshot is None:
        return {
            "checkpoint_path": str(Path(checkpoint_path).resolve()),
            "status": "stub",
            "snapshot_meta": snapshot_meta,
            "seed_results": [],
            "mean_score": 0.0,
            "std_score": 0.0,
            "mean_episode_length": 0.0,
            "invalid_rate": 1.0,
        }

    adapter = RLEnvAdapter(
        backend=str(backend),
        seed=str(seeds[0] if seeds else "AAAAAAA"),
        timeout_sec=_safe_float(env_config.get("timeout_sec"), 8.0),
        max_steps_per_episode=max(1, _safe_int(env_config.get("max_steps_per_episode"), max_steps_per_episode)),
        max_auto_steps=max(1, _safe_int(env_config.get("max_auto_steps"), 8)),
        max_ante=max(0, _safe_int(env_config.get("max_ante"), 0)),
        auto_advance=bool(env_config.get("auto_advance", True)),
        reward_config=reward_config,
    )
    try:
        seed_rows: list[dict[str, Any]] = []
        all_scores: list[float] = []
        all_lengths: list[int] = []
        all_rewards: list[float] = []
        invalid_steps = 0
        total_steps = 0
        for seed in seeds:
            episode_scores: list[float] = []
            episode_lengths: list[int] = []
            episode_rewards: list[float] = []
            for episode_idx in range(1, max(1, int(episodes_per_seed)) + 1):
                obs, info = adapter.reset(seed=f"{seed}-eval-{episode_idx:04d}")
                final_score = 0.0
                episode_reward = 0.0
                episode_steps = 0
                episode_invalid = 0
                for _ in range(max(1, int(max_steps_per_episode))):
                    action_mask = adapter.get_action_mask(obs, info)
                    action, _logprob, _value_pred, _meta = actor.act(obs=obs, action_mask=action_mask, info=info)
                    next_obs, reward, terminated, truncated, step_info = adapter.step(action)
                    total_steps += 1
                    episode_steps += 1
                    episode_reward += float(reward)
                    if bool(step_info.get("invalid_action")):
                        invalid_steps += 1
                        episode_invalid += 1
                    final_score = _safe_float(
                        ((step_info.get("episode_metrics_partial") or {}).get("score")),
                        final_score,
                    )
                    obs, info = next_obs, step_info
                    if terminated or truncated:
                        break
                episode_scores.append(float(final_score))
                episode_lengths.append(int(episode_steps))
                episode_rewards.append(float(episode_reward))
                all_scores.append(float(final_score))
                all_lengths.append(int(episode_steps))
                all_rewards.append(float(episode_reward))
            seed_rows.append(
                {
                    "seed": str(seed),
                    "mean_score": float(statistics.mean(episode_scores)) if episode_scores else 0.0,
                    "std_score": float(statistics.pstdev(episode_scores)) if len(episode_scores) > 1 else 0.0,
                    "mean_episode_length": float(statistics.mean(episode_lengths)) if episode_lengths else 0.0,
                    "mean_reward": float(statistics.mean(episode_rewards)) if episode_rewards else 0.0,
                    "invalid_rate": float(episode_invalid) / float(max(1, sum(episode_lengths))),
                }
            )
        return {
            "checkpoint_path": str(Path(checkpoint_path).resolve()),
            "status": "ok",
            "snapshot_meta": snapshot_meta,
            "seed_results": seed_rows,
            "mean_score": float(statistics.mean(all_scores)) if all_scores else 0.0,
            "std_score": float(statistics.pstdev(all_scores)) if len(all_scores) > 1 else 0.0,
            "mean_episode_length": float(statistics.mean(all_lengths)) if all_lengths else 0.0,
            "mean_reward": float(statistics.mean(all_rewards)) if all_rewards else 0.0,
            "invalid_rate": float(invalid_steps) / float(max(1, total_steps)),
        }
    finally:
        adapter.close()


def run_multi_seed_evaluation(
    *,
    checkpoint_paths: list[str | Path] | str | Path,
    seeds: list[str] | None = None,
    episodes_per_seed: int = 1,
    max_steps_per_episode: int = 180,
    backend: str = "sim",
    reward_config: dict[str, Any] | None = None,
    env_config: dict[str, Any] | None = None,
    greedy: bool = True,
    run_id: str = "",
    out_root: str | Path = "docs/artifacts/p44/eval",
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint_list: list[str | Path]
    if isinstance(checkpoint_paths, (str, Path)):
        checkpoint_list = [checkpoint_paths]
    else:
        checkpoint_list = list(checkpoint_paths)
    checkpoint_list = [path for path in checkpoint_list if str(path).strip()]
    seed_list = [str(seed).strip() for seed in (seeds or ["AAAAAAA", "BBBBBBB", "CCCCCCC", "DDDDDDD"]) if str(seed).strip()]
    if not seed_list:
        seed_list = ["AAAAAAA", "BBBBBBB"]

    run_token = str(run_id or _now_stamp())
    run_dir = Path(out_dir).resolve() if out_dir else (Path(out_root).resolve() / run_token)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_results_path = run_dir / "seed_results.json"

    checkpoint_results = [
        _evaluate_single_checkpoint(
            checkpoint_path=checkpoint_path,
            seeds=seed_list,
            episodes_per_seed=max(1, int(episodes_per_seed)),
            max_steps_per_episode=max(1, int(max_steps_per_episode)),
            backend=str(backend),
            reward_config=dict(reward_config or {}),
            env_config=dict(env_config or {}),
            greedy=bool(greedy),
        )
        for checkpoint_path in checkpoint_list
    ]
    ok_rows = [row for row in checkpoint_results if str(row.get("status")) == "ok"]
    sorted_rows = sorted(
        checkpoint_results,
        key=lambda row: (
            str(row.get("status")) != "ok",
            -_safe_float(row.get("mean_score"), 0.0),
            _safe_float(row.get("invalid_rate"), 1.0),
            _safe_float(row.get("std_score"), 999999.0),
        ),
    )
    best_checkpoint = str(sorted_rows[0].get("checkpoint_path") or "") if sorted_rows else ""
    payload = {
        "schema": "p44_multi_seed_eval_v1",
        "generated_at": _now_iso(),
        "run_id": run_token,
        "status": "ok" if ok_rows else "stub",
        "backend": str(backend),
        "seeds": seed_list,
        "episodes_per_seed": int(episodes_per_seed),
        "max_steps_per_episode": int(max_steps_per_episode),
        "best_checkpoint": best_checkpoint,
        "checkpoint_results": sorted_rows,
    }
    _write_json(seed_results_path, payload)
    return {
        "status": str(payload.get("status")),
        "run_id": run_token,
        "run_dir": str(run_dir),
        "seed_results_json": str(seed_results_path),
        "best_checkpoint": best_checkpoint,
        "checkpoint_results": sorted_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P44 multi-seed RL evaluation.")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--checkpoints", default="", help="Optional comma-separated list of checkpoints.")
    parser.add_argument("--out-root", default="docs/artifacts/p44/eval")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--seeds", default="AAAAAAA,BBBBBBB,CCCCCCC,DDDDDDD")
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--max-steps-per-episode", type=int, default=180)
    parser.add_argument("--backend", default="sim")
    parser.add_argument("--greedy", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoints = [token.strip() for token in str(args.checkpoints or "").split(",") if token.strip()]
    if str(args.checkpoint or "").strip():
        checkpoints.append(str(args.checkpoint).strip())
    summary = run_multi_seed_evaluation(
        checkpoint_paths=checkpoints,
        seeds=[token.strip() for token in str(args.seeds).split(",") if token.strip()],
        episodes_per_seed=max(1, int(args.episodes_per_seed)),
        max_steps_per_episode=max(1, int(args.max_steps_per_episode)),
        backend=str(args.backend),
        greedy=bool(args.greedy),
        run_id=str(args.run_id or ""),
        out_root=str(args.out_root),
        out_dir=(str(args.out_dir) if str(args.out_dir).strip() else None),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if str(summary.get("status")) in {"ok", "stub"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
