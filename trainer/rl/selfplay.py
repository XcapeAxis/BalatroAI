from __future__ import annotations

import json
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from trainer.rl.rollout_buffer import RolloutBuffer


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class RLPolicyLike(Protocol):
    def act(self, obs: dict[str, Any]) -> tuple[int, float | None, float | None]:
        ...


def _resolve_action(
    *,
    policy: RLPolicyLike | None,
    obs: dict[str, Any],
    rng: random.Random,
) -> tuple[int, float | None, float | None]:
    if policy is not None:
        return policy.act(obs)
    legal_ids = [int(x) for x in (obs.get("legal_action_ids") or [])]
    if legal_ids:
        action = int(rng.choice(legal_ids))
    else:
        action = 0
    return action, None, None


def run_selfplay(
    *,
    policy: RLPolicyLike | None,
    env,
    episodes: int,
    seed: str,
    gamma: float = 0.99,
    run_id: str = "",
    out_root: str | Path = "docs/artifacts/p37/selfplay",
    out_dir: str | Path | None = None,
    max_steps_per_episode: int = 320,
) -> dict[str, Any]:
    ep_count = max(1, int(episodes))
    max_steps = max(1, int(max_steps_per_episode))
    run_token = str(run_id or _now_stamp())
    if out_dir is None:
        run_dir = Path(out_root).resolve() / run_token
    else:
        run_dir = Path(out_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    progress_path = run_dir / "progress.jsonl"
    rollout_path = run_dir / "rollout.jsonl"
    episodes_path = run_dir / "episodes.json"
    summary_path = run_dir / "summary.json"

    rng = random.Random(str(seed))
    buffer = RolloutBuffer()
    episode_rows: list[dict[str, Any]] = []
    started_ts = time.perf_counter()

    for ep_idx in range(ep_count):
        ep_seed = f"{seed}-ep{ep_idx:04d}"
        obs = env.reset(seed=ep_seed)
        done = False
        total_reward = 0.0
        length = 0
        episode_started = time.perf_counter()

        while not done and length < max_steps:
            action, logprob, value = _resolve_action(policy=policy, obs=obs, rng=rng)
            next_obs, reward, done, info = env.step(action)
            buffer.add(
                obs=list(obs.get("vector") or []),
                action=int(action),
                reward=float(reward),
                done=bool(done),
                legal_action_ids=[int(x) for x in (obs.get("legal_action_ids") or [])],
                logprob=logprob,
                value=value,
                episode_idx=ep_idx,
                seed=ep_seed,
            )
            _append_jsonl(
                rollout_path,
                {
                    "episode_idx": ep_idx,
                    "seed": ep_seed,
                    "step_idx": length,
                    "action": int(action),
                    "reward": float(reward),
                    "done": bool(done),
                    "phase": obs.get("phase"),
                    "info": info if isinstance(info, dict) else {},
                },
            )
            total_reward += float(reward)
            length += 1
            obs = next_obs

        wall_time = float(time.perf_counter() - episode_started)
        row = {
            "episode_idx": int(ep_idx),
            "seed": ep_seed,
            "total_reward": float(total_reward),
            "length": int(length),
            "wall_time": wall_time,
        }
        episode_rows.append(row)
        _append_jsonl(progress_path, row)

    returns = buffer.compute_returns(gamma=float(gamma))
    rewards = [float(x.get("total_reward") or 0.0) for x in episode_rows]
    lengths = [int(x.get("length") or 0) for x in episode_rows]
    avg_reward = float(statistics.mean(rewards)) if rewards else 0.0
    std_reward = float(statistics.pstdev(rewards)) if len(rewards) >= 2 else 0.0
    best_reward = float(max(rewards)) if rewards else 0.0
    avg_length = float(statistics.mean(lengths)) if lengths else 0.0

    summary = {
        "schema": "p37_selfplay_v1",
        "generated_at": _now_iso(),
        "run_id": run_token,
        "seed": str(seed),
        "episodes": int(ep_count),
        "steps": int(len(buffer)),
        "gamma": float(gamma),
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "best_episode_reward": best_reward,
        "avg_episode_length": avg_length,
        "elapsed_sec": float(time.perf_counter() - started_ts),
        "run_dir": str(run_dir),
        "paths": {
            "progress_jsonl": str(progress_path),
            "rollout_jsonl": str(rollout_path),
            "episodes_json": str(episodes_path),
            "summary_json": str(summary_path),
        },
    }
    _write_json(episodes_path, episode_rows)
    _write_json(summary_path, summary)
    _write_json(run_dir / "returns.json", {"returns": returns})
    return {
        "buffer": buffer,
        "episode_rows": episode_rows,
        "summary": summary,
        "run_dir": str(run_dir),
    }

