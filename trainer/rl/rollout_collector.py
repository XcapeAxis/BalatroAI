from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from trainer.rl.action_mask import action_mask_density
from trainer.rl.env_adapter import RLEnvAdapter
from trainer.rl.reward_config import RewardConfig
from trainer.rl.rollout_schema import RolloutStepRecord


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


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


class RolloutPolicy(Protocol):
    def act(
        self,
        *,
        obs: dict[str, Any],
        action_mask: list[int],
        info: dict[str, Any],
    ) -> tuple[int, float | None, float | None, dict[str, Any] | None]:
        ...


class RandomLegalPolicy:
    def __init__(self, *, seed: str = "AAAAAAA") -> None:
        self._rng = random.Random(str(seed))

    def act(
        self,
        *,
        obs: dict[str, Any],
        action_mask: list[int],
        info: dict[str, Any],
    ) -> tuple[int, float | None, float | None, dict[str, Any] | None]:
        legal_ids = [idx for idx, flag in enumerate(action_mask) if int(flag) > 0]
        if legal_ids:
            action = int(self._rng.choice(legal_ids))
        else:
            action = 0
        return action, None, None, {"policy": "random_legal"}


def _stats_markdown(stats: dict[str, Any]) -> str:
    lines = [
        "# P42 Rollout Stats",
        "",
        f"- run_id: `{stats.get('run_id')}`",
        f"- status: `{stats.get('status')}`",
        f"- step_count: `{int(stats.get('step_count') or 0)}`",
        f"- episode_count: `{int(stats.get('episode_count') or 0)}`",
        f"- avg_reward: `{float(stats.get('avg_reward') or 0.0):.6f}`",
        f"- avg_episode_length: `{float(stats.get('avg_episode_length') or 0.0):.6f}`",
        f"- invalid_action_rate: `{float(stats.get('invalid_action_rate') or 0.0):.6f}`",
        f"- mask_density_mean: `{float(stats.get('mask_density_mean') or 0.0):.6f}`",
    ]
    early = str(stats.get("early_stop_reason") or "").strip()
    if early:
        lines.append(f"- early_stop_reason: `{early}`")
    warnings = stats.get("warnings") if isinstance(stats.get("warnings"), list) else []
    if warnings:
        lines.extend(["", "## Warnings"])
        lines.extend([f"- {str(w)}" for w in warnings])
    return "\n".join(lines) + "\n"


def run_rollout_collection(
    *,
    policy: RolloutPolicy | None = None,
    policy_id: str = "random_legal",
    seeds: list[str] | None = None,
    episodes_per_seed: int = 1,
    max_steps_per_episode: int = 120,
    total_steps_cap: int = 2000,
    early_stop_invalid_rate: float = 0.40,
    min_steps_before_early_stop: int = 64,
    run_id: str = "",
    out_root: str | Path = "docs/artifacts/p42/rollouts",
    out_dir: str | Path | None = None,
    backend: str = "sim",
    reward_config: dict[str, Any] | RewardConfig | None = None,
    include_steps_in_result: bool = False,
) -> dict[str, Any]:
    seed_list = [str(s).strip() for s in (seeds or ["AAAAAAA", "BBBBBBB"]) if str(s).strip()]
    if not seed_list:
        seed_list = ["AAAAAAA", "BBBBBBB"]
    run_token = str(run_id or _now_stamp())
    run_dir = Path(out_dir).resolve() if out_dir else Path(out_root).resolve() / run_token
    run_dir.mkdir(parents=True, exist_ok=True)

    reward_cfg = (
        reward_config
        if isinstance(reward_config, RewardConfig)
        else RewardConfig.from_mapping(reward_config if isinstance(reward_config, dict) else {})
    )
    actor = policy if policy is not None else RandomLegalPolicy(seed=seed_list[0])
    adapter = RLEnvAdapter(
        backend=backend,
        seed=seed_list[0],
        max_steps_per_episode=max(8, int(max_steps_per_episode)),
        reward_config=reward_cfg,
    )

    rollout_steps_path = run_dir / "rollout_steps.jsonl"
    manifest_path = run_dir / "rollout_manifest.json"
    stats_json_path = run_dir / "rollout_stats.json"
    stats_md_path = run_dir / "rollout_stats.md"

    all_steps: list[dict[str, Any]] = []
    warnings: list[str] = []
    per_seed_episode_rewards: dict[str, list[float]] = {seed: [] for seed in seed_list}
    per_seed_episode_lengths: dict[str, list[int]] = {seed: [] for seed in seed_list}
    invalid_steps = 0
    step_count = 0
    mask_densities: list[float] = []
    episode_count = 0
    exception_count = 0
    early_stop_reason = ""

    try:
        for seed in seed_list:
            for ep_idx in range(max(1, int(episodes_per_seed))):
                episode_id = f"{seed}-ep{ep_idx + 1:04d}"
                try:
                    obs, info = adapter.reset(seed=episode_id)
                except Exception as exc:
                    exception_count += 1
                    warnings.append(f"reset_failed:{episode_id}:{exc}")
                    continue

                episode_count += 1
                ep_reward = 0.0
                ep_steps = 0
                terminated = False
                truncated = False
                for step_id in range(max(1, int(max_steps_per_episode))):
                    if int(total_steps_cap) > 0 and step_count >= int(total_steps_cap):
                        early_stop_reason = "total_steps_cap_reached"
                        break
                    action_mask = adapter.get_action_mask(obs, info)
                    mask_density = action_mask_density(action_mask)
                    mask_densities.append(mask_density)
                    legal_ids = [idx for idx, flag in enumerate(action_mask) if int(flag) > 0]
                    action, logprob, value_pred, policy_meta = actor.act(obs=obs, action_mask=action_mask, info=info)
                    next_obs, reward, terminated, truncated, step_info = adapter.step(action)
                    step = RolloutStepRecord(
                        seed=seed,
                        episode_id=episode_id,
                        step_id=int(step_id),
                        obs_vector=[float(x) for x in (obs.get("vector") or [])],
                        action=int(action),
                        action_logprob=logprob,
                        value_pred=value_pred,
                        reward=float(reward),
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                        action_mask_density=float(mask_density),
                        action_mask_legal_count=int(sum(1 for x in action_mask if int(x) > 0)),
                        legal_action_ids=legal_ids,
                        invalid_action=bool(step_info.get("invalid_action")),
                        phase=str(step_info.get("phase") or ""),
                        score_delta=float(step_info.get("score_delta") or 0.0),
                        ante=_safe_int(step_info.get("ante"), 0),
                        round_num=_safe_int(step_info.get("round"), 0),
                        info_summary={
                            "action_applied": step_info.get("action_applied"),
                            "action_resolution": step_info.get("action_resolution"),
                            "mask_density": step_info.get("mask_density"),
                            "policy_meta": policy_meta if isinstance(policy_meta, dict) else {},
                        },
                    ).to_dict()
                    _append_jsonl(rollout_steps_path, step)
                    if include_steps_in_result:
                        all_steps.append(step)
                    step_count += 1
                    ep_steps += 1
                    ep_reward += float(reward)
                    if bool(step.get("invalid_action")):
                        invalid_steps += 1
                    obs, info = next_obs, step_info
                    if terminated or truncated:
                        break

                per_seed_episode_rewards[seed].append(float(ep_reward))
                per_seed_episode_lengths[seed].append(int(ep_steps))
                if (
                    step_count >= int(min_steps_before_early_stop)
                    and step_count > 0
                    and (float(invalid_steps) / float(step_count)) > float(early_stop_invalid_rate)
                ):
                    early_stop_reason = "invalid_action_rate_above_threshold"
                    break
            if early_stop_reason:
                break
    finally:
        adapter.close()

    all_rewards = [float(x) for rows in per_seed_episode_rewards.values() for x in rows]
    all_lengths = [int(x) for rows in per_seed_episode_lengths.values() for x in rows]
    avg_reward = float(statistics.mean(all_rewards)) if all_rewards else 0.0
    avg_episode_length = float(statistics.mean(all_lengths)) if all_lengths else 0.0
    invalid_action_rate = (float(invalid_steps) / float(step_count)) if step_count > 0 else 0.0
    mask_density_mean = float(statistics.mean(mask_densities)) if mask_densities else 0.0
    mask_density_min = float(min(mask_densities)) if mask_densities else 0.0
    mask_density_max = float(max(mask_densities)) if mask_densities else 0.0

    stats = {
        "schema": "p42_rollout_stats_v1",
        "generated_at": _now_iso(),
        "run_id": run_token,
        "status": "ok" if step_count > 0 else "stub",
        "policy_id": str(policy_id),
        "seed_count": len(seed_list),
        "seeds": seed_list,
        "episode_count": int(episode_count),
        "step_count": int(step_count),
        "avg_reward": float(avg_reward),
        "avg_episode_length": float(avg_episode_length),
        "invalid_action_count": int(invalid_steps),
        "invalid_action_rate": float(invalid_action_rate),
        "mask_density_mean": float(mask_density_mean),
        "mask_density_min": float(mask_density_min),
        "mask_density_max": float(mask_density_max),
        "exception_count": int(exception_count),
        "early_stop_reason": str(early_stop_reason),
        "warnings": warnings,
        "per_seed": {
            seed: {
                "episode_count": len(per_seed_episode_rewards.get(seed, [])),
                "avg_reward": (
                    float(statistics.mean(per_seed_episode_rewards[seed]))
                    if per_seed_episode_rewards.get(seed)
                    else 0.0
                ),
                "avg_episode_length": (
                    float(statistics.mean(per_seed_episode_lengths[seed]))
                    if per_seed_episode_lengths.get(seed)
                    else 0.0
                ),
            }
            for seed in seed_list
        },
    }
    manifest = {
        "schema": "p42_rollout_manifest_v1",
        "generated_at": _now_iso(),
        "run_id": run_token,
        "run_dir": str(run_dir),
        "backend": str(backend),
        "policy_id": str(policy_id),
        "reward_config": reward_cfg.to_dict(),
        "config": {
            "seeds": seed_list,
            "episodes_per_seed": int(episodes_per_seed),
            "max_steps_per_episode": int(max_steps_per_episode),
            "total_steps_cap": int(total_steps_cap),
            "early_stop_invalid_rate": float(early_stop_invalid_rate),
            "min_steps_before_early_stop": int(min_steps_before_early_stop),
        },
        "paths": {
            "rollout_steps_jsonl": str(rollout_steps_path),
            "rollout_manifest_json": str(manifest_path),
            "rollout_stats_json": str(stats_json_path),
            "rollout_stats_md": str(stats_md_path),
        },
    }
    _write_json(manifest_path, manifest)
    _write_json(stats_json_path, stats)
    stats_md_path.write_text(_stats_markdown(stats), encoding="utf-8")

    result = {
        "status": str(stats.get("status")),
        "run_id": run_token,
        "run_dir": str(run_dir),
        "rollout_manifest": str(manifest_path),
        "rollout_steps": str(rollout_steps_path),
        "rollout_stats_json": str(stats_json_path),
        "rollout_stats_md": str(stats_md_path),
        "step_count": int(step_count),
        "episode_count": int(episode_count),
        "invalid_action_rate": float(invalid_action_rate),
    }
    if include_steps_in_result:
        result["steps"] = all_steps
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P42 online rollout collector.")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--out-root", default="docs/artifacts/p42/rollouts")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--seeds", default="AAAAAAA,BBBBBBB")
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--max-steps-per-episode", type=int, default=80)
    parser.add_argument("--total-steps-cap", type=int, default=500)
    parser.add_argument("--early-stop-invalid-rate", type=float, default=0.45)
    parser.add_argument("--backend", default="sim")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [s.strip() for s in str(args.seeds).split(",") if s.strip()]
    summary = run_rollout_collection(
        seeds=seeds,
        episodes_per_seed=max(1, int(args.episodes_per_seed)),
        max_steps_per_episode=max(1, int(args.max_steps_per_episode)),
        total_steps_cap=max(0, int(args.total_steps_cap)),
        early_stop_invalid_rate=max(0.0, float(args.early_stop_invalid_rate)),
        run_id=str(args.run_id),
        out_root=args.out_root,
        out_dir=(str(args.out_dir) if str(args.out_dir).strip() else None),
        backend=str(args.backend),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if str(summary.get("status")) in {"ok", "stub"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
