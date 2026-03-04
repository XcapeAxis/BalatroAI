from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trainer.rl.action_mask import (
    action_mask_density,
    apply_action_mask_policy,
    build_action_mask,
    normalize_legal_action_ids,
)
from trainer.rl.env import BalatroEnv
from trainer.rl.reward_config import RewardConfig, compute_shaped_reward


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


@dataclass
class RLEnvAdapterConfig:
    backend: str = "sim"
    seed: str = "AAAAAAA"
    stake: str = "WHITE"
    timeout_sec: float = 8.0
    max_steps_per_episode: int = 320
    max_auto_steps: int = 8
    max_ante: int = 0
    auto_advance: bool = True


class RLEnvAdapter:
    """P42 RL adapter over the existing sim-aligned BalatroEnv contract."""

    def __init__(
        self,
        *,
        backend: str = "sim",
        seed: str = "AAAAAAA",
        stake: str = "WHITE",
        timeout_sec: float = 8.0,
        max_steps_per_episode: int = 320,
        max_auto_steps: int = 8,
        max_ante: int = 0,
        auto_advance: bool = True,
        reward_config: RewardConfig | dict[str, Any] | None = None,
    ) -> None:
        self.config = RLEnvAdapterConfig(
            backend=str(backend),
            seed=str(seed),
            stake=str(stake),
            timeout_sec=float(timeout_sec),
            max_steps_per_episode=max(1, int(max_steps_per_episode)),
            max_auto_steps=max(1, int(max_auto_steps)),
            max_ante=max(0, int(max_ante)),
            auto_advance=bool(auto_advance),
        )
        self.reward_config = (
            reward_config
            if isinstance(reward_config, RewardConfig)
            else RewardConfig.from_mapping(reward_config if isinstance(reward_config, dict) else {})
        )
        self._env = BalatroEnv(
            backend=self.config.backend,
            seed=self.config.seed,
            stake=self.config.stake,
            timeout_sec=self.config.timeout_sec,
            reward_mode="score_delta",
            max_steps_per_episode=self.config.max_steps_per_episode,
            max_auto_steps=self.config.max_auto_steps,
            max_ante=self.config.max_ante,
            auto_advance=self.config.auto_advance,
        )
        self._last_obs: dict[str, Any] | None = None
        self._episode_index = 0
        self._episode_shaped_return = 0.0
        self._step_index = 0

    @property
    def action_dim(self) -> int:
        return int(self._env.action_dim)

    @property
    def obs_dim(self) -> int:
        return int(self._env.obs_dim)

    def close(self) -> None:
        self._env.close()

    def reset(self, seed: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        obs = self._env.reset(seed=seed)
        self._last_obs = obs
        self._episode_index += 1
        self._episode_shaped_return = 0.0
        self._step_index = 0
        info = {
            "seed": str(seed or self.config.seed),
            "episode_index": int(self._episode_index),
            "phase": str(obs.get("phase") or ""),
            "ante": _safe_int(obs.get("ante_num"), 0),
            "round": _safe_int(obs.get("round_num"), 0),
            "score": _safe_float(obs.get("score"), 0.0),
            "score_delta": 0.0,
            "invalid_action": False,
            "episode_metrics_partial": {
                "episode_length": 0,
                "episode_return_raw": 0.0,
                "episode_return_shaped": 0.0,
                "score": _safe_float(obs.get("score"), 0.0),
                "ante": _safe_int(obs.get("ante_num"), 0),
                "phase": str(obs.get("phase") or ""),
            },
        }
        return obs, info

    def get_action_mask(self, obs: dict[str, Any], info: dict[str, Any] | None = None) -> list[int]:
        dim = max(1, int(obs.get("action_dim") or self.action_dim))
        raw = obs.get("action_mask")
        if isinstance(raw, list) and len(raw) == dim:
            return [1 if int(x) > 0 else 0 for x in raw]
        legal_ids = normalize_legal_action_ids(obs.get("legal_action_ids"), action_dim=dim)
        return build_action_mask(action_dim=dim, legal_action_ids=legal_ids)

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if not isinstance(self._last_obs, dict):
            raise RuntimeError("reset() must be called before step()")

        action_mask = self.get_action_mask(self._last_obs, None)
        chosen_action, requested_invalid, action_resolution = apply_action_mask_policy(
            requested_action=int(action),
            action_mask=action_mask,
            strategy=self.reward_config.invalid_action_mode,
        )

        next_obs, env_reward_raw, done, env_info = self._env.step(chosen_action)
        env_info = env_info if isinstance(env_info, dict) else {}
        truncated = bool(env_info.get("truncated", False))
        terminated = bool(done and not truncated)
        phase_after = str(env_info.get("phase_after") or next_obs.get("phase") or "")
        score_after = _safe_float(env_info.get("score_after"), _safe_float(next_obs.get("score"), 0.0))
        score_delta = _safe_float(env_info.get("score_delta"), _safe_float(env_reward_raw, 0.0))
        invalid_action_env = bool(env_info.get("invalid_action", False))
        invalid_action = bool(invalid_action_env or requested_invalid)
        is_win = bool(terminated and score_after >= float(self.reward_config.win_score_threshold))

        shaped_reward, reward_components = compute_shaped_reward(
            score_delta=score_delta,
            terminated=terminated,
            truncated=truncated,
            is_win=is_win,
            invalid_action=invalid_action_env,
            invalid_action_requested=requested_invalid,
            config=self.reward_config,
        )

        self._step_index += 1
        self._episode_shaped_return += float(shaped_reward)
        step_mask = self.get_action_mask(next_obs, env_info)
        info = {
            "score_delta": float(score_delta),
            "round": _safe_int(next_obs.get("round_num"), 0),
            "ante": _safe_int(next_obs.get("ante_num"), 0),
            "phase": phase_after,
            "invalid_action": bool(invalid_action),
            "invalid_action_env": bool(invalid_action_env),
            "invalid_action_requested": bool(requested_invalid),
            "action_requested": int(action),
            "action_applied": int(chosen_action),
            "action_resolution": action_resolution,
            "mask_density": action_mask_density(step_mask),
            "reward_components": reward_components,
            "episode_metrics_partial": {
                "episode_length": _safe_int(env_info.get("episode_length"), self._step_index),
                "episode_return_raw": _safe_float(env_info.get("episode_return"), 0.0),
                "episode_return_shaped": float(self._episode_shaped_return),
                "score": score_after,
                "ante": _safe_int(next_obs.get("ante_num"), 0),
                "phase": phase_after,
            },
            "backend_info": env_info.get("backend_info") if isinstance(env_info.get("backend_info"), dict) else {},
        }

        self._last_obs = next_obs
        return next_obs, float(shaped_reward), bool(terminated), bool(truncated), info

