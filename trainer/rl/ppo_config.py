from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_seed_list(raw: Any, default: list[str]) -> list[str]:
    if not isinstance(raw, list):
        return list(default)
    out = [str(x).strip() for x in raw if str(x).strip()]
    return out if out else list(default)


@dataclass
class PPOTrainConfig:
    ppo_epochs: int = 2
    minibatch_size: int = 128
    clip_range: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    lr: float = 3e-4
    grad_clip_norm: float = 0.5
    normalize_advantage: bool = True
    max_updates: int = 3
    invalid_action_warn_threshold: float = 0.15
    max_kl_warn: float = 0.20
    nan_fail_fast: bool = True


@dataclass
class PPORolloutConfig:
    episodes_per_seed: int = 2
    max_steps_per_episode: int = 120
    total_steps_cap: int = 1200
    early_stop_invalid_rate: float = 0.45
    min_steps_before_early_stop: int = 64


@dataclass
class PPOEnvConfig:
    backend: str = "sim"
    timeout_sec: float = 8.0
    max_steps_per_episode: int = 320
    max_auto_steps: int = 8
    max_ante: int = 0
    auto_advance: bool = True
    reward: dict[str, Any] = field(default_factory=dict)


@dataclass
class PPOConfig:
    schema: str = "p42_ppo_lite_config_v1"
    run_id: str = ""
    seeds: list[str] = field(default_factory=lambda: ["AAAAAAA", "BBBBBBB"])
    output_artifacts_root: str = "docs/artifacts/p42/rl_train"
    env: PPOEnvConfig = field(default_factory=PPOEnvConfig)
    rollout: PPORolloutConfig = field(default_factory=PPORolloutConfig)
    train: PPOTrainConfig = field(default_factory=PPOTrainConfig)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "PPOConfig":
        payload = raw if isinstance(raw, dict) else {}
        env_raw = payload.get("env") if isinstance(payload.get("env"), dict) else {}
        rollout_raw = payload.get("rollout") if isinstance(payload.get("rollout"), dict) else {}
        train_raw = payload.get("train") if isinstance(payload.get("train"), dict) else {}
        env_cfg = PPOEnvConfig(
            backend=str(env_raw.get("backend") or "sim"),
            timeout_sec=_safe_float(env_raw.get("timeout_sec"), 8.0),
            max_steps_per_episode=max(1, _safe_int(env_raw.get("max_steps_per_episode"), 320)),
            max_auto_steps=max(1, _safe_int(env_raw.get("max_auto_steps"), 8)),
            max_ante=max(0, _safe_int(env_raw.get("max_ante"), 0)),
            auto_advance=bool(env_raw.get("auto_advance", True)),
            reward=env_raw.get("reward") if isinstance(env_raw.get("reward"), dict) else {},
        )
        rollout_cfg = PPORolloutConfig(
            episodes_per_seed=max(1, _safe_int(rollout_raw.get("episodes_per_seed"), 2)),
            max_steps_per_episode=max(1, _safe_int(rollout_raw.get("max_steps_per_episode"), 120)),
            total_steps_cap=max(0, _safe_int(rollout_raw.get("total_steps_cap"), 1200)),
            early_stop_invalid_rate=max(0.0, _safe_float(rollout_raw.get("early_stop_invalid_rate"), 0.45)),
            min_steps_before_early_stop=max(1, _safe_int(rollout_raw.get("min_steps_before_early_stop"), 64)),
        )
        train_cfg = PPOTrainConfig(
            ppo_epochs=max(1, _safe_int(train_raw.get("ppo_epochs"), 2)),
            minibatch_size=max(8, _safe_int(train_raw.get("minibatch_size"), 128)),
            clip_range=max(0.01, _safe_float(train_raw.get("clip_range"), 0.2)),
            gamma=min(1.0, max(0.0, _safe_float(train_raw.get("gamma"), 0.99))),
            gae_lambda=min(1.0, max(0.0, _safe_float(train_raw.get("gae_lambda"), 0.95))),
            entropy_coef=max(0.0, _safe_float(train_raw.get("entropy_coef"), 0.01)),
            value_coef=max(0.0, _safe_float(train_raw.get("value_coef"), 0.5)),
            lr=max(1e-6, _safe_float(train_raw.get("lr"), 3e-4)),
            grad_clip_norm=max(0.0, _safe_float(train_raw.get("grad_clip_norm"), 0.5)),
            normalize_advantage=bool(train_raw.get("normalize_advantage", True)),
            max_updates=max(1, _safe_int(train_raw.get("max_updates"), 3)),
            invalid_action_warn_threshold=max(0.0, _safe_float(train_raw.get("invalid_action_warn_threshold"), 0.15)),
            max_kl_warn=max(0.0, _safe_float(train_raw.get("max_kl_warn"), 0.20)),
            nan_fail_fast=bool(train_raw.get("nan_fail_fast", True)),
        )
        return cls(
            schema=str(payload.get("schema") or "p42_ppo_lite_config_v1"),
            run_id=str(payload.get("run_id") or ""),
            seeds=_as_seed_list(payload.get("seeds"), ["AAAAAAA", "BBBBBBB"]),
            output_artifacts_root=str(payload.get("output_artifacts_root") or "docs/artifacts/p42/rl_train"),
            env=env_cfg,
            rollout=rollout_cfg,
            train=train_cfg,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

