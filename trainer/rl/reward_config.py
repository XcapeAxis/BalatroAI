from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y"}:
        return True
    if token in {"0", "false", "no", "n"}:
        return False
    return bool(default)


@dataclass
class RewardConfig:
    score_delta_weight: float = 1.0
    survival_bonus: float = 0.0
    terminal_win_bonus: float = 0.0
    terminal_loss_penalty: float = 0.0
    invalid_action_penalty: float = 0.05
    clip_abs: float = 0.0
    penalize_requested_invalid: bool = True
    win_score_threshold: float = 1.0
    invalid_action_mode: str = "fallback_first_legal"

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "RewardConfig":
        payload = raw if isinstance(raw, dict) else {}
        return cls(
            score_delta_weight=_safe_float(payload.get("score_delta_weight"), 1.0),
            survival_bonus=_safe_float(payload.get("survival_bonus"), 0.0),
            terminal_win_bonus=_safe_float(payload.get("terminal_win_bonus"), 0.0),
            terminal_loss_penalty=max(0.0, _safe_float(payload.get("terminal_loss_penalty"), 0.0)),
            invalid_action_penalty=max(0.0, _safe_float(payload.get("invalid_action_penalty"), 0.05)),
            clip_abs=max(0.0, _safe_float(payload.get("clip_abs"), 0.0)),
            penalize_requested_invalid=_safe_bool(payload.get("penalize_requested_invalid"), True),
            win_score_threshold=_safe_float(payload.get("win_score_threshold"), 1.0),
            invalid_action_mode=str(payload.get("invalid_action_mode") or "fallback_first_legal").strip().lower(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clip_reward(value: float, clip_abs: float) -> float:
    if clip_abs <= 0.0:
        return float(value)
    bound = abs(float(clip_abs))
    return max(-bound, min(bound, float(value)))


def compute_shaped_reward(
    *,
    score_delta: float,
    terminated: bool,
    truncated: bool,
    is_win: bool,
    invalid_action: bool,
    invalid_action_requested: bool,
    config: RewardConfig,
) -> tuple[float, dict[str, float]]:
    components: dict[str, float] = {
        "score_delta_term": float(config.score_delta_weight) * float(score_delta),
        "survival_bonus_term": 0.0,
        "terminal_bonus_term": 0.0,
        "invalid_penalty_term": 0.0,
    }

    if not terminated and not truncated:
        components["survival_bonus_term"] = float(config.survival_bonus)

    if terminated and not truncated:
        if bool(is_win):
            components["terminal_bonus_term"] += float(config.terminal_win_bonus)
        else:
            components["terminal_bonus_term"] -= float(config.terminal_loss_penalty)

    should_penalize_invalid = bool(invalid_action)
    if bool(config.penalize_requested_invalid) and bool(invalid_action_requested):
        should_penalize_invalid = True
    if should_penalize_invalid:
        components["invalid_penalty_term"] = -float(config.invalid_action_penalty)

    reward = float(sum(float(v) for v in components.values()))
    clipped_reward = clip_reward(reward, float(config.clip_abs))
    components["reward_unclipped"] = reward
    components["reward_clipped"] = clipped_reward
    return clipped_reward, components

