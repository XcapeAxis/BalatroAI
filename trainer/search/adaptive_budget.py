from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


@dataclass(frozen=True)
class AdaptiveBudgetPlan:
    risk_score: float
    risk_bucket: str
    depth_override: int
    rollout_count: int
    pruning_threshold: float
    budget_multiplier: float
    time_budget_ms: float
    hand_max_candidates: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_adaptive_budget(
    *,
    risk_score: float,
    risk_bucket: str,
    adaptive_cfg: dict[str, Any] | None = None,
    budget_multiplier: float = 1.0,
    depth_override: int | None = None,
) -> AdaptiveBudgetPlan:
    cfg = adaptive_cfg or {}
    base_depth = _safe_int(cfg.get("base_depth"), 2)
    base_rollouts = _safe_int(cfg.get("base_rollouts"), 48)
    base_pruning = _safe_float(cfg.get("base_pruning_threshold"), 0.08)
    base_time_budget_ms = _safe_float(cfg.get("base_time_budget_ms"), 15.0)
    base_candidates = _safe_int(cfg.get("base_hand_max_candidates"), 80)

    medium_depth_bonus = _safe_int(cfg.get("medium_depth_bonus"), 1)
    high_depth_bonus = _safe_int(cfg.get("high_depth_bonus"), 2)
    medium_rollout_multiplier = _safe_float(cfg.get("medium_rollout_multiplier"), 1.35)
    high_rollout_multiplier = _safe_float(cfg.get("high_rollout_multiplier"), 1.8)
    medium_budget_multiplier = _safe_float(cfg.get("medium_budget_multiplier"), 1.2)
    high_budget_multiplier = _safe_float(cfg.get("high_budget_multiplier"), 1.5)
    medium_pruning_multiplier = _safe_float(cfg.get("medium_pruning_multiplier"), 0.85)
    high_pruning_multiplier = _safe_float(cfg.get("high_pruning_multiplier"), 0.70)

    bucket = str(risk_bucket or "low").lower()
    depth = int(base_depth)
    rollouts = int(base_rollouts)
    pruning = float(base_pruning)
    bucket_budget_multiplier = 1.0

    if bucket == "medium":
        depth += int(medium_depth_bonus)
        rollouts = int(round(rollouts * medium_rollout_multiplier))
        pruning *= medium_pruning_multiplier
        bucket_budget_multiplier = medium_budget_multiplier
    elif bucket == "high":
        depth += int(high_depth_bonus)
        rollouts = int(round(rollouts * high_rollout_multiplier))
        pruning *= high_pruning_multiplier
        bucket_budget_multiplier = high_budget_multiplier

    # Extra smooth adjustment from continuous risk score.
    risk = _clamp(float(risk_score), 0.0, 1.0)
    rollouts = int(round(rollouts * (1.0 + 0.3 * risk)))
    budget_mul = _clamp(float(budget_multiplier) * bucket_budget_multiplier * (1.0 + 0.15 * risk), 0.5, 3.0)
    depth = max(1, int(depth_override) if depth_override is not None else int(depth))
    rollouts = max(8, min(512, int(rollouts)))
    pruning = _clamp(pruning, 0.005, 0.25)
    time_budget_ms = max(1.0, float(base_time_budget_ms) * budget_mul)
    hand_max_candidates = max(8, min(256, int(round(base_candidates * budget_mul))))

    return AdaptiveBudgetPlan(
        risk_score=risk,
        risk_bucket=bucket,
        depth_override=depth,
        rollout_count=rollouts,
        pruning_threshold=pruning,
        budget_multiplier=budget_mul,
        time_budget_ms=time_budget_ms,
        hand_max_candidates=hand_max_candidates,
    )


def to_search_kwargs(plan: AdaptiveBudgetPlan) -> dict[str, Any]:
    return {
        "max_depth": int(plan.depth_override),
        "max_branch": int(plan.hand_max_candidates),
        "time_budget_ms": float(plan.time_budget_ms),
        # Included for callers that support richer budget controls.
        "rollout_count": int(plan.rollout_count),
        "pruning_threshold": float(plan.pruning_threshold),
        "budget_multiplier": float(plan.budget_multiplier),
    }

