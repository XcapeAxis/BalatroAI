from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RolloutStepRecord:
    seed: str
    episode_id: str
    step_id: int
    obs_vector: list[float]
    action: int
    action_logprob: float | None
    value_pred: float | None
    reward: float
    terminated: bool
    truncated: bool
    action_mask_density: float
    action_mask_legal_count: int
    legal_action_ids: list[int]
    invalid_action: bool
    phase: str
    score_delta: float
    ante: int
    round_num: int
    info_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "p42_rollout_step_v1",
            "seed": str(self.seed),
            "episode_id": str(self.episode_id),
            "step_id": int(self.step_id),
            "obs_vector": [float(x) for x in self.obs_vector],
            "action": int(self.action),
            "action_logprob": (float(self.action_logprob) if self.action_logprob is not None else None),
            "value_pred": (float(self.value_pred) if self.value_pred is not None else None),
            "reward": float(self.reward),
            "terminated": bool(self.terminated),
            "truncated": bool(self.truncated),
            "action_mask_density": float(self.action_mask_density),
            "action_mask_legal_count": int(self.action_mask_legal_count),
            "legal_action_ids": [int(x) for x in self.legal_action_ids],
            "invalid_action": bool(self.invalid_action),
            "phase": str(self.phase),
            "score_delta": float(self.score_delta),
            "ante": int(self.ante),
            "round_num": int(self.round_num),
            "info_summary": self.info_summary if isinstance(self.info_summary, dict) else {},
        }
