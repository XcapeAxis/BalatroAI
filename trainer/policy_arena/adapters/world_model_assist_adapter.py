from __future__ import annotations

from typing import Any

from trainer.policy_arena.adapters.heuristic_adapter import HeuristicAdapter
from trainer.policy_arena.policy_adapter import AdapterDescriptor, BasePolicyAdapter, normalize_action, phase_from_obs
from trainer.world_model.planning_hook import WorldModelPlanningHook


class WorldModelAssistAdapter(BasePolicyAdapter):
    def __init__(
        self,
        *,
        name: str = "heuristic_wm_assist",
        base_policy: str = "heuristic_baseline",
        world_model_checkpoint: str = "",
        assist_mode: str = "one_step_heuristic",
        weight: float = 0.35,
        uncertainty_penalty: float = 0.5,
    ) -> None:
        self.base_policy = str(base_policy or "heuristic_baseline")
        self.world_model_checkpoint = str(world_model_checkpoint or "")
        self.assist_mode = str(assist_mode or "one_step_heuristic")
        self.weight = float(weight)
        self.uncertainty_penalty = float(uncertainty_penalty)
        self._base = HeuristicAdapter(name=self.base_policy)
        self._planner = WorldModelPlanningHook(
            checkpoint_path=self.world_model_checkpoint,
            assist_mode=self.assist_mode,
            weight=self.weight,
            uncertainty_penalty=self.uncertainty_penalty,
        )
        available = bool(self._planner.available)
        super().__init__(
            descriptor=AdapterDescriptor(
                name=name,
                family="world_model_assist",
                status=("active" if available else "stub"),
                supports_batch=False,
                supports_shop=True,
                supports_consumables=True,
                supports_position_actions=False,
                notes=(
                    "Heuristic baseline with world-model one-step reranking."
                    if available
                    else "World-model checkpoint unavailable; falls back to heuristic baseline."
                ),
            )
        )

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload["adapter"]["base_policy"] = self.base_policy
        payload["adapter"]["world_model_assist"] = bool(self._planner.available)
        payload["adapter"]["world_model_checkpoint"] = self.world_model_checkpoint
        payload["adapter"]["assist_mode"] = self.assist_mode
        payload["adapter"]["weight"] = float(self.weight)
        payload["adapter"]["uncertainty_penalty"] = float(self.uncertainty_penalty)
        return payload

    def reset(self, seed: str | int | None = None) -> None:
        super().reset(seed)
        self._base.reset(seed)

    def act(self, obs: dict[str, Any], legal_actions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        base_action = self._base.act(obs, legal_actions=legal_actions)
        phase = phase_from_obs(obs)
        if not self._planner.available or not isinstance(legal_actions, list) or not legal_actions:
            return normalize_action(base_action, phase=phase)

        ranked = self._planner.score_candidates(
            obs=obs,
            legal_actions=legal_actions,
            fallback_action=base_action,
        )
        if ranked:
            top = ranked[0]
            action = top.get("action") if isinstance(top.get("action"), dict) else {}
            return normalize_action(action, phase=phase)
        return normalize_action(base_action, phase=phase)

    def close(self) -> None:
        self._base.close()
        super().close()
