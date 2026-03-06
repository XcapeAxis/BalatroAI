from __future__ import annotations

from typing import Any

from trainer.policy_arena.adapters.heuristic_adapter import HeuristicAdapter
from trainer.policy_arena.adapters.model_adapter import ModelAdapter
from trainer.policy_arena.adapters.search_adapter import SearchAdapter
from trainer.policy_arena.policy_adapter import AdapterDescriptor, BasePolicyAdapter, normalize_action, phase_from_obs
from trainer.world_model.lookahead_planner import WorldModelLookaheadPlanner


class WMRerankAdapter(BasePolicyAdapter):
    def __init__(
        self,
        *,
        name: str = "heuristic_wm_rerank",
        base_policy: str = "heuristic_baseline",
        candidate_source: str = "",
        model_path: str = "",
        world_model_checkpoint: str = "",
        top_k: int = 4,
        horizon: int = 1,
        gamma: float = 0.95,
        uncertainty_penalty: float = 0.5,
        reward_weight: float = 1.0,
        score_weight: float = 0.5,
        value_weight: float = 0.15,
        terminal_bonus: float = 0.0,
        search_max_branch: int = 80,
        search_max_depth: int = 2,
        search_time_budget_ms: float = 15.0,
    ) -> None:
        self.base_policy = str(base_policy or "heuristic_baseline")
        self.candidate_source = str(candidate_source or self.base_policy)
        self.model_path = str(model_path or "")
        self.world_model_checkpoint = str(world_model_checkpoint or "")
        self.top_k = max(1, int(top_k))
        self.horizon = max(1, int(horizon))
        self.gamma = float(gamma)
        self.uncertainty_penalty = float(uncertainty_penalty)
        self.reward_weight = float(reward_weight)
        self.score_weight = float(score_weight)
        self.value_weight = float(value_weight)
        self.terminal_bonus = float(terminal_bonus)
        self.search_max_branch = int(search_max_branch)
        self.search_max_depth = int(search_max_depth)
        self.search_time_budget_ms = float(search_time_budget_ms)

        self._base = self._make_adapter(self.base_policy)
        self._candidate_adapter = self._base if self.candidate_source == self.base_policy else self._make_adapter(self.candidate_source)
        self._planner = WorldModelLookaheadPlanner(
            checkpoint_path=self.world_model_checkpoint,
            horizon=self.horizon,
            gamma=self.gamma,
            uncertainty_penalty=self.uncertainty_penalty,
            reward_weight=self.reward_weight,
            score_weight=self.score_weight,
            value_weight=self.value_weight,
            terminal_bonus=self.terminal_bonus,
        )
        available = bool(self._planner.available)
        super().__init__(
            descriptor=AdapterDescriptor(
                name=name,
                family="wm_rerank",
                status=("active" if available else "stub"),
                supports_batch=False,
                supports_shop=True,
                supports_consumables=True,
                supports_position_actions=False,
                notes=(
                    "World-model rerank wrapper around existing candidate generators."
                    if available
                    else "World-model checkpoint unavailable; falls back to base adapter."
                ),
            )
        )

    def _make_adapter(self, policy_id: str) -> BasePolicyAdapter:
        token = str(policy_id or "").strip().lower()
        if token in {"heuristic", "heuristic_baseline", "baseline", "rule", "heuristic_candidates"}:
            return HeuristicAdapter(name=policy_id)
        if token in {"search", "search_expert", "search_candidates"}:
            return SearchAdapter(
                name=policy_id,
                max_branch=self.search_max_branch,
                max_depth=self.search_max_depth,
                time_budget_ms=self.search_time_budget_ms,
            )
        return ModelAdapter(name=policy_id, model_path=self.model_path, strategy=token or "bc")

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload["adapter"]["base_policy"] = self.base_policy
        payload["adapter"]["candidate_source"] = self.candidate_source
        payload["adapter"]["world_model_assist"] = bool(self._planner.available)
        payload["adapter"]["wm_assist_enabled"] = bool(self._planner.available)
        payload["adapter"]["assist_mode"] = "rerank"
        payload["adapter"]["world_model_checkpoint"] = self.world_model_checkpoint
        payload["adapter"]["model_path"] = self.model_path
        payload["adapter"]["top_k"] = int(self.top_k)
        payload["adapter"]["horizon"] = int(self.horizon)
        payload["adapter"]["uncertainty_penalty"] = float(self.uncertainty_penalty)
        payload["adapter"]["gamma"] = float(self.gamma)
        return payload

    def reset(self, seed: str | int | None = None) -> None:
        super().reset(seed)
        self._base.reset(seed)
        if self._candidate_adapter is not self._base:
            self._candidate_adapter.reset(seed)

    def candidate_actions(
        self,
        obs: dict[str, Any],
        legal_actions: list[dict[str, Any]] | None = None,
        *,
        top_k: int = 4,
    ) -> list[dict[str, Any]]:
        rows = self._candidate_adapter.candidate_actions(obs, legal_actions=legal_actions, top_k=max(1, int(top_k)))
        if not self._planner.available:
            return rows
        reranked = self._planner.rerank_candidates(obs=obs, candidates=rows)
        return list(reranked.get("ranked_candidates") or rows)

    def act(self, obs: dict[str, Any], legal_actions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        base_action = self._base.act(obs, legal_actions=legal_actions)
        phase = phase_from_obs(obs)
        if not self._planner.available:
            return normalize_action(base_action, phase=phase)
        candidates = self._candidate_adapter.candidate_actions(obs, legal_actions=legal_actions, top_k=self.top_k)
        reranked = self._planner.rerank_candidates(obs=obs, candidates=candidates, baseline_action=base_action)
        ranked = list(reranked.get("ranked_candidates") or [])
        if ranked:
            return normalize_action(ranked[0].get("action") if isinstance(ranked[0].get("action"), dict) else {}, phase=phase)
        return normalize_action(base_action, phase=phase)

    def close(self) -> None:
        self._base.close()
        if self._candidate_adapter is not self._base:
            self._candidate_adapter.close()
        super().close()
