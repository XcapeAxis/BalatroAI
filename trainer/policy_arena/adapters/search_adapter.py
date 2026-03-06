from __future__ import annotations

import time
from typing import Any

from trainer.candidates_hand import generate_hand_candidates
from trainer.expert_policy import choose_action as choose_hand_action
from trainer.expert_policy_shop import choose_shop_action
from trainer.search_expert import _search_value, _simulate_action, _state_value, choose_action as choose_search_action
from trainer.policy_arena.adapters.heuristic_adapter import _macro_to_action
from trainer.policy_arena.adapters.heuristic_adapter import HeuristicAdapter
from trainer.policy_arena.policy_adapter import AdapterDescriptor, BasePolicyAdapter, normalize_action, phase_default_action


class SearchAdapter(BasePolicyAdapter):
    def __init__(
        self,
        *,
        name: str = "search_expert",
        max_branch: int = 80,
        max_depth: int = 2,
        time_budget_ms: float = 15.0,
    ) -> None:
        super().__init__(
            descriptor=AdapterDescriptor(
                name=name,
                family="search",
                status="active",
                supports_batch=True,
                supports_shop=True,
                supports_consumables=False,
                supports_position_actions=False,
                notes="Search is used for hand phase; other phases fallback to heuristic policy.",
            )
        )
        self.max_branch = int(max_branch)
        self.max_depth = int(max_depth)
        self.time_budget_ms = float(time_budget_ms)
        self._fallback = HeuristicAdapter(name=f"{name}_fallback")

    def act(self, obs: dict, legal_actions: list[dict] | None = None) -> dict:
        phase = str(obs.get("state") or "UNKNOWN").upper()
        if phase == "SELECTING_HAND":
            round_info = obs.get("round") if isinstance(obs.get("round"), dict) else {}
            hands_left = int(round_info.get("hands_left") or 0)
            discards_left = int(round_info.get("discards_left") or 0)
            if hands_left <= 0 and discards_left <= 0:
                return normalize_action({"action_type": "WAIT"}, phase=phase)
            if hands_left <= 0 and discards_left > 0:
                return self._heuristic_fallback(obs, phase)
            decision = choose_search_action(
                obs,
                max_branch=self.max_branch,
                max_depth=self.max_depth,
                time_budget_ms=self.time_budget_ms,
                seed=self._seed,
            )
            return normalize_action(
                {"action_type": decision.action_type, "indices": list(decision.indices or [])},
                phase=phase,
            )

        if phase in {"SHOP", "SMODS_BOOSTER_OPENED"} or "PACK" in phase or "BOOSTER" in phase:
            shop_decision = choose_shop_action(obs)
            action = dict(shop_decision.action) if isinstance(shop_decision.action, dict) else {"action_type": "WAIT"}
            return normalize_action(action, phase=phase)

        return self._heuristic_fallback(obs, phase)

    def _heuristic_fallback(self, obs: dict, phase: str) -> dict:
        decision = choose_hand_action(obs, start_seed=self._seed)
        if decision.macro_action:
            return normalize_action(
                _macro_to_action(decision.macro_action, decision.macro_params, fallback_seed=self._seed),
                phase=phase,
            )
        return phase_default_action(obs, seed=self._seed)

    def reset(self, seed: str | int | None = None) -> None:
        super().reset(seed)
        self._fallback.reset(seed)

    def candidate_actions(
        self,
        obs: dict[str, Any],
        legal_actions: list[dict[str, Any]] | None = None,
        *,
        top_k: int = 4,
    ) -> list[dict[str, Any]]:
        phase = str(obs.get("state") or "UNKNOWN").upper()
        limit = max(1, int(top_k))
        if phase != "SELECTING_HAND":
            return self._fallback.candidate_actions(obs, legal_actions=legal_actions, top_k=limit)

        round_info = obs.get("round") if isinstance(obs.get("round"), dict) else {}
        hands_left = int(round_info.get("hands_left") or 0)
        discards_left = int(round_info.get("discards_left") or 0)
        if hands_left <= 0 and discards_left > 0:
            return self._fallback.candidate_actions(obs, legal_actions=legal_actions, top_k=limit)

        seed = str(self._seed or "SEARCH")
        scored: list[dict[str, Any]] = []
        candidates = generate_hand_candidates(obs, max_candidates=max(limit * 4, limit + 2))
        for action in candidates:
            try:
                next_state, reward, done = _simulate_action(obs, dict(action), seed=seed)
                value = float(reward) + 0.05 * float(_state_value(next_state))
                considered = 1
                if not done and self.max_depth > 1:
                    start = time.perf_counter()
                    tail, tail_considered = _search_value(
                        next_state,
                        depth=1,
                        max_depth=self.max_depth,
                        max_branch=self.max_branch,
                        seed=seed,
                        t0=start,
                        time_budget_sec=max(0.01, float(self.time_budget_ms) / 1000.0),
                    )
                    value += 0.9 * float(tail)
                    considered += int(tail_considered)
            except Exception:
                value = float("-inf")
                considered = 0
            scored.append(
                {
                    "action": normalize_action(action, phase=phase),
                    "source": "search_candidates",
                    "source_rank": 0,
                    "source_score": float(value),
                    "legal": value > float("-inf"),
                    "metadata": {"considered": int(considered)},
                }
            )
        scored.sort(key=lambda row: float(row.get("source_score") or float("-inf")), reverse=True)
        for rank, row in enumerate(scored, start=1):
            row["source_rank"] = int(rank)
        return scored[:limit] if scored else self._fallback.candidate_actions(obs, legal_actions=legal_actions, top_k=limit)

    def close(self) -> None:
        self._fallback.close()
        super().close()
