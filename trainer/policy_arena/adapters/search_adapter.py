from __future__ import annotations

from trainer.expert_policy import choose_action as choose_hand_action
from trainer.expert_policy_shop import choose_shop_action
from trainer.search_expert import choose_action as choose_search_action
from trainer.policy_arena.adapters.heuristic_adapter import _macro_to_action
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
