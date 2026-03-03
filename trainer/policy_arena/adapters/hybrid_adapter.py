from __future__ import annotations

from trainer.policy_arena.adapters.heuristic_adapter import HeuristicAdapter
from trainer.policy_arena.adapters.search_adapter import SearchAdapter
from trainer.policy_arena.policy_adapter import AdapterDescriptor, BasePolicyAdapter


class HybridAdapter(BasePolicyAdapter):
    def __init__(
        self,
        *,
        name: str = "hybrid_search_heuristic",
        search_threshold_hand_size: int = 4,
    ) -> None:
        super().__init__(
            descriptor=AdapterDescriptor(
                name=name,
                family="hybrid",
                status="active",
                supports_batch=False,
                supports_shop=True,
                supports_consumables=True,
                supports_position_actions=False,
                notes="Heuristic policy with search rerank on larger hands.",
            )
        )
        self.search_threshold_hand_size = max(1, int(search_threshold_hand_size))
        self._heuristic = HeuristicAdapter(name=f"{name}_heuristic")
        self._search = SearchAdapter(name=f"{name}_search")

    def reset(self, seed: str | int | None = None) -> None:
        super().reset(seed)
        self._heuristic.reset(seed)
        self._search.reset(seed)

    def act(self, obs: dict, legal_actions: list[dict] | None = None) -> dict:
        phase = str(obs.get("state") or "UNKNOWN").upper()
        if phase == "SELECTING_HAND":
            hand_cards = (obs.get("hand") or {}).get("cards") if isinstance(obs.get("hand"), dict) else []
            hand_size = len(hand_cards or [])
            if hand_size >= self.search_threshold_hand_size:
                return self._search.act(obs, legal_actions=legal_actions)
        return self._heuristic.act(obs, legal_actions=legal_actions)

    def close(self) -> None:
        self._heuristic.close()
        self._search.close()
        super().close()

