from __future__ import annotations

from typing import Any

from trainer import action_space
from trainer.expert_policy import choose_action as choose_hand_action
from trainer.expert_policy_shop import choose_shop_action
from trainer.policy_arena.policy_adapter import AdapterDescriptor, BasePolicyAdapter, normalize_action, phase_default_action


def _macro_to_action(
    macro_action: str,
    macro_params: dict[str, Any] | None,
    *,
    fallback_seed: str,
) -> dict[str, Any]:
    params = dict(macro_params or {})
    macro = str(macro_action or "wait").strip().lower()
    if macro == "select":
        return {"action_type": "SELECT", "index": int(params.get("index") or 0)}
    if macro == "cash_out":
        return {"action_type": "CASH_OUT"}
    if macro == "next_round":
        return {"action_type": "NEXT_ROUND"}
    if macro == "start":
        return {
            "action_type": "START",
            "seed": str(params.get("seed") or fallback_seed),
            "deck": str(params.get("deck") or "RED"),
            "stake": str(params.get("stake") or "WHITE"),
        }
    return {"action_type": "WAIT"}


class HeuristicAdapter(BasePolicyAdapter):
    def __init__(self, *, name: str = "heuristic_baseline") -> None:
        super().__init__(
            descriptor=AdapterDescriptor(
                name=name,
                family="heuristic",
                status="active",
                supports_batch=True,
                supports_shop=True,
                supports_consumables=True,
                supports_position_actions=False,
                notes="Wraps trainer.expert_policy + trainer.expert_policy_shop without model dependencies.",
            )
        )

    def act(self, obs: dict[str, Any], legal_actions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        phase = str(obs.get("state") or "UNKNOWN").upper()

        if phase in {"SHOP", "SMODS_BOOSTER_OPENED"} or "PACK" in phase or "BOOSTER" in phase:
            shop_decision = choose_shop_action(obs)
            action = dict(shop_decision.action) if isinstance(shop_decision.action, dict) else {"action_type": "WAIT"}
            return normalize_action(action, phase=phase)

        decision = choose_hand_action(obs, start_seed=self._seed)
        if decision.action_type in {action_space.PLAY, action_space.DISCARD} and decision.mask_int is not None:
            hand_cards = (obs.get("hand") or {}).get("cards") if isinstance(obs.get("hand"), dict) else []
            hand_size = min(len(hand_cards or []), action_space.MAX_HAND)
            if hand_size > 0:
                indices = action_space.mask_to_indices(int(decision.mask_int), hand_size)
                return normalize_action({"action_type": decision.action_type, "indices": indices}, phase=phase)

        if decision.macro_action:
            action = _macro_to_action(decision.macro_action, decision.macro_params, fallback_seed=self._seed)
            return normalize_action(action, phase=phase)

        return phase_default_action(obs, seed=self._seed)

