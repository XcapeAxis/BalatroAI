from __future__ import annotations

from pathlib import Path
from typing import Any

from trainer.policy_arena.adapters.heuristic_adapter import HeuristicAdapter
from trainer.policy_arena.policy_adapter import AdapterDescriptor, BasePolicyAdapter, normalize_action


class ModelAdapter(BasePolicyAdapter):
    def __init__(
        self,
        *,
        name: str = "model_policy",
        model_path: str = "",
        strategy: str = "bc",
    ) -> None:
        self.model_path = str(model_path or "").strip()
        self.strategy = str(strategy or "bc").strip().lower()
        self._fallback = HeuristicAdapter(name=f"{name}_fallback")
        self._available = bool(self.model_path) and Path(self.model_path).exists()

        note = "Model checkpoint loaded."
        status = "active"
        if not self._available:
            status = "stub"
            note = (
                "Model adapter running in stub mode (checkpoint unavailable). "
                "Actions fallback to heuristic adapter while preserving unified interface."
            )
        super().__init__(
            descriptor=AdapterDescriptor(
                name=name,
                family="model",
                status=status,
                supports_batch=False,
                supports_shop=True,
                supports_consumables=True,
                supports_position_actions=False,
                notes=note,
            )
        )

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload["adapter"]["strategy"] = self.strategy
        payload["adapter"]["model_path"] = self.model_path
        payload["adapter"]["available"] = bool(self._available)
        return payload

    def reset(self, seed: str | int | None = None) -> None:
        super().reset(seed)
        self._fallback.reset(seed)

    def act(self, obs: dict[str, Any], legal_actions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        # P39 v1: keep a stable adapter contract even without model checkpoint.
        # Once a stable model inference entrypoint is standardized, replace this fallback path.
        action = self._fallback.act(obs, legal_actions=legal_actions)
        return normalize_action(action, phase=str(obs.get("state") or "UNKNOWN"))

    def close(self) -> None:
        self._fallback.close()
        super().close()

