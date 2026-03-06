from __future__ import annotations

from importlib import import_module

__all__ = ["HeuristicAdapter", "SearchAdapter", "ModelAdapter", "HybridAdapter", "WorldModelAssistAdapter", "WMRerankAdapter"]


_MODULE_MAP = {
    "HeuristicAdapter": "trainer.policy_arena.adapters.heuristic_adapter",
    "SearchAdapter": "trainer.policy_arena.adapters.search_adapter",
    "ModelAdapter": "trainer.policy_arena.adapters.model_adapter",
    "HybridAdapter": "trainer.policy_arena.adapters.hybrid_adapter",
    "WorldModelAssistAdapter": "trainer.policy_arena.adapters.world_model_assist_adapter",
    "WMRerankAdapter": "trainer.policy_arena.adapters.wm_rerank_adapter",
}


def __getattr__(name: str):
    module_name = _MODULE_MAP.get(name)
    if not module_name:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
