from __future__ import annotations

from importlib import import_module

__all__ = [
    "AdaptiveHybridController",
    "RuleBasedHybridRouter",
    "build_controller_registry",
    "collect_sample_states",
    "extract_routing_features",
    "run_hybrid_controller_pipeline",
    "run_hybrid_controller_smoke",
]


_MODULE_MAP = {
    "AdaptiveHybridController": "trainer.hybrid.hybrid_controller",
    "RuleBasedHybridRouter": "trainer.hybrid.router",
    "build_controller_registry": "trainer.hybrid.controller_registry",
    "collect_sample_states": "trainer.hybrid.routing_features",
    "extract_routing_features": "trainer.hybrid.routing_features",
    "run_hybrid_controller_pipeline": "trainer.hybrid.hybrid_controller",
    "run_hybrid_controller_smoke": "trainer.hybrid.hybrid_controller",
}


def __getattr__(name: str):
    module_name = _MODULE_MAP.get(name)
    if not module_name:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
