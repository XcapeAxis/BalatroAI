from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

from trainer.search.adaptive_budget import AdaptiveBudgetPlan, plan_adaptive_budget

from .context_features import ContextFeatures, extract_context_features
from .risk_model import RiskEstimate, estimate_risk

ProviderFn = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class RouterDecision:
    action: dict[str, Any]
    selected_strategy: str
    reason: str
    features: dict[str, Any]
    risk: dict[str, Any]
    adaptive_budget: dict[str, Any]
    search_depth: int
    budget_multiplier: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": dict(self.action),
            "selected_strategy": self.selected_strategy,
            "reason": self.reason,
            "features": dict(self.features),
            "risk": dict(self.risk),
            "adaptive_budget": dict(self.adaptive_budget),
            "search_depth": int(self.search_depth),
            "budget_multiplier": float(self.budget_multiplier),
        }


def _default_router_config() -> dict[str, Any]:
    return {
        "schema": "p31_router_config_v1",
        "router": {
            "policy_preferred_phases": ["SHOP", "SHOPPING", "TRANSITION"],
            "search_phases": ["SELECTING_HAND"],
            "hand_strength_search_threshold": 0.45,
            "economy_pressure_threshold": 0.25,
            "budget_low_ms": 6.0,
            "risk_fallback_on_high": True,
            "prefer_deep_search_on_medium": True,
        },
        "search": {
            "shallow_depth": 2,
            "deep_depth": 4,
        },
        "risk_model": {},
        "adaptive_budget": {},
    }


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is not None:
            payload = yaml.safe_load(text)
        else:
            # Keep router usable in minimal environments: if YAML parser is
            # unavailable, fall back to a same-stem JSON config when present.
            sidecar_json = path.with_suffix(".json")
            if sidecar_json.exists():
                payload = json.loads(sidecar_json.read_text(encoding="utf-8"))
            else:
                payload = {}
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"router config must be a mapping: {path}")
    return payload


def load_router_config(config_path: str | Path | None = None) -> dict[str, Any]:
    cfg = _default_router_config()
    if config_path is None:
        return cfg
    path = Path(config_path)
    if not path.exists():
        return cfg
    loaded = _load_yaml_or_json(path)
    # shallow merge by top-level sections for predictability.
    for key, value in loaded.items():
        if key in cfg and isinstance(cfg[key], dict) and isinstance(value, dict):
            merged = dict(cfg[key])
            merged.update(value)
            cfg[key] = merged
        else:
            cfg[key] = value
    return cfg


class DecisionRouter:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or _default_router_config()

    def _provider(self, providers: dict[str, ProviderFn], name: str) -> ProviderFn | None:
        fn = providers.get(name)
        if callable(fn):
            return fn
        return None

    def _choose_strategy(
        self,
        *,
        features: ContextFeatures,
        risk: RiskEstimate,
        adaptive_budget: AdaptiveBudgetPlan,
    ) -> tuple[str, str]:
        router_cfg = self.config.get("router") if isinstance(self.config.get("router"), dict) else {}
        search_cfg = self.config.get("search") if isinstance(self.config.get("search"), dict) else {}

        policy_phases = {str(x).upper() for x in (router_cfg.get("policy_preferred_phases") or [])}
        search_phases = {str(x).upper() for x in (router_cfg.get("search_phases") or [])}
        hand_threshold = float(router_cfg.get("hand_strength_search_threshold") or 0.45)
        econ_threshold = float(router_cfg.get("economy_pressure_threshold") or 0.25)
        budget_low_ms = float(router_cfg.get("budget_low_ms") or 6.0)
        risk_fallback_on_high = bool(router_cfg.get("risk_fallback_on_high", True))
        prefer_deep_search_on_medium = bool(router_cfg.get("prefer_deep_search_on_medium", True))
        shallow_depth = int(search_cfg.get("shallow_depth") or 2)
        deep_depth = int(search_cfg.get("deep_depth") or 4)

        phase = str(features.phase).upper()
        risk_bucket = str(risk.risk_bucket)
        low_budget = float(features.budget_ms) < budget_low_ms

        if risk_bucket == "high" and risk_fallback_on_high:
            return "risk_fallback", "high_risk_fallback"

        if phase in search_phases:
            if low_budget:
                return "heuristic", "low_budget_heuristic"
            if risk_bucket == "medium":
                if prefer_deep_search_on_medium or features.hand_strength_proxy < hand_threshold:
                    return "search", f"medium_risk_search_d{max(deep_depth, adaptive_budget.depth_override)}"
                return "search", f"medium_risk_search_d{max(shallow_depth, adaptive_budget.depth_override)}"
            if features.hand_strength_proxy < hand_threshold:
                return "search", f"weak_hand_search_d{max(shallow_depth, adaptive_budget.depth_override)}"
            if features.economy_level < econ_threshold:
                return "search", f"low_economy_search_d{max(shallow_depth, adaptive_budget.depth_override)}"
            return "policy", "hand_policy_default"

        if phase in policy_phases and risk_bucket != "high":
            return "policy", "policy_phase_preferred"

        if risk_bucket == "medium":
            return "search", f"medium_risk_search_d{max(shallow_depth, adaptive_budget.depth_override)}"

        return "heuristic", "default_heuristic"

    def decide(
        self,
        state: dict[str, Any],
        *,
        providers: dict[str, ProviderFn],
        seed: str = "",
        budget_multiplier: float = 1.0,
        depth_override: int | None = None,
    ) -> RouterDecision:
        # deterministic: no random branch, selection only from state/config.
        features = extract_context_features(state, default_budget_ms=float(state.get("budget_ms") or 15.0))
        risk_cfg = self.config.get("risk_model") if isinstance(self.config.get("risk_model"), dict) else {}
        risk = estimate_risk(features, config=risk_cfg)
        adaptive_cfg = self.config.get("adaptive_budget") if isinstance(self.config.get("adaptive_budget"), dict) else {}
        adaptive_plan = plan_adaptive_budget(
            risk_score=float(risk.risk_score),
            risk_bucket=str(risk.risk_bucket),
            adaptive_cfg=adaptive_cfg,
            budget_multiplier=budget_multiplier,
            depth_override=depth_override,
        )
        selected, reason = self._choose_strategy(features=features, risk=risk, adaptive_budget=adaptive_plan)

        provider_name = selected
        if provider_name == "search":
            provider_name = "search"
        elif provider_name == "risk_fallback":
            provider_name = "risk_fallback"
        elif provider_name == "policy":
            provider_name = "policy"
        else:
            provider_name = "heuristic"

        fn = self._provider(providers, provider_name)
        if fn is None:
            fn = self._provider(providers, "heuristic")
            provider_name = "heuristic"
            reason = reason + "|fallback_to_heuristic_provider"
        if fn is None:
            action = {"action_type": "WAIT", "sleep": 0.01}
            reason = reason + "|no_provider_wait"
            provider_name = "wait"
        else:
            hint = {
                "seed": seed,
                "risk_score": risk.risk_score,
                "risk_bucket": risk.risk_bucket,
                "depth_override": adaptive_plan.depth_override,
                "rollout_count": adaptive_plan.rollout_count,
                "pruning_threshold": adaptive_plan.pruning_threshold,
                "budget_multiplier": adaptive_plan.budget_multiplier,
                "time_budget_ms": adaptive_plan.time_budget_ms,
                "search_max_branch": adaptive_plan.hand_max_candidates,
            }
            action = fn(state, hint) or {"action_type": "WAIT", "sleep": 0.01}
        return RouterDecision(
            action=action,
            selected_strategy=provider_name,
            reason=reason,
            features=features.to_dict(),
            risk=risk.to_dict(),
            adaptive_budget=adaptive_plan.to_dict(),
            search_depth=int(adaptive_plan.depth_override if provider_name == "search" else 0),
            budget_multiplier=float(adaptive_plan.budget_multiplier),
        )
