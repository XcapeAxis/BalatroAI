"""Weakness taxonomy utilities for P29 weakness mining v3."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BucketTemplate:
    category: str
    affected_phase: str
    likely_fix_type: str
    description: str


TAXONOMY: dict[str, BucketTemplate] = {
    "resource_exhaustion": BucketTemplate(
        category="resource_exhaustion",
        affected_phase="HAND",
        likely_fix_type="heuristic",
        description="hands/discards exhausted before target chips",
    ),
    "misplay_hand_value": BucketTemplate(
        category="misplay_hand_value",
        affected_phase="HAND",
        likely_fix_type="model",
        description="hand selection underestimates value progression",
    ),
    "shop_econ": BucketTemplate(
        category="shop_econ",
        affected_phase="SHOP",
        likely_fix_type="data",
        description="shop buy/sell/reroll economy choices degrade future rounds",
    ),
    "risky_discard": BucketTemplate(
        category="risky_discard",
        affected_phase="HAND",
        likely_fix_type="search",
        description="discard timing introduces avoidable risk",
    ),
    "blind_score_shortfall": BucketTemplate(
        category="blind_score_shortfall",
        affected_phase="HAND",
        likely_fix_type="model",
        description="cannot reach target chips by blind end",
    ),
    "economy_collapse": BucketTemplate(
        category="economy_collapse",
        affected_phase="SHOP",
        likely_fix_type="heuristic",
        description="money spiral limits future scaling",
    ),
    "runtime_latency": BucketTemplate(
        category="runtime_latency",
        affected_phase="TRANSITION",
        likely_fix_type="search",
        description="runtime regression limits iteration velocity",
    ),
    "seed_fragility": BucketTemplate(
        category="seed_fragility",
        affected_phase="TRANSITION",
        likely_fix_type="simulator coverage",
        description="high variance across fixed seeds / flaky outcomes",
    ),
    "risk_overconservative": BucketTemplate(
        category="risk_overconservative",
        affected_phase="TRANSITION",
        likely_fix_type="heuristic",
        description="risk controller blocks useful aggression",
    ),
    "policy_confusion": BucketTemplate(
        category="policy_confusion",
        affected_phase="HAND",
        likely_fix_type="model",
        description="policy oscillates between suboptimal action families",
    ),
    "action_illegality": BucketTemplate(
        category="action_illegality",
        affected_phase="HAND",
        likely_fix_type="simulator coverage",
        description="illegal/invalid action handling causes drops",
    ),
    "simulator_coverage_gap": BucketTemplate(
        category="simulator_coverage_gap",
        affected_phase="TRANSITION",
        likely_fix_type="simulator coverage",
        description="missing scenario coverage or weak diagnostics",
    ),
}


FIX_TYPE_WEIGHT: dict[str, float] = {
    "model": 1.20,
    "data": 1.15,
    "search": 1.10,
    "heuristic": 1.00,
    "simulator coverage": 0.90,
}


def normalize_category(category: str) -> str:
    token = (category or "").strip().lower().replace("-", "_")
    if token in TAXONOMY:
        return token
    return "policy_confusion"


def template_for(category: str) -> BucketTemplate:
    key = normalize_category(category)
    return TAXONOMY.get(key, TAXONOMY["policy_confusion"])


def priority_score(*, category: str, frequency: float, avg_penalty_proxy: float) -> float:
    tpl = template_for(category)
    weight = FIX_TYPE_WEIGHT.get(tpl.likely_fix_type, 1.0)
    return float(max(0.0, frequency) * (1.0 + max(0.0, avg_penalty_proxy)) * weight)


def classify_alert(record: dict[str, Any]) -> str:
    metric = str(record.get("metric_name") or "").lower()
    strategy = str(record.get("strategy") or "").lower()
    gate = str(record.get("gate_name") or "").lower()

    if "runtime" in metric:
        return "runtime_latency"
    if "win_rate" in metric:
        if "risk" in strategy:
            return "risk_overconservative"
        return "shop_econ"
    if "median_ante" in metric:
        return "risky_discard"
    if "avg_ante" in metric:
        if "candidate" in gate or "ranking" in gate:
            return "misplay_hand_value"
        return "blind_score_shortfall"
    if "flake" in metric:
        return "seed_fragility"
    return "policy_confusion"


def classify_seed_metrics(*, avg_ante: float | None, median_ante: float | None, win_rate: float | None) -> list[str]:
    out: list[str] = []
    if avg_ante is not None and avg_ante < 3.30:
        out.append("misplay_hand_value")
    if median_ante is not None and median_ante < 3.20:
        out.append("risky_discard")
    if win_rate is not None and win_rate < 0.35:
        out.append("shop_econ")
    if not out:
        out.append("policy_confusion")
    return out


def bucket_id(category: str, suffix: str = "") -> str:
    core = normalize_category(category)
    if suffix:
        return f"wb3_{core}_{suffix}"
    return f"wb3_{core}"


def default_buckets() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, tpl in TAXONOMY.items():
        rows.append(
            {
                "bucket_id": bucket_id(key),
                "category": tpl.category,
                "frequency": 0.0,
                "avg_penalty_proxy": 0.0,
                "affected_phase": tpl.affected_phase,
                "likely_fix_type": tpl.likely_fix_type,
                "priority_score": 0.0,
                "impact": {"avg_ante_loss_proxy": 0.0, "median_ante_loss_proxy": 0.0, "win_rate_loss_proxy": 0.0},
                "signals": [],
            }
        )
    return rows
