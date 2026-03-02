"""Sampling helpers for P29 targeted dataset generation."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass
class WeightedItem:
    key: str
    weight: float


def normalize_weights(mapping: dict[str, float]) -> list[WeightedItem]:
    rows: list[WeightedItem] = []
    total = 0.0
    for key, value in mapping.items():
        w = max(0.0, float(value))
        if w <= 0:
            continue
        rows.append(WeightedItem(key=key, weight=w))
        total += w
    if not rows:
        return [WeightedItem(key="default", weight=1.0)]
    if total <= 0:
        return [WeightedItem(key=rows[0].key, weight=1.0)]
    return [WeightedItem(key=r.key, weight=r.weight / total) for r in rows]


def weighted_pick(items: list[WeightedItem], rng: random.Random) -> str:
    if not items:
        return "default"
    p = rng.random()
    acc = 0.0
    last = items[-1].key
    for item in items:
        acc += max(0.0, float(item.weight))
        if p <= acc:
            return item.key
    return last


def build_bucket_weights(report: dict[str, Any], *, top_n: int = 10, floor_weight: float = 0.05) -> list[WeightedItem]:
    rows = report.get("top_buckets") if isinstance(report.get("top_buckets"), list) else []
    weights: dict[str, float] = {}
    for row in rows[:top_n]:
        if not isinstance(row, dict):
            continue
        key = str(row.get("bucket_id") or row.get("category") or "default")
        score = row.get("priority_score")
        try:
            val = max(float(score), floor_weight)
        except Exception:
            val = floor_weight
        weights[key] = val
    if not weights:
        weights = {
            "wb3_misplay_hand_value": 1.0,
            "wb3_shop_econ": 0.8,
            "wb3_risky_discard": 0.6,
            "wb3_runtime_latency": 0.4,
        }
    return normalize_weights(weights)


def build_teacher_weights(cfg: dict[str, Any]) -> list[WeightedItem]:
    mix = cfg.get("teacher_mix") if isinstance(cfg.get("teacher_mix"), dict) else {}
    if not mix:
        mix = {
            "heuristic_teacher": 0.35,
            "search_expert": 0.30,
            "risk_aware_fallback": 0.20,
            "failure_replay": 0.10,
            "champion_prior": 0.05,
        }
    return normalize_weights({str(k): float(v) for k, v in mix.items()})


def derive_phase_plan(cfg: dict[str, Any]) -> dict[str, float]:
    phase_cfg = cfg.get("phase_balance") if isinstance(cfg.get("phase_balance"), dict) else {}
    hand_w = float(phase_cfg.get("hand_weight") or 0.75)
    shop_w = float(phase_cfg.get("shop_weight") or 0.25)
    total = hand_w + shop_w
    if total <= 0:
        return {"HAND": 0.75, "SHOP": 0.25}
    return {"HAND": hand_w / total, "SHOP": shop_w / total}


def derive_stake_weights(cfg: dict[str, Any]) -> list[WeightedItem]:
    stakes = cfg.get("stake_mix") if isinstance(cfg.get("stake_mix"), dict) else {}
    if not stakes:
        stakes = {"gold": 0.6, "orange": 0.25, "white": 0.15}
    return normalize_weights({str(k): float(v) for k, v in stakes.items()})


def derive_ante_weights(cfg: dict[str, Any]) -> list[WeightedItem]:
    tiers = cfg.get("ante_tier_mix") if isinstance(cfg.get("ante_tier_mix"), dict) else {}
    if not tiers:
        tiers = {"low": 0.3, "mid": 0.45, "high": 0.25}
    return normalize_weights({str(k): float(v) for k, v in tiers.items()})
