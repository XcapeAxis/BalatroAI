from __future__ import annotations

import math
from collections import Counter
from typing import Any


FEATURE_ENCODER_SCHEMA = "p52_router_feature_encoder_v1"
DATASET_SAMPLE_SCHEMA = "p52_router_sample_v1"

SUPPORTED_CONTROLLER_IDS = (
    "policy_baseline",
    "policy_plus_wm_rerank",
    "search_baseline",
    "heuristic_baseline",
)

BOOLEAN_FEATURE_KEYS = (
    "policy_available",
    "heuristic_available",
    "search_available",
    "wm_available",
    "fallback_available",
)

NUMERIC_FEATURE_KEYS = (
    "policy_margin",
    "policy_entropy",
    "policy_top1_prob",
    "policy_top2_prob",
    "policy_topk_concentration",
    "policy_candidate_count",
    "wm_uncertainty",
    "wm_predicted_return",
    "wm_score",
    "wm_probe_count",
    "search_time_budget_ms",
    "search_max_depth",
    "search_max_branch",
    "legal_action_count",
    "round_num",
    "ante_num",
    "money",
)

CATEGORICAL_FEATURE_KEYS = (
    "phase",
    "policy_primary_source",
    "budget_level",
    "slice_stage",
    "slice_resource_pressure",
    "slice_action_type",
    "slice_position_sensitive",
    "slice_stateful_joker_present",
)

DEFAULT_CATEGORICAL_VOCABS: dict[str, tuple[str, ...]] = {
    "phase": (
        "BLIND_SELECT",
        "SELECTING_HAND",
        "SHOP",
        "ROUND_EVAL",
        "SMODS_BOOSTER_OPENED",
        "UNKNOWN",
        "unknown",
        "missing",
    ),
    "policy_primary_source": (
        "policy_topk",
        "search",
        "heuristic",
        "unknown",
        "missing",
    ),
    "budget_level": ("low", "medium", "high", "unknown", "missing"),
    "slice_stage": ("early", "mid", "late", "unknown", "missing"),
    "slice_resource_pressure": ("low", "medium", "high", "unknown", "missing"),
    "slice_action_type": ("play", "discard", "shop", "consumable", "transition", "unknown", "missing"),
    "slice_position_sensitive": ("true", "false", "unknown", "missing"),
    "slice_stateful_joker_present": ("true", "false", "unknown", "missing"),
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return float(default)
    if math.isnan(result) or math.isinf(result):
        return float(default)
    return result


def _normalize_bool_token(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value in {None, ""}:
        return "missing"
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return "true"
    if token in {"0", "false", "no", "n", "off"}:
        return "false"
    if token in {"unknown", "missing"}:
        return token
    return "unknown"


def _normalize_category_value(key: str, value: Any) -> str:
    if key in {"slice_position_sensitive", "slice_stateful_joker_present"}:
        return _normalize_bool_token(value)
    if value in {None, ""}:
        return "missing"
    token = str(value).strip()
    return token if token else "missing"


def _canonical_policy_like_id(token: str) -> str:
    raw = str(token or "").strip().lower()
    if not raw:
        return ""
    if raw in {"policy_baseline", "policy", "model_policy", "model"}:
        return "policy_baseline"
    if raw in {"policy_plus_wm_rerank", "model_wm_rerank", "policy_wm_rerank"} or "wm_rerank" in raw:
        return "policy_plus_wm_rerank"
    if raw in {"search_baseline", "search", "search_expert"}:
        return "search_baseline"
    if raw in {"heuristic_baseline", "heuristic", "baseline", "rule"}:
        return "heuristic_baseline"
    return raw


def canonicalize_controller_id(value: Any) -> str:
    token = _canonical_policy_like_id(str(value or ""))
    return token if token in SUPPORTED_CONTROLLER_IDS else ""


def supported_controller_ids() -> list[str]:
    return list(SUPPORTED_CONTROLLER_IDS)


def normalize_available_controllers(raw: Any) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    values: list[Any]
    if isinstance(raw, dict):
        values = list(raw.keys())
    elif isinstance(raw, (list, tuple, set)):
        values = list(raw)
    elif raw in {None, ""}:
        values = []
    else:
        values = [raw]
    for value in values:
        token = canonicalize_controller_id(value)
        if not token or token in seen:
            continue
        seen.add(token)
        rows.append(token)
    return rows


def available_controller_mask(raw: Any, controller_ids: list[str] | tuple[str, ...] | None = None) -> list[float]:
    chosen_ids = list(controller_ids or SUPPORTED_CONTROLLER_IDS)
    available = set(normalize_available_controllers(raw))
    return [1.0 if controller_id in available else 0.0 for controller_id in chosen_ids]


def routing_features_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(sample, dict):
        return {}
    for key in ("routing_features", "features", "key_feature_values"):
        payload = sample.get(key)
        if isinstance(payload, dict):
            return dict(payload)
    return {}


def normalize_routing_features(sample: dict[str, Any]) -> dict[str, Any]:
    features = routing_features_from_sample(sample)
    normalized: dict[str, Any] = {}
    for key in BOOLEAN_FEATURE_KEYS:
        normalized[key] = _normalize_bool_token(features.get(key))
    for key in NUMERIC_FEATURE_KEYS:
        normalized[key] = _safe_float(features.get(key), 0.0)
    for key in CATEGORICAL_FEATURE_KEYS:
        normalized[key] = _normalize_category_value(key, features.get(key))
    return normalized


def build_feature_encoder(samples: list[dict[str, Any]]) -> dict[str, Any]:
    categorical_values: dict[str, set[str]] = {
        key: set(DEFAULT_CATEGORICAL_VOCABS.get(key, ("unknown", "missing"))) for key in CATEGORICAL_FEATURE_KEYS
    }
    numeric_values: dict[str, list[float]] = {key: [] for key in (*BOOLEAN_FEATURE_KEYS, *NUMERIC_FEATURE_KEYS)}

    for sample in samples:
        normalized = normalize_routing_features(sample)
        for key in CATEGORICAL_FEATURE_KEYS:
            categorical_values[key].add(_normalize_category_value(key, normalized.get(key)))
        for key in BOOLEAN_FEATURE_KEYS:
            numeric_values[key].append(1.0 if str(normalized.get(key)) == "true" else 0.0)
        for key in NUMERIC_FEATURE_KEYS:
            numeric_values[key].append(_safe_float(normalized.get(key), 0.0))

    categorical_maps = {
        key: {token: idx for idx, token in enumerate(sorted(values))}
        for key, values in categorical_values.items()
    }
    numeric_stats = {}
    for key, values in numeric_values.items():
        clean = [float(value) for value in values if math.isfinite(float(value))]
        if not clean:
            clean = [0.0]
        mean = float(sum(clean) / max(1, len(clean)))
        variance = float(sum((value - mean) ** 2 for value in clean) / max(1, len(clean)))
        numeric_stats[key] = {
            "mean": mean,
            "std": math.sqrt(variance) if variance > 0.0 else 1.0,
            "min": min(clean),
            "max": max(clean),
        }

    feature_names: list[str] = []
    for key in (*BOOLEAN_FEATURE_KEYS, *NUMERIC_FEATURE_KEYS):
        feature_names.append(key)
    for key in CATEGORICAL_FEATURE_KEYS:
        mapping = categorical_maps.get(key) or {}
        for token in sorted(mapping.keys(), key=lambda item: mapping[item]):
            feature_names.append(f"{key}={token}")

    return {
        "schema": FEATURE_ENCODER_SCHEMA,
        "controller_ids": list(SUPPORTED_CONTROLLER_IDS),
        "boolean_feature_keys": list(BOOLEAN_FEATURE_KEYS),
        "numeric_feature_keys": list(NUMERIC_FEATURE_KEYS),
        "categorical_feature_keys": list(CATEGORICAL_FEATURE_KEYS),
        "categorical_maps": categorical_maps,
        "numeric_stats": numeric_stats,
        "feature_names": feature_names,
        "output_dim": len(feature_names),
    }


def _numeric_feature_value(key: str, normalized_features: dict[str, Any]) -> float:
    if key in BOOLEAN_FEATURE_KEYS:
        return 1.0 if str(normalized_features.get(key)) == "true" else 0.0
    return _safe_float(normalized_features.get(key), 0.0)


def encode_routing_features(sample: dict[str, Any], encoder: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_routing_features(sample)
    numeric_stats = encoder.get("numeric_stats") if isinstance(encoder.get("numeric_stats"), dict) else {}
    categorical_maps = encoder.get("categorical_maps") if isinstance(encoder.get("categorical_maps"), dict) else {}

    vector: list[float] = []
    missing_keys: list[str] = []
    unknown_categorical_keys: list[str] = []
    ood_z_scores: list[float] = []

    for key in (*BOOLEAN_FEATURE_KEYS, *NUMERIC_FEATURE_KEYS):
        value = _numeric_feature_value(key, normalized)
        vector.append(value)
        stats = numeric_stats.get(key) if isinstance(numeric_stats.get(key), dict) else {}
        mean = _safe_float(stats.get("mean"), 0.0)
        std = max(_safe_float(stats.get("std"), 1.0), 1e-6)
        ood_z_scores.append(abs((value - mean) / std))
        if key not in normalized:
            missing_keys.append(key)

    for key in CATEGORICAL_FEATURE_KEYS:
        mapping = categorical_maps.get(key) if isinstance(categorical_maps.get(key), dict) else {}
        token = _normalize_category_value(key, normalized.get(key))
        if token not in mapping:
            fallback = "unknown" if "unknown" in mapping else ("missing" if "missing" in mapping else "")
            if fallback:
                token = fallback
            unknown_categorical_keys.append(key)
        index = int(mapping.get(token, 0))
        width = max(1, len(mapping))
        for offset in range(width):
            vector.append(1.0 if offset == index else 0.0)

    completeness = 1.0 - (float(len(missing_keys) + len(unknown_categorical_keys)) / max(1, len(BOOLEAN_FEATURE_KEYS) + len(NUMERIC_FEATURE_KEYS) + len(CATEGORICAL_FEATURE_KEYS)))
    return {
        "vector": vector,
        "normalized_features": normalized,
        "completeness": max(0.0, min(1.0, completeness)),
        "missing_keys": missing_keys,
        "unknown_categorical_keys": unknown_categorical_keys,
        "ood_score": max(ood_z_scores) if ood_z_scores else 0.0,
    }


def sample_label_index(sample: dict[str, Any], controller_ids: list[str] | None = None) -> int:
    chosen_ids = list(controller_ids or SUPPORTED_CONTROLLER_IDS)
    target = canonicalize_controller_id(sample.get("target_controller_label"))
    if not target:
        return -1
    try:
        return chosen_ids.index(target)
    except ValueError:
        return -1


def normalize_target_scores(sample: dict[str, Any], controller_ids: list[str] | None = None) -> list[float]:
    chosen_ids = list(controller_ids or SUPPORTED_CONTROLLER_IDS)
    raw = sample.get("target_controller_scores") if isinstance(sample.get("target_controller_scores"), dict) else {}
    available = normalize_available_controllers(sample.get("available_controllers"))
    scores: list[float] = []
    for controller_id in chosen_ids:
        if available and controller_id not in available:
            scores.append(0.0)
            continue
        scores.append(max(0.0, _safe_float(raw.get(controller_id), 0.0)))
    total = sum(scores)
    if total > 0.0:
        return [value / total for value in scores]
    label_idx = sample_label_index(sample, controller_ids=chosen_ids)
    out = [0.0 for _ in chosen_ids]
    if 0 <= label_idx < len(out):
        out[label_idx] = 1.0
    return out


def controller_distribution(rows: list[dict[str, Any]], *, key: str) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for row in rows:
        token = canonicalize_controller_id(row.get(key))
        counter[token or "unknown"] += 1
    total = sum(counter.values())
    return [
        {"controller_id": controller_id, "count": int(count), "ratio": float(count) / max(1, total)}
        for controller_id, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def slice_value(sample: dict[str, Any], key: str) -> str:
    features = normalize_routing_features(sample)
    return str(features.get(key) or "unknown")
