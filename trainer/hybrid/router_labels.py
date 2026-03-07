from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trainer.hybrid.router_schema import (
    CATEGORICAL_FEATURE_KEYS,
    canonicalize_controller_id,
    normalize_available_controllers,
    normalize_routing_features,
    supported_controller_ids,
)


SLICE_KEYS = (
    "slice_stage",
    "slice_resource_pressure",
    "slice_action_type",
    "slice_position_sensitive",
    "slice_stateful_joker_present",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return float(default)
    if math.isnan(result) or math.isinf(result):
        return float(default)
    return result


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _new_metric_cell() -> dict[str, Any]:
    return {
        "weighted_score_sum": 0.0,
        "weighted_win_sum": 0.0,
        "weight_sum": 0.0,
        "count_sum": 0,
        "source_paths": [],
    }


def _append_metric(cell: dict[str, Any], *, mean_total_score: float, win_rate: float, count: int, source_path: str) -> None:
    weight = max(1, int(count))
    cell["weighted_score_sum"] = _safe_float(cell.get("weighted_score_sum"), 0.0) + (mean_total_score * weight)
    cell["weighted_win_sum"] = _safe_float(cell.get("weighted_win_sum"), 0.0) + (win_rate * weight)
    cell["weight_sum"] = _safe_float(cell.get("weight_sum"), 0.0) + float(weight)
    cell["count_sum"] = _safe_int(cell.get("count_sum"), 0) + weight
    rows = cell.get("source_paths") if isinstance(cell.get("source_paths"), list) else []
    if source_path and source_path not in rows:
        rows.append(source_path)
    cell["source_paths"] = rows[:24]


def _finalize_metric(cell: dict[str, Any]) -> dict[str, Any]:
    weight = max(_safe_float(cell.get("weight_sum"), 0.0), 1.0)
    return {
        "mean_total_score": _safe_float(cell.get("weighted_score_sum"), 0.0) / weight,
        "win_rate": _safe_float(cell.get("weighted_win_sum"), 0.0) / weight,
        "count": _safe_int(cell.get("count_sum"), 0),
        "source_paths": list(cell.get("source_paths") or []),
    }


def build_knowledge_base(
    *,
    summary_paths: list[Path] | None = None,
    bucket_paths: list[Path] | None = None,
) -> dict[str, Any]:
    overall_accum: dict[str, dict[str, Any]] = defaultdict(_new_metric_cell)
    slice_accum: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(_new_metric_cell))
    )
    used_summary_paths: list[str] = []
    used_bucket_paths: list[str] = []

    for path in summary_paths or []:
        payload = _read_json(path)
        if not isinstance(payload, list):
            continue
        used_summary_paths.append(str(path.resolve()))
        for row in payload:
            if not isinstance(row, dict):
                continue
            controller_id = canonicalize_controller_id(row.get("policy_id"))
            if not controller_id:
                continue
            _append_metric(
                overall_accum[controller_id],
                mean_total_score=_safe_float(row.get("mean_total_score"), 0.0),
                win_rate=_safe_float(row.get("win_rate"), 0.0),
                count=max(1, _safe_int(row.get("episodes"), _safe_int(row.get("count"), 1))),
                source_path=str(path.resolve()),
            )

    for path in bucket_paths or []:
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        policies = payload.get("policies") if isinstance(payload.get("policies"), list) else []
        if not policies:
            continue
        used_bucket_paths.append(str(path.resolve()))
        for policy_row in policies:
            if not isinstance(policy_row, dict):
                continue
            controller_id = canonicalize_controller_id(policy_row.get("policy_id"))
            if not controller_id:
                continue
            slice_metrics = policy_row.get("slice_metrics") if isinstance(policy_row.get("slice_metrics"), dict) else {}
            for slice_key, rows in slice_metrics.items():
                if slice_key not in SLICE_KEYS:
                    continue
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    label = str(row.get("slice_label") or "unknown")
                    _append_metric(
                        slice_accum[controller_id][slice_key][label],
                        mean_total_score=_safe_float(row.get("mean_total_score"), 0.0),
                        win_rate=_safe_float(row.get("win_rate"), 0.0),
                        count=max(1, _safe_int(row.get("count"), 1)),
                        source_path=str(path.resolve()),
                    )

    overall = {controller_id: _finalize_metric(cell) for controller_id, cell in overall_accum.items()}
    slice_scores = {
        controller_id: {
            slice_key: {label: _finalize_metric(cell) for label, cell in labels.items()}
            for slice_key, labels in key_rows.items()
        }
        for controller_id, key_rows in slice_accum.items()
    }
    return {
        "schema": "p52_router_knowledge_base_v1",
        "generated_at": _now_iso(),
        "overall": overall,
        "slice_scores": slice_scores,
        "source_refs": {
            "summary_paths": used_summary_paths,
            "bucket_paths": used_bucket_paths,
        },
    }


def _lookup_metric(
    table: dict[str, Any],
    *,
    controller_id: str,
    slice_key: str = "",
    slice_label: str = "",
) -> dict[str, Any]:
    if slice_key:
        slice_scores = table.get("slice_scores") if isinstance(table.get("slice_scores"), dict) else {}
        controller_rows = slice_scores.get(controller_id) if isinstance(slice_scores.get(controller_id), dict) else {}
        label_rows = controller_rows.get(slice_key) if isinstance(controller_rows.get(slice_key), dict) else {}
        result = label_rows.get(slice_label)
        return dict(result or {}) if isinstance(result, dict) else {}
    overall = table.get("overall") if isinstance(table.get("overall"), dict) else {}
    result = overall.get(controller_id)
    return dict(result or {}) if isinstance(result, dict) else {}


def _score_breakdown_as_rule_prior(sample: dict[str, Any], controllers: list[str]) -> dict[str, float]:
    raw = sample.get("routing_score_breakdown") if isinstance(sample.get("routing_score_breakdown"), dict) else {}
    values = [_safe_float(raw.get(controller_id), -999.0) for controller_id in controllers]
    finite = [value for value in values if value > -900.0]
    if not finite:
        chosen = canonicalize_controller_id(sample.get("chosen_controller_rule") or sample.get("selected_controller"))
        if not chosen:
            return {}
        return {controller_id: (100.0 if controller_id == chosen else 0.0) for controller_id in controllers}
    low = min(finite)
    high = max(finite)
    width = max(1e-6, high - low)
    return {
        controller_id: (0.0 if value <= -900.0 else ((value - low) / width) * 100.0)
        for controller_id, value in zip(controllers, values)
    }


def _arena_score_candidates(
    *,
    table: dict[str, Any],
    features: dict[str, Any],
    controllers: list[str],
    scope: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    scores: dict[str, float] = {}
    evidence: list[dict[str, Any]] = []
    for controller_id in controllers:
        total = 0.0
        weight = 0.0
        for slice_key in SLICE_KEYS:
            slice_label = str(features.get(slice_key) or "unknown")
            metric = _lookup_metric(table, controller_id=controller_id, slice_key=slice_key, slice_label=slice_label)
            if not metric:
                continue
            metric_weight = 0.70 * max(0.25, min(1.0, _safe_int(metric.get("count"), 0) / 4.0))
            total += _safe_float(metric.get("mean_total_score"), 0.0) * metric_weight
            weight += metric_weight
            evidence.append(
                {
                    "scope": scope,
                    "source": "arena",
                    "controller_id": controller_id,
                    "slice_key": slice_key,
                    "slice_label": slice_label,
                    "mean_total_score": _safe_float(metric.get("mean_total_score"), 0.0),
                    "count": _safe_int(metric.get("count"), 0),
                    "source_paths": list(metric.get("source_paths") or []),
                }
            )
        overall_metric = _lookup_metric(table, controller_id=controller_id)
        if overall_metric:
            metric_weight = 0.35 * max(0.25, min(1.0, _safe_int(overall_metric.get("count"), 0) / 8.0))
            total += _safe_float(overall_metric.get("mean_total_score"), 0.0) * metric_weight
            weight += metric_weight
            evidence.append(
                {
                    "scope": scope,
                    "source": "arena",
                    "controller_id": controller_id,
                    "slice_key": "",
                    "slice_label": "",
                    "mean_total_score": _safe_float(overall_metric.get("mean_total_score"), 0.0),
                    "count": _safe_int(overall_metric.get("count"), 0),
                    "source_paths": list(overall_metric.get("source_paths") or []),
                }
            )
        if weight > 0.0:
            scores[controller_id] = total / weight
    return scores, evidence


def _fill_missing_scores_with_prior(
    *,
    scores: dict[str, float],
    rule_prior: dict[str, float],
    controllers: list[str],
    base_weight: float,
) -> dict[str, float]:
    rows = dict(scores)
    for controller_id in controllers:
        if controller_id in rows:
            continue
        if controller_id in rule_prior:
            rows[controller_id] = float(rule_prior[controller_id]) * float(base_weight)
    return rows


def _normalize_score_map(raw: dict[str, float], controllers: list[str]) -> dict[str, float]:
    rows = {controller_id: max(0.0, _safe_float(raw.get(controller_id), 0.0)) for controller_id in controllers}
    total = sum(rows.values())
    if total > 0.0:
        return {controller_id: value / total for controller_id, value in rows.items()}
    return rows


def _score_margin(scores: dict[str, float], controllers: list[str]) -> float:
    ordered = sorted([_safe_float(scores.get(controller_id), 0.0) for controller_id in controllers], reverse=True)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    return ordered[0] - ordered[1]


def infer_router_label(
    *,
    sample: dict[str, Any],
    global_knowledge: dict[str, Any],
    local_knowledge: dict[str, Any] | None = None,
    allow_rule_labels: bool = True,
) -> dict[str, Any]:
    controllers = normalize_available_controllers(sample.get("available_controllers"))
    if not controllers:
        controllers = normalize_available_controllers(sample.get("routing_score_breakdown"))
    if not controllers:
        controllers = supported_controller_ids()

    features = normalize_routing_features(sample)
    local_scores, local_evidence = _arena_score_candidates(
        table=(local_knowledge if isinstance(local_knowledge, dict) else {}),
        features=features,
        controllers=controllers,
        scope="local",
    )
    global_scores, global_evidence = _arena_score_candidates(
        table=global_knowledge,
        features=features,
        controllers=controllers,
        scope="global",
    )
    rule_prior = _score_breakdown_as_rule_prior(sample, controllers)

    merged_scores: dict[str, float] = {}
    for controller_id in controllers:
        local_score = local_scores.get(controller_id)
        global_score = global_scores.get(controller_id)
        if local_score is not None and global_score is not None:
            merged_scores[controller_id] = (local_score * 0.75) + (global_score * 0.25)
        elif local_score is not None:
            merged_scores[controller_id] = local_score
        elif global_score is not None:
            merged_scores[controller_id] = global_score

    label_evidence = [*local_evidence, *global_evidence]
    label_source = "arena" if merged_scores else ""
    if allow_rule_labels and rule_prior:
        if merged_scores:
            for controller_id in controllers:
                merged_scores[controller_id] = _safe_float(merged_scores.get(controller_id), 0.0) + (rule_prior.get(controller_id, 0.0) * 0.10)
            label_source = "mixed"
            for controller_id, value in rule_prior.items():
                label_evidence.append(
                    {
                        "scope": "row",
                        "source": "rule",
                        "controller_id": controller_id,
                        "mean_total_score": float(value),
                        "count": 1,
                        "source_paths": [],
                    }
                )
        else:
            merged_scores = _fill_missing_scores_with_prior(scores={}, rule_prior=rule_prior, controllers=controllers, base_weight=1.0)
            label_source = "rule"

    normalized_scores = _normalize_score_map(merged_scores, controllers)
    if not normalized_scores or max(normalized_scores.values() or [0.0]) <= 0.0:
        return {
            "target_controller_label": "",
            "target_controller_scores": {},
            "label_source": "missing",
            "label_confidence": 0.0,
            "label_evidence": [],
            "valid_for_training": False,
        }

    target_controller = max(controllers, key=lambda controller_id: _safe_float(normalized_scores.get(controller_id), 0.0))
    margin = _score_margin(normalized_scores, controllers)
    arena_evidence_count = sum(1 for row in label_evidence if str(row.get("source") or "") == "arena")
    local_evidence_count = sum(1 for row in label_evidence if str(row.get("scope") or "") == "local")
    global_evidence_count = sum(1 for row in label_evidence if str(row.get("scope") or "") == "global")
    rule_used = any(str(row.get("source") or "") == "rule" for row in label_evidence)

    if label_source == "arena":
        base_confidence = 0.85 if local_evidence_count >= 2 else 0.72
    elif label_source == "mixed":
        base_confidence = 0.68 if arena_evidence_count > 0 else 0.45
    elif label_source == "rule":
        base_confidence = 0.35
    else:
        base_confidence = 0.10
    confidence = min(0.98, max(0.15, base_confidence + min(0.15, margin * 0.25) + min(0.05, global_evidence_count * 0.01)))

    return {
        "target_controller_label": target_controller,
        "target_controller_scores": {controller_id: float(normalized_scores.get(controller_id, 0.0)) for controller_id in controllers},
        "label_source": label_source or ("rule" if rule_used else "missing"),
        "label_confidence": float(confidence),
        "label_evidence": label_evidence[:24],
        "valid_for_training": bool(target_controller),
    }
