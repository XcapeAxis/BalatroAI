from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from trainer.closed_loop.replay_manifest import now_iso, write_json, write_markdown


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _pick_baseline_run(current_run_dir: Path) -> Path | None:
    parent = current_run_dir.parent
    if not parent.exists():
        return None
    dirs = sorted([p for p in parent.iterdir() if p.is_dir() and p.name != current_run_dir.name], key=lambda p: p.name)
    if not dirs:
        return None
    return dirs[-1]


def _load_slice_breakdown(decision_payload: dict[str, Any]) -> dict[str, Any]:
    path_raw = str(decision_payload.get("slice_decision_breakdown_json") or "").strip()
    if path_raw:
        path = Path(path_raw)
        if path.exists():
            payload = _read_json(path)
            if isinstance(payload, dict):
                return payload
    champion_rules = decision_payload.get("champion_rules_payload")
    if isinstance(champion_rules, dict):
        path_raw = str(champion_rules.get("slice_decision_breakdown_json") or "").strip()
        if path_raw:
            path = Path(path_raw)
            if path.exists():
                payload = _read_json(path)
                if isinstance(payload, dict):
                    return payload
    return {"rows": []}


def _curriculum_signature(candidate_manifest: dict[str, Any]) -> dict[str, Any]:
    plan = candidate_manifest.get("curriculum_plan") if isinstance(candidate_manifest.get("curriculum_plan"), dict) else {}
    phases = plan.get("phases") if isinstance(plan.get("phases"), list) else []
    rows: list[dict[str, Any]] = []
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        rows.append(
            {
                "name": str(phase.get("name") or ""),
                "source_weights": phase.get("source_weights") if isinstance(phase.get("source_weights"), dict) else {},
                "slice_weights": phase.get("slice_weights") if isinstance(phase.get("slice_weights"), dict) else {},
            }
        )
    return {"phase_count": len(rows), "phases": rows}


def _entry_slice_value(entry: dict[str, Any], slice_key: str) -> str:
    if slice_key in entry and entry.get(slice_key) not in {None, ""}:
        return str(entry.get(slice_key))
    labels = entry.get("slice_labels") if isinstance(entry.get("slice_labels"), dict) else {}
    token = labels.get(slice_key)
    if token in {None, ""}:
        return "unknown"
    return str(token)


def _source_attribution(
    replay_manifest: dict[str, Any],
    degraded_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    entries = replay_manifest.get("selected_entries") if isinstance(replay_manifest.get("selected_entries"), list) else []
    source_totals: dict[str, int] = {}
    source_degraded: dict[str, int] = {}
    seed_totals: dict[str, int] = {}
    seed_degraded: dict[str, int] = {}
    source_seed_degraded: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    degrade_matchers = []
    for row in degraded_rows:
        if not isinstance(row, dict):
            continue
        degrade_matchers.append((str(row.get("slice_key") or ""), str(row.get("slice_label") or "")))

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        source_type = str(entry.get("source_type") or "unknown")
        source_seed = str(entry.get("source_seed") or "unknown")
        sample_count = max(0, int(entry.get("sample_count") or 0))
        if sample_count <= 0:
            continue
        source_totals[source_type] = int(source_totals.get(source_type, 0)) + sample_count
        seed_totals[source_seed] = int(seed_totals.get(source_seed, 0)) + sample_count

        matched_degraded = False
        for slice_key, slice_label in degrade_matchers:
            if slice_key and _entry_slice_value(entry, slice_key) == slice_label:
                matched_degraded = True
                break
        if matched_degraded:
            source_degraded[source_type] = int(source_degraded.get(source_type, 0)) + sample_count
            seed_degraded[source_seed] = int(seed_degraded.get(source_seed, 0)) + sample_count
            source_seed_degraded[source_type][source_seed] += sample_count

    source_rows: list[dict[str, Any]] = []
    for source_type in sorted(source_totals.keys()):
        total = int(source_totals[source_type])
        deg = int(source_degraded.get(source_type, 0))
        seed_rows = []
        for seed, deg_count in sorted(source_seed_degraded.get(source_type, {}).items(), key=lambda kv: (-int(kv[1]), kv[0]))[:5]:
            seed_rows.append(
                {
                    "seed": str(seed),
                    "degraded_slice_sample_count": int(deg_count),
                }
            )
        source_rows.append(
            {
                "source_type": source_type,
                "sample_count": total,
                "degraded_slice_sample_count": deg,
                "degraded_slice_ratio": (float(deg) / total) if total > 0 else 0.0,
                "top_degraded_seeds": seed_rows,
            }
        )
    source_rows.sort(key=lambda r: (-float(r.get("degraded_slice_sample_count") or 0.0), str(r.get("source_type") or "")))

    seed_rows: list[dict[str, Any]] = []
    for seed in sorted(seed_totals.keys()):
        total = int(seed_totals.get(seed, 0))
        deg = int(seed_degraded.get(seed, 0))
        seed_rows.append(
            {
                "source_seed": str(seed),
                "sample_count": total,
                "degraded_slice_sample_count": deg,
                "degraded_slice_ratio": (float(deg) / total) if total > 0 else 0.0,
            }
        )
    seed_rows.sort(key=lambda r: (-float(r.get("degraded_slice_sample_count") or 0.0), str(r.get("source_seed") or "")))

    return {
        "source_rows": source_rows,
        "seed_rows": seed_rows,
    }


def _imagined_source_impact(
    replay_manifest: dict[str, Any],
    candidate_manifest: dict[str, Any],
    degraded_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    entries = replay_manifest.get("selected_entries") if isinstance(replay_manifest.get("selected_entries"), list) else []
    imagined_entries = [entry for entry in entries if isinstance(entry, dict) and str(entry.get("source_type") or "") == "imagined_world_model"]
    imagined_sample_count = int(sum(int(entry.get("sample_count") or 0) for entry in imagined_entries))
    uncertainty_values: list[float] = []
    gate_true = 0
    gate_false = 0
    for entry in imagined_entries:
        try:
            uncertainty_values.append(float(entry.get("uncertainty_score") or 0.0))
        except Exception:
            pass
        if entry.get("uncertainty_gate_passed") is True:
            gate_true += 1
        elif entry.get("uncertainty_gate_passed") is False:
            gate_false += 1

    imagined_source_row = next(
        (row for row in source_rows if isinstance(row, dict) and str(row.get("source_type") or "") == "imagined_world_model"),
        {},
    )
    top_slices: list[dict[str, Any]] = []
    for row in degraded_rows:
        if not isinstance(row, dict):
            continue
        slice_key = str(row.get("slice_key") or "")
        slice_label = str(row.get("slice_label") or "")
        matched = 0
        for entry in imagined_entries:
            if _entry_slice_value(entry, slice_key) == slice_label:
                matched += int(entry.get("sample_count") or 0)
        if matched <= 0:
            continue
        top_slices.append(
            {
                "slice_key": slice_key,
                "slice_label": slice_label,
                "imagined_sample_count": int(matched),
            }
        )
    top_slices.sort(key=lambda item: (-int(item.get("imagined_sample_count") or 0), str(item.get("slice_key") or "")))

    filter_mode = str(candidate_manifest.get("imagined_filter_mode") or "")
    imagination_recipe = str(candidate_manifest.get("imagination_recipe") or "")
    imagined_fraction = _safe_float(candidate_manifest.get("imagined_fraction"), 0.0)
    suggestions: list[str] = []
    degraded_ratio = _safe_float(imagined_source_row.get("degraded_slice_ratio"), 0.0)
    if imagined_sample_count > 0 and degraded_ratio > 0.4:
        suggestions.append("reduce_imagined_fraction")
    if imagined_sample_count > 0 and filter_mode not in {"uncertainty_gate", "filtered"}:
        suggestions.append("enable_uncertainty_filter")
    if top_slices:
        suggestions.append("review_top_degrading_imagined_slices")
    if gate_false > gate_true and imagined_sample_count > 0:
        suggestions.append("tighten_uncertainty_threshold")

    impact = {
        "imagined_enabled": bool(candidate_manifest.get("imagined_enabled")) or imagined_sample_count > 0,
        "imagination_recipe": imagination_recipe,
        "imagined_filter_mode": filter_mode,
        "imagined_fraction": imagined_fraction,
        "imagined_sample_count": imagined_sample_count,
        "degraded_slice_sample_count": int(imagined_source_row.get("degraded_slice_sample_count") or 0),
        "degraded_slice_ratio": degraded_ratio,
        "mean_uncertainty": (sum(uncertainty_values) / max(1, len(uncertainty_values))),
        "uncertainty_gate_true": int(gate_true),
        "uncertainty_gate_false": int(gate_false),
    }
    return impact, top_slices[:5], suggestions


def _lineage_quality(replay_manifest: dict[str, Any]) -> dict[str, Any]:
    entries = replay_manifest.get("selected_entries") if isinstance(replay_manifest.get("selected_entries"), list) else []
    if not entries:
        return {"entry_count": 0, "invalid_ratio": 0.0}
    invalid = 0
    for entry in entries:
        if isinstance(entry, dict) and not bool(entry.get("valid_for_training", True)):
            invalid += 1
    return {"entry_count": len(entries), "invalid_ratio": float(invalid) / max(1, len(entries))}


def _extract_lineage_health(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {
            "status": "baseline_missing",
            "lineage_health_json": "",
            "required_field_missing_ratio": 0.0,
            "missing_source_paths": 0,
        }
    ref_path = run_dir / "replay_lineage_health_ref.json"
    ref_payload = _read_json(ref_path)
    if not isinstance(ref_payload, dict):
        return {
            "status": "missing",
            "lineage_health_json": "",
            "required_field_missing_ratio": 0.0,
            "missing_source_paths": 0,
        }
    summary = ref_payload.get("summary") if isinstance(ref_payload.get("summary"), dict) else {}
    lineage_health_json = str(summary.get("lineage_health_json") or "")
    status = str(summary.get("status") or "unknown")
    required_field_missing_ratio = 0.0
    missing_source_paths = 0
    if lineage_health_json:
        loaded = _read_json(Path(lineage_health_json))
        if isinstance(loaded, dict):
            status = str(loaded.get("status") or status)
            required_field_missing_ratio = _safe_float(loaded.get("required_field_missing_ratio"), 0.0)
            missing_source_paths = _safe_int(loaded.get("missing_source_paths"), 0)
    return {
        "status": status,
        "lineage_health_json": lineage_health_json,
        "required_field_missing_ratio": required_field_missing_ratio,
        "missing_source_paths": missing_source_paths,
    }


def _p47_manifest_block(run_manifest: dict[str, Any]) -> dict[str, Any]:
    if isinstance(run_manifest.get("world_model_rerank"), dict):
        return dict(run_manifest.get("world_model_rerank") or {})
    if isinstance(run_manifest.get("model_based_search"), dict):
        return dict(run_manifest.get("model_based_search") or {})
    return {}


def _p47_summary_rows(run_manifest: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    paths = run_manifest.get("paths") if isinstance(run_manifest.get("paths"), dict) else {}
    raw = str(paths.get("summary_table_json") or "").strip()
    path = Path(raw).resolve() if raw else (run_dir / "summary_table.json")
    payload = _read_json(path)
    return [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []


def _find_policy_row(rows: list[dict[str, Any]], policy_id: str) -> dict[str, Any]:
    token = str(policy_id or "").strip()
    for row in rows:
        if str(row.get("policy_id") or "") == token:
            return row
    return {}


def _p47_assist_map(wm_block: dict[str, Any]) -> dict[str, dict[str, Any]]:
    path_raw = str(wm_block.get("policy_assist_map_json") or "").strip()
    if not path_raw:
        return {}
    payload = _read_json(Path(path_raw))
    if not isinstance(payload, dict):
        return {}
    return {str(key): dict(value) for key, value in payload.items() if isinstance(value, dict)}


def _p47_world_model_assist_impact(
    run_manifest: dict[str, Any],
    run_dir: Path,
    current_decision: dict[str, Any],
    degraded_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    wm_block = _p47_manifest_block(run_manifest)
    if not wm_block:
        return (
            {
                "wm_assist_enabled": False,
                "wm_checkpoint": "",
                "wm_eval_ref": "",
                "baseline_policy": "",
                "candidate_policy": "",
            },
            [],
            [],
            [],
        )

    summary_rows = _p47_summary_rows(run_manifest, run_dir)
    assist_map = _p47_assist_map(wm_block)
    baseline_policy = str(wm_block.get("baseline_policy") or current_decision.get("champion_policy_id") or "")
    candidate_policy = str(wm_block.get("candidate_policy") or current_decision.get("candidate_policy_id") or "")
    baseline_row = _find_policy_row(summary_rows, baseline_policy)
    candidate_row = _find_policy_row(summary_rows, candidate_policy)
    baseline_score = _safe_float(baseline_row.get("mean_total_score"), 0.0)
    best_variant_row = max(
        [row for row in summary_rows if str(row.get("policy_id") or "") != baseline_policy],
        key=lambda row: _safe_float(row.get("mean_total_score"), 0.0),
        default={},
    )

    uncertainty_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    for policy_id, config in assist_map.items():
        row = _find_policy_row(summary_rows, policy_id)
        delta = _safe_float(row.get("mean_total_score"), 0.0) - baseline_score
        uncertainty_rows.append(
            {
                "policy_id": policy_id,
                "uncertainty_penalty": _safe_float(config.get("uncertainty_penalty"), 0.0),
                "score_delta_vs_baseline": delta,
                "invalid_delta_vs_baseline": _safe_float(row.get("invalid_action_rate"), 0.0) - _safe_float(baseline_row.get("invalid_action_rate"), 0.0),
            }
        )
        horizon_rows.append(
            {
                "policy_id": policy_id,
                "horizon": _safe_int(config.get("horizon"), 0),
                "score_delta_vs_baseline": delta,
                "invalid_delta_vs_baseline": _safe_float(row.get("invalid_action_rate"), 0.0) - _safe_float(baseline_row.get("invalid_action_rate"), 0.0),
            }
        )
    uncertainty_rows.sort(key=lambda row: (float(row.get("uncertainty_penalty") or 0.0), str(row.get("policy_id") or "")))
    horizon_rows.sort(key=lambda row: (int(row.get("horizon") or 0), str(row.get("policy_id") or "")))

    top_slices = [
        {
            "slice_key": str(row.get("slice_key") or ""),
            "slice_label": str(row.get("slice_label") or ""),
            "score_delta": _safe_float(((row.get("metrics") or {}).get("mean_total_score_delta")) if isinstance(row.get("metrics"), dict) else 0.0, 0.0),
            "win_rate_delta": _safe_float(((row.get("metrics") or {}).get("win_rate_delta")) if isinstance(row.get("metrics"), dict) else 0.0, 0.0),
        }
        for row in degraded_rows
        if isinstance(row, dict)
    ][:5]

    impact = {
        "wm_assist_enabled": True,
        "wm_checkpoint": str(wm_block.get("wm_checkpoint") or ""),
        "wm_eval_ref": str(wm_block.get("wm_eval_ref") or ""),
        "baseline_policy": baseline_policy,
        "candidate_policy": candidate_policy,
        "candidate_score_delta_vs_baseline": _safe_float(candidate_row.get("mean_total_score"), 0.0) - baseline_score,
        "candidate_invalid_delta_vs_baseline": _safe_float(candidate_row.get("invalid_action_rate"), 0.0) - _safe_float(baseline_row.get("invalid_action_rate"), 0.0),
        "best_variant_policy": str(best_variant_row.get("policy_id") or ""),
        "best_variant_score_delta_vs_baseline": _safe_float(best_variant_row.get("mean_total_score"), 0.0) - baseline_score,
    }
    return impact, uncertainty_rows, horizon_rows, top_slices


def _p48_manifest_block(run_manifest: dict[str, Any]) -> dict[str, Any]:
    if isinstance(run_manifest.get("hybrid_controller"), dict):
        return dict(run_manifest.get("hybrid_controller") or {})
    return {}


def _p48_routing_summary(run_dir: Path, hybrid_block: dict[str, Any]) -> dict[str, Any]:
    raw = str(hybrid_block.get("routing_summary_json") or "").strip()
    path = Path(raw).resolve() if raw else (run_dir / "routing_summary.json")
    payload = _read_json(path)
    return payload if isinstance(payload, dict) else {}


def _p48_trace_rows(run_dir: Path) -> list[dict[str, Any]]:
    candidates = sorted(run_dir.glob("router_traces/**/routing_trace.jsonl"), key=lambda path: str(path))
    if not candidates:
        return []
    rows: list[dict[str, Any]] = []
    for line in candidates[-1].read_text(encoding="utf-8-sig").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _p48_hybrid_impact(
    run_manifest: dict[str, Any],
    run_dir: Path,
    current_decision: dict[str, Any],
    degraded_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    hybrid_block = _p48_manifest_block(run_manifest)
    if not hybrid_block:
        return (
            {
                "hybrid_enabled": False,
                "baseline_policy": "",
                "hybrid_policy": "",
                "search_policy": "",
                "wm_policy": "",
            },
            [],
            [],
            {"selection_distribution": [], "fallback_rate": 0.0},
        )

    summary_rows = _p47_summary_rows(run_manifest, run_dir)
    baseline_policy = str(current_decision.get("champion_policy_id") or "policy_baseline")
    hybrid_policy = str(current_decision.get("candidate_policy_id") or "hybrid_controller_v1")
    search_policy = "search_baseline"
    wm_policy = "policy_plus_wm_rerank"
    baseline_row = _find_policy_row(summary_rows, baseline_policy)
    hybrid_row = _find_policy_row(summary_rows, hybrid_policy)
    search_row = _find_policy_row(summary_rows, search_policy)
    wm_row = _find_policy_row(summary_rows, wm_policy)
    baseline_score = _safe_float(baseline_row.get("mean_total_score"), 0.0)

    routing_summary = _p48_routing_summary(run_dir, hybrid_block)
    trace_rows = _p48_trace_rows(run_dir)
    selection_distribution = routing_summary.get("controller_selection_distribution") if isinstance(routing_summary.get("controller_selection_distribution"), list) else []
    fallback_rate = _safe_float(((routing_summary.get("routing_decision_impact") or {}).get("fallback_rate")) if isinstance(routing_summary.get("routing_decision_impact"), dict) else 0.0, 0.0)
    if not selection_distribution and trace_rows:
        counts = defaultdict(int)
        for row in trace_rows:
            counts[str(row.get("selected_controller") or "unknown")] += 1
        total = sum(counts.values())
        selection_distribution = [
            {"controller_id": key, "count": int(value), "ratio": float(value) / max(1, total)}
            for key, value in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
        fallback_rate = float(sum(1 for row in trace_rows if bool(row.get("fallback_used")))) / max(1, len(trace_rows))

    top_slices = [
        {
            "slice_key": str(row.get("slice_key") or ""),
            "slice_label": str(row.get("slice_label") or ""),
            "score_delta": _safe_float(((row.get("metrics") or {}).get("mean_total_score_delta")) if isinstance(row.get("metrics"), dict) else 0.0, 0.0),
            "win_rate_delta": _safe_float(((row.get("metrics") or {}).get("win_rate_delta")) if isinstance(row.get("metrics"), dict) else 0.0, 0.0),
        }
        for row in degraded_rows
        if isinstance(row, dict)
    ][:5]

    search_selected_ratio = 0.0
    wm_disabled_count = 0
    for row in selection_distribution:
        if not isinstance(row, dict):
            continue
        if str(row.get("controller_id") or "") == "search_baseline":
            search_selected_ratio = _safe_float(row.get("ratio"), 0.0)
    for row in trace_rows:
        reason = str(row.get("routing_reason") or "")
        if "wm" in reason and "uncertainty" in reason:
            wm_disabled_count += 1
        explainability = row.get("features") if isinstance(row.get("features"), dict) else {}
        if _safe_float(explainability.get("wm_uncertainty"), 0.0) >= 1.2:
            wm_disabled_count += 0

    routing_decision_impact = {
        "hybrid_enabled": True,
        "baseline_policy": baseline_policy,
        "hybrid_policy": hybrid_policy,
        "search_policy": search_policy,
        "wm_policy": wm_policy,
        "hybrid_score_delta_vs_baseline": _safe_float(hybrid_row.get("mean_total_score"), 0.0) - baseline_score,
        "hybrid_score_delta_vs_search": _safe_float(hybrid_row.get("mean_total_score"), 0.0) - _safe_float(search_row.get("mean_total_score"), 0.0),
        "hybrid_score_delta_vs_wm_rerank": _safe_float(hybrid_row.get("mean_total_score"), 0.0) - _safe_float(wm_row.get("mean_total_score"), 0.0),
        "fallback_rate": fallback_rate,
    }
    search_budget_sensitivity = [
        {
            "policy_id": search_policy,
            "score_delta_vs_hybrid": _safe_float(search_row.get("mean_total_score"), 0.0) - _safe_float(hybrid_row.get("mean_total_score"), 0.0),
            "search_selected_ratio": search_selected_ratio,
        }
    ]
    wm_gate_impact = [
        {
            "wm_uncertainty_gate_trigger_count": int(wm_disabled_count),
            "decision_count": len(trace_rows),
            "wm_selected_ratio": next(
                (
                    _safe_float(row.get("ratio"), 0.0)
                    for row in selection_distribution
                    if isinstance(row, dict) and str(row.get("controller_id") or "") == "policy_plus_wm_rerank"
                ),
                0.0,
            ),
        }
    ]
    controller_distribution = {
        "selection_distribution": selection_distribution,
        "fallback_rate": fallback_rate,
    }
    return routing_decision_impact, selection_distribution, top_slices, {
        "search_budget_sensitivity": search_budget_sensitivity,
        "wm_uncertainty_gate_impact": wm_gate_impact,
        "controller_distribution": controller_distribution,
    }


def run_regression_triage(
    *,
    current_run_dir: str | Path,
    out_dir: str | Path | None = None,
    baseline_run_dir: str | Path | None = None,
) -> dict[str, Any]:
    current_dir = Path(current_run_dir).resolve()
    if not current_dir.exists():
        raise FileNotFoundError(f"current_run_dir not found: {current_dir}")

    baseline_dir: Path | None = None
    if baseline_run_dir:
        baseline_dir = Path(baseline_run_dir).resolve()
        if not baseline_dir.exists():
            baseline_dir = None
    if baseline_dir is None:
        baseline_dir = _pick_baseline_run(current_dir)

    current_decision = _read_json(current_dir / "promotion_decision.json")
    current_replay_ref = _read_json(current_dir / "replay_mix_manifest_ref.json")
    current_candidate_ref = _read_json(current_dir / "candidate_train_ref.json")
    current_manifest = _read_json(current_dir / "run_manifest.json")

    current_replay_manifest_path = ""
    if isinstance(current_replay_ref, dict):
        summary = current_replay_ref.get("summary") if isinstance(current_replay_ref.get("summary"), dict) else {}
        current_replay_manifest_path = str(summary.get("replay_mix_manifest") or "")
    current_replay_manifest = _read_json(Path(current_replay_manifest_path)) if current_replay_manifest_path else {}
    if not isinstance(current_replay_manifest, dict):
        current_replay_manifest = {}

    current_candidate_manifest_path = ""
    if isinstance(current_candidate_ref, dict):
        summary = current_candidate_ref.get("summary") if isinstance(current_candidate_ref.get("summary"), dict) else {}
        current_candidate_manifest_path = str(summary.get("candidate_train_manifest") or "")
    current_candidate_manifest = _read_json(Path(current_candidate_manifest_path)) if current_candidate_manifest_path else {}
    if not isinstance(current_candidate_manifest, dict):
        current_candidate_manifest = {}

    baseline_missing = baseline_dir is None
    baseline_decision: dict[str, Any] = {}
    baseline_candidate_manifest: dict[str, Any] = {}
    baseline_replay_manifest: dict[str, Any] = {}
    if baseline_dir is not None:
        payload = _read_json(baseline_dir / "promotion_decision.json")
        if isinstance(payload, dict):
            baseline_decision = payload
        replay_ref = _read_json(baseline_dir / "replay_mix_manifest_ref.json")
        if isinstance(replay_ref, dict):
            summary = replay_ref.get("summary") if isinstance(replay_ref.get("summary"), dict) else {}
            replay_path = str(summary.get("replay_mix_manifest") or "")
            replay_payload = _read_json(Path(replay_path)) if replay_path else None
            if isinstance(replay_payload, dict):
                baseline_replay_manifest = replay_payload
        cand_ref = _read_json(baseline_dir / "candidate_train_ref.json")
        if isinstance(cand_ref, dict):
            summary = cand_ref.get("summary") if isinstance(cand_ref.get("summary"), dict) else {}
            cand_path = str(summary.get("candidate_train_manifest") or "")
            cand_payload = _read_json(Path(cand_path)) if cand_path else None
            if isinstance(cand_payload, dict):
                baseline_candidate_manifest = cand_payload

    current_decision_dict = current_decision if isinstance(current_decision, dict) else {}
    baseline_decision_dict = baseline_decision if isinstance(baseline_decision, dict) else {}
    current_slice = _load_slice_breakdown(current_decision_dict)
    degraded_rows = [
        row
        for row in (current_slice.get("rows") if isinstance(current_slice.get("rows"), list) else [])
        if isinstance(row, dict) and bool((row.get("signals") or {}).get("degraded_significant"))
    ]
    degraded_rows = degraded_rows[:5]

    attribution = _source_attribution(current_replay_manifest, degraded_rows)
    current_source_attr = attribution.get("source_rows") if isinstance(attribution.get("source_rows"), list) else []
    current_seed_attr = attribution.get("seed_rows") if isinstance(attribution.get("seed_rows"), list) else []
    imagined_source_impact, top_degrading_imagined_slices, suggested_imagination_adjustments = _imagined_source_impact(
        current_replay_manifest,
        current_candidate_manifest,
        degraded_rows,
        current_source_attr,
    )
    world_model_assist_impact, uncertainty_penalty_sensitivity, horizon_sensitivity, top_degrading_slices_with_wm = _p47_world_model_assist_impact(
        current_manifest if isinstance(current_manifest, dict) else {},
        current_dir,
        current_decision_dict,
        degraded_rows,
    )
    routing_decision_impact, controller_selection_distribution, top_degrading_slices_for_hybrid, hybrid_aux = _p48_hybrid_impact(
        current_manifest if isinstance(current_manifest, dict) else {},
        current_dir,
        current_decision_dict,
        degraded_rows,
    )
    search_budget_sensitivity = hybrid_aux.get("search_budget_sensitivity") if isinstance(hybrid_aux, dict) else []
    wm_uncertainty_gate_impact = hybrid_aux.get("wm_uncertainty_gate_impact") if isinstance(hybrid_aux, dict) else []
    curr_signature = _curriculum_signature(current_candidate_manifest)
    prev_signature = _curriculum_signature(baseline_candidate_manifest)

    curriculum_change = {
        "current_phase_count": int(curr_signature.get("phase_count") or 0),
        "baseline_phase_count": int(prev_signature.get("phase_count") or 0),
        "phase_count_delta": int(curr_signature.get("phase_count") or 0) - int(prev_signature.get("phase_count") or 0),
        "current_phases": curr_signature.get("phases"),
        "baseline_phases": prev_signature.get("phases"),
    }

    current_quality = _lineage_quality(current_replay_manifest)
    baseline_quality = _lineage_quality(baseline_replay_manifest)
    quality_delta = _safe_float(current_quality.get("invalid_ratio")) - _safe_float(baseline_quality.get("invalid_ratio"))
    current_lineage_health = _extract_lineage_health(current_dir)
    baseline_lineage_health = _extract_lineage_health(baseline_dir)
    lineage_missing_ratio_delta = _safe_float(current_lineage_health.get("required_field_missing_ratio"), 0.0) - _safe_float(
        baseline_lineage_health.get("required_field_missing_ratio"),
        0.0,
    )
    lineage_missing_paths_delta = _safe_int(current_lineage_health.get("missing_source_paths"), 0) - _safe_int(
        baseline_lineage_health.get("missing_source_paths"),
        0,
    )

    quality_warnings: list[str] = []
    if quality_delta > 0.01:
        quality_warnings.append("invalid_for_training_ratio_increased")
    if lineage_missing_ratio_delta > 0.01:
        quality_warnings.append("lineage_required_field_missing_ratio_increased")
    if lineage_missing_paths_delta > 0:
        quality_warnings.append("lineage_missing_source_paths_increased")
    if str(current_lineage_health.get("status") or "").lower() in {"warn", "error"}:
        quality_warnings.append("current_lineage_health_non_ok")

    overall = {
        "current_candidate_score": _safe_float(current_decision_dict.get("candidate_score"), 0.0),
        "current_champion_score": _safe_float(current_decision_dict.get("champion_score"), 0.0),
        "current_score_delta": _safe_float(current_decision_dict.get("score_delta"), 0.0),
        "baseline_score_delta": _safe_float(baseline_decision_dict.get("score_delta"), 0.0),
        "score_delta_change_vs_baseline": _safe_float(current_decision_dict.get("score_delta"), 0.0)
        - _safe_float(baseline_decision_dict.get("score_delta"), 0.0),
        "recommendation": str(current_decision_dict.get("recommendation") or ""),
    }

    payload = {
        "schema": "p41_regression_triage_v1",
        "generated_at": now_iso(),
        "current_run_dir": str(current_dir),
        "baseline_run_dir": str(baseline_dir) if baseline_dir else "",
        "baseline_missing": baseline_missing,
        "overall": overall,
        "degraded_slices_topk": degraded_rows,
        "source_attribution": current_source_attr,
        "seed_attribution": current_seed_attr,
        "imagined_source_impact": imagined_source_impact,
        "top_degrading_imagined_slices": top_degrading_imagined_slices,
        "suggested_imagination_adjustments": suggested_imagination_adjustments,
        "world_model_assist_impact": world_model_assist_impact,
        "uncertainty_penalty_sensitivity": uncertainty_penalty_sensitivity,
        "horizon_sensitivity": horizon_sensitivity,
        "top_degrading_slices_with_wm": top_degrading_slices_with_wm,
        "routing_decision_impact": routing_decision_impact,
        "controller_selection_distribution": controller_selection_distribution,
        "top_degrading_slices_for_hybrid": top_degrading_slices_for_hybrid,
        "search_budget_sensitivity": search_budget_sensitivity,
        "wm_uncertainty_gate_impact": wm_uncertainty_gate_impact,
        "curriculum_change": curriculum_change,
        "data_quality": {
            "current": current_quality,
            "baseline": baseline_quality,
            "invalid_ratio_delta": quality_delta,
            "lineage_health": {
                "current": current_lineage_health,
                "baseline": baseline_lineage_health,
                "required_field_missing_ratio_delta": lineage_missing_ratio_delta,
                "missing_source_paths_delta": lineage_missing_paths_delta,
            },
            "warnings": quality_warnings,
        },
        "refs": {
            "current_run_manifest": str(current_dir / "run_manifest.json"),
            "current_promotion_decision": str(current_dir / "promotion_decision.json"),
            "current_slice_breakdown": str(current_decision_dict.get("slice_decision_breakdown_json") or ""),
            "current_lineage_health_ref": str(current_dir / "replay_lineage_health_ref.json"),
        },
    }

    output_dir = Path(out_dir).resolve() if out_dir else current_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "triage_report.json"
    md_path = output_dir / "triage_report.md"
    write_json(json_path, payload)

    md_lines = [
        "# P41 Regression Triage Report",
        "",
        f"- current_run_dir: `{payload.get('current_run_dir')}`",
        f"- baseline_run_dir: `{payload.get('baseline_run_dir')}`",
        f"- baseline_missing: `{payload.get('baseline_missing')}`",
        "",
        "## Overall",
        f"- current_score_delta: {float((overall or {}).get('current_score_delta') or 0.0):.6f}",
        f"- baseline_score_delta: {float((overall or {}).get('baseline_score_delta') or 0.0):.6f}",
        f"- score_delta_change_vs_baseline: {float((overall or {}).get('score_delta_change_vs_baseline') or 0.0):.6f}",
        f"- recommendation: `{(overall or {}).get('recommendation')}`",
        "",
        "## Degraded Slices (Top-K)",
    ]
    if degraded_rows:
        for row in degraded_rows:
            if not isinstance(row, dict):
                continue
            metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            md_lines.append(
                "- {slice_key}:{slice_label} score_delta={score:.6f} win_delta={win:.6f}".format(
                    slice_key=row.get("slice_key"),
                    slice_label=row.get("slice_label"),
                    score=float(metrics.get("mean_total_score_delta") or 0.0),
                    win=float(metrics.get("win_rate_delta") or 0.0),
                )
            )
    else:
        md_lines.append("- none")

    md_lines.extend(["", "## Source Attribution"])
    if current_source_attr:
        for row in current_source_attr:
            if not isinstance(row, dict):
                continue
            md_lines.append(
                "- {source}: samples={samples}, degraded_slice_samples={deg}, ratio={ratio:.3f}".format(
                    source=row.get("source_type"),
                    samples=int(row.get("sample_count") or 0),
                    deg=int(row.get("degraded_slice_sample_count") or 0),
                    ratio=float(row.get("degraded_slice_ratio") or 0.0),
                )
            )
    else:
        md_lines.append("- none")
    md_lines.extend(["", "## Imagined Source Impact"])
    md_lines.append(
        "- enabled={enabled} recipe={recipe} filter={filter_mode} fraction={fraction:.3f} degraded_ratio={ratio:.3f} mean_uncertainty={unc:.4f}".format(
            enabled=str(bool(imagined_source_impact.get("imagined_enabled"))).lower(),
            recipe=imagined_source_impact.get("imagination_recipe"),
            filter_mode=imagined_source_impact.get("imagined_filter_mode"),
            fraction=float(imagined_source_impact.get("imagined_fraction") or 0.0),
            ratio=float(imagined_source_impact.get("degraded_slice_ratio") or 0.0),
            unc=float(imagined_source_impact.get("mean_uncertainty") or 0.0),
        )
    )
    if top_degrading_imagined_slices:
        for row in top_degrading_imagined_slices:
            if not isinstance(row, dict):
                continue
            md_lines.append(
                "- top_imagined_slice {key}:{label} samples={count}".format(
                    key=row.get("slice_key"),
                    label=row.get("slice_label"),
                    count=int(row.get("imagined_sample_count") or 0),
                )
            )
    if suggested_imagination_adjustments:
        md_lines.extend(["", "## Suggested Imagination Adjustments"])
        for item in suggested_imagination_adjustments:
            md_lines.append(f"- {item}")
    md_lines.extend(["", "## World Model Assist Impact"])
    md_lines.append(
        "- enabled={enabled} baseline={baseline} candidate={candidate} delta_vs_baseline={delta:.6f} best_variant={best} best_delta={best_delta:.6f}".format(
            enabled=str(bool(world_model_assist_impact.get("wm_assist_enabled"))).lower(),
            baseline=world_model_assist_impact.get("baseline_policy"),
            candidate=world_model_assist_impact.get("candidate_policy"),
            delta=float(world_model_assist_impact.get("candidate_score_delta_vs_baseline") or 0.0),
            best=world_model_assist_impact.get("best_variant_policy"),
            best_delta=float(world_model_assist_impact.get("best_variant_score_delta_vs_baseline") or 0.0),
        )
    )
    if uncertainty_penalty_sensitivity:
        md_lines.extend(["", "## Uncertainty Penalty Sensitivity"])
        for row in uncertainty_penalty_sensitivity:
            if not isinstance(row, dict):
                continue
            md_lines.append(
                "- {policy}: unc_penalty={unc:.3f} score_delta={score:.6f} invalid_delta={invalid:.6f}".format(
                    policy=row.get("policy_id"),
                    unc=float(row.get("uncertainty_penalty") or 0.0),
                    score=float(row.get("score_delta_vs_baseline") or 0.0),
                    invalid=float(row.get("invalid_delta_vs_baseline") or 0.0),
                )
            )
    if horizon_sensitivity:
        md_lines.extend(["", "## Horizon Sensitivity"])
        for row in horizon_sensitivity:
            if not isinstance(row, dict):
                continue
            md_lines.append(
                "- {policy}: horizon={horizon} score_delta={score:.6f} invalid_delta={invalid:.6f}".format(
                    policy=row.get("policy_id"),
                    horizon=int(row.get("horizon") or 0),
                    score=float(row.get("score_delta_vs_baseline") or 0.0),
                    invalid=float(row.get("invalid_delta_vs_baseline") or 0.0),
                )
            )
    if top_degrading_slices_with_wm:
        md_lines.extend(["", "## Top Degrading Slices With WM"])
        for row in top_degrading_slices_with_wm:
            if not isinstance(row, dict):
                continue
            md_lines.append(
                "- {key}:{label} score_delta={score:.6f} win_delta={win:.6f}".format(
                    key=row.get("slice_key"),
                    label=row.get("slice_label"),
                    score=float(row.get("score_delta") or 0.0),
                    win=float(row.get("win_rate_delta") or 0.0),
                )
            )
    md_lines.extend(["", "## Hybrid Routing Impact"])
    md_lines.append(
        "- enabled={enabled} baseline={baseline} hybrid={hybrid} delta_vs_baseline={delta:.6f} delta_vs_search={search_delta:.6f}".format(
            enabled=str(bool(routing_decision_impact.get("hybrid_enabled"))).lower(),
            baseline=routing_decision_impact.get("baseline_policy"),
            hybrid=routing_decision_impact.get("hybrid_policy"),
            delta=float(routing_decision_impact.get("hybrid_score_delta_vs_baseline") or 0.0),
            search_delta=float(routing_decision_impact.get("hybrid_score_delta_vs_search") or 0.0),
        )
    )
    if controller_selection_distribution:
        md_lines.extend(["", "## Controller Selection Distribution"])
        for row in controller_selection_distribution:
            if not isinstance(row, dict):
                continue
            md_lines.append(
                "- {controller}: count={count} ratio={ratio:.3f}".format(
                    controller=row.get("controller_id"),
                    count=int(row.get("count") or 0),
                    ratio=float(row.get("ratio") or 0.0),
                )
            )
    if top_degrading_slices_for_hybrid:
        md_lines.extend(["", "## Top Degrading Slices For Hybrid"])
        for row in top_degrading_slices_for_hybrid:
            if not isinstance(row, dict):
                continue
            md_lines.append(
                "- {key}:{label} score_delta={score:.6f} win_delta={win:.6f}".format(
                    key=row.get("slice_key"),
                    label=row.get("slice_label"),
                    score=float(row.get("score_delta") or 0.0),
                    win=float(row.get("win_rate_delta") or 0.0),
                )
            )
    if search_budget_sensitivity:
        md_lines.extend(["", "## Search Budget Sensitivity"])
        for row in search_budget_sensitivity:
            if not isinstance(row, dict):
                continue
            md_lines.append(
                "- {policy}: score_delta_vs_hybrid={score:.6f} search_selected_ratio={ratio:.3f}".format(
                    policy=row.get("policy_id"),
                    score=float(row.get("score_delta_vs_hybrid") or 0.0),
                    ratio=float(row.get("search_selected_ratio") or 0.0),
                )
            )
    if wm_uncertainty_gate_impact:
        md_lines.extend(["", "## WM Uncertainty Gate Impact"])
        for row in wm_uncertainty_gate_impact:
            if not isinstance(row, dict):
                continue
            md_lines.append(
                "- trigger_count={count} decision_count={decisions} wm_selected_ratio={ratio:.3f}".format(
                    count=int(row.get("wm_uncertainty_gate_trigger_count") or 0),
                    decisions=int(row.get("decision_count") or 0),
                    ratio=float(row.get("wm_selected_ratio") or 0.0),
                )
            )
    md_lines.extend(["", "## Seed Attribution"])
    if current_seed_attr:
        for row in current_seed_attr[:10]:
            if not isinstance(row, dict):
                continue
            md_lines.append(
                "- {seed}: samples={samples}, degraded_slice_samples={deg}, ratio={ratio:.3f}".format(
                    seed=row.get("source_seed"),
                    samples=int(row.get("sample_count") or 0),
                    deg=int(row.get("degraded_slice_sample_count") or 0),
                    ratio=float(row.get("degraded_slice_ratio") or 0.0),
                )
            )
    else:
        md_lines.append("- none")
    md_lines.extend(
        [
            "",
            "## Curriculum Change",
            f"- current_phase_count: {int(curriculum_change.get('current_phase_count') or 0)}",
            f"- baseline_phase_count: {int(curriculum_change.get('baseline_phase_count') or 0)}",
            f"- phase_count_delta: {int(curriculum_change.get('phase_count_delta') or 0)}",
            "",
            "## Data Quality",
            f"- current_invalid_ratio: {float((current_quality or {}).get('invalid_ratio') or 0.0):.6f}",
            f"- baseline_invalid_ratio: {float((baseline_quality or {}).get('invalid_ratio') or 0.0):.6f}",
            f"- invalid_ratio_delta: {float(quality_delta):.6f}",
            f"- current_lineage_health_status: `{current_lineage_health.get('status')}`",
            f"- baseline_lineage_health_status: `{baseline_lineage_health.get('status')}`",
            "- lineage_required_field_missing_ratio_delta: {0:.6f}".format(float(lineage_missing_ratio_delta)),
            "- lineage_missing_source_paths_delta: {0}".format(int(lineage_missing_paths_delta)),
        ]
    )
    if quality_warnings:
        md_lines.extend(["", "## Data Quality Warnings"])
        for warning in quality_warnings:
            md_lines.append(f"- {warning}")
    write_markdown(md_path, md_lines)
    return {
        "status": "ok",
        "triage_report_json": str(json_path),
        "triage_report_md": str(md_path),
        "baseline_missing": bool(baseline_missing),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P41 regression triage for closed-loop candidate degradation.")
    parser.add_argument("--current-run-dir", required=True)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--baseline-run-dir", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_regression_triage(
        current_run_dir=args.current_run_dir,
        out_dir=(args.out_dir if str(args.out_dir).strip() else None),
        baseline_run_dir=(args.baseline_run_dir if str(args.baseline_run_dir).strip() else None),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
