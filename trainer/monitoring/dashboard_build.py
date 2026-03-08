from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as fp:
        for line in fp:
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _campaign_stage_summary(payload: dict[str, Any]) -> dict[str, Any]:
    stages = [dict(item) for item in (payload.get("stages") or []) if isinstance(item, dict)]
    active = next((item for item in stages if str(item.get("status") or "") == "running"), None)
    failed = next((item for item in stages if str(item.get("status") or "") == "failed"), None)
    latest = active or failed or (stages[-1] if stages else {})
    return {
        "campaign_id": str(payload.get("campaign_id") or ""),
        "run_id": str(payload.get("run_id") or ""),
        "experiment_id": str(payload.get("experiment_id") or ""),
        "seed": str(payload.get("seed") or ""),
        "stage_id": str((latest or {}).get("stage_id") or ""),
        "status": str((latest or {}).get("status") or ""),
        "state_path": str(payload.get("state_path") or ""),
    }


def _registry_summary(input_root: Path) -> dict[str, Any]:
    registry_path = input_root / "registry" / "checkpoints_registry.json"
    payload = _read_json(registry_path)
    items = payload.get("items") if isinstance(payload, dict) and isinstance(payload.get("items"), list) else []
    counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    latest_items = []
    learned_router_items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        token = str(item.get("status") or "draft")
        family = str(item.get("family") or "other")
        counts[token] = int(counts.get(token, 0)) + 1
        family_counts[family] = int(family_counts.get(family, 0)) + 1
    for item in items[:8]:
        if isinstance(item, dict):
            latest_items.append(
                {
                    "checkpoint_id": str(item.get("checkpoint_id") or ""),
                    "family": str(item.get("family") or ""),
                    "status": str(item.get("status") or ""),
                }
            )
    for item in items:
        if not isinstance(item, dict) or str(item.get("family") or "") != "learned_router":
            continue
        learned_router_items.append(
            {
                "checkpoint_id": str(item.get("checkpoint_id") or ""),
                "status": str(item.get("status") or ""),
                "artifact_path": str(item.get("artifact_path") or ""),
                "calibration_ref": str(item.get("calibration_ref") or ""),
                "guard_tuning_ref": str(item.get("guard_tuning_ref") or ""),
                "canary_eval_ref": str(item.get("canary_eval_ref") or ""),
                "deployment_mode_recommendation": str(item.get("deployment_mode_recommendation") or ""),
            }
        )
        if len(learned_router_items) >= 6:
            break
    return {
        "registry_path": str(registry_path.resolve()),
        "count": len(items),
        "status_counts": counts,
        "family_counts": family_counts,
        "latest_items": latest_items,
        "learned_router_items": learned_router_items,
    }


def _latest_matching_json(input_root: Path, pattern: str, *, required_tokens: tuple[str, ...] = ()) -> tuple[str, dict[str, Any] | list[Any] | None]:
    candidates = []
    for path in input_root.glob(pattern):
        token = str(path).lower().replace("\\", "/")
        if any(required not in token for required in required_tokens):
            continue
        candidates.append(path)
    candidates.sort(key=lambda item: (item.parent.name, str(item)), reverse=True)
    if not candidates:
        return "", None
    path = candidates[0].resolve()
    return str(path), _read_json(path)


def _latest_matching_json_by_prefix(
    input_root: Path,
    pattern: str,
    *,
    prefixes: tuple[str, ...],
    extra_tokens: tuple[str, ...] = (),
) -> tuple[str, dict[str, Any] | list[Any] | None]:
    for prefix in prefixes:
        path, payload = _latest_matching_json(input_root, pattern, required_tokens=(prefix, *extra_tokens))
        if path:
            return path, payload
    return "", None


def _guarded_variant(payload: dict[str, Any]) -> dict[str, Any]:
    variants = payload.get("variants") if isinstance(payload.get("variants"), list) else []
    return next(
        (
            row
            for row in variants
            if isinstance(row, dict) and str(row.get("policy_id") or "") == "hybrid_controller_learned_with_rule_guard"
        ),
        {},
    )


def _canary_variant(payload: dict[str, Any]) -> dict[str, Any]:
    variants = payload.get("variants") if isinstance(payload.get("variants"), list) else []
    return next(
        (
            row
            for row in variants
            if isinstance(row, dict) and str(row.get("policy_id") or "") == "hybrid_controller_canary_learned_router"
        ),
        {},
    )


def _collect_learned_router_dashboard_data(input_root: Path, campaign_states: list[dict[str, Any]], registry_summary: dict[str, Any]) -> dict[str, Any]:
    prefixes = ("p56/", "p54/", "p52/")
    dataset_path, dataset_payload = _latest_matching_json_by_prefix(input_root, "**/router_dataset_stats.json", prefixes=prefixes)
    train_path, train_payload = _latest_matching_json_by_prefix(input_root, "**/metrics.json", prefixes=prefixes, extra_tokens=("router_train/",))
    promotion_path, promotion_payload = _latest_matching_json_by_prefix(input_root, "**/promotion_decision.json", prefixes=prefixes, extra_tokens=("arena_ablation/",))
    routing_path, routing_payload = _latest_matching_json_by_prefix(input_root, "**/routing_summary.json", prefixes=prefixes, extra_tokens=("arena_ablation/",))
    slice_eval_path, slice_eval_payload = _latest_matching_json_by_prefix(input_root, "**/slice_eval.json", prefixes=prefixes, extra_tokens=("arena_ablation/",))
    queue_path, queue_payload = _latest_matching_json_by_prefix(input_root, "**/promotion_queue.json", prefixes=prefixes)
    ablation_dir = Path(promotion_path).parent if promotion_path else None
    summary_rows_payload = _read_json(ablation_dir / "summary_table.json") if isinstance(ablation_dir, Path) else None
    summary_rows = [row for row in summary_rows_payload if isinstance(row, dict)] if isinstance(summary_rows_payload, list) else []
    guarded_variant = _guarded_variant(routing_payload if isinstance(routing_payload, dict) else {})
    canary_variant = _canary_variant(routing_payload if isinstance(routing_payload, dict) else {})
    path_tokens = " ".join(
        str(value).lower().replace("\\", "/")
        for value in (dataset_path, promotion_path, routing_path, queue_path)
        if value
    )
    if "/p56/" in path_tokens:
        family_prefix = "p56"
    elif "/p54/" in path_tokens:
        family_prefix = "p54"
    else:
        family_prefix = "p52"
    campaign_rows = [
        row
        for row in campaign_states
        if isinstance(row, dict)
        and (
            family_prefix in str(row.get("campaign_id") or "").lower()
            or family_prefix in str(row.get("experiment_id") or "").lower()
        )
    ]
    train_checkpoint_id = str(train_payload.get("checkpoint_id") or "") if isinstance(train_payload, dict) else ""
    return {
        "family_prefix": family_prefix,
        "dataset": {
            "path": dataset_path,
            "payload": dataset_payload if isinstance(dataset_payload, dict) else {},
        },
        "train": {
            "path": train_path,
            "payload": train_payload if isinstance(train_payload, dict) else {},
            "checkpoint_id": train_checkpoint_id,
        },
        "ablation": {
            "promotion_decision_path": promotion_path,
            "promotion_decision": promotion_payload if isinstance(promotion_payload, dict) else {},
            "routing_summary_path": routing_path,
            "routing_summary": routing_payload if isinstance(routing_payload, dict) else {},
            "slice_eval_path": slice_eval_path,
            "slice_eval": slice_eval_payload if isinstance(slice_eval_payload, dict) else {},
            "summary_rows": summary_rows,
            "guarded_variant": guarded_variant if isinstance(guarded_variant, dict) else {},
            "canary_variant": canary_variant if isinstance(canary_variant, dict) else {},
        },
        "campaign_states": campaign_rows,
        "promotion_queue": {
            "path": queue_path,
            "payload": queue_payload if isinstance(queue_payload, dict) else {},
        },
        "registry": {
            "learned_router_items": registry_summary.get("learned_router_items") if isinstance(registry_summary.get("learned_router_items"), list) else [],
        },
    }


def _collect_p56_dashboard_data(input_root: Path, campaign_states: list[dict[str, Any]], registry_summary: dict[str, Any]) -> dict[str, Any]:
    calibration_path, calibration_payload = _latest_matching_json(input_root, "**/calibration_metrics.json", required_tokens=("p56/",))
    reliability_path, reliability_payload = _latest_matching_json(input_root, "**/reliability_bins.json", required_tokens=("p56/",))
    guard_path, guard_payload = _latest_matching_json(input_root, "**/guard_tuning_results.json", required_tokens=("p56/",))
    guard_cfg_path, guard_cfg_payload = _latest_matching_json(input_root, "**/recommended_guard_config.json", required_tokens=("p56/",))
    canary_path, canary_payload = _latest_matching_json(input_root, "**/canary_eval_summary.json", required_tokens=("p56/",))
    benchmark_path, benchmark_payload = _latest_matching_json(input_root, "**/benchmark_summary.json", required_tokens=("p56/",))
    promotion_path, promotion_payload = _latest_matching_json(input_root, "**/promotion_decision.json", required_tokens=("p56/", "arena_ablation/"))
    routing_path, routing_payload = _latest_matching_json(input_root, "**/routing_summary.json", required_tokens=("p56/", "arena_ablation/"))
    queue_path, queue_payload = _latest_matching_json(input_root, "**/promotion_queue.json", required_tokens=("p56/",))
    campaign_rows = [
        row
        for row in campaign_states
        if isinstance(row, dict) and ("p56" in str(row.get("campaign_id") or "").lower() or "p56" in str(row.get("experiment_id") or "").lower())
    ]
    registry_items = [
        row
        for row in (registry_summary.get("learned_router_items") if isinstance(registry_summary.get("learned_router_items"), list) else [])
        if any(str(row.get(key) or "").strip() for key in ("calibration_ref", "guard_tuning_ref", "canary_eval_ref"))
        or str(row.get("deployment_mode_recommendation") or "").strip()
    ]
    return {
        "calibration": {"path": calibration_path, "payload": calibration_payload if isinstance(calibration_payload, dict) else {}},
        "reliability": {"path": reliability_path, "payload": reliability_payload if isinstance(reliability_payload, dict) else {}},
        "guard_tuning": {"path": guard_path, "payload": guard_payload if isinstance(guard_payload, dict) else {}},
        "recommended_guard": {"path": guard_cfg_path, "payload": guard_cfg_payload if isinstance(guard_cfg_payload, dict) else {}},
        "canary_eval": {"path": canary_path, "payload": canary_payload if isinstance(canary_payload, dict) else {}},
        "benchmark": {"path": benchmark_path, "payload": benchmark_payload if isinstance(benchmark_payload, dict) else {}},
        "promotion": {"path": promotion_path, "payload": promotion_payload if isinstance(promotion_payload, dict) else {}},
        "routing": {"path": routing_path, "payload": routing_payload if isinstance(routing_payload, dict) else {}},
        "promotion_queue": {"path": queue_path, "payload": queue_payload if isinstance(queue_payload, dict) else {}},
        "campaign_states": campaign_rows,
        "registry": {"learned_router_items": registry_items},
    }


def _collect_p53_dashboard_data(input_root: Path, campaign_states: list[dict[str, Any]]) -> dict[str, Any]:
    window_state_path = input_root / "p53" / "window_supervisor" / "latest" / "window_state.json"
    background_path = input_root / "p53" / "background_mode_validation" / "latest" / "background_mode_validation.json"
    ops_ui_path = input_root / "p53" / "ops_ui" / "latest" / "ops_ui_state.json"
    audit_path = input_root / "p53" / "ops_audit" / "ops_audit.jsonl"
    metadata_path, metadata_payload = _latest_matching_json(input_root, "**/ops_ui_metadata.json", required_tokens=("p53/",))
    window_payload = _read_json(window_state_path)
    background_payload = _read_json(background_path)
    ops_ui_payload = _read_json(ops_ui_path)
    audit_rows = _read_jsonl(audit_path)[-12:]
    dominant_mode = ""
    rows = (
        window_payload.get("window_mode_after")
        if isinstance(window_payload, dict) and isinstance(window_payload.get("window_mode_after"), list)
        else (window_payload.get("windows") if isinstance(window_payload, dict) else [])
    )
    if isinstance(rows, list):
        for role in ("game_main", "other_balatro", "diagnostic_console"):
            for row in rows:
                if isinstance(row, dict) and str(row.get("role") or "") == role and str(row.get("mode") or "").strip():
                    dominant_mode = str(row.get("mode") or "")
                    break
            if dominant_mode:
                break
    p53_campaign_states = [
        row
        for row in campaign_states
        if isinstance(row, dict)
        and (
            "p53" in str(row.get("campaign_id") or "").lower()
            or "p53" in str(row.get("experiment_id") or "").lower()
        )
    ]
    return {
        "window_state": {"path": str(window_state_path.resolve()), "payload": window_payload if isinstance(window_payload, dict) else {}, "dominant_mode": dominant_mode},
        "background_validation": {"path": str(background_path.resolve()), "payload": background_payload if isinstance(background_payload, dict) else {}},
        "ops_ui": {"path": str(ops_ui_path.resolve()), "payload": ops_ui_payload if isinstance(ops_ui_payload, dict) else {}},
        "ops_ui_metadata": {"path": metadata_path, "payload": metadata_payload if isinstance(metadata_payload, dict) else {}},
        "audit_tail": audit_rows,
        "campaign_states": p53_campaign_states,
    }


def _collect_config_sync_status(input_root: Path) -> dict[str, Any]:
    """P55: Find the latest config sidecar sync report and return a brief status summary."""
    repo_root = input_root.parent
    sync_root = repo_root / "docs" / "artifacts" / "p55" / "config_sidecar_sync"
    if not sync_root.exists():
        return {"available": False}
    runs = sorted([p for p in sync_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not runs:
        return {"available": False}
    latest = runs[-1]
    report = _read_json(latest / "sidecar_sync_report.json")
    if not isinstance(report, dict):
        return {"available": False, "report_dir": str(latest)}
    return {
        "available": True,
        "timestamp": report.get("timestamp"),
        "mode": report.get("mode"),
        "overall_status": report.get("overall_status"),
        "total": report.get("total"),
        "in_sync": report.get("in_sync"),
        "drifted": report.get("drifted"),
        "missing": report.get("missing"),
        "errors": report.get("errors"),
        "report_json": str(latest / "sidecar_sync_report.json"),
        "report_md": str(latest / "sidecar_sync_report.md"),
    }


def collect_dashboard_data(input_root: Path) -> dict[str, Any]:
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    warnings: list[dict[str, Any]] = []
    for path in input_root.glob("**/*progress*.jsonl"):
        for row in _read_jsonl(path):
            if str(row.get("schema") or "") != "p49_progress_event_v1":
                continue
            key = (
                str(row.get("run_id") or ""),
                str(row.get("component") or ""),
                str(row.get("seed") or ""),
            )
            latest[key] = row
            if str(row.get("warning") or "").strip():
                warnings.append(row)
    p22_runs_root = input_root / "p22" / "runs"
    latest_p22_summary = []
    latest_p22_config_provenance: dict[str, Any] = {}
    if p22_runs_root.exists():
        runs = sorted([path for path in p22_runs_root.iterdir() if path.is_dir()], key=lambda path: path.name)
        if runs:
            summary_raw = _read_json(runs[-1] / "summary_table.json")
            if isinstance(summary_raw, list):
                latest_p22_summary = [row for row in summary_raw if isinstance(row, dict)]
            elif isinstance(summary_raw, dict):
                # P55 format: {"config_provenance": {...}, "rows": [...]}
                latest_p22_config_provenance = summary_raw.get("config_provenance") or {}
                rows = summary_raw.get("rows") or []
                if isinstance(rows, list):
                    latest_p22_summary = [row for row in rows if isinstance(row, dict)]
            # Also try to extract provenance from run plan
            if not latest_p22_config_provenance:
                run_plan = _read_json(runs[-1] / "run_plan.json")
                if isinstance(run_plan, dict):
                    latest_p22_config_provenance = run_plan.get("config_provenance") or {}
            # Also try from report_p23.json
            if not latest_p22_config_provenance:
                for fname in ("report_p23.json", "report_p22.json"):
                    rpt = _read_json(runs[-1] / fname)
                    if isinstance(rpt, dict) and rpt.get("config_provenance"):
                        latest_p22_config_provenance = rpt.get("config_provenance") or {}
                        break
    campaign_states = []
    for path in sorted(input_root.glob("**/campaign_state.json"), key=lambda item: str(item), reverse=True)[:12]:
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        payload["state_path"] = str(path.resolve())
        campaign_states.append(_campaign_stage_summary(payload))
    registry_summary = _registry_summary(input_root)
    learned_router_payload = _collect_learned_router_dashboard_data(input_root, campaign_states, registry_summary)
    p56_payload = _collect_p56_dashboard_data(input_root, campaign_states, registry_summary)
    # P55: collect latest config sidecar sync report
    config_sync_status = _collect_config_sync_status(input_root)

    return {
        "schema": "p49_dashboard_data_v1",
        "generated_at": _now_iso(),
        "input_root": str(input_root),
        "latest_events": sorted(latest.values(), key=lambda row: (str(row.get("run_id") or ""), str(row.get("component") or ""))),
        "warnings": warnings[-20:],
        "latest_p22_summary": latest_p22_summary,
        "config_provenance": latest_p22_config_provenance,
        "config_sync_status": config_sync_status,
        "campaign_states": campaign_states,
        "registry_summary": registry_summary,
        "learned_router": learned_router_payload,
        "p52": learned_router_payload,
        "p56": p56_payload,
        "p53": _collect_p53_dashboard_data(input_root, campaign_states),
    }


def build_dashboard(input_root: Path, output_dir: Path) -> dict[str, Any]:
    data = collect_dashboard_data(input_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dashboard_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rows_html = []
    for row in data.get("latest_events") if isinstance(data.get("latest_events"), list) else []:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('run_id') or ''))}</td>"
            f"<td>{html.escape(str(row.get('component') or ''))}</td>"
            f"<td>{html.escape(str(row.get('phase') or ''))}</td>"
            f"<td>{html.escape(str(row.get('status') or ''))}</td>"
            f"<td>{html.escape(str(row.get('learner_device') or ''))}</td>"
            f"<td>{html.escape(str(row.get('rollout_device') or ''))}</td>"
            f"<td>{html.escape(str(row.get('throughput') if row.get('throughput') is not None else '-'))}</td>"
            f"<td>{html.escape(str(row.get('gpu_mem_mb') if row.get('gpu_mem_mb') is not None else '-'))}</td>"
            f"<td><code>{html.escape(json.dumps(metrics, ensure_ascii=False)[:180])}</code></td>"
            "</tr>"
        )

    warnings_html = []
    for row in data.get("warnings") if isinstance(data.get("warnings"), list) else []:
        warnings_html.append(
            "<li>"
            f"{html.escape(str(row.get('run_id') or ''))} / {html.escape(str(row.get('component') or ''))}: "
            f"{html.escape(str(row.get('warning') or ''))}"
            "</li>"
        )

    p22_html = []
    for row in data.get("latest_p22_summary") if isinstance(data.get("latest_p22_summary"), list) else []:
        if not isinstance(row, dict):
            continue
        p22_html.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('exp_id') or ''))}</td>"
            f"<td>{html.escape(str(row.get('status') or ''))}</td>"
            f"<td>{html.escape(str(row.get('mean') or ''))}</td>"
            f"<td>{html.escape(str(row.get('seed_count') or ''))}</td>"
            "</tr>"
        )

    # P55: config provenance panel data
    prov = data.get("config_provenance") if isinstance(data.get("config_provenance"), dict) else {}
    sync_st = data.get("config_sync_status") if isinstance(data.get("config_sync_status"), dict) else {}
    prov_src_type = str(prov.get("config_source_type") or "-")
    prov_hash = str(prov.get("config_hash") or "")[:16]
    prov_sidecar_used = str(prov.get("sidecar_used") or False)
    prov_sidecar_sync = str(prov.get("sidecar_in_sync") or True)
    sync_overall = str(sync_st.get("overall_status") or "-")
    sync_total = str(sync_st.get("total") or "-")
    sync_drifted = str(sync_st.get("drifted") or 0)
    sync_missing = str(sync_st.get("missing") or 0)
    sync_report_link = html.escape(str(sync_st.get("report_md") or ""))

    campaign_html = []
    for row in data.get("campaign_states") if isinstance(data.get("campaign_states"), list) else []:
        if not isinstance(row, dict):
            continue
        campaign_html.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('campaign_id') or ''))}</td>"
            f"<td>{html.escape(str(row.get('experiment_id') or ''))}</td>"
            f"<td>{html.escape(str(row.get('seed') or ''))}</td>"
            f"<td>{html.escape(str(row.get('stage_id') or ''))}</td>"
            f"<td>{html.escape(str(row.get('status') or ''))}</td>"
            "</tr>"
        )

    registry_summary = data.get("registry_summary") if isinstance(data.get("registry_summary"), dict) else {}
    registry_counts = registry_summary.get("status_counts") if isinstance(registry_summary.get("status_counts"), dict) else {}
    registry_items = registry_summary.get("latest_items") if isinstance(registry_summary.get("latest_items"), list) else []
    registry_html = []
    for row in registry_items:
        if not isinstance(row, dict):
            continue
        registry_html.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('checkpoint_id') or ''))}</td>"
            f"<td>{html.escape(str(row.get('family') or ''))}</td>"
            f"<td>{html.escape(str(row.get('status') or ''))}</td>"
            "</tr>"
        )

    router_payload = data.get("learned_router") if isinstance(data.get("learned_router"), dict) else {}
    router_family_prefix = str(router_payload.get("family_prefix") or "p52")
    router_label = f"{router_family_prefix.upper()} Learned Router"
    router_dataset = router_payload.get("dataset") if isinstance(router_payload.get("dataset"), dict) else {}
    router_dataset_payload = router_dataset.get("payload") if isinstance(router_dataset.get("payload"), dict) else {}
    router_train = router_payload.get("train") if isinstance(router_payload.get("train"), dict) else {}
    router_train_payload = router_train.get("payload") if isinstance(router_train.get("payload"), dict) else {}
    router_ablation = router_payload.get("ablation") if isinstance(router_payload.get("ablation"), dict) else {}
    router_promotion = router_ablation.get("promotion_decision") if isinstance(router_ablation.get("promotion_decision"), dict) else {}
    router_guarded_variant = router_ablation.get("guarded_variant") if isinstance(router_ablation.get("guarded_variant"), dict) else {}
    router_summary_rows = router_ablation.get("summary_rows") if isinstance(router_ablation.get("summary_rows"), list) else []
    router_slice_eval = router_ablation.get("slice_eval") if isinstance(router_ablation.get("slice_eval"), dict) else {}
    router_queue = router_payload.get("promotion_queue") if isinstance(router_payload.get("promotion_queue"), dict) else {}
    router_queue_payload = router_queue.get("payload") if isinstance(router_queue.get("payload"), dict) else {}
    router_campaign_rows = router_payload.get("campaign_states") if isinstance(router_payload.get("campaign_states"), list) else []
    router_registry = router_payload.get("registry") if isinstance(router_payload.get("registry"), dict) else {}
    router_registry_items = router_registry.get("learned_router_items") if isinstance(router_registry.get("learned_router_items"), list) else []
    p56_payload = data.get("p56") if isinstance(data.get("p56"), dict) else {}
    p56_calibration = p56_payload.get("calibration") if isinstance(p56_payload.get("calibration"), dict) else {}
    p56_calibration_payload = p56_calibration.get("payload") if isinstance(p56_calibration.get("payload"), dict) else {}
    p56_guard = p56_payload.get("guard_tuning") if isinstance(p56_payload.get("guard_tuning"), dict) else {}
    p56_guard_payload = p56_guard.get("payload") if isinstance(p56_guard.get("payload"), dict) else {}
    p56_guard_cfg = p56_payload.get("recommended_guard") if isinstance(p56_payload.get("recommended_guard"), dict) else {}
    p56_guard_cfg_payload = p56_guard_cfg.get("payload") if isinstance(p56_guard_cfg.get("payload"), dict) else {}
    p56_canary = p56_payload.get("canary_eval") if isinstance(p56_payload.get("canary_eval"), dict) else {}
    p56_canary_payload = p56_canary.get("payload") if isinstance(p56_canary.get("payload"), dict) else {}
    p56_benchmark = p56_payload.get("benchmark") if isinstance(p56_payload.get("benchmark"), dict) else {}
    p56_benchmark_payload = p56_benchmark.get("payload") if isinstance(p56_benchmark.get("payload"), dict) else {}
    p56_promotion = p56_payload.get("promotion") if isinstance(p56_payload.get("promotion"), dict) else {}
    p56_promotion_payload = p56_promotion.get("payload") if isinstance(p56_promotion.get("payload"), dict) else {}
    p56_campaign_rows = p56_payload.get("campaign_states") if isinstance(p56_payload.get("campaign_states"), list) else []
    p53_payload = data.get("p53") if isinstance(data.get("p53"), dict) else {}
    p53_window = p53_payload.get("window_state") if isinstance(p53_payload.get("window_state"), dict) else {}
    p53_window_payload = p53_window.get("payload") if isinstance(p53_window.get("payload"), dict) else {}
    p53_background = p53_payload.get("background_validation") if isinstance(p53_payload.get("background_validation"), dict) else {}
    p53_background_payload = p53_background.get("payload") if isinstance(p53_background.get("payload"), dict) else {}
    p53_ops_ui = p53_payload.get("ops_ui") if isinstance(p53_payload.get("ops_ui"), dict) else {}
    p53_ops_ui_payload = p53_ops_ui.get("payload") if isinstance(p53_ops_ui.get("payload"), dict) else {}
    p53_ops_meta = p53_payload.get("ops_ui_metadata") if isinstance(p53_payload.get("ops_ui_metadata"), dict) else {}
    p53_ops_meta_payload = p53_ops_meta.get("payload") if isinstance(p53_ops_meta.get("payload"), dict) else {}
    p53_campaign_rows = p53_payload.get("campaign_states") if isinstance(p53_payload.get("campaign_states"), list) else []
    p53_audit_rows = p53_payload.get("audit_tail") if isinstance(p53_payload.get("audit_tail"), list) else []

    p52_summary_html = []
    for row in router_summary_rows:
        if not isinstance(row, dict):
            continue
        p52_summary_html.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('policy_id') or ''))}</td>"
            f"<td>{html.escape(str(row.get('mean_total_score') or ''))}</td>"
            f"<td>{html.escape(str(row.get('win_rate') or ''))}</td>"
            f"<td>{html.escape(str(row.get('invalid_action_rate') or ''))}</td>"
            "</tr>"
        )

    p52_campaign_html = []
    for row in router_campaign_rows:
        if not isinstance(row, dict):
            continue
        p52_campaign_html.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('campaign_id') or ''))}</td>"
            f"<td>{html.escape(str(row.get('stage_id') or ''))}</td>"
            f"<td>{html.escape(str(row.get('status') or ''))}</td>"
            f"<td>{html.escape(str(row.get('state_path') or ''))}</td>"
            "</tr>"
        )

    p52_registry_html = []
    for row in router_registry_items:
        if not isinstance(row, dict):
            continue
        p52_registry_html.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('checkpoint_id') or ''))}</td>"
            f"<td>{html.escape(str(row.get('status') or ''))}</td>"
            f"<td><code>{html.escape(str(row.get('artifact_path') or ''))}</code></td>"
            "</tr>"
        )

    p56_campaign_html = []
    for row in p56_campaign_rows:
        if not isinstance(row, dict):
            continue
        p56_campaign_html.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('campaign_id') or ''))}</td>"
            f"<td>{html.escape(str(row.get('stage_id') or ''))}</td>"
            f"<td>{html.escape(str(row.get('status') or ''))}</td>"
            f"<td>{html.escape(str(row.get('state_path') or ''))}</td>"
            "</tr>"
        )

    p52_slice_eval_rows = router_slice_eval.get("comparisons") if isinstance(router_slice_eval.get("comparisons"), list) else []
    p52_slice_eval_html = []
    for row in p52_slice_eval_rows[:10]:
        if not isinstance(row, dict):
            continue
        p52_slice_eval_html.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('target_policy') or ''))}</td>"
            f"<td>{html.escape(str(row.get('slice_key') or ''))}</td>"
            f"<td>{html.escape(str(row.get('slice_label') or ''))}</td>"
            f"<td>{html.escape(str(row.get('score_delta') or ''))}</td>"
            f"<td>{html.escape(str(row.get('win_rate_delta') or ''))}</td>"
            "</tr>"
        )

    p53_campaign_html = []
    for row in p53_campaign_rows:
        if not isinstance(row, dict):
            continue
        p53_campaign_html.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('campaign_id') or ''))}</td>"
            f"<td>{html.escape(str(row.get('experiment_id') or ''))}</td>"
            f"<td>{html.escape(str(row.get('stage_id') or ''))}</td>"
            f"<td>{html.escape(str(row.get('status') or ''))}</td>"
            "</tr>"
        )

    p53_audit_html = []
    for row in p53_audit_rows:
        if not isinstance(row, dict):
            continue
        p53_audit_html.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('timestamp') or ''))}</td>"
            f"<td>{html.escape(str(row.get('action') or ''))}</td>"
            f"<td>{html.escape(str(row.get('success') or ''))}</td>"
            f"<td>{html.escape(str(row.get('target') or ''))}</td>"
            "</tr>"
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>P49/Learned-Router/P53 Dashboard</title>
  <style>
    :root {{ --bg: #f4f0e8; --panel: #fffaf2; --ink: #241d16; --muted: #8a745d; --warn: #9d3c2f; }}
    body {{ margin: 0; padding: 24px; font-family: Georgia, 'Times New Roman', serif; background: linear-gradient(180deg, #efe6d6 0%, var(--bg) 100%); color: var(--ink); }}
    h1, h2 {{ margin: 0 0 12px; }}
    .panel {{ background: var(--panel); border: 1px solid #d9ccb6; border-radius: 14px; padding: 18px; margin-bottom: 18px; box-shadow: 0 8px 24px rgba(36,29,22,0.08); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e8dcc8; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 700; }}
    .muted {{ color: var(--muted); }}
    code {{ font-family: Consolas, monospace; font-size: 12px; }}
  </style>
</head>
<body>
  <div class="panel">
    <h1>P49/P51/Learned-Router/P53 Dashboard</h1>
    <p class="muted">Generated from unified progress events and the latest P22 summary.</p>
    <p><strong>Input:</strong> <code>{html.escape(str(input_root))}</code></p>
    <p><strong>Data:</strong> <code>{html.escape(str((output_dir / "dashboard_data.json").resolve()))}</code></p>
    <p><strong>Ops UI:</strong> <a href="{html.escape(str(p53_ops_ui_payload.get('url') or p53_ops_meta_payload.get('ops_ui_url') or 'http://127.0.0.1:8765/'))}">{html.escape(str(p53_ops_ui_payload.get('url') or p53_ops_meta_payload.get('ops_ui_url') or 'http://127.0.0.1:8765/'))}</a></p>
  </div>
  <div class="panel">
    <h2>Active / Latest Progress</h2>
    <table>
      <thead>
        <tr><th>Run</th><th>Component</th><th>Phase</th><th>Status</th><th>Learner</th><th>Rollout</th><th>Throughput</th><th>GPU MB</th><th>Metrics</th></tr>
      </thead>
      <tbody>
        {''.join(rows_html) or '<tr><td colspan="9">No unified progress events found.</td></tr>'}
      </tbody>
    </table>
  </div>
  <div class="panel">
    <h2>Warnings</h2>
    <ul>
      {''.join(warnings_html) or '<li>No warnings captured.</li>'}
    </ul>
  </div>
  <div class="panel">
    <h2>Latest P22 Summary</h2>
    <table>
      <thead><tr><th>Experiment</th><th>Status</th><th>Mean</th><th>Seeds</th></tr></thead>
      <tbody>
        {''.join(p22_html) or '<tr><td colspan="4">No P22 summary found.</td></tr>'}
      </tbody>
    </table>
  </div>
  <div class="panel">
    <h2>Config Provenance (P55)</h2>
    <table>
      <thead><tr><th>Field</th><th>Value</th></tr></thead>
      <tbody>
        <tr><td>source_type</td><td><code>{html.escape(prov_src_type)}</code></td></tr>
        <tr><td>config_hash</td><td><code>{html.escape(prov_hash)}</code></td></tr>
        <tr><td>sidecar_used</td><td>{html.escape(prov_sidecar_used)}</td></tr>
        <tr><td>sidecar_in_sync</td><td>{html.escape(prov_sidecar_sync)}</td></tr>
        <tr><td>sync_overall_status</td><td><strong>{html.escape(sync_overall)}</strong></td></tr>
        <tr><td>sync_total_yamls</td><td>{html.escape(sync_total)}</td></tr>
        <tr><td>sync_drifted</td><td>{html.escape(sync_drifted)}</td></tr>
        <tr><td>sync_missing_sidecars</td><td>{html.escape(sync_missing)}</td></tr>
        <tr><td>sync_report</td><td><code>{sync_report_link}</code></td></tr>
      </tbody>
    </table>
  </div>
  <div class="panel">
    <h2>{html.escape(router_label)}</h2>
    <p class="muted">dataset: <code>{html.escape(str(router_dataset.get('path') or ''))}</code></p>
    <p class="muted">train: <code>{html.escape(str(router_train.get('path') or ''))}</code></p>
    <p class="muted">promotion_decision: <code>{html.escape(str(router_ablation.get('promotion_decision_path') or ''))}</code></p>
    <p>dataset_samples={html.escape(str(router_dataset_payload.get('sample_count') or 0))} valid={html.escape(str(router_dataset_payload.get('valid_for_training_count') or 0))} mean_label_confidence={html.escape(str(router_dataset_payload.get('mean_label_confidence') or 0.0))}</p>
    <p>train_checkpoint_id=<code>{html.escape(str(router_train.get('checkpoint_id') or ''))}</code> val_top1={html.escape(str(router_train_payload.get('val_top1_accuracy') or ''))} val_topk={html.escape(str(router_train_payload.get('val_topk_accuracy') or ''))} learner={html.escape(str(router_train_payload.get('learner_device') or ''))}</p>
    <p>recommendation=<code>{html.escape(str(router_promotion.get('recommendation') or ''))}</code> score_delta={html.escape(str(router_promotion.get('score_delta') or ''))} guard_trigger_rate={html.escape(str(router_guarded_variant.get('guard_trigger_rate') or ''))}</p>
  </div>
  <div class="panel">
    <h2>{html.escape(router_family_prefix.upper())} Arena Summary</h2>
    <table>
      <thead><tr><th>Policy</th><th>Mean Score</th><th>Win Rate</th><th>Invalid Rate</th></tr></thead>
      <tbody>
        {''.join(p52_summary_html) or '<tr><td colspan="4">No learned-router arena summary found.</td></tr>'}
      </tbody>
    </table>
  </div>
  <div class="panel">
    <h2>{html.escape(router_family_prefix.upper())} Slice Eval</h2>
    <table>
      <thead><tr><th>Target</th><th>Slice Key</th><th>Slice Label</th><th>Score Delta</th><th>Win Delta</th></tr></thead>
      <tbody>
        {''.join(p52_slice_eval_html) or '<tr><td colspan="5">No learned-router slice eval found.</td></tr>'}
      </tbody>
    </table>
  </div>
  <div class="panel">
    <h2>Campaign Stages</h2>
    <table>
      <thead><tr><th>Campaign</th><th>Experiment</th><th>Seed</th><th>Stage</th><th>Status</th></tr></thead>
      <tbody>
        {''.join(campaign_html) or '<tr><td colspan="5">No campaign state found.</td></tr>'}
      </tbody>
    </table>
  </div>
  <div class="panel">
    <h2>Checkpoint Registry</h2>
    <p class="muted">registry: <code>{html.escape(str(registry_summary.get('registry_path') or ''))}</code></p>
    <p class="muted">count={html.escape(str(registry_summary.get('count') or 0))} status_counts={html.escape(json.dumps(registry_counts, ensure_ascii=False))} family_counts={html.escape(json.dumps(registry_summary.get('family_counts') or {}, ensure_ascii=False))}</p>
    <table>
      <thead><tr><th>Checkpoint ID</th><th>Family</th><th>Status</th></tr></thead>
      <tbody>
        {''.join(registry_html) or '<tr><td colspan="3">No registry entries found.</td></tr>'}
      </tbody>
    </table>
  </div>
  <div class="panel">
    <h2>{html.escape(router_family_prefix.upper())} Campaigns And Queue</h2>
    <p class="muted">promotion_queue: <code>{html.escape(str(router_queue.get('path') or ''))}</code></p>
    <p class="muted">queue_counts={html.escape(json.dumps((router_queue_payload.get('counts') or {}), ensure_ascii=False))}</p>
    <table>
      <thead><tr><th>Campaign</th><th>Stage</th><th>Status</th><th>State Path</th></tr></thead>
      <tbody>
        {''.join(p52_campaign_html) or '<tr><td colspan="4">No learned-router campaigns found.</td></tr>'}
      </tbody>
    </table>
    <table>
      <thead><tr><th>Checkpoint ID</th><th>Status</th><th>Artifact</th></tr></thead>
      <tbody>
        {''.join(p52_registry_html) or '<tr><td colspan="3">No learned-router registry entries found.</td></tr>'}
      </tbody>
    </table>
  </div>
  <div class="panel">
    <h2>P56 Calibration / Guard / Canary</h2>
    <p class="muted">calibration: <code>{html.escape(str(p56_calibration.get('path') or ''))}</code></p>
    <p class="muted">guard_tuning: <code>{html.escape(str(p56_guard.get('path') or ''))}</code></p>
    <p class="muted">canary_eval: <code>{html.escape(str(p56_canary.get('path') or ''))}</code></p>
    <p>bias=<code>{html.escape(str(p56_calibration_payload.get('calibration_bias') or 'n/a'))}</code> ece={html.escape(str(p56_calibration_payload.get('ece') or ''))} accuracy={html.escape(str(p56_calibration_payload.get('accuracy') or ''))}</p>
    <p>recommended_guard=<code>{html.escape(json.dumps(p56_guard_cfg_payload.get('guard_config') or {}, ensure_ascii=False))}</code></p>
    <p>deployment_mode=<code>{html.escape(str(p56_canary_payload.get('deployment_mode_recommendation') or p56_promotion_payload.get('deployment_mode_recommendation') or ''))}</code> canary_usage_rate={html.escape(str(p56_canary_payload.get('canary_usage_rate') or ''))} fallback_rate={html.escape(str(p56_canary_payload.get('canary_fallback_rate') or ''))}</p>
    <p>benchmark_policies={html.escape(str(len(p56_benchmark_payload.get('policy_rows') or [])))} benchmark_seeds={html.escape(str(((p56_benchmark_payload.get('evaluation_budget') or {}).get('seed_count') or 0)))} </p>
    <table>
      <thead><tr><th>Campaign</th><th>Stage</th><th>Status</th><th>State Path</th></tr></thead>
      <tbody>
        {''.join(p56_campaign_html) or '<tr><td colspan="4">No P56 campaigns found.</td></tr>'}
      </tbody>
    </table>
  </div>
  <div class="panel">
    <h2>P53 Background Execution</h2>
    <p class="muted">window_state: <code>{html.escape(str(p53_window.get('path') or ''))}</code></p>
    <p class="muted">background_validation: <code>{html.escape(str(p53_background.get('path') or ''))}</code></p>
    <p class="muted">ops_ui_state: <code>{html.escape(str(p53_ops_ui.get('path') or ''))}</code></p>
    <p>current_window_mode=<code>{html.escape(str(p53_window.get('dominant_mode') or ''))}</code> recommended_default_mode=<code>{html.escape(str(p53_background_payload.get('recommended_default_mode') or ''))}</code> fallback=<code>{html.escape(str(p53_background_payload.get('window_mode_fallback') or ''))}</code></p>
    <p>ops_ui_url=<a href="{html.escape(str(p53_ops_ui_payload.get('url') or p53_ops_meta_payload.get('ops_ui_url') or 'http://127.0.0.1:8765/'))}">{html.escape(str(p53_ops_ui_payload.get('url') or p53_ops_meta_payload.get('ops_ui_url') or 'http://127.0.0.1:8765/'))}</a></p>
  </div>
  <div class="panel">
    <h2>P53 Campaigns</h2>
    <table>
      <thead><tr><th>Campaign</th><th>Experiment</th><th>Stage</th><th>Status</th></tr></thead>
      <tbody>
        {''.join(p53_campaign_html) or '<tr><td colspan="4">No P53 campaigns found.</td></tr>'}
      </tbody>
    </table>
  </div>
  <div class="panel">
    <h2>P53 Ops Audit</h2>
    <table>
      <thead><tr><th>Timestamp</th><th>Action</th><th>Success</th><th>Target</th></tr></thead>
      <tbody>
        {''.join(p53_audit_html) or '<tr><td colspan="4">No P53 audit rows found.</td></tr>'}
      </tbody>
    </table>
  </div>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")
    return {
        "status": "ok",
        "output_dir": str(output_dir),
        "index_html": str((output_dir / "index.html").resolve()),
        "dashboard_data_json": str((output_dir / "dashboard_data.json").resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static HTML dashboard from unified P49 progress events.")
    parser.add_argument("--input", default="docs/artifacts")
    parser.add_argument("--output", default="docs/artifacts/dashboard/latest")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    input_root = Path(args.input)
    if not input_root.is_absolute():
        input_root = (repo_root / input_root).resolve()
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()
    summary = build_dashboard(input_root, output_dir)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
