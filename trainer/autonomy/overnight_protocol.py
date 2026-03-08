from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from trainer.autonomy.attention_queue import add_attention_item, attention_root, load_attention_queue
from trainer.autonomy.decision_policy import evaluate_autonomy, load_decision_policy
from trainer.autonomy.morning_summary import write_morning_summary
from trainer.campaigns.campaign_schema import P57_OVERNIGHT_STAGE_IDS
from trainer.campaigns.resume_state import (
    load_campaign_state,
    mark_stage_blocked,
    mark_stage_completed,
    mark_stage_failed,
    mark_stage_started,
    save_campaign_state,
    set_stage_autonomy,
    should_skip_stage,
    unresolved_human_gate,
)
from trainer.monitoring.dashboard_build import build_dashboard
from trainer.ops_ui.state_loader import campaign_rows, latest_promotion_queue, latest_readiness_report, registry_entries
from trainer.registry.checkpoint_registry import list_entries, snapshot_registry
from trainer.registry.promotion_queue import build_promotion_queue_summary


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _latest_config_sync_status(repo_root: Path) -> dict[str, Any]:
    root = repo_root / "docs" / "artifacts" / "p55" / "config_sidecar_sync"
    if not root.exists():
        return {"path": "", "payload": {}}
    runs = sorted([path for path in root.iterdir() if path.is_dir()], key=lambda path: path.name)
    if not runs:
        return {"path": "", "payload": {}}
    latest = runs[-1] / "sidecar_sync_report.json"
    payload = _read_json(latest)
    return {"path": str(latest.resolve()), "payload": payload if isinstance(payload, dict) else {}}


def _campaign_resume_lines(payload: dict[str, Any]) -> list[str]:
    lines = [
        "# P57 Campaign Resume Report",
        "",
        f"- campaign_id: `{payload.get('campaign_id') or ''}`",
        f"- run_id: `{payload.get('run_id') or ''}`",
        f"- experiment_id: `{payload.get('experiment_id') or ''}`",
        f"- seed: `{payload.get('seed') or ''}`",
        "",
        "## Stages",
        "",
    ]
    for stage in payload.get("stages") or []:
        if not isinstance(stage, dict):
            continue
        lines.append(
            "- {stage}: status={status} attempts={attempts} decision={decision} continue_allowed={allowed} attention={attention}".format(
                stage=str(stage.get("stage_id") or ""),
                status=str(stage.get("status") or ""),
                attempts=int(stage.get("attempt_count") or 0),
                decision=str(stage.get("autonomy_decision") or ""),
                allowed=str(bool(stage.get("continue_allowed", True))).lower(),
                attention=str(stage.get("attention_item_ref") or ""),
            )
        )
    return lines


def run_overnight_protocol_seed_experiment(
    *,
    repo_root: Path,
    run_id: str,
    exp: dict[str, Any],
    exp_dir: Path,
    seed: str,
    seed_idx: int,
    seed_total: int,
    config_path: Path,
    config_payload: dict[str, Any],
    readiness_report_path: str = "",
) -> dict[str, Any]:
    decision_cfg = config_payload.get("decision_policy") if isinstance(config_payload.get("decision_policy"), dict) else {}
    campaign_cfg = config_payload.get("campaign") if isinstance(config_payload.get("campaign"), dict) else {}
    output_cfg = config_payload.get("output") if isinstance(config_payload.get("output"), dict) else {}
    simulation_cfg = config_payload.get("simulation") if isinstance(config_payload.get("simulation"), dict) else {}
    policy_path = (repo_root / str(decision_cfg.get("path") or "configs/runtime/decision_policy.yaml")).resolve()
    policy = load_decision_policy(policy_path)
    autonomy_mode = str(config_payload.get("autonomy_mode") or exp.get("autonomy_mode") or "overnight_smoke")
    stage_ids = list(campaign_cfg.get("stage_ids") or P57_OVERNIGHT_STAGE_IDS)
    force_block_stage = str(simulation_cfg.get("force_human_gate_stage") or "").strip()

    queue_root = (repo_root / str(output_cfg.get("attention_root") or "docs/artifacts/attention_required")).resolve()
    summary_root = (repo_root / str(output_cfg.get("morning_summary_root") or "docs/artifacts/morning_summary")).resolve()
    campaign_root = exp_dir / "campaign_runs" / f"seed_{seed_idx:03d}_{seed}"
    campaign_root.mkdir(parents=True, exist_ok=True)
    state_path = campaign_root / "campaign_state.json"
    resume_report_path = campaign_root / "campaign_resume_report.md"
    campaign_summary_path = campaign_root / "campaign_summary.json"
    registry_snapshot_path = campaign_root / "checkpoint_registry_snapshot.json"
    promotion_queue_path = campaign_root / "promotion_queue.json"

    campaign_state = load_campaign_state(
        state_path=state_path,
        campaign_id=f"{str(exp.get('id') or 'p57_overnight_protocol')}-{seed}",
        run_id=run_id,
        experiment_id=str(exp.get("id") or ""),
        seed=seed,
        stage_ids=stage_ids,
        metadata={
            "config_path": str(config_path),
            "decision_policy_path": str(policy_path),
            "autonomy_mode": autonomy_mode,
            "training_python": str(exp.get("training_python") or ""),
            "readiness_report_path": str(readiness_report_path or ""),
        },
    )

    def _existing_stage(stage_id: str) -> dict[str, Any]:
        for stage in campaign_state.get("stages") or []:
            if isinstance(stage, dict) and str(stage.get("stage_id") or "") == str(stage_id):
                return stage
        return {}

    existing_human_gate = unresolved_human_gate(campaign_state)
    warnings: list[str] = []
    attention_refs: list[str] = []
    human_gate_triggered = existing_human_gate is not None
    dashboard_path = str((repo_root / "docs" / "artifacts" / "dashboard" / "latest" / "index.html").resolve())
    latest_morning_summary_path = str(
        (
            _existing_stage("morning_summary").get("artifacts")
            if isinstance(_existing_stage("morning_summary").get("artifacts"), dict)
            else {}
        ).get("morning_summary_path")
        or ""
    )

    def _new_attention(
        *,
        severity: str,
        category: str,
        title: str,
        summary: str,
        stage_id: str,
        attempted_actions: list[str],
        recommended_options: list[dict[str, Any]],
        recommended_default: str,
        required_human_input: list[str],
        artifact_refs: list[str],
    ) -> dict[str, Any]:
        item = add_attention_item(
            {
                "severity": severity,
                "category": category,
                "title": title,
                "summary": summary,
                "blocking_stage": stage_id,
                "attempted_actions": attempted_actions,
                "recommended_options": recommended_options,
                "recommended_default": recommended_default,
                "required_human_input": required_human_input,
                "artifact_refs": artifact_refs,
                "source_run_id": run_id,
                "source_experiment_id": str(exp.get("id") or ""),
                "source_campaign_id": str(campaign_state.get("campaign_id") or ""),
            },
            root=queue_root,
        )
        attention_refs.append(str(item.get("attention_file") or item.get("attention_id") or ""))
        return item

    def _finalize_stage(
        stage_id: str,
        *,
        artifacts: dict[str, Any],
        autonomy: dict[str, Any],
        attention_item: dict[str, Any] | None = None,
        failed: bool = False,
    ) -> None:
        nonlocal campaign_state, human_gate_triggered
        attention_ref = str((attention_item or {}).get("attention_file") or (attention_item or {}).get("attention_id") or "")
        if failed:
            campaign_state = mark_stage_failed(
                campaign_state,
                stage_id,
                error_summary=str(autonomy.get("reason") or "stage_failed"),
                artifacts=artifacts,
                resume_safe=True,
            )
        elif str(autonomy.get("decision") or "") == "stop_and_queue_attention":
            human_gate_triggered = True
            campaign_state = mark_stage_blocked(
                campaign_state,
                stage_id,
                reason=str(autonomy.get("reason") or "human_gate_triggered"),
                attention_item_ref=attention_ref,
                artifacts=artifacts,
                resume_safe=True,
            )
        else:
            campaign_state = mark_stage_completed(campaign_state, stage_id, artifacts=artifacts, resume_safe=True, skipped=False)
            campaign_state = set_stage_autonomy(
                campaign_state,
                stage_id,
                decision=str(autonomy.get("decision") or ""),
                reason=str(autonomy.get("reason") or ""),
                continue_allowed=bool(autonomy.get("continue_allowed", True)),
                human_gate_triggered=bool(autonomy.get("human_gate_triggered", False)),
                attention_item_ref=attention_ref,
            )
        if str(autonomy.get("decision") or "") == "continue_with_warning":
            warnings.append(str(autonomy.get("reason") or stage_id))
        save_campaign_state(state_path, campaign_state)

    def _run_stage(stage_id: str, handler: Callable[[], tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]], *, finalizer: bool = False) -> bool:
        nonlocal campaign_state
        if not finalizer and human_gate_triggered:
            return False
        if should_skip_stage(campaign_state, stage_id, force_rerun=stage_id in {str(item).strip() for item in (campaign_cfg.get("force_rerun_stages") or [])}):
            return True
        campaign_state = mark_stage_started(campaign_state, stage_id)
        save_campaign_state(state_path, campaign_state)
        try:
            artifacts, autonomy, attention_item = handler()
        except Exception as exc:
            attention_item = _new_attention(
                severity="block",
                category="regression",
                title=f"Stage `{stage_id}` failed during overnight protocol",
                summary=str(exc),
                stage_id=stage_id,
                attempted_actions=[stage_id],
                recommended_options=[{"label": "inspect_logs", "description": "Inspect the stage artifacts and rerun after review."}],
                recommended_default="inspect_logs",
                required_human_input=["Confirm whether the failure is expected and safe to resume."],
                artifact_refs=[str(state_path)],
            )
            autonomy = {
                "decision": "stop_and_queue_attention",
                "reason": str(exc),
                "continue_allowed": False,
                "human_gate_triggered": True,
            }
            _finalize_stage(stage_id, artifacts={"error": str(exc)}, autonomy=autonomy, attention_item=attention_item, failed=False)
            return False
        _finalize_stage(stage_id, artifacts=artifacts, autonomy=autonomy, attention_item=attention_item, failed=False)
        return bool(autonomy.get("continue_allowed", True))

    def _readiness_handler() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        report = latest_readiness_report()
        report_path = readiness_report_path or str(report.get("path") or "")
        payload = report.get("payload") if isinstance(report.get("payload"), dict) else {}
        forced_status = str(simulation_cfg.get("force_readiness_status") or "").strip().lower()
        allow_missing = bool(simulation_cfg.get("allow_missing_readiness_report", False))
        readiness_status = forced_status or str(payload.get("status") or "").strip().lower()
        if force_block_stage == "readiness_guard":
            attention_item = _new_attention(
                severity="block",
                category="environment",
                title="Readiness guard forced a human stop",
                summary="P57 validation forced a blocking readiness scenario.",
                stage_id="readiness_guard",
                attempted_actions=["probe_service_readiness"],
                recommended_options=[{"label": "confirm_recovery", "description": "Confirm the service environment is healthy before resume."}],
                recommended_default="confirm_recovery",
                required_human_input=["Confirm the real backend is safe to use."],
                artifact_refs=[report_path] if report_path else [],
            )
            autonomy = evaluate_autonomy(policy=policy, conditions=["readiness_guard_failed_no_fallback"], summary="forced_readiness_block")
            return {"readiness_report_path": report_path, "readiness_status": readiness_status}, autonomy, attention_item
        if not readiness_status and allow_missing:
            autonomy = evaluate_autonomy(policy=policy, actions=["resume_campaign"], summary="readiness_report_missing_allowed")
            return {"readiness_report_path": report_path, "readiness_status": "missing_allowed"}, autonomy, None
        if readiness_status != "ready":
            attention_item = _new_attention(
                severity="block",
                category="environment",
                title="Readiness guard failed",
                summary="The service readiness report is not ready, so the overnight runner cannot safely continue.",
                stage_id="readiness_guard",
                attempted_actions=["probe_service_readiness"],
                recommended_options=[{"label": "repair_service", "description": "Repair the service/backend and resume the campaign."}],
                recommended_default="repair_service",
                required_human_input=["Confirm the service/backend is healthy."],
                artifact_refs=[report_path] if report_path else [],
            )
            autonomy = evaluate_autonomy(policy=policy, conditions=["readiness_guard_failed_no_fallback"], summary="readiness_not_ready")
            return {"readiness_report_path": report_path, "readiness_status": readiness_status or "missing"}, autonomy, attention_item
        autonomy = evaluate_autonomy(policy=policy, actions=["resume_campaign"], summary="readiness_ok")
        return {"readiness_report_path": report_path, "readiness_status": readiness_status}, autonomy, None

    def _decision_policy_handler() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        snapshot_path = campaign_root / "decision_policy_snapshot.json"
        snapshot = {
            "policy_path": str(policy_path),
            "policy": policy,
            "sample_actions": {
                "resume_campaign": evaluate_autonomy(policy=policy, actions=["resume_campaign"]),
                "promote_candidate": evaluate_autonomy(policy=policy, actions=["promote_candidate"], conditions=["promotion_review_pending"]),
                "install_dependencies": evaluate_autonomy(policy=policy, actions=["install_dependencies"]),
            },
        }
        _write_json(snapshot_path, snapshot)
        autonomy = evaluate_autonomy(policy=policy, actions=["resume_campaign"], summary="decision_policy_loaded")
        return {"decision_policy_path": str(policy_path), "decision_policy_snapshot": str(snapshot_path.resolve())}, autonomy, None

    def _config_handler() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        report = _latest_config_sync_status(repo_root)
        payload = report.get("payload") if isinstance(report.get("payload"), dict) else {}
        anomaly = bool(simulation_cfg.get("simulate_config_provenance_anomaly")) or force_block_stage == "config_provenance_scan"
        if anomaly or str(payload.get("overall_status") or "").lower() not in {"clean", "in_sync", "ok"}:
            attention_item = _new_attention(
                severity="block",
                category="config",
                title="Config provenance anomaly detected",
                summary="A config sidecar or provenance mismatch requires human review before the overnight campaign should continue.",
                stage_id="config_provenance_scan",
                attempted_actions=["update_config_sidecars", "verify_config_hash"],
                recommended_options=[
                    {"label": "investigate_drift", "description": "Inspect the drift and confirm the intended config source."},
                    {"label": "accept_sync", "description": "Confirm the new sidecar/provenance state and rerun."},
                ],
                recommended_default="investigate_drift",
                required_human_input=["Confirm whether the config drift is expected."],
                artifact_refs=[str(report.get("path") or "")] if str(report.get("path") or "") else [],
            )
            autonomy = evaluate_autonomy(policy=policy, conditions=["config_provenance_unrecoverable"], summary="config_anomaly")
            return {"config_sync_report_path": str(report.get("path") or ""), "config_sync_status": str(payload.get("overall_status") or "")}, autonomy, attention_item
        autonomy = evaluate_autonomy(policy=policy, actions=["update_config_sidecars"], summary="config_provenance_clean")
        return {"config_sync_report_path": str(report.get("path") or ""), "config_sync_status": str(payload.get("overall_status") or "")}, autonomy, None

    def _promotion_handler() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        snapshot_registry(out_path=registry_snapshot_path)
        queue_payload = build_promotion_queue_summary(list_entries())
        _write_json(promotion_queue_path, queue_payload)
        review_rows = [row for row in (queue_payload.get("promotion_review") or []) if isinstance(row, dict)]
        if force_block_stage == "promotion_queue_scan":
            attention_item = _new_attention(
                severity="block",
                category="promotion",
                title="Promotion scan forced a blocking human gate",
                summary="P57 validation forced a blocking promotion decision to verify campaign stop semantics.",
                stage_id="promotion_queue_scan",
                attempted_actions=["refresh_registry", "build_promotion_queue"],
                recommended_options=[{"label": "confirm_route", "description": "Choose whether the candidate should be promoted or deferred."}],
                recommended_default="confirm_route",
                required_human_input=["Choose whether to promote, canary, guard, or defer."],
                artifact_refs=[str(promotion_queue_path.resolve())],
            )
            autonomy = evaluate_autonomy(policy=policy, actions=["promote_checkpoint_live"], summary="forced_promotion_block")
            return {"registry_snapshot_path": str(registry_snapshot_path.resolve()), "promotion_queue_path": str(promotion_queue_path.resolve()), "promotion_review_count": len(review_rows)}, autonomy, attention_item
        if review_rows:
            attention_item = _new_attention(
                severity="warn",
                category="promotion",
                title="Promotion review items are waiting for a human",
                summary="Promotion review exists, but the overnight system must only suggest the action rather than apply it.",
                stage_id="promotion_queue_scan",
                attempted_actions=["refresh_registry", "build_promotion_queue"],
                recommended_options=[
                    {"label": "review_candidates", "description": "Review promotion_review candidates in the morning."},
                    {"label": "defer", "description": "Keep candidates in review and gather more evidence."},
                ],
                recommended_default="review_candidates",
                required_human_input=["Confirm whether any candidate should become the live or default choice."],
                artifact_refs=[str(promotion_queue_path.resolve())],
            )
            autonomy = evaluate_autonomy(policy=policy, actions=["promote_candidate"], conditions=["promotion_review_pending"], summary="promotion_review_pending")
            return {"registry_snapshot_path": str(registry_snapshot_path.resolve()), "promotion_queue_path": str(promotion_queue_path.resolve()), "promotion_review_count": len(review_rows)}, autonomy, attention_item
        autonomy = evaluate_autonomy(policy=policy, actions=["update_registry"], summary="promotion_queue_clear")
        return {"registry_snapshot_path": str(registry_snapshot_path.resolve()), "promotion_queue_path": str(promotion_queue_path.resolve()), "promotion_review_count": 0}, autonomy, None

    def _campaign_health_handler() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        rows = campaign_rows(limit=48)
        blocked_rows = [row for row in rows if isinstance(row, dict) and str(row.get("status") or "") == "blocked"]
        if blocked_rows and bool(campaign_cfg.get("block_on_existing_blocked_campaigns", False)):
            attention_item = _new_attention(
                severity="block",
                category="regression",
                title="Blocked campaigns already exist",
                summary="One or more campaigns are already blocked by a human gate, so the overnight protocol stopped this branch.",
                stage_id="campaign_health_scan",
                attempted_actions=["scan_campaign_state"],
                recommended_options=[{"label": "review_blocked_campaigns", "description": "Resolve or ignore the blocked campaigns before resuming."}],
                recommended_default="review_blocked_campaigns",
                required_human_input=["Resolve the blocked campaigns or explicitly ignore them."],
                artifact_refs=[str(row.get("state_path") or "") for row in blocked_rows[:6]],
            )
            autonomy = evaluate_autonomy(policy=policy, conditions=["unresolved_attention_item"], summary="blocked_campaigns_present")
            return {"blocked_campaign_count": len(blocked_rows)}, autonomy, attention_item
        if blocked_rows:
            attention_item = _new_attention(
                severity="warn",
                category="ambiguity",
                title="Blocked campaigns are open in the repo",
                summary="Existing blocked campaigns were detected. The overnight protocol continued, but they should be reviewed in the morning.",
                stage_id="campaign_health_scan",
                attempted_actions=["scan_campaign_state"],
                recommended_options=[{"label": "review_blocked_campaigns", "description": "Inspect the blocked campaigns and resolve their attention items."}],
                recommended_default="review_blocked_campaigns",
                required_human_input=["Review blocked campaigns."],
                artifact_refs=[str(row.get("state_path") or "") for row in blocked_rows[:6]],
            )
            autonomy = evaluate_autonomy(policy=policy, conditions=["advisory_attention_item_open"], summary="blocked_campaigns_advisory")
            return {"blocked_campaign_count": len(blocked_rows)}, autonomy, attention_item
        autonomy = evaluate_autonomy(policy=policy, actions=["resume_campaign"], summary="campaign_health_clear")
        return {"blocked_campaign_count": 0}, autonomy, None

    def _dashboard_handler() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        summary = build_dashboard(repo_root / "docs" / "artifacts", repo_root / "docs" / "artifacts" / "dashboard" / "latest")
        autonomy = evaluate_autonomy(policy=policy, actions=["run_dashboard_build"], summary="dashboard_built")
        return summary, autonomy, None

    def _morning_summary_handler() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        nonlocal latest_morning_summary_path
        summary = write_morning_summary(repo=repo_root)
        latest_morning_summary_path = str(summary.get("latest_md") or "")
        autonomy = evaluate_autonomy(policy=policy, actions=["build_morning_summary"], summary="morning_summary_built")
        return {
            "morning_summary_path": str(summary.get("latest_md") or ""),
            "morning_summary_json": str(summary.get("latest_json") or ""),
        }, autonomy, None

    stage_handlers: dict[str, tuple[Callable[[], tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]], bool]] = {
        "readiness_guard": (_readiness_handler, False),
        "decision_policy_audit": (_decision_policy_handler, False),
        "config_provenance_scan": (_config_handler, False),
        "promotion_queue_scan": (_promotion_handler, False),
        "campaign_health_scan": (_campaign_health_handler, False),
        "dashboard_build": (_dashboard_handler, True),
        "morning_summary": (_morning_summary_handler, True),
    }

    for stage_id in stage_ids:
        handler, finalizer = stage_handlers.get(stage_id, (None, False))
        if handler is None:
            continue
        _run_stage(stage_id, handler, finalizer=finalizer)

    resume_report_path.write_text("\n".join(_campaign_resume_lines(campaign_state)).rstrip() + "\n", encoding="utf-8")
    queue_payload = load_attention_queue(queue_root)
    open_items = [item for item in (queue_payload.get("items") or []) if isinstance(item, dict) and str(item.get("status") or "") == "open"]
    stage_statuses = [str(stage.get("status") or "") for stage in (campaign_state.get("stages") or []) if isinstance(stage, dict)]
    completed_count = sum(1 for status in stage_statuses if status in {"completed", "skipped"})
    blocked_count = sum(1 for status in stage_statuses if status == "blocked")
    total_stages = max(1, len(stage_statuses))
    score = max(0.0, round((completed_count / total_stages) * 100.0 - (10.0 * blocked_count), 4))
    summary_payload = {
        "schema": "p57_campaign_seed_summary_v1",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "run_id": run_id,
        "seed": seed,
        "run_dir": str(campaign_root.resolve()),
        "campaign_state_path": str(state_path.resolve()),
        "decision_policy_path": str(policy_path),
        "attention_queue_path": str((queue_root / "attention_queue.json").resolve()),
        "morning_summary_path": latest_morning_summary_path,
        "registry_snapshot_path": str(registry_snapshot_path.resolve()),
        "promotion_queue_path": str(promotion_queue_path.resolve()),
        "resume_report_md": str(resume_report_path.resolve()),
        "dashboard_path": dashboard_path,
        "autonomy_mode": autonomy_mode,
        "human_gate_triggered": human_gate_triggered,
        "open_attention_count": len(open_items),
        "warning_count": len(warnings),
        "attention_refs": attention_refs,
    }
    _write_json(campaign_summary_path, summary_payload)

    status = "ok" if not human_gate_triggered else "blocked"
    metrics = {
        "score": score,
        "avg_reward": score,
        "best_episode_reward": score,
        "avg_ante_reached": score,
        "median_ante": score,
        "win_rate": 1.0 if not human_gate_triggered else 0.0,
        "illegal_action_rate": 0.0,
        "p57_campaign_state_path": str(state_path.resolve()),
        "p57_registry_snapshot_path": str(registry_snapshot_path.resolve()),
        "p57_promotion_queue_path": str(promotion_queue_path.resolve()),
        "p57_resume_report_md": str(resume_report_path.resolve()),
        "p57_dashboard_path": dashboard_path,
        "p57_readiness_report_path": str(readiness_report_path or latest_readiness_report().get("path") or ""),
        "p57_decision_policy_path": str(policy_path),
        "p57_attention_queue_path": str((queue_root / "attention_queue.json").resolve()),
        "p57_morning_summary_path": latest_morning_summary_path,
        "p57_autonomy_mode": autonomy_mode,
        "p57_human_gate_triggered": human_gate_triggered,
        "p57_open_attention_count": len(open_items),
    }
    return {
        "status": status,
        "metrics": metrics,
        "summary": summary_payload,
    }
