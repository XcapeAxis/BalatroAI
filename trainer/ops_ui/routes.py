from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from trainer.ops_ui.state_loader import path_in_repo


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def artifact_link(path: str, label: str = "") -> str:
    target = path_in_repo(path)
    if not isinstance(target, Path):
        return f"<code>{esc(path)}</code>"
    href = "/artifact?path=" + quote(str(target))
    return f'<a href="{href}">{esc(label or target.name)}</a>'


def table(headers: list[str], rows: list[list[str]]) -> str:
    head_html = "".join([f"<th>{esc(head)}</th>" for head in headers])
    row_html = []
    for row in rows:
        row_html.append("<tr>" + "".join([f"<td>{cell}</td>" for cell in row]) + "</tr>")
    if not row_html:
        row_html.append(f'<tr><td colspan="{len(headers)}">No rows.</td></tr>')
    return "<table><thead><tr>{}</tr></thead><tbody>{}</tbody></table>".format(head_html, "".join(row_html))


def actions_panel(current_path: str, resume_command: str, ops_ui_url: str) -> str:
    return """
    <section class="panel">
      <h2>Controls</h2>
      <div class="actions">
        <form method="post" action="/actions/run_p22_quick"><input type="hidden" name="return_to" value="{return_to}"><button type="submit">Start P22 Quick</button></form>
        <form method="post" action="/actions/run_p22_p53"><input type="hidden" name="return_to" value="{return_to}"><button type="submit">Start P53 Smoke</button></form>
        <form method="post" action="/actions/run_p22_p57"><input type="hidden" name="return_to" value="{return_to}"><button type="submit">Start P57 Smoke</button></form>
        <form method="post" action="/actions/run_p22_overnight"><input type="hidden" name="return_to" value="{return_to}"><button type="submit">Start Overnight</button></form>
        <form method="post" action="/actions/run_autonomy_quick"><input type="hidden" name="return_to" value="{return_to}"><button type="submit">Start Autonomy Quick</button></form>
        <form method="post" action="/actions/run_autonomy_overnight"><input type="hidden" name="return_to" value="{return_to}"><button type="submit">Start Autonomy Overnight</button></form>
        <form method="post" action="/actions/run_autonomy_resume"><input type="hidden" name="return_to" value="{return_to}"><button type="submit">Resume Via Autonomy</button></form>
        <form method="post" action="/actions/run_fast_checks"><input type="hidden" name="return_to" value="{return_to}"><button type="submit">Run Fast Checks</button></form>
        <form method="post" action="/actions/run_certification_latest"><input type="hidden" name="return_to" value="{return_to}"><button type="submit">Run Latest Certification</button></form>
        <form method="post" action="/actions/run_doctor"><input type="hidden" name="return_to" value="{return_to}"><button type="submit">Run Doctor</button></form>
        <form method="post" action="/actions/resume_latest_campaign"><input type="hidden" name="return_to" value="{return_to}"><button type="submit">Resume Latest Campaign</button></form>
        <form method="post" action="/actions/rebuild_dashboard"><input type="hidden" name="return_to" value="{return_to}"><button type="submit">Rebuild Dashboard</button></form>
        <form method="post" action="/actions/refresh_registry"><input type="hidden" name="return_to" value="{return_to}"><button type="submit">Refresh Registry Snapshot</button></form>
      </div>
      <p class="muted">resume command: <code>{resume}</code></p>
      <p class="muted">ops ui url: <code>{ops_ui}</code></p>
    </section>
    """.format(return_to=esc(current_path), resume=esc(resume_command), ops_ui=esc(ops_ui_url))


def render_overview(state: dict[str, Any], *, current_path: str) -> str:
    p22 = state.get("p22") if isinstance(state.get("p22"), dict) else {}
    readiness = state.get("readiness") if isinstance(state.get("readiness"), dict) else {}
    readiness_payload = readiness.get("payload") if isinstance(readiness.get("payload"), dict) else {}
    window_state = state.get("window_state") if isinstance(state.get("window_state"), dict) else {}
    background = state.get("background_validation") if isinstance(state.get("background_validation"), dict) else {}
    background_payload = background.get("payload") if isinstance(background.get("payload"), dict) else {}
    dashboard = state.get("dashboard") if isinstance(state.get("dashboard"), dict) else {}
    dashboard_payload = dashboard.get("payload") if isinstance(dashboard.get("payload"), dict) else {}
    learned_router = dashboard_payload.get("learned_router") if isinstance(dashboard_payload.get("learned_router"), dict) else {}
    learned_router_family = str(learned_router.get("family_prefix") or "p52").upper()
    learned_router_dataset = learned_router.get("dataset") if isinstance(learned_router.get("dataset"), dict) else {}
    learned_router_dataset_payload = learned_router_dataset.get("payload") if isinstance(learned_router_dataset.get("payload"), dict) else {}
    learned_router_train = learned_router.get("train") if isinstance(learned_router.get("train"), dict) else {}
    learned_router_train_payload = learned_router_train.get("payload") if isinstance(learned_router_train.get("payload"), dict) else {}
    learned_router_ablation = learned_router.get("ablation") if isinstance(learned_router.get("ablation"), dict) else {}
    learned_router_promotion = learned_router_ablation.get("promotion_decision") if isinstance(learned_router_ablation.get("promotion_decision"), dict) else {}
    learned_router_guarded = learned_router_ablation.get("guarded_variant") if isinstance(learned_router_ablation.get("guarded_variant"), dict) else {}
    learned_router_canary = learned_router_ablation.get("canary_variant") if isinstance(learned_router_ablation.get("canary_variant"), dict) else {}
    p56 = dashboard_payload.get("p56") if isinstance(dashboard_payload.get("p56"), dict) else {}
    p56_calibration = p56.get("calibration") if isinstance(p56.get("calibration"), dict) else {}
    p56_calibration_payload = p56_calibration.get("payload") if isinstance(p56_calibration.get("payload"), dict) else {}
    p56_guard = p56.get("guard_tuning") if isinstance(p56.get("guard_tuning"), dict) else {}
    p56_guard_payload = p56_guard.get("payload") if isinstance(p56_guard.get("payload"), dict) else {}
    p56_guard_cfg = p56.get("recommended_guard") if isinstance(p56.get("recommended_guard"), dict) else {}
    p56_guard_cfg_payload = p56_guard_cfg.get("payload") if isinstance(p56_guard_cfg.get("payload"), dict) else {}
    p56_canary = p56.get("canary_eval") if isinstance(p56.get("canary_eval"), dict) else {}
    p56_canary_payload = p56_canary.get("payload") if isinstance(p56_canary.get("payload"), dict) else {}
    # P55: config provenance from dashboard payload
    config_prov = dashboard_payload.get("config_provenance") if isinstance(dashboard_payload.get("config_provenance"), dict) else {}
    config_sync_st = dashboard_payload.get("config_sync_status") if isinstance(dashboard_payload.get("config_sync_status"), dict) else {}
    campaigns = state.get("campaigns") if isinstance(state.get("campaigns"), list) else []
    progress = state.get("progress") if isinstance(state.get("progress"), list) else []
    registry = state.get("registry") if isinstance(state.get("registry"), dict) else {}
    attention_queue = state.get("attention_queue") if isinstance(state.get("attention_queue"), dict) else {}
    attention_open = attention_queue.get("open_items") if isinstance(attention_queue.get("open_items"), list) else []
    morning_summary = state.get("morning_summary") if isinstance(state.get("morning_summary"), dict) else {}
    morning_payload = morning_summary.get("payload") if isinstance(morning_summary.get("payload"), dict) else {}
    blocked_campaigns = state.get("blocked_campaigns") if isinstance(state.get("blocked_campaigns"), list) else []
    resume_target = state.get("latest_resume_target") if isinstance(state.get("latest_resume_target"), dict) else {}
    ops_ui_state = state.get("ops_ui") if isinstance(state.get("ops_ui"), dict) else {}
    ops_ui_payload = ops_ui_state.get("payload") if isinstance(ops_ui_state.get("payload"), dict) else {}
    environment = state.get("environment") if isinstance(state.get("environment"), dict) else {}
    doctor = state.get("doctor") if isinstance(state.get("doctor"), dict) else {}
    bootstrap = state.get("bootstrap") if isinstance(state.get("bootstrap"), dict) else {}
    agents = state.get("agents") if isinstance(state.get("agents"), dict) else {}
    autonomy_entry = state.get("autonomy_entry") if isinstance(state.get("autonomy_entry"), dict) else {}
    autonomy_payload = autonomy_entry.get("payload") if isinstance(autonomy_entry.get("payload"), dict) else {}
    agents_consistency = state.get("agents_consistency") if isinstance(state.get("agents_consistency"), dict) else {}
    agents_consistency_payload = agents_consistency.get("payload") if isinstance(agents_consistency.get("payload"), dict) else {}
    fast_checks = state.get("fast_checks") if isinstance(state.get("fast_checks"), dict) else {}
    fast_checks_payload = fast_checks.get("payload") if isinstance(fast_checks.get("payload"), dict) else {}
    certification_queue = state.get("certification_queue") if isinstance(state.get("certification_queue"), dict) else {}
    certification_state = certification_queue.get("state_payload") if isinstance(certification_queue.get("state_payload"), dict) else {}
    pending_certification = certification_queue.get("pending_items") if isinstance(certification_queue.get("pending_items"), list) else []
    latest_certified_runs = certification_queue.get("latest_certified_runs") if isinstance(certification_queue.get("latest_certified_runs"), list) else []

    latest_rows = []
    for row in (p22.get("summary_rows") or [])[:8]:
        if not isinstance(row, dict):
            continue
        latest_rows.append(
            [
                esc(row.get("exp_id") or ""),
                esc(row.get("status") or ""),
                esc(row.get("mean") or ""),
                esc(row.get("window_mode") or ""),
                artifact_link(str(row.get("run_dir") or ""), "run"),
            ]
        )

    progress_rows = []
    for row in progress[:8]:
        if not isinstance(row, dict):
            continue
        progress_rows.append(
            [
                esc(row.get("run_id") or ""),
                esc(row.get("component") or ""),
                esc(row.get("status") or ""),
                esc(row.get("learner_device") or ""),
                esc(row.get("rollout_device") or ""),
                esc(row.get("throughput") or ""),
                esc(row.get("warning") or ""),
            ]
        )

    campaign_rows = []
    for row in campaigns[:8]:
        if not isinstance(row, dict):
            continue
        campaign_rows.append(
            [
                esc(row.get("campaign_id") or ""),
                esc(row.get("experiment_id") or ""),
                esc(row.get("stage_id") or ""),
                esc(row.get("status") or ""),
                artifact_link(str(row.get("state_path") or ""), "state"),
            ]
        )

    cards = f"""
    <section class="panel cards">
      <div class="card"><h2>Latest P22</h2><p><strong>{esc(p22.get('run_id') or 'n/a')}</strong></p><p class="muted">{artifact_link(str(p22.get('summary_path') or ''), 'summary_table.json')}</p></div>
      <div class="card"><h2>Readiness</h2><p><strong>{esc(readiness_payload.get('status') or 'n/a')}</strong></p><p class="muted">window={esc(window_state.get('dominant_mode') or '')}</p></div>
      <div class="card"><h2>Background Mode</h2><p><strong>{esc(background_payload.get('recommended_default_mode') or 'n/a')}</strong></p><p class="muted">fallback={esc(background_payload.get('window_mode_fallback') or '')}</p></div>
      <div class="card"><h2>Registry</h2><p><strong>{esc(sum((registry.get('counts') or {}).values()) if isinstance(registry.get('counts'), dict) else 0)}</strong></p><p class="muted">{esc(json.dumps(registry.get('counts') or {}, ensure_ascii=False))}</p></div>
      <div class="card"><h2>P57 Attention</h2><p><strong>{esc(len(attention_open))}</strong></p><p class="muted">blocked_campaigns={esc(len(blocked_campaigns))}</p></div>
      <div class="card"><h2>P58 Environment</h2><p><strong>{esc(environment.get('status') or 'n/a')}</strong></p><p class="muted">mode={esc(environment.get('recommended_mode') or '')}</p></div>
      <div class="card"><h2>P60 Autonomy</h2><p><strong>{esc(autonomy_payload.get('autonomy_state') or 'n/a')}</strong></p><p class="muted">plan={esc(autonomy_payload.get('selected_plan') or '')}</p></div>
      <div class="card"><h2>P61 Fast Loop</h2><p><strong>{esc(fast_checks_payload.get('fast_check_status') or 'n/a')}</strong></p><p class="muted">tiers={esc(','.join([str(item) for item in (fast_checks_payload.get('validation_tiers_completed') or [])]))}</p></div>
      <div class="card"><h2>P61 Certification</h2><p><strong>{esc(certification_state.get('status') or 'n/a')}</strong></p><p class="muted">pending={esc(len(pending_certification))}</p></div>
    </section>
    """
    return (
        actions_panel(
            current_path,
            str(resume_target.get("resume_command") or ""),
            str(ops_ui_payload.get("url") or "http://127.0.0.1:8765/"),
        )
        + cards
        + f"""
        <section class="panel">
          <h2>P61 Validation Workflow</h2>
          <p>fast_check: {artifact_link(str(fast_checks.get('json_path') or ''), 'latest_fast_check_report.json')}</p>
          <p>certification_queue: {artifact_link(str(certification_queue.get('queue_path') or ''), 'certification_queue.json')} | {artifact_link(str(certification_queue.get('state_path') or ''), 'certification_state.json')}</p>
          <p>fast_check_status=<strong>{esc(fast_checks_payload.get('fast_check_status') or '')}</strong> validation_tiers=<code>{esc(','.join([str(item) for item in (fast_checks_payload.get('validation_tiers_completed') or [])]))}</code></p>
          <p>certification_status=<strong>{esc(certification_state.get('status') or '')}</strong> pending={esc(len(pending_certification))} certified_runs={esc(len(latest_certified_runs))} next_gate=<code>{esc(fast_checks_payload.get('recommended_next_gate') or '')}</code></p>
        </section>
        <section class="panel">
          <h2>P58 Windows Bootstrap</h2>
          <p>doctor: {artifact_link(str(doctor.get('json_path') or ''), 'latest_doctor.json')}</p>
          <p>bootstrap: {artifact_link(str(bootstrap.get('path') or ''), 'latest_bootstrap_state.json')}</p>
          <p>status=<strong>{esc(environment.get('status') or '')}</strong> recommended_mode=<code>{esc(environment.get('recommended_mode') or '')}</code> ready={esc(environment.get('ready_for_continuation'))}</p>
          <p>training_python=<code>{esc(environment.get('selected_training_python') or '')}</code> env=<code>{esc(environment.get('training_env_name') or '')}</code> source=<code>{esc(environment.get('training_env_source') or '')}</code></p>
          <p>next_steps=<code>{esc(' | '.join([str(item) for item in (environment.get('next_steps') or [])[:3]]))}</code></p>
        </section>
        <section class="panel">
          <h2>Dashboard / Paths</h2>
          <p>dashboard: {artifact_link(str(dashboard.get('index_path') or ''), 'latest dashboard')}</p>
          <p>readiness: {artifact_link(str(readiness.get('path') or ''), 'latest readiness')}</p>
          <p>window state: {artifact_link(str(window_state.get('path') or ''), 'window_state.json')}</p>
          <p>background validation: {artifact_link(str(background.get('path') or ''), 'background_mode_validation.json')}</p>
        </section>
        <section class="panel">
          <h2>{esc(learned_router_family)} Learned Router</h2>
          <p class="muted">dataset: {artifact_link(str(learned_router_dataset.get('path') or ''), 'router_dataset_stats.json')}</p>
          <p class="muted">train: {artifact_link(str(learned_router_train.get('path') or ''), 'metrics.json')}</p>
          <p class="muted">promotion: {artifact_link(str(learned_router_ablation.get('promotion_decision_path') or ''), 'promotion_decision.json')}</p>
          <p>dataset_samples={esc(learned_router_dataset_payload.get('sample_count') or 0)} valid={esc(learned_router_dataset_payload.get('valid_for_training_count') or 0)} mean_label_confidence={esc(learned_router_dataset_payload.get('mean_label_confidence') or 0.0)}</p>
          <p>checkpoint_id=<code>{esc(learned_router_train.get('checkpoint_id') or '')}</code> val_top1={esc(learned_router_train_payload.get('val_top1_accuracy') or '')} learner={esc(learned_router_train_payload.get('learner_device') or '')}</p>
          <p>recommendation=<code>{esc(learned_router_promotion.get('recommendation') or '')}</code> guard_trigger_rate={esc(learned_router_guarded.get('guard_trigger_rate') or '')} canary_usage_rate={esc(learned_router_canary.get('canary_usage_rate') or '')}</p>
        </section>
        <section class="panel">
          <h2>P56 Calibration / Guard / Canary</h2>
          <p>calibration: {artifact_link(str(p56_calibration.get('path') or ''), 'calibration_metrics.json')}</p>
          <p>guard_tuning: {artifact_link(str(p56_guard.get('path') or ''), 'guard_tuning_results.json')}</p>
          <p>recommended_guard: {artifact_link(str(p56_guard_cfg.get('path') or ''), 'recommended_guard_config.json')}</p>
          <p>canary_eval: {artifact_link(str(p56_canary.get('path') or ''), 'canary_eval_summary.json')}</p>
          <p>bias=<code>{esc(p56_calibration_payload.get('calibration_bias') or '')}</code> ece={esc(p56_calibration_payload.get('ece') or '')} accuracy={esc(p56_calibration_payload.get('accuracy') or '')}</p>
          <p>deployment_mode=<code>{esc(p56_canary_payload.get('deployment_mode_recommendation') or '')}</code> canary_usage_rate={esc(p56_canary_payload.get('canary_usage_rate') or '')} fallback_rate={esc(p56_canary_payload.get('canary_fallback_rate') or '')}</p>
          <p>guard_config=<code>{esc(json.dumps(p56_guard_cfg_payload.get('guard_config') or {}, ensure_ascii=False))}</code> candidates={esc(len(p56_guard_payload.get('results') or []) if isinstance(p56_guard_payload.get('results'), list) else 0)}</p>
        </section>
        <section class="panel">
          <h2>Config Provenance (P55)</h2>
          <p>source_type=<code>{esc(config_prov.get('config_source_type') or '-')}</code>
             hash=<code>{esc(str(config_prov.get('config_hash') or '')[:16])}</code>
             sidecar_used={esc(str(config_prov.get('sidecar_used') or False))}
             sidecar_in_sync={esc(str(config_prov.get('sidecar_in_sync') or True))}</p>
          <p>sync_status=<strong>{esc(str(config_sync_st.get('overall_status') or '-'))}</strong>
             total={esc(str(config_sync_st.get('total') or '-'))}
             drifted={esc(str(config_sync_st.get('drifted') or 0))}
             missing={esc(str(config_sync_st.get('missing') or 0))}</p>
          <p>report: {artifact_link(str(config_sync_st.get('report_md') or ''), 'sidecar_sync_report.md')}</p>
        </section>
        <section class="panel">
          <h2>P57 Overnight Autonomy</h2>
          <p>attention_queue: {artifact_link(str(attention_queue.get('path') or ''), 'attention_queue.json')}</p>
          <p>morning_summary: {artifact_link(str(morning_summary.get('md_path') or ''), 'latest.md')}</p>
          <p>open_attention={esc(len(attention_open))} blocked_campaigns={esc(len(blocked_campaigns))}</p>
          <p>recommended_first_action=<code>{esc(morning_payload.get('recommended_first_action') or '')}</code></p>
        </section>
        <section class="panel">
          <h2>P60 AGENTS / Autonomy</h2>
          <p>root_agents: {artifact_link(str((agents.get('paths') or {}).get('root') or ''), 'AGENTS.md')}</p>
          <p>autonomy_entry: {artifact_link(str(autonomy_entry.get('json_path') or ''), 'latest_autonomy_entry.json')}</p>
          <p>consistency: {artifact_link(str(agents_consistency.get('json_path') or ''), 'latest_agents_consistency.json')}</p>
          <p>autonomy_state=<strong>{esc(autonomy_payload.get('autonomy_state') or '')}</strong> selected_plan=<code>{esc(autonomy_payload.get('selected_plan') or '')}</code> requested_mode=<code>{esc(autonomy_payload.get('requested_mode') or '')}</code></p>
          <p>root_present={esc(agents.get('root_present'))} subdir_agents={esc(agents.get('subdir_present_count') or 0)} consistency_status=<code>{esc(agents_consistency_payload.get('status') or '')}</code></p>
          <p>decision_policy_path=<code>{esc(autonomy_payload.get('decision_policy_path') or '')}</code></p>
        </section>
        <section class="panel">
          <h2>Latest P22 Summary</h2>
          {table(["Experiment", "Status", "Mean", "Window Mode", "Run"], latest_rows)}
        </section>
        <section class="panel">
          <h2>Campaigns</h2>
          {table(["Campaign", "Experiment", "Stage", "Status", "State"], campaign_rows)}
        </section>
        <section class="panel">
          <h2>Runs / Metrics</h2>
          {table(["Run", "Component", "Status", "Learner", "Rollout", "Throughput", "Warning"], progress_rows)}
        </section>
        """
    )


def render_campaigns(state: dict[str, Any]) -> str:
    rows = []
    for row in state.get("campaigns") or []:
        if not isinstance(row, dict):
            continue
        rows.append(
            [
                esc(row.get("campaign_id") or ""),
                esc(row.get("experiment_id") or ""),
                esc(row.get("seed") or ""),
                esc(row.get("stage_id") or ""),
                esc(row.get("status") or ""),
                esc(row.get("autonomy_decision") or ""),
                esc(row.get("human_gate_triggered") or ""),
                f"<code>{esc(row.get('resume_command') or '')}</code>",
                artifact_link(str(row.get("state_path") or ""), "state"),
            ]
        )
    return f'<section class="panel"><h2>Campaigns</h2>{table(["Campaign", "Experiment", "Seed", "Stage", "Status", "Decision", "Human Gate", "Resume", "State"], rows)}</section>'


def render_registry(state: dict[str, Any]) -> str:
    registry = state.get("registry") if isinstance(state.get("registry"), dict) else {}
    rows = []
    for row in registry.get("entries") or []:
        if not isinstance(row, dict):
            continue
        rows.append(
            [
                esc(row.get("checkpoint_id") or ""),
                esc(row.get("family") or ""),
                esc(row.get("status") or ""),
                esc(row.get("created_at") or ""),
                esc(row.get("source_run_id") or ""),
                artifact_link(str(row.get("metrics_ref") or row.get("artifact_path") or ""), "artifact"),
                artifact_link(str(row.get("calibration_ref") or ""), "calibration"),
                artifact_link(str(row.get("guard_tuning_ref") or ""), "guard"),
                artifact_link(str(row.get("canary_eval_ref") or ""), "canary"),
                esc(row.get("deployment_mode_recommendation") or ""),
            ]
        )
    return (
        f'<section class="panel"><h2>Registry</h2><p class="muted">{artifact_link(str(registry.get("path") or ""), "checkpoints_registry.json")}</p>'
        f'<p class="muted">status_counts={esc(json.dumps(registry.get("counts") or {}, ensure_ascii=False))} family_counts={esc(json.dumps(registry.get("family_counts") or {}, ensure_ascii=False))}</p>'
        f'{table(["Checkpoint", "Family", "Status", "Created", "Source Run", "Artifact", "Calibration", "Guard", "Canary", "Deploy"], rows)}</section>'
    )


def render_promotion_queue(state: dict[str, Any]) -> str:
    queue = state.get("promotion_queue") if isinstance(state.get("promotion_queue"), dict) else {}
    payload = queue.get("payload") if isinstance(queue.get("payload"), dict) else {}
    rows = []
    for row in (payload.get("promotion_review") or [])[:24]:
        if not isinstance(row, dict):
            continue
        rows.append(
            [
                esc(row.get("checkpoint_id") or ""),
                esc(row.get("family") or ""),
                esc(row.get("status") or ""),
                esc(row.get("source_run_id") or ""),
                artifact_link(str(row.get("arena_ref") or row.get("triage_ref") or row.get("metrics_ref") or ""), "ref"),
                esc(row.get("deployment_mode_recommendation") or ""),
            ]
        )
    return (
        f'<section class="panel"><h2>Promotion Queue</h2><p class="muted">{artifact_link(str(queue.get("path") or ""), "promotion_queue.json")}</p>'
        f'<p class="muted">counts={esc(json.dumps(payload.get("counts") or {}, ensure_ascii=False))}</p>'
        f'{table(["Checkpoint", "Family", "Status", "Source Run", "Reference", "Deploy"], rows)}</section>'
    )


def render_router_calibration(state: dict[str, Any]) -> str:
    dashboard = state.get("dashboard") if isinstance(state.get("dashboard"), dict) else {}
    payload = dashboard.get("payload") if isinstance(dashboard.get("payload"), dict) else {}
    p56 = payload.get("p56") if isinstance(payload.get("p56"), dict) else {}
    calibration = p56.get("calibration") if isinstance(p56.get("calibration"), dict) else {}
    calibration_payload = calibration.get("payload") if isinstance(calibration.get("payload"), dict) else {}
    reliability = p56.get("reliability") if isinstance(p56.get("reliability"), dict) else {}
    reliability_payload = reliability.get("payload") if isinstance(reliability.get("payload"), dict) else {}
    rows = []
    for row in (reliability_payload.get("bins") or [])[:12]:
        if not isinstance(row, dict):
            continue
        rows.append([
            esc(row.get("bucket_start") or ""),
            esc(row.get("bucket_end") or ""),
            esc(row.get("count") or ""),
            esc(row.get("accuracy") or ""),
            esc(row.get("avg_confidence") or ""),
        ])
    return (
        f'<section class="panel"><h2>P56 Calibration</h2>'
        f'<p>{artifact_link(str(calibration.get("path") or ""), "calibration_metrics.json")} ? {artifact_link(str(reliability.get("path") or ""), "reliability_bins.json")}</p>'
        f'<p>checkpoint_id=<code>{esc(calibration_payload.get("checkpoint_id") or "")}</code> bias=<code>{esc(calibration_payload.get("calibration_bias") or "")}</code> ece={esc(calibration_payload.get("ece") or "")} accuracy={esc(calibration_payload.get("accuracy") or "")}</p>'
        f'{table(["Start", "End", "Count", "Accuracy", "Confidence"], rows)}</section>'
    )


def render_router_guard_canary(state: dict[str, Any]) -> str:
    dashboard = state.get("dashboard") if isinstance(state.get("dashboard"), dict) else {}
    payload = dashboard.get("payload") if isinstance(dashboard.get("payload"), dict) else {}
    p56 = payload.get("p56") if isinstance(payload.get("p56"), dict) else {}
    guard = p56.get("guard_tuning") if isinstance(p56.get("guard_tuning"), dict) else {}
    guard_payload = guard.get("payload") if isinstance(guard.get("payload"), dict) else {}
    guard_cfg = p56.get("recommended_guard") if isinstance(p56.get("recommended_guard"), dict) else {}
    guard_cfg_payload = guard_cfg.get("payload") if isinstance(guard_cfg.get("payload"), dict) else {}
    canary = p56.get("canary_eval") if isinstance(p56.get("canary_eval"), dict) else {}
    canary_payload = canary.get("payload") if isinstance(canary.get("payload"), dict) else {}
    rows = []
    for row in (canary_payload.get("per_slice_distribution") or [])[:16]:
        if not isinstance(row, dict):
            continue
        rows.append([
            esc(row.get("slice_stage") or ""),
            esc(row.get("slice_resource_pressure") or ""),
            esc(row.get("count") or ""),
            esc(row.get("canary_used_rate") or ""),
            esc(row.get("fallback_rate") or ""),
        ])
    return (
        f'<section class="panel"><h2>P56 Guard / Canary</h2>'
        f'<p>{artifact_link(str(guard.get("path") or ""), "guard_tuning_results.json")} ? {artifact_link(str(guard_cfg.get("path") or ""), "recommended_guard_config.json")} ? {artifact_link(str(canary.get("path") or ""), "canary_eval_summary.json")}</p>'
        f'<p>recommended_guard=<code>{esc(json.dumps(guard_cfg_payload.get("guard_config") or {}, ensure_ascii=False))}</code></p>'
        f'<p>deployment=<code>{esc(canary_payload.get("deployment_mode_recommendation") or "")}</code> canary_usage_rate={esc(canary_payload.get("canary_usage_rate") or "")} fallback_rate={esc(canary_payload.get("canary_fallback_rate") or "")} candidate_count={esc(len(guard_payload.get("results") or []) if isinstance(guard_payload.get("results"), list) else 0)}</p>'
        f'{table(["Stage", "Pressure", "Count", "Used", "Fallback"], rows)}</section>'
    )


def render_runs(state: dict[str, Any]) -> str:
    rows = []
    for row in state.get("progress") or []:
        if not isinstance(row, dict):
            continue
        rows.append(
            [
                esc(row.get("timestamp") or ""),
                esc(row.get("run_id") or ""),
                esc(row.get("component") or ""),
                esc(row.get("phase") or ""),
                esc(row.get("status") or ""),
                esc(row.get("learner_device") or ""),
                esc(row.get("rollout_device") or ""),
                esc(row.get("gpu_mem_mb") or ""),
                esc(row.get("eta_sec") or ""),
                esc(row.get("warning") or ""),
            ]
        )
    return f'<section class="panel"><h2>Runs / Metrics</h2>{table(["Timestamp", "Run", "Component", "Phase", "Status", "Learner", "Rollout", "GPU MB", "ETA", "Warning"], rows)}</section>'


def render_environment(state: dict[str, Any]) -> str:
    environment = state.get("environment") if isinstance(state.get("environment"), dict) else {}
    doctor = state.get("doctor") if isinstance(state.get("doctor"), dict) else {}
    bootstrap = state.get("bootstrap") if isinstance(state.get("bootstrap"), dict) else {}
    doctor_payload = doctor.get("payload") if isinstance(doctor.get("payload"), dict) else {}
    bootstrap_payload = bootstrap.get("payload") if isinstance(bootstrap.get("payload"), dict) else {}
    rows = [
        ["status", esc(environment.get("status") or "")],
        ["recommended_mode", esc(environment.get("recommended_mode") or "")],
        ["ready_for_continuation", esc(environment.get("ready_for_continuation"))],
        ["selected_training_python", f"<code>{esc(environment.get('selected_training_python') or '')}</code>"],
        ["training_env_name", esc(environment.get("training_env_name") or "")],
        ["training_env_source", esc(environment.get("training_env_source") or "")],
        ["doctor_report", artifact_link(str(doctor.get("json_path") or ""), "latest_doctor.json")],
        ["doctor_report_md", artifact_link(str(doctor.get("md_path") or ""), "latest_doctor.md")],
        ["bootstrap_state", artifact_link(str(bootstrap.get("path") or ""), "latest_bootstrap_state.json")],
        ["bootstrap_complete", esc(bootstrap_payload.get("bootstrap_complete"))],
    ]
    blockers = "".join([f"<li>{esc(item)}</li>" for item in (environment.get("blocking_reasons") or [])[:12]]) or "<li>No blocking reasons.</li>"
    warnings = "".join([f"<li>{esc(item)}</li>" for item in (environment.get("warnings") or [])[:12]]) or "<li>No warnings.</li>"
    next_steps = "".join([f"<li><code>{esc(item)}</code></li>" for item in (environment.get("next_steps") or [])[:12]]) or "<li>No next steps.</li>"
    return (
        f'<section class="panel"><h2>Environment Summary</h2>{table(["Field", "Value"], rows)}</section>'
        f'<section class="panel"><h2>Doctor Blocking Reasons</h2><ul>{blockers}</ul></section>'
        f'<section class="panel"><h2>Doctor Warnings</h2><ul>{warnings}</ul></section>'
        f'<section class="panel"><h2>Recommended Next Steps</h2><ul>{next_steps}</ul></section>'
        f'<section class="panel"><h2>Doctor Payload</h2><pre>{esc(json.dumps(doctor_payload, ensure_ascii=False, indent=2))}</pre></section>'
    )


def render_autonomy(state: dict[str, Any]) -> str:
    agents = state.get("agents") if isinstance(state.get("agents"), dict) else {}
    autonomy = state.get("autonomy_entry") if isinstance(state.get("autonomy_entry"), dict) else {}
    autonomy_payload = autonomy.get("payload") if isinstance(autonomy.get("payload"), dict) else {}
    consistency = state.get("agents_consistency") if isinstance(state.get("agents_consistency"), dict) else {}
    consistency_payload = consistency.get("payload") if isinstance(consistency.get("payload"), dict) else {}
    attention = state.get("attention_queue") if isinstance(state.get("attention_queue"), dict) else {}
    morning = state.get("morning_summary") if isinstance(state.get("morning_summary"), dict) else {}
    blocked = state.get("blocked_campaigns") if isinstance(state.get("blocked_campaigns"), list) else []
    fast_checks = state.get("fast_checks") if isinstance(state.get("fast_checks"), dict) else {}
    fast_checks_payload = fast_checks.get("payload") if isinstance(fast_checks.get("payload"), dict) else {}
    certification_queue = state.get("certification_queue") if isinstance(state.get("certification_queue"), dict) else {}
    certification_state = certification_queue.get("state_payload") if isinstance(certification_queue.get("state_payload"), dict) else {}
    pending_certification = certification_queue.get("pending_items") if isinstance(certification_queue.get("pending_items"), list) else []
    latest_certified_runs = certification_queue.get("latest_certified_runs") if isinstance(certification_queue.get("latest_certified_runs"), list) else []

    agent_rows = []
    for name, path in (agents.get("paths") or {}).items():
        agent_rows.append([esc(name), esc((agents.get("present") or {}).get(name)), artifact_link(str(path), "open")])

    attention_rows = []
    for item in (attention.get("open_items") or [])[:12]:
        if not isinstance(item, dict):
            continue
        attention_rows.append(
            [
                esc(item.get("attention_id") or ""),
                esc(item.get("severity") or ""),
                esc(item.get("category") or ""),
                esc(item.get("blocking_scope") or ""),
                esc(item.get("title") or ""),
            ]
        )

    blocked_rows = []
    for row in blocked[:12]:
        if not isinstance(row, dict):
            continue
        blocked_rows.append(
            [
                esc(row.get("campaign_id") or ""),
                esc(row.get("stage_id") or ""),
                esc(row.get("status") or ""),
                esc(row.get("autonomy_decision") or ""),
                artifact_link(str(row.get("state_path") or ""), "state"),
            ]
        )

    return (
        f'<section class="panel"><h2>Autonomy Overview</h2>'
        f'<p>autonomy_entry: {artifact_link(str(autonomy.get("json_path") or ""), "latest_autonomy_entry.json")}</p>'
        f'<p>agents_consistency: {artifact_link(str(consistency.get("json_path") or ""), "latest_agents_consistency.json")}</p>'
        f'<p>morning_summary: {artifact_link(str(morning.get("md_path") or ""), "latest.md")}</p>'
        f'<p>attention_queue: {artifact_link(str(attention.get("path") or ""), "attention_queue.json")}</p>'
        f'<p>fast_check_report: {artifact_link(str(fast_checks.get("json_path") or ""), "latest_fast_check_report.json")}</p>'
        f'<p>certification_queue: {artifact_link(str(certification_queue.get("queue_path") or ""), "certification_queue.json")} | {artifact_link(str(certification_queue.get("state_path") or ""), "certification_state.json")}</p>'
        f'<p>autonomy_state=<strong>{esc(autonomy_payload.get("autonomy_state") or "")}</strong> selected_plan=<code>{esc(autonomy_payload.get("selected_plan") or "")}</code> requested_mode=<code>{esc(autonomy_payload.get("requested_mode") or "")}</code></p>'
        f'<p>reason=<code>{esc(autonomy_payload.get("reason") or "")}</code></p>'
        f'<p>consistency_status=<code>{esc(consistency_payload.get("status") or "")}</code> warnings={esc(len(consistency_payload.get("warnings") or []))} errors={esc(len(consistency_payload.get("errors") or []))}</p>'
        f'<p>fast_check_status=<code>{esc(fast_checks_payload.get("fast_check_status") or "")}</code> validation_tiers=<code>{esc(",".join([str(item) for item in (fast_checks_payload.get("validation_tiers_completed") or [])]))}</code></p>'
        f'<p>certification_status=<code>{esc(certification_state.get("status") or "")}</code> pending_certification={esc(len(pending_certification))} latest_certified_runs={esc(len(latest_certified_runs))}</p>'
        f'<p>decision_policy_path=<code>{esc(autonomy_payload.get("decision_policy_path") or "")}</code></p></section>'
        f'<section class="panel"><h2>AGENTS Hierarchy</h2>{table(["Scope", "Present", "Path"], agent_rows)}</section>'
        f'<section class="panel"><h2>Open Attention Items</h2>{table(["Attention ID", "Severity", "Category", "Scope", "Title"], attention_rows)}</section>'
        f'<section class="panel"><h2>Blocked Campaigns</h2>{table(["Campaign", "Stage", "Status", "Decision", "State"], blocked_rows)}</section>'
    )


def render_windows(state: dict[str, Any], *, current_path: str) -> str:
    window_state = state.get("window_state") if isinstance(state.get("window_state"), dict) else {}
    payload = window_state.get("payload") if isinstance(window_state.get("payload"), dict) else {}
    background = state.get("background_validation") if isinstance(state.get("background_validation"), dict) else {}
    background_payload = background.get("payload") if isinstance(background.get("payload"), dict) else {}
    rows = []
    for row in (payload.get("window_mode_after") or payload.get("windows") or []):
        if not isinstance(row, dict):
            continue
        rows.append(
            [
                esc(row.get("role") or ""),
                esc(row.get("title") or ""),
                esc(row.get("class_name") or ""),
                esc(row.get("mode") or ""),
                esc(row.get("visible") or ""),
                esc(row.get("pid") or ""),
            ]
        )
    validation_rows = []
    for row in (background_payload.get("mode_results") or []):
        if not isinstance(row, dict):
            continue
        validation_rows.append(
            [
                esc(row.get("requested_mode") or ""),
                esc(row.get("effective_mode") or ""),
                esc(row.get("status") or ""),
                esc(row.get("smoke_passed") or ""),
                esc(row.get("matches_visible_baseline") or ""),
                artifact_link(str(row.get("report_path") or ""), "report"),
            ]
        )
    buttons = []
    for mode in ("visible", "offscreen", "minimized", "hidden", "restore"):
        buttons.append(
            f'<form method="post" action="/actions/window_mode"><input type="hidden" name="return_to" value="{esc(current_path)}"><input type="hidden" name="mode" value="{mode}"><button type="submit">Set {mode}</button></form>'
        )
    return (
        f'<section class="panel"><h2>Background Execution / Windows</h2><p>dominant_mode=<strong>{esc(window_state.get("dominant_mode") or "")}</strong></p>'
        f'<p class="muted">{artifact_link(str(window_state.get("path") or ""), "window_state.json")} | {artifact_link(str(background.get("path") or ""), "background_mode_validation.json")}</p>'
        f'<div class="actions">{"".join(buttons)}</div>'
        f'{table(["Role", "Title", "Class", "Mode", "Visible", "PID"], rows)}'
        f'<h3>Validation</h3>{table(["Requested", "Effective", "Status", "Smoke Passed", "Matches Visible", "Report"], validation_rows)}</section>'
    )


def render_jobs(state: dict[str, Any]) -> str:
    jobs = []
    for row in state.get("jobs") or []:
        if not isinstance(row, dict):
            continue
        jobs.append(
            [
                esc(row.get("job_id") or ""),
                esc(row.get("action") or ""),
                esc(row.get("status") or ""),
                esc(row.get("pid") or ""),
                artifact_link(str(row.get("log_path") or ""), "log"),
                esc(row.get("command") or ""),
            ]
        )
    audits = []
    for row in state.get("audit_tail") or []:
        if not isinstance(row, dict):
            continue
        audits.append(
            [
                esc(row.get("timestamp") or ""),
                esc(row.get("action") or ""),
                esc(row.get("success") or ""),
                esc(row.get("target") or ""),
                artifact_link(str(row.get("output_ref") or ""), "output"),
            ]
        )
    return (
        f'<section class="panel"><h2>Ops Jobs</h2>{table(["Job", "Action", "Status", "PID", "Log", "Command"], jobs)}</section>'
        f'<section class="panel"><h2>Audit Trail</h2>{table(["Timestamp", "Action", "Success", "Target", "Output"], audits)}</section>'
    )


def render_attention_queue(state: dict[str, Any]) -> str:
    queue = state.get("attention_queue") if isinstance(state.get("attention_queue"), dict) else {}
    payload = queue.get("payload") if isinstance(queue.get("payload"), dict) else {}
    rows = []
    for row in payload.get("items") or []:
        if not isinstance(row, dict):
            continue
        resolve_form = ""
        if str(row.get("status") or "") == "open":
            resolve_form = (
                '<form method="post" action="/actions/resolve_attention">'
                '<input type="hidden" name="return_to" value="/attention-queue">'
                f'<input type="hidden" name="attention_id" value="{esc(row.get("attention_id") or "")}">'
                '<input type="hidden" name="resolution_note" value="resolved in ops ui">'
                '<button type="submit">Resolve</button>'
                "</form>"
            )
        rows.append(
            [
                esc(row.get("attention_id") or ""),
                esc(row.get("severity") or ""),
                esc(row.get("category") or ""),
                esc(row.get("blocking_scope") or ""),
                esc(row.get("status") or ""),
                esc(row.get("title") or ""),
                artifact_link(str(row.get("item_md_path") or ""), "item"),
                resolve_form,
            ]
        )
    return (
        f'<section class="panel"><h2>Attention Queue</h2><p class="muted">{artifact_link(str(queue.get("path") or ""), "attention_queue.json")}</p>'
        f'{table(["Attention ID", "Severity", "Category", "Scope", "Status", "Title", "Artifact", "Action"], rows)}</section>'
    )


def render_morning_summary(state: dict[str, Any]) -> str:
    summary = state.get("morning_summary") if isinstance(state.get("morning_summary"), dict) else {}
    payload = summary.get("payload") if isinstance(summary.get("payload"), dict) else {}
    blocked = state.get("blocked_campaigns") if isinstance(state.get("blocked_campaigns"), list) else []
    blocked_rows = []
    for row in blocked[:16]:
        if not isinstance(row, dict):
            continue
        blocked_rows.append(
            [
                esc(row.get("campaign_id") or ""),
                esc(row.get("experiment_id") or ""),
                esc(row.get("stage_id") or ""),
                esc(row.get("status") or ""),
                artifact_link(str(row.get("state_path") or ""), "state"),
            ]
        )
    return (
        f'<section class="panel"><h2>Morning Summary</h2><p>{artifact_link(str(summary.get("md_path") or ""), "latest.md")} | {artifact_link(str(summary.get("json_path") or ""), "latest.json")}</p>'
        f'<p>task_summary=<code>{esc(json.dumps(payload.get("task_summary") or {}, ensure_ascii=False))}</code></p>'
        f'<p>recommended_first_action=<code>{esc(payload.get("recommended_first_action") or "")}</code></p>'
        f'<pre>{esc(summary.get("excerpt") or "")}</pre></section>'
        f'<section class="panel"><h2>Blocked Campaigns</h2>{table(["Campaign", "Experiment", "Stage", "Status", "State"], blocked_rows)}</section>'
    )


def render_blocked_campaigns(state: dict[str, Any]) -> str:
    blocked = state.get("blocked_campaigns") if isinstance(state.get("blocked_campaigns"), list) else []
    rows = []
    for row in blocked[:32]:
        if not isinstance(row, dict):
            continue
        rows.append(
            [
                esc(row.get("campaign_id") or ""),
                esc(row.get("experiment_id") or ""),
                esc(row.get("seed") or ""),
                esc(row.get("stage_id") or ""),
                esc(row.get("status") or ""),
                esc(row.get("autonomy_decision") or ""),
                artifact_link(str(row.get("attention_item_ref") or ""), "attention"),
                artifact_link(str(row.get("state_path") or ""), "state"),
            ]
        )
    return f'<section class="panel"><h2>Blocked Campaigns</h2>{table(["Campaign", "Experiment", "Seed", "Stage", "Status", "Decision", "Attention", "State"], rows)}</section>'
