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
    campaigns = state.get("campaigns") if isinstance(state.get("campaigns"), list) else []
    progress = state.get("progress") if isinstance(state.get("progress"), list) else []
    registry = state.get("registry") if isinstance(state.get("registry"), dict) else {}
    resume_target = state.get("latest_resume_target") if isinstance(state.get("latest_resume_target"), dict) else {}
    ops_ui_state = state.get("ops_ui") if isinstance(state.get("ops_ui"), dict) else {}
    ops_ui_payload = ops_ui_state.get("payload") if isinstance(ops_ui_state.get("payload"), dict) else {}

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
          <h2>Dashboard / Paths</h2>
          <p>dashboard: {artifact_link(str(dashboard.get('index_path') or ''), 'latest dashboard')}</p>
          <p>readiness: {artifact_link(str(readiness.get('path') or ''), 'latest readiness')}</p>
          <p>window state: {artifact_link(str(window_state.get('path') or ''), 'window_state.json')}</p>
          <p>background validation: {artifact_link(str(background.get('path') or ''), 'background_mode_validation.json')}</p>
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
                f"<code>{esc(row.get('resume_command') or '')}</code>",
                artifact_link(str(row.get("state_path") or ""), "state"),
            ]
        )
    return f'<section class="panel"><h2>Campaigns</h2>{table(["Campaign", "Experiment", "Seed", "Stage", "Status", "Resume", "State"], rows)}</section>'


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
            ]
        )
    return (
        f'<section class="panel"><h2>Registry</h2><p class="muted">{artifact_link(str(registry.get("path") or ""), "checkpoints_registry.json")}</p>'
        f'<p class="muted">status_counts={esc(json.dumps(registry.get("counts") or {}, ensure_ascii=False))} family_counts={esc(json.dumps(registry.get("family_counts") or {}, ensure_ascii=False))}</p>'
        f'{table(["Checkpoint", "Family", "Status", "Created", "Source Run", "Artifact"], rows)}</section>'
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
            ]
        )
    return (
        f'<section class="panel"><h2>Promotion Queue</h2><p class="muted">{artifact_link(str(queue.get("path") or ""), "promotion_queue.json")}</p>'
        f'<p class="muted">counts={esc(json.dumps(payload.get("counts") or {}, ensure_ascii=False))}</p>'
        f'{table(["Checkpoint", "Family", "Status", "Source Run", "Reference"], rows)}</section>'
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
