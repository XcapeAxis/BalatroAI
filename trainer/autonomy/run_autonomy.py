from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trainer.autonomy.agents_consistency_check import write_consistency_report
from trainer.autonomy.attention_queue import open_attention_item
from trainer.autonomy.decision_policy import evaluate_autonomy, load_decision_policy
from trainer.autonomy.morning_summary import write_morning_summary
from trainer.ops_ui.state_loader import build_ops_state, repo_root as ops_repo_root

AUTONOMY_MILESTONE = "P60"
AUTONOMY_ARTIFACT_FAMILY = "p60"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_repo_root() -> Path:
    preferred = Path("D:/MYFILES/BalatroAI")
    if preferred.exists():
        return preferred.resolve()
    return ops_repo_root()


def _out_root(repo_root: Path) -> Path:
    return (repo_root / "docs" / "artifacts" / AUTONOMY_ARTIFACT_FAMILY).resolve()


def _render_md(payload: dict[str, Any]) -> str:
    lines = [
        f"# {AUTONOMY_MILESTONE} Autonomy Entry",
        "",
        f"- generated_at: `{payload.get('generated_at')}`",
        f"- repo_root: `{payload.get('repo_root')}`",
        f"- requested_mode: `{payload.get('requested_mode')}`",
        f"- autonomy_state: `{payload.get('autonomy_state')}`",
        f"- selected_plan: `{payload.get('selected_plan')}`",
        f"- continue_allowed: `{payload.get('continue_allowed')}`",
        f"- reason: {payload.get('reason') or ''}",
        f"- decision_policy_path: `{payload.get('decision_policy_path') or ''}`",
        f"- attention_queue_path: `{payload.get('attention_queue_path') or ''}`",
        f"- morning_summary_path: `{payload.get('morning_summary_path') or ''}`",
        f"- dashboard_path: `{payload.get('dashboard_path') or ''}`",
        f"- autonomy_entry_ref: `{payload.get('latest_json') or payload.get('json_path') or ''}`",
        "",
        "## AGENTS",
        "",
        "- root_present=`{}` subdir_present=`{}`".format(
            (payload.get("agents") or {}).get("root_present"),
            (payload.get("agents") or {}).get("subdir_present_count"),
        ),
        "",
        "## Selected Command",
        "",
    ]
    command_argv = payload.get("selected_command_argv") or []
    if command_argv:
        lines.append("- `{}`".format(" ".join([str(item) for item in command_argv])))
    else:
        lines.append("- No command selected.")
    lines += ["", "## Attention Items", ""]
    for item in payload.get("open_attention_items") or []:
        if not isinstance(item, dict):
            continue
        lines.append(
            "- `{}` severity=`{}` category=`{}` title={}".format(
                item.get("attention_id") or "",
                item.get("severity") or "",
                item.get("category") or "",
                item.get("title") or "",
            )
        )
    if not (payload.get("open_attention_items") or []):
        lines.append("- None.")
    execution = payload.get("execution") if isinstance(payload.get("execution"), dict) else {}
    if execution:
        lines += [
            "",
            "## Execution",
            "",
            f"- executed: `{execution.get('executed')}`",
            f"- status: `{execution.get('status') or ''}`",
            f"- safe_run_summary_path: `{execution.get('safe_run_summary_path') or ''}`",
        ]
    return "\n".join(lines).rstrip() + "\n"


def _write_payload(out_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = out_root / f"autonomy_entry_{stamp}.json"
    md_path = out_root / f"autonomy_entry_{stamp}.md"
    latest_json = out_root / "latest_autonomy_entry.json"
    latest_md = out_root / "latest_autonomy_entry.md"
    out_root.mkdir(parents=True, exist_ok=True)
    payload["json_path"] = str(json_path.resolve())
    payload["md_path"] = str(md_path.resolve())
    payload["latest_json"] = str(latest_json.resolve())
    payload["latest_md"] = str(latest_md.resolve())
    text = _render_md(payload)
    json_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    json_path.write_text(json_text, encoding="utf-8")
    md_path.write_text(text, encoding="utf-8")
    latest_json.write_text(json_text, encoding="utf-8")
    latest_md.write_text(text, encoding="utf-8")
    return payload


def _command_for_mode(mode: str, state: dict[str, Any]) -> tuple[str, list[str], str]:
    attention_queue = state.get("attention_queue") if isinstance(state.get("attention_queue"), dict) else {}
    open_items = attention_queue.get("open_items") if isinstance(attention_queue.get("open_items"), list) else []
    blocking_items = [
        item
        for item in open_items
        if isinstance(item, dict)
        and str(item.get("severity") or "").strip().lower() == "block"
        and str(item.get("blocking_scope") or "").strip().lower() not in {"validation_smoke", "validation_only"}
    ]
    blocked_campaigns = state.get("blocked_campaigns") if isinstance(state.get("blocked_campaigns"), list) else []
    resume_target = state.get("latest_resume_target") if isinstance(state.get("latest_resume_target"), dict) else {}

    if blocking_items:
        return "blocked_by_attention_queue", [], "Open blocking attention items require a human decision before autonomy may continue."
    if any(bool(item.get("human_gate_open")) for item in blocked_campaigns if isinstance(item, dict)):
        return "blocked_by_campaign_human_gate", [], "A campaign is blocked by an unresolved human gate."
    if mode == "resume-latest":
        resume_command = str(resume_target.get("resume_command") or "").strip()
        if resume_command:
            return (
                "resume_latest_campaign",
                [
                    "powershell",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    "scripts\\run_p22.ps1",
                    "-ResumeLatestCampaign",
                ],
                "A resumable campaign exists and there are no blocking attention items.",
            )
        return "next_mainline_task", [], "No resumable campaign was found. The next mainline task is a new autonomy smoke run."
    if mode == "overnight":
        return (
            "start_overnight",
            [
                "powershell",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                "scripts\\run_p22.ps1",
                "-Overnight",
            ],
            "No blocking attention items were found, so the overnight autonomy lane may continue.",
        )
    if str(resume_target.get("status") or "").strip().lower() in {"failed", "running"} and str(resume_target.get("resume_command") or "").strip():
        return (
            "resume_latest_campaign",
            [
                "powershell",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                "scripts\\run_p22.ps1",
                "-ResumeLatestCampaign",
            ],
            "A resumable campaign exists, so autonomy will resume it before starting a new smoke run.",
        )
    return (
        "start_autonomy_smoke",
        [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            "scripts\\run_p22.ps1",
            "-RunP57",
        ],
        "No blockers or resumable failures were found, so autonomy will start the smoke lane.",
    )


def build_autonomy_entry(*, repo_root: str | Path | None = None, mode: str = "quick", execute: bool = False, timeout_sec: int = 0) -> dict[str, Any]:
    root = Path(repo_root).resolve() if repo_root else resolve_repo_root()
    out_root = _out_root(root)
    policy = load_decision_policy()
    state = build_ops_state()
    consistency = write_consistency_report(repo_root=root, out_root=out_root)
    morning = write_morning_summary(repo=root)

    attention_queue = state.get("attention_queue") if isinstance(state.get("attention_queue"), dict) else {}
    open_attention_items = [item for item in (attention_queue.get("open_items") or []) if isinstance(item, dict)]
    agents = {
        "root_path": str((root / "AGENTS.md").resolve()),
        "root_present": (root / "AGENTS.md").exists(),
        "subdir_paths": {name: str((root / name / "AGENTS.md").resolve()) for name in ("trainer", "sim", "scripts", "docs", "configs")},
        "subdir_present_count": sum(1 for name in ("trainer", "sim", "scripts", "docs", "configs") if (root / name / "AGENTS.md").exists()),
    }

    selected_plan, command_argv, reason = _command_for_mode(mode, state)
    policy_eval = evaluate_autonomy(
        policy=policy,
        actions=["run_autonomy_entry", "resume_campaign" if "resume" in selected_plan else "build_morning_summary"],
        conditions=["unresolved_human_gate"] if selected_plan.startswith("blocked") else [],
        summary=reason,
    )

    attention_item_ref = ""
    if consistency.get("status") == "error":
        created = open_attention_item(
            severity="block",
            category="ambiguity",
            title="AGENTS / decision-policy consistency check failed",
            summary="The autonomy entry found missing or conflicting AGENTS/decision-policy files.",
            summary_for_human="Autonomy cannot continue because the repository rule layer is incomplete or inconsistent. Fix the AGENTS and decision-policy files first.",
            blocking_stage="agents_consistency_check",
            blocking_scope="repo_rules",
            attempted_actions=["run_agents_consistency_check"],
            recommended_options=[
                {"label": "repair_agents", "description": "Restore the missing AGENTS or decision-policy files."},
                {"label": "review_doc_links", "description": "Make README and rule docs point to the same entrypoints."},
            ],
            recommended_default="repair_agents",
            required_human_input=["Confirm the AGENTS hierarchy after repairing the missing files."],
            artifact_refs=[str(consistency.get("json_path") or ""), str(consistency.get("md_path") or "")],
            suggested_commands=[
                "python -m trainer.autonomy.agents_consistency_check",
                "powershell -ExecutionPolicy Bypass -File scripts\\run_autonomy.ps1 -DryRun",
            ],
            decision_deadline_hint="before next autonomy run",
            dedupe_key="p60-agents-consistency-error",
            root=root / "docs" / "artifacts" / "attention_required",
        )
        attention_item_ref = str(created.get("item_md_path") or created.get("attention_file") or "")
        selected_plan = "blocked_by_agents_consistency"
        command_argv = []
        reason = "AGENTS or decision-policy consistency check failed."
        policy_eval = evaluate_autonomy(policy=policy, actions=["run_autonomy_entry"], conditions=["agents_consistency_error"], summary=reason)

    autonomy_state = "idle"
    human_gate_triggered = bool(selected_plan.startswith("blocked") or not policy_eval.get("continue_allowed", True))
    if human_gate_triggered:
        autonomy_state = "blocked"
    elif selected_plan == "resume_latest_campaign":
        autonomy_state = "resume-ready"
    elif execute and command_argv:
        autonomy_state = "running"

    payload: dict[str, Any] = {
        "schema": "p60_autonomy_entry_v1",
        "generated_at": _now_iso(),
        "repo_root": str(root),
        "requested_mode": mode,
        "autonomy_state": autonomy_state,
        "selected_plan": selected_plan,
        "reason": reason,
        "continue_allowed": not human_gate_triggered,
        "human_gate_triggered": human_gate_triggered,
        "selected_command_argv": command_argv,
        "decision_policy_path": str(policy.get("source_path") or ""),
        "attention_queue_path": str(attention_queue.get("path") or ""),
        "morning_summary_path": str(morning.get("latest_md") or ""),
        "dashboard_path": str(((state.get("dashboard") or {}).get("index_path")) or ""),
        "latest_resume_target": state.get("latest_resume_target") if isinstance(state.get("latest_resume_target"), dict) else {},
        "open_attention_items": open_attention_items[:12],
        "blocked_campaigns": state.get("blocked_campaigns") if isinstance(state.get("blocked_campaigns"), list) else [],
        "agents": agents,
        "agents_consistency": {
            "status": consistency.get("status"),
            "json_path": consistency.get("json_path"),
            "md_path": consistency.get("md_path"),
            "errors": consistency.get("errors") or [],
            "warnings": consistency.get("warnings") or [],
        },
        "policy_evaluation": policy_eval,
        "attention_item_ref": attention_item_ref,
        "next_suggested_command": (
            "powershell -ExecutionPolicy Bypass -File scripts\\run_p22.ps1 -RunP57"
            if selected_plan == "next_mainline_task"
            else ""
        ),
        "execution": {
            "executed": False,
            "status": "not_requested",
            "safe_run_summary_path": "",
        },
    }
    payload = _write_payload(out_root, payload)

    if execute and command_argv and not human_gate_triggered:
        safe_run_summary = out_root / f"safe_run_autonomy_{mode}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        env = os.environ.copy()
        env["BALATRO_AUTONOMY_MODE"] = str(mode)
        env["BALATRO_DECISION_POLICY_PATH"] = str(payload.get("decision_policy_path") or "")
        env["BALATRO_ATTENTION_QUEUE_PATH"] = str(payload.get("attention_queue_path") or "")
        env["BALATRO_MORNING_SUMMARY_PATH"] = str(payload.get("morning_summary_path") or "")
        env["BALATRO_AUTONOMY_ENTRY_REF"] = str(payload.get("latest_json") or payload.get("json_path") or "")
        env["BALATRO_AGENTS_ROOT_PRESENT"] = "true" if agents.get("root_present") else "false"
        safe_run_script = root / "scripts" / "safe_run.ps1"
        timeout_value = int(timeout_sec) if int(timeout_sec or 0) > 0 else (14400 if mode == "overnight" else 7200)
        cmd = [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(safe_run_script),
            "-TimeoutSec",
            str(timeout_value),
            "-SummaryJson",
            str(safe_run_summary.resolve()),
            "--",
            *command_argv,
        ]
        proc = subprocess.run(cmd, cwd=str(root), env=env, check=False)
        safe_payload: dict[str, Any] = {}
        if safe_run_summary.exists():
            try:
                safe_payload = json.loads(safe_run_summary.read_text(encoding="utf-8-sig"))
            except Exception:
                safe_payload = {}
        payload["execution"] = {
            "executed": True,
            "status": "passed" if int(proc.returncode or 0) == 0 else "failed",
            "exit_code": int(proc.returncode or 0),
            "safe_run_summary_path": str(safe_run_summary.resolve()),
            "safe_run_status": str(safe_payload.get("status") or ""),
            "completed_at": _now_iso(),
        }
        payload["autonomy_state"] = "idle"
        payload = _write_payload(out_root, payload)
        morning = write_morning_summary(repo=root)
        payload["morning_summary_path"] = str(morning.get("latest_md") or "")
        payload = _write_payload(out_root, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="P60 autonomy entry")
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--mode", choices=("quick", "overnight", "resume-latest"), default="quick")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=0)
    args = parser.parse_args()
    payload = build_autonomy_entry(repo_root=args.repo_root or None, mode=args.mode, execute=args.execute, timeout_sec=args.timeout_sec)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
