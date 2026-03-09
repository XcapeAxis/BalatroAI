from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trainer.autonomy.certification_queue import queue_json_path, upsert_from_fast_check
from trainer.runtime.change_scope import resolve_repo_root
from trainer.runtime.validation_planner import build_validation_plan, write_validation_plan_artifacts


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _tail(text: str, limit: int = 4000) -> str:
    return str(text or "")[-max(1, int(limit)) :]


def _run_capture(command: list[str], *, cwd: Path, timeout_sec: int, env: dict[str, str] | None = None) -> dict[str, Any]:
    start = time.time()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            env=env,
            check=False,
        )
        return {
            "status": "passed" if int(proc.returncode or 0) == 0 else "failed",
            "returncode": int(proc.returncode or 0),
            "stdout": str(proc.stdout or ""),
            "stderr": str(proc.stderr or ""),
            "elapsed_sec": time.time() - start,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "failed",
            "returncode": 124,
            "stdout": str(exc.stdout or ""),
            "stderr": str(exc.stderr or ""),
            "elapsed_sec": time.time() - start,
            "timed_out": True,
        }


def _run_safe(command_argv: list[str], *, cwd: Path, timeout_sec: int, summary_path: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    safe_run_script = cwd / "scripts" / "safe_run.ps1"
    cmd = [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(safe_run_script),
        "-TimeoutSec",
        str(timeout_sec),
        "-SummaryJson",
        str(summary_path.resolve()),
        "--",
        *command_argv,
    ]
    start = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, check=False)
    payload = _read_json(summary_path) or {}
    return {
        "status": "passed" if int(proc.returncode or 0) == 0 else "failed",
        "returncode": int(proc.returncode or 0),
        "elapsed_sec": float(payload.get("duration_sec") or (time.time() - start)),
        "timed_out": bool(payload.get("timed_out")),
        "safe_run_summary_path": str(summary_path.resolve()),
    }


def _check_env(repo_root: Path, *, planned_tiers: list[str], recommended_next_gate: str) -> dict[str, str]:
    return {
        **os.environ,
        "BALATRO_CERTIFICATION_QUEUE_REF": str(queue_json_path(repo_root).resolve()),
        "BALATRO_RECOMMENDED_NEXT_GATE": recommended_next_gate,
        "BALATRO_PLANNED_VALIDATION_TIERS": ",".join(planned_tiers),
    }


def _check_command(check_id: str) -> list[str]:
    mapping = {
        "readme_lint": ["powershell", "-ExecutionPolicy", "Bypass", "-File", "scripts\\lint_readme_p25.ps1"],
        "p22_dry_run": ["powershell", "-ExecutionPolicy", "Bypass", "-File", "scripts\\run_p22.ps1", "-DryRun"],
        "router_smoke": ["powershell", "-ExecutionPolicy", "Bypass", "-File", "scripts\\run_p22.ps1", "-Only", "p54_learned_router_smoke"],
        "world_model_smoke": ["powershell", "-ExecutionPolicy", "Bypass", "-File", "scripts\\run_p22.ps1", "-Only", "p47_wm_search_smoke"],
        "imagination_smoke": ["powershell", "-ExecutionPolicy", "Bypass", "-File", "scripts\\run_p22.ps1", "-Only", "p46_imagination_smoke"],
        "rl_smoke": ["powershell", "-ExecutionPolicy", "Bypass", "-File", "scripts\\run_p22.ps1", "-Only", "p42_rl_candidate_smoke"],
        "closed_loop_smoke": ["powershell", "-ExecutionPolicy", "Bypass", "-File", "scripts\\run_p22.ps1", "-Only", "p40_closed_loop_smoke"],
        "p22_quick_gate": ["powershell", "-ExecutionPolicy", "Bypass", "-File", "scripts\\run_p22.ps1", "-Quick"],
        "sim_subsystem_gate": ["powershell", "-ExecutionPolicy", "Bypass", "-File", "scripts\\run_regressions.ps1", "-RunP22"],
    }
    return list(mapping.get(check_id) or [])


def execute_fast_checks(
    *,
    repo_root: str | Path | None = None,
    changed_files: list[str] | None = None,
    gate_plan_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    root = resolve_repo_root(repo_root)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = root / "docs" / "artifacts" / "p61"
    run_dir = out_root / "fast_checks" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    plan = build_validation_plan(repo_root=root, changed_files=changed_files, gate_plan_path=gate_plan_path)
    plan_paths = write_validation_plan_artifacts(plan, out_root=out_root, stamp=stamp)
    gate_snapshot_path = out_root / f"gate_plan_snapshot_{stamp}.json"
    gate_snapshot_path.write_text(
        json.dumps(
            {
                "schema": "p61_gate_plan_snapshot_v1",
                "generated_at": _now_iso(),
                "gate_plan_path": str(plan.get("gate_plan_path") or ""),
                "selected_checks": plan.get("selected_checks") or [],
                "deferred_certification": plan.get("deferred_certification") or [],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    selected_checks = [item for item in (plan.get("selected_checks") or []) if isinstance(item, dict)]
    completed_tiers: list[str] = []
    report_rows: list[dict[str, Any]] = []
    overall_status = "passed"
    recommended_next_gate = str(((plan.get("deferred_certification") or [{}])[0] or {}).get("command_template") or "")
    env_base = _check_env(root, planned_tiers=list(plan.get("validation_tiers_required") or []), recommended_next_gate=recommended_next_gate)

    if not dry_run:
        for item in selected_checks:
            check_id = str(item.get("check_id") or "")
            tier = str(item.get("tier") or "")
            timeout_sec = int(item.get("timeout_sec") or 600)
            blocking_behavior = str(item.get("blocking_behavior") or "must-pass")
            row: dict[str, Any] = {
                "check_id": check_id,
                "tier": tier,
                "blocking_behavior": blocking_behavior,
                "reasons": list(item.get("reasons") or []),
                "command": str(item.get("command_template") or ""),
            }
            if tier not in completed_tiers:
                completed_tiers.append(tier)
            if check_id == "py_compile_changed_python":
                files = [str(root / path) for path in (plan.get("change_scope") or {}).get("changed_python_files") or []]
                if files:
                    result = _run_capture([sys.executable, "-m", "py_compile", *files], cwd=root, timeout_sec=timeout_sec, env=env_base)
                else:
                    result = {"status": "skipped", "returncode": 0, "stdout": "", "stderr": "", "elapsed_sec": 0.0}
                row.update({"status": result.get("status"), "returncode": result.get("returncode"), "elapsed_sec": result.get("elapsed_sec"), "stdout_tail": _tail(str(result.get("stdout") or "")), "stderr_tail": _tail(str(result.get("stderr") or ""))})
            elif check_id == "config_sidecar_consistency":
                result = _run_capture([sys.executable, "-B", "-m", "trainer.experiments.config_sidecar_sync", "--check", "--quiet"], cwd=root, timeout_sec=timeout_sec, env=env_base)
                row.update({"status": result.get("status"), "returncode": result.get("returncode"), "elapsed_sec": result.get("elapsed_sec"), "stdout_tail": _tail(str(result.get("stdout") or "")), "stderr_tail": _tail(str(result.get("stderr") or ""))})
            elif check_id == "agents_consistency":
                result = _run_capture([sys.executable, "-B", "-m", "trainer.autonomy.agents_consistency_check"], cwd=root, timeout_sec=timeout_sec, env=env_base)
                row.update({"status": result.get("status"), "returncode": result.get("returncode"), "elapsed_sec": result.get("elapsed_sec"), "stdout_tail": _tail(str(result.get("stdout") or "")), "stderr_tail": _tail(str(result.get("stderr") or ""))})
            elif check_id == "decision_policy_smoke":
                result = _run_capture([sys.executable, "-B", "-m", "trainer.runtime.decision_policy_check", "--action", "run_autonomy_entry"], cwd=root, timeout_sec=timeout_sec, env=env_base)
                row.update({"status": result.get("status"), "returncode": result.get("returncode"), "elapsed_sec": result.get("elapsed_sec"), "stdout_tail": _tail(str(result.get("stdout") or "")), "stderr_tail": _tail(str(result.get("stderr") or ""))})
            elif check_id == "doctor_precheck":
                result = _run_capture(["powershell", "-ExecutionPolicy", "Bypass", "-File", "scripts\\doctor.ps1", "-Emit", "json"], cwd=root, timeout_sec=timeout_sec, env=env_base)
                row.update({"status": result.get("status"), "returncode": result.get("returncode"), "elapsed_sec": result.get("elapsed_sec"), "stdout_tail": _tail(str(result.get("stdout") or "")), "stderr_tail": _tail(str(result.get("stderr") or ""))})
            else:
                command_argv = _check_command(check_id)
                safe_summary_path = run_dir / f"{check_id}_safe_run.json"
                result = _run_safe(command_argv, cwd=root, timeout_sec=timeout_sec, summary_path=safe_summary_path, env=env_base)
                row.update({"status": result.get("status"), "returncode": result.get("returncode"), "elapsed_sec": result.get("elapsed_sec"), "timed_out": result.get("timed_out"), "safe_run_summary_path": result.get("safe_run_summary_path")})
            report_rows.append(row)
            if str(row.get("status") or "") == "failed" and blocking_behavior == "must-pass":
                overall_status = "failed"
                break

    pending_certification = bool(plan.get("deferred_certification")) and overall_status == "passed"
    certification_status = "pending" if pending_certification else ("failed" if overall_status == "failed" else "not_required")
    required_next_step = "blocked" if overall_status == "failed" else ("certify" if pending_certification else "can_merge")
    report_payload: dict[str, Any] = {
        "schema": "p61_fast_check_report_v1",
        "generated_at": _now_iso(),
        "run_id": stamp,
        "repo_root": str(root),
        "change_scope": plan.get("change_scope"),
        "validation_plan_ref": plan_paths.get("json_path"),
        "validation_tiers_completed": completed_tiers if not dry_run else list(plan.get("validation_tiers_required") or []),
        "fast_check_status": "dry_run" if dry_run else overall_status,
        "pending_certification": pending_certification,
        "certification_status": certification_status,
        "required_next_step": required_next_step,
        "recommended_next_gate": recommended_next_gate,
        "checks": report_rows,
        "deferred_certification": plan.get("deferred_certification") or [],
        "artifact_refs": [str(run_dir.resolve()), str(plan_paths.get("json_path") or ""), str(plan_paths.get("md_path") or ""), str(gate_snapshot_path.resolve())],
    }
    queue_state = upsert_from_fast_check(report_payload, repo_root=root) if pending_certification else {}
    report_payload["certification_queue_ref"] = str(queue_json_path(root).resolve())
    report_payload["certification_state"] = queue_state

    json_path = run_dir / "fast_check_report.json"
    md_path = run_dir / "fast_check_report.md"
    latest_json = out_root / "latest_fast_check_report.json"
    latest_md = out_root / "latest_fast_check_report.md"
    report_payload["json_path"] = str(json_path.resolve())
    report_payload["md_path"] = str(md_path.resolve())

    md_lines = [
        "# P61 Fast Check Report",
        "",
        f"- generated_at: `{report_payload.get('generated_at')}`",
        f"- run_id: `{report_payload.get('run_id')}`",
        f"- fast_check_status: `{report_payload.get('fast_check_status')}`",
        f"- validation_tiers_completed: `{', '.join([str(item) for item in report_payload.get('validation_tiers_completed') or []])}`",
        f"- pending_certification: `{report_payload.get('pending_certification')}`",
        f"- certification_status: `{report_payload.get('certification_status')}`",
        f"- required_next_step: `{report_payload.get('required_next_step')}`",
        f"- certification_queue_ref: `{report_payload.get('certification_queue_ref') or ''}`",
        f"- recommended_next_gate: `{report_payload.get('recommended_next_gate') or ''}`",
        "",
        "## Checks",
        "",
    ]
    for row in report_rows:
        md_lines.append(
            "- `{}` tier=`{}` status=`{}` elapsed_sec=`{:.3f}`".format(
                row.get("check_id") or "",
                row.get("tier") or "",
                row.get("status") or "",
                float(row.get("elapsed_sec") or 0.0),
            )
        )
    if not report_rows:
        md_lines.append("- No checks executed.")
    if report_payload.get("deferred_certification"):
        md_lines += ["", "## Deferred Certification", ""]
        for item in report_payload.get("deferred_certification") or []:
            if not isinstance(item, dict):
                continue
            md_lines.append(
                "- `{}` command=`{}` reasons=`{}`".format(
                    item.get("check_id") or "",
                    item.get("command_template") or "",
                    " | ".join([str(reason) for reason in (item.get("reasons") or [])]),
                )
            )
    text = "\n".join(md_lines).rstrip() + "\n"
    json_text = json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n"
    json_path.write_text(json_text, encoding="utf-8")
    md_path.write_text(text, encoding="utf-8")
    latest_json.write_text(json_text, encoding="utf-8")
    latest_md.write_text(text, encoding="utf-8")
    return report_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="P61 fast validation loop")
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--gate-plan", default="")
    parser.add_argument("--changed-file", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    payload = execute_fast_checks(
        repo_root=args.repo_root or None,
        changed_files=list(args.changed_file or []),
        gate_plan_path=args.gate_plan or None,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
