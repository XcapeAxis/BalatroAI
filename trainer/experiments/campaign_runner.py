from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trainer.experiments.campaign import CampaignSpec, load_campaign
from trainer.experiments.campaign_report import render_campaign_summary_md, write_json
from trainer.experiments.champion import update_p24_from_ranking


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_process(
    args: list[str],
    *,
    cwd: Path,
    timeout_sec: int | None = None,
) -> dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
        return {
            "returncode": int(proc.returncode),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "elapsed_sec": time.time() - started,
            "timed_out": False,
            "cmd": args,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": 124,
            "stdout": str(exc.stdout or ""),
            "stderr": str(exc.stderr or ""),
            "elapsed_sec": time.time() - started,
            "timed_out": True,
            "cmd": args,
        }


def latest_dir(path: Path) -> Path | None:
    if not path.exists():
        return None
    dirs = sorted([p for p in path.iterdir() if p.is_dir()], key=lambda p: p.name)
    return dirs[-1] if dirs else None


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def write_status(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = now_iso()
    write_json(path, payload)


def copy_stage_experiments(stage_run_root: Path, campaign_run_root: Path, stage_id: str) -> list[str]:
    copied: list[str] = []
    for p in sorted([x for x in stage_run_root.iterdir() if x.is_dir()], key=lambda x: x.name):
        if not (p / "run_manifest.json").exists():
            continue
        target_name = f"{stage_id}__{p.name}"
        target = campaign_run_root / target_name
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(p, target)
        copied.append(target_name)
    return copied


def merge_stage_telemetry(
    *,
    src_telemetry: Path,
    dst_telemetry: Path,
    campaign_id: str,
    stage_id: str,
) -> int:
    if not src_telemetry.exists():
        return 0
    count = 0
    for raw in src_telemetry.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                continue
        except Exception:
            continue
        payload["campaign_id"] = campaign_id
        payload["campaign_stage"] = stage_id
        append_jsonl(dst_telemetry, payload)
        count += 1
    return count


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P24 campaign runner")
    p.add_argument("--campaign-config", required=True)
    p.add_argument("--out-root", default="docs/artifacts/p24")
    p.add_argument("--run-id", default="")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--headless-dashboard", action="store_true")
    p.add_argument("--dashboard-out", default="")
    p.add_argument("--ranking-config", default="configs/experiments/ranking_p24.yaml")
    p.add_argument("--python", default=sys.executable)
    return p.parse_args()


def run_post_action(
    *,
    name: str,
    enabled: bool,
    python_exe: str,
    repo_root: Path,
    run_root: Path,
    ranking_config: str,
    headless_dashboard: bool,
    dashboard_out: str,
) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False, "status": "skipped"}

    if name == "coverage":
        cmd = [
            python_exe,
            "-B",
            "-m",
            "trainer.experiments.coverage",
            "--run-root",
            str(run_root),
            "--out-dir",
            str(run_root),
        ]
        result = run_process(cmd, cwd=repo_root, timeout_sec=600)
        return {"enabled": True, "status": "pass" if result["returncode"] == 0 else "fail", "cmd": cmd}

    if name == "triage":
        out_dir = run_root / "triage_latest"
        cmd = [
            python_exe,
            "-B",
            "-m",
            "trainer.experiments.triage",
            "--run-root",
            str(run_root),
            "--out-dir",
            str(out_dir),
        ]
        result = run_process(cmd, cwd=repo_root, timeout_sec=600)
        return {"enabled": True, "status": "pass" if result["returncode"] == 0 else "fail", "cmd": cmd}

    if name == "bisect":
        out_dir = run_root / "bisect_latest"
        cmd = [
            python_exe,
            "-B",
            "-m",
            "trainer.experiments.bisect_lite",
            "--run-root",
            str(run_root),
            "--mode",
            "seed_bisect",
            "--out-dir",
            str(out_dir),
        ]
        result = run_process(cmd, cwd=repo_root, timeout_sec=600)
        return {"enabled": True, "status": "pass" if result["returncode"] == 0 else "fail", "cmd": cmd}

    if name == "ranking":
        out_dir = run_root / "ranking_latest"
        cmd = [
            python_exe,
            "-B",
            "-m",
            "trainer.experiments.ranking",
            "--run-root",
            str(run_root),
            "--config",
            ranking_config,
            "--out-dir",
            str(out_dir),
        ]
        result = run_process(cmd, cwd=repo_root, timeout_sec=600)
        return {"enabled": True, "status": "pass" if result["returncode"] == 0 else "fail", "cmd": cmd}

    if name == "dashboard":
        if not headless_dashboard:
            return {"enabled": True, "status": "skipped", "reason": "headless_not_enabled"}
        out_path = (
            Path(dashboard_out).resolve()
            if dashboard_out
            else (run_root / "dashboard_headless_log.txt")
        )
        cmd = [
            python_exe,
            "-B",
            "-m",
            "trainer.experiments.dashboard_tui",
            "--watch",
            str(run_root / "telemetry.jsonl"),
            "--headless-log",
            "--out",
            str(out_path),
        ]
        result = run_process(cmd, cwd=repo_root, timeout_sec=600)
        return {"enabled": True, "status": "pass" if result["returncode"] == 0 else "fail", "cmd": cmd, "out": str(out_path)}

    return {"enabled": True, "status": "skipped", "reason": "unknown_action"}


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    campaign_path = (repo_root / args.campaign_config).resolve()
    out_root = (repo_root / args.out_root).resolve()
    runs_root = out_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    spec: CampaignSpec = load_campaign(campaign_path)

    run_id = str(args.run_id or "").strip()
    if args.resume:
        latest = latest_dir(runs_root)
        if latest is not None:
            run_id = latest.name
    if not run_id:
        run_id = now_stamp()
    run_root = runs_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    status_path = run_root / "campaign_status.json"
    summary_path = run_root / "campaign_summary.json"
    summary_md_path = run_root / "campaign_summary.md"
    telemetry_path = run_root / "telemetry.jsonl"

    plan_payload = {
        "schema": "p24_campaign_plan_v1",
        "generated_at": now_iso(),
        "run_id": run_id,
        "campaign_config": str(campaign_path),
        "run_root": str(run_root),
        "campaign": spec.to_dict(),
    }
    write_json(run_root / "campaign_plan.json", plan_payload)

    status_payload: dict[str, Any] = {
        "schema": "p24_campaign_status_v1",
        "campaign_id": spec.campaign_id,
        "run_id": run_id,
        "state": "queued",
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "stage_status": [],
        "completed_stage_ids": [],
    }
    write_status(status_path, status_payload)

    stage_rows: list[dict[str, Any]] = []
    stage_states: dict[str, str] = {}
    started_ts = time.time()
    max_wall_sec = max(60, spec.budget.max_wall_time_minutes * 60)

    status_payload["state"] = "running"
    write_status(status_path, status_payload)

    for stage in spec.stages:
        if time.time() - started_ts > max_wall_sec:
            status_payload["state"] = "budget_cut"
            write_status(status_path, status_payload)
            break

        dep_failed = False
        for dep in stage.depends_on:
            if stage_states.get(dep) not in {"stage_pass"}:
                dep_failed = True
                break
        if dep_failed:
            row = {
                "stage_id": stage.stage_id,
                "purpose": stage.purpose,
                "status": "skipped_dependency",
                "run_root": "",
                "experiment_count": 0,
            }
            stage_states[stage.stage_id] = "skipped_dependency"
            stage_rows.append(row)
            status_payload["stage_status"].append(row)
            write_status(status_path, status_payload)
            continue

        stage_out_root = run_root / "stages" / stage.stage_id
        stage_out_root.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.python,
            "-B",
            "-m",
            "trainer.experiments.orchestrator",
            "--config",
            stage.matrix_ref,
            "--out-root",
            str(stage_out_root),
            "--mode",
            stage.mode or spec.mode,
            "--max-parallel",
            str(spec.budget.max_parallel),
            "--max-experiments",
            str(stage.max_experiments or spec.budget.max_experiments),
            "--seed-policy-config",
            spec.seed_policy_config,
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        if args.resume:
            cmd.append("--resume")
        if stage.include:
            cmd += ["--only", ",".join(stage.include)]
        if stage.exclude:
            cmd += ["--exclude", ",".join(stage.exclude)]
        if stage.seed_limit > 0:
            cmd += ["--seed-limit", str(stage.seed_limit)]

        result = run_process(cmd, cwd=repo_root, timeout_sec=max_wall_sec)
        stage_runs_root = stage_out_root / "runs"
        latest_stage_run = latest_dir(stage_runs_root)
        latest_stage_run_path = str(latest_stage_run) if latest_stage_run is not None else ""
        report = read_json((latest_stage_run or stage_out_root) / "report_p23.json")
        row_status = "stage_pass" if result["returncode"] == 0 and str(report.get("status")) == "PASS" else "stage_fail"
        experiment_count = len(report.get("rows") or [])

        copied_exps: list[str] = []
        merged_lines = 0
        if latest_stage_run is not None:
            copied_exps = copy_stage_experiments(latest_stage_run, run_root, stage.stage_id)
            merged_lines = merge_stage_telemetry(
                src_telemetry=latest_stage_run / "telemetry.jsonl",
                dst_telemetry=telemetry_path,
                campaign_id=spec.campaign_id,
                stage_id=stage.stage_id,
            )

        row = {
            "stage_id": stage.stage_id,
            "purpose": stage.purpose,
            "status": row_status,
            "run_root": latest_stage_run_path,
            "experiment_count": experiment_count,
            "copied_experiments": copied_exps,
            "telemetry_events_merged": merged_lines,
            "returncode": result["returncode"],
            "elapsed_sec": result["elapsed_sec"],
            "failure_policy": stage.failure_policy,
        }
        stage_rows.append(row)
        stage_states[stage.stage_id] = row_status
        status_payload["stage_status"].append(row)
        if row_status == "stage_pass":
            status_payload["completed_stage_ids"].append(stage.stage_id)
        write_status(status_path, status_payload)

        if row_status != "stage_pass" and stage.failure_policy == "fail_fast":
            status_payload["state"] = "aborted"
            write_status(status_path, status_payload)
            break

    if status_payload.get("state") not in {"aborted", "budget_cut"}:
        if all(r.get("status") in {"stage_pass", "skipped_dependency"} for r in stage_rows):
            status_payload["state"] = "completed"
        else:
            status_payload["state"] = "stage_fail"
        write_status(status_path, status_payload)

    post_action_results: dict[str, Any] = {}
    for action_name, enabled in spec.post_actions.items():
        post_action_results[action_name] = run_post_action(
            name=action_name,
            enabled=bool(enabled),
            python_exe=args.python,
            repo_root=repo_root,
            run_root=run_root,
            ranking_config=args.ranking_config,
            headless_dashboard=bool(args.headless_dashboard),
            dashboard_out=args.dashboard_out,
        )

    if bool(spec.post_actions.get("champion_update", True)):
        ranking_summary = read_json(run_root / "ranking_latest" / "ranking_summary.json")
        champion_update = update_p24_from_ranking(
            out_root=out_root,
            run_id=run_id,
            ranking_summary=ranking_summary,
        )
        post_action_results["champion_update"] = {
            "enabled": True,
            "status": "pass",
            "result": champion_update,
        }

    summary_payload = {
        "schema": "p24_campaign_summary_v1",
        "generated_at": now_iso(),
        "campaign_id": spec.campaign_id,
        "run_id": run_id,
        "mode": spec.mode,
        "status": status_payload.get("state"),
        "elapsed_sec": time.time() - started_ts,
        "stages": stage_rows,
        "post_actions": post_action_results,
        "run_root": str(run_root),
    }
    write_json(summary_path, summary_payload)
    summary_md_path.write_text(render_campaign_summary_md(summary_payload), encoding="utf-8")

    print(f"[P24-campaign] run_id={run_id} status={summary_payload['status']} run_root={run_root}")
    return 0 if summary_payload["status"] in {"completed", "stage_fail"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

