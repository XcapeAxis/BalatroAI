from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from trainer.autonomy.attention_queue import open_attention_item
from trainer.runtime.bootstrap_env import (
    detect_nvidia_smi,
    latest_bootstrap_state_path,
    load_bootstrap_state,
    now_iso,
    now_stamp,
    probe_python_interpreter,
    resolve_repo_root,
    write_json,
)
from trainer.runtime.python_resolver import resolve_training_python


def _run_command(command: list[str], *, cwd: Path, timeout_sec: int = 60) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=max(5, int(timeout_sec)),
            encoding="utf-8",
            errors="replace",
        )
        return {
            "returncode": int(proc.returncode),
            "stdout": str(proc.stdout or ""),
            "stderr": str(proc.stderr or ""),
            "command": command,
        }
    except Exception as exc:
        return {
            "returncode": 1,
            "stdout": "",
            "stderr": repr(exc),
            "command": command,
        }


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _latest_matching_path(root: Path, pattern: str) -> Path | None:
    paths = sorted(root.glob(pattern), key=lambda item: (item.stat().st_mtime if item.exists() else 0.0, str(item)), reverse=True)
    return paths[0].resolve() if paths else None


def _git_status(repo_root: Path) -> dict[str, Any]:
    git_path = shutil.which("git")
    if not git_path:
        return {
            "available": False,
            "git_path": "",
            "branch": "",
            "head": "",
            "clean": False,
            "status_output": "",
            "remote_output": "",
        }
    branch = _run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root, timeout_sec=20)
    head = _run_command(["git", "rev-parse", "HEAD"], cwd=repo_root, timeout_sec=20)
    status = _run_command(["git", "status", "--short", "--branch"], cwd=repo_root, timeout_sec=20)
    remote = _run_command(["git", "remote", "-v"], cwd=repo_root, timeout_sec=20)
    status_output = str(status.get("stdout") or "").strip()
    dirty_lines = [line for line in status_output.splitlines() if line and not line.startswith("##")]
    return {
        "available": True,
        "git_path": git_path,
        "branch": str(branch.get("stdout") or "").strip(),
        "head": str(head.get("stdout") or "").strip(),
        "clean": len(dirty_lines) == 0,
        "status_output": status_output,
        "remote_output": str(remote.get("stdout") or "").strip(),
    }


def _py_launcher_inventory(repo_root: Path) -> dict[str, Any]:
    py_path = shutil.which("py")
    if not py_path:
        return {"available": False, "path": "", "output": ""}
    result = _run_command(["py", "-0p"], cwd=repo_root, timeout_sec=20)
    return {
        "available": True,
        "path": py_path,
        "output": str(result.get("stdout") or result.get("stderr") or "").strip(),
    }


def _system_python_inventory(repo_root: Path) -> list[str]:
    rows: list[str] = []
    where_path = shutil.which("where.exe")
    if where_path:
        result = _run_command(["where.exe", "python"], cwd=repo_root, timeout_sec=20)
        rows.extend([line.strip() for line in str(result.get("stdout") or "").splitlines() if line.strip()])
    return rows[:8]


def _latest_p55_report(repo_root: Path) -> tuple[Path | None, dict[str, Any]]:
    root = repo_root / "docs" / "artifacts" / "p55" / "config_sidecar_sync"
    path = _latest_matching_path(root, "**/sidecar_sync_report.json")
    payload = _read_json(path) if isinstance(path, Path) else {}
    return path, payload if isinstance(payload, dict) else {}


def _run_config_sidecar_check(repo_root: Path, selected_python: str, *, skip: bool = False) -> dict[str, Any]:
    latest_before, _ = _latest_p55_report(repo_root)
    if skip:
        path, payload = _latest_p55_report(repo_root)
        return {
            "status": "skipped",
            "returncode": 0,
            "path": str(path.resolve()) if isinstance(path, Path) else "",
            "payload": payload,
            "latest_before": str(latest_before.resolve()) if isinstance(latest_before, Path) else "",
        }
    if not str(selected_python or "").strip():
        return {"status": "failed", "returncode": 1, "path": "", "payload": {}, "error": "selected_python_missing"}
    result = _run_command(
        [selected_python, "-B", "-m", "trainer.experiments.config_sidecar_sync", "--check", "--quiet"],
        cwd=repo_root,
        timeout_sec=600,
    )
    path, payload = _latest_p55_report(repo_root)
    return {
        "status": "ok" if int(result.get("returncode") or 0) == 0 else "failed",
        "returncode": int(result.get("returncode") or 0),
        "path": str(path.resolve()) if isinstance(path, Path) else "",
        "payload": payload,
        "overall_status": str(payload.get("overall_status") or payload.get("status") or "").lower() if isinstance(payload, dict) else "",
        "stdout": str(result.get("stdout") or "").strip(),
        "stderr": str(result.get("stderr") or "").strip(),
    }


def _probe_runtime_paths(repo_root: Path) -> dict[str, Any]:
    artifacts_root = repo_root / "docs" / "artifacts"
    dashboard_index = artifacts_root / "dashboard" / "latest" / "index.html"
    dashboard_data = artifacts_root / "dashboard" / "latest" / "dashboard_data.json"
    ops_ui_state = artifacts_root / "p53" / "ops_ui" / "latest" / "ops_ui_state.json"
    registry_path = artifacts_root / "registry" / "checkpoints_registry.json"
    readiness_latest = _latest_matching_path(artifacts_root / "p49" / "readiness", "**/service_readiness_report.json")
    dashboard_payload = _read_json(dashboard_data) if dashboard_data.exists() else {}
    ops_ui_payload = _read_json(ops_ui_state) if ops_ui_state.exists() else {}
    registry_payload = _read_json(registry_path) if registry_path.exists() else {}
    readiness_payload = _read_json(readiness_latest) if isinstance(readiness_latest, Path) else {}
    registry_entries = registry_payload.get("entries") if isinstance(registry_payload, dict) else []
    return {
        "dashboard": {
            "index_path": str(dashboard_index.resolve()),
            "data_path": str(dashboard_data.resolve()),
            "exists": dashboard_index.exists() and dashboard_data.exists(),
            "generated_at": str(dashboard_payload.get("generated_at") or "") if isinstance(dashboard_payload, dict) else "",
            "latest_run_id": str(dashboard_payload.get("latest_run_id") or "") if isinstance(dashboard_payload, dict) else "",
        },
        "ops_ui": {
            "state_path": str(ops_ui_state.resolve()),
            "exists": ops_ui_state.exists(),
            "generated_at": str(ops_ui_payload.get("generated_at") or "") if isinstance(ops_ui_payload, dict) else "",
            "url": str(ops_ui_payload.get("url") or "") if isinstance(ops_ui_payload, dict) else "",
        },
        "registry": {
            "path": str(registry_path.resolve()),
            "exists": registry_path.exists(),
            "entry_count": len(registry_entries) if isinstance(registry_entries, list) else 0,
            "counts": dict(registry_payload.get("counts") or {}) if isinstance(registry_payload, dict) else {},
        },
        "latest_readiness": {
            "path": str(readiness_latest.resolve()) if isinstance(readiness_latest, Path) else "",
            "status": str(readiness_payload.get("status") or "") if isinstance(readiness_payload, dict) else "",
            "generated_at": str(readiness_payload.get("generated_at") or "") if isinstance(readiness_payload, dict) else "",
        },
    }


def _live_readiness_probe(repo_root: Path, selected_python: str, *, base_url: str) -> dict[str, Any]:
    if not str(selected_python or "").strip():
        return {"status": "skipped", "reason": "selected_python_missing", "path": "", "payload": {}}
    run_id = f"doctor-{now_stamp()}"
    out_dir = repo_root / "docs" / "artifacts" / "p58" / "readiness_probe"
    result = _run_command(
        [
            selected_python,
            "-B",
            "-m",
            "trainer.runtime.service_readiness",
            "--base-url",
            base_url,
            "--out-dir",
            str(out_dir),
            "--run-id",
            run_id,
            "--max-retries",
            "1",
            "--retry-interval-sec",
            "0.1",
            "--warmup-grace-sec",
            "0",
            "--consecutive-successes",
            "1",
            "--timeout-sec",
            "1.5",
        ],
        cwd=repo_root,
        timeout_sec=90,
    )
    report_path = out_dir / run_id / "service_readiness_report.json"
    payload = _read_json(report_path) if report_path.exists() else {}
    return {
        "status": "ready" if int(result.get("returncode") or 0) == 0 else "unreachable",
        "path": str(report_path.resolve()) if report_path.exists() else "",
        "payload": payload if isinstance(payload, dict) else {},
        "stdout": str(result.get("stdout") or "").strip(),
        "stderr": str(result.get("stderr") or "").strip(),
    }


def _env_probe_table(repo_root: Path) -> dict[str, Any]:
    return {
        "cpu": probe_python_interpreter(repo_root / ".venv_trainer", repo_root=repo_root, label=".venv_trainer"),
        "cuda": probe_python_interpreter(repo_root / ".venv_trainer_cuda", repo_root=repo_root, label=".venv_trainer_cuda"),
    }


def _recommend_mode(env_probes: dict[str, Any]) -> str:
    cpu = env_probes.get("cpu") if isinstance(env_probes.get("cpu"), dict) else {}
    cuda = env_probes.get("cuda") if isinstance(env_probes.get("cuda"), dict) else {}
    if bool(cuda.get("ok")) and bool(cuda.get("torch_available")) and bool(cuda.get("cuda_available")):
        return "cuda_mainline"
    if bool(cpu.get("ok")) and bool(cpu.get("torch_available")):
        return "cpu_safe"
    return "blocked"


def _overlay_bootstrap_probe(env_probe: dict[str, Any], bootstrap_probe: dict[str, Any]) -> dict[str, Any]:
    merged = dict(env_probe)
    if not (
        bool(env_probe.get("ok"))
        and not bool(env_probe.get("torch_available"))
        and bool(bootstrap_probe.get("ok"))
        and bool(bootstrap_probe.get("torch_available"))
        and str(env_probe.get("python") or "").strip().lower() == str(bootstrap_probe.get("python") or "").strip().lower()
    ):
        return merged
    live_error = str(env_probe.get("error") or "").strip()
    merged["torch_available"] = bool(bootstrap_probe.get("torch_available"))
    merged["torch_version"] = bootstrap_probe.get("torch_version")
    merged["cuda_available"] = bool(bootstrap_probe.get("cuda_available"))
    merged["device_count"] = int(bootstrap_probe.get("device_count") or 0)
    merged["device_name"] = bootstrap_probe.get("device_name")
    merged["env_type"] = str(bootstrap_probe.get("env_type") or merged.get("env_type") or "")
    merged["health_status"] = str(bootstrap_probe.get("health_status") or merged.get("health_status") or "")
    warnings = [str(item) for item in (merged.get("warnings") or []) if str(item).strip() and str(item) not in {"torch_missing", "cuda_unavailable"}]
    warnings.append("torch_probe_fallback_bootstrap_state")
    if live_error:
        warnings.append("live_torch_probe_timeout")
        merged["live_probe_error"] = live_error
    merged["warnings"] = warnings
    merged["error"] = ""
    return merged


def _build_md(payload: dict[str, Any]) -> str:
    env_probes = payload.get("env_probes") if isinstance(payload.get("env_probes"), dict) else {}
    blocking = payload.get("blocking_reasons") if isinstance(payload.get("blocking_reasons"), list) else []
    warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
    next_steps = payload.get("next_steps") if isinstance(payload.get("next_steps"), list) else []
    lines = [
        "# P58 Doctor Report",
        "",
        f"- generated_at: `{payload.get('generated_at') or ''}`",
        f"- status: `{payload.get('status') or ''}`",
        f"- requested_mode: `{payload.get('requested_mode') or ''}`",
        f"- recommended_mode: `{payload.get('recommended_mode') or ''}`",
        f"- ready_for_continuation: `{payload.get('ready_for_continuation')}`",
        f"- selected_training_python: `{((payload.get('resolver') or {}).get('selected') or {}).get('python') or ''}`",
        f"- bootstrap_state: `{payload.get('bootstrap_state_path') or ''}`",
        "",
        "## Blocking Reasons",
        "",
    ]
    if blocking:
        for item in blocking:
            lines.append(f"- {item}")
    else:
        lines.append("- None.")
    lines += ["", "## Warnings", ""]
    if warnings:
        for item in warnings:
            lines.append(f"- {item}")
    else:
        lines.append("- None.")
    lines += [
        "",
        "## Environment Probes",
        "",
        "| Env | Python | Torch | PyYAML | CUDA | Health |",
        "|---|---|---|---|---|---|",
    ]
    for role in ("cpu", "cuda"):
        row = env_probes.get(role) if isinstance(env_probes.get(role), dict) else {}
        lines.append(
            "| {role} | `{python}` | `{torch}` | `{yaml}` | `{cuda}` | `{health}` |".format(
                role=role,
                python=row.get("python") or "",
                torch=row.get("torch_version") or "",
                yaml=row.get("yaml_version") or "",
                cuda=row.get("cuda_available"),
                health=row.get("health_status") or "",
            )
        )
    lines += ["", "## Next Steps", ""]
    if next_steps:
        for item in next_steps:
            lines.append(f"- `{item}`")
    else:
        lines.append("- No follow-up required.")
    if payload.get("attention_item_ref"):
        lines += ["", "## Attention", "", f"- attention_item: `{payload.get('attention_item_ref')}`"]
    return "\n".join(lines).rstrip() + "\n"


def run_doctor(
    *,
    repo_root: Path | None = None,
    requested_mode: str = "auto",
    explicit_python: str = "",
    explicit_env: str = "",
    out_root: str | Path | None = None,
    queue_attention_on_block: bool = False,
    skip_config_check: bool = False,
    base_url: str = "http://127.0.0.1:12346",
) -> dict[str, Any]:
    root = repo_root.resolve() if isinstance(repo_root, Path) else resolve_repo_root()
    output_root = Path(out_root).resolve() if out_root else (root / "docs" / "artifacts" / "p58").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    bootstrap_state = load_bootstrap_state(root)
    bootstrap_envs = bootstrap_state.get("envs") if isinstance(bootstrap_state.get("envs"), dict) else {}
    env_probes = _env_probe_table(root)
    for role in ("cpu", "cuda"):
        probe = env_probes.get(role) if isinstance(env_probes.get(role), dict) else {}
        bootstrap_probe = bootstrap_envs.get(role) if isinstance(bootstrap_envs.get(role), dict) else {}
        if probe:
            env_probes[role] = _overlay_bootstrap_probe(probe, bootstrap_probe)
    recommended_mode = _recommend_mode(env_probes)
    require_cuda = str(requested_mode or "auto").lower() == "cuda"
    resolver = resolve_training_python(
        repo_root=root,
        explicit_python=explicit_python,
        explicit_env=explicit_env,
        prefer_cuda=True,
        require_cuda=require_cuda,
        allow_cpu_fallback=True,
    )
    selected = resolver.get("selected") if isinstance(resolver.get("selected"), dict) else {}
    selected_python = str(selected.get("python") or "")
    config_sync = _run_config_sidecar_check(root, selected_python, skip=skip_config_check)
    runtime_paths = _probe_runtime_paths(root)
    readiness_live = _live_readiness_probe(root, selected_python, base_url=base_url)
    bootstrap_complete = bool(bootstrap_state.get("bootstrap_complete")) if isinstance(bootstrap_state, dict) else False
    git_state = _git_status(root)
    py_launcher = _py_launcher_inventory(root)
    system_python = _system_python_inventory(root)
    blocking_reasons: list[str] = []
    warnings: list[str] = []
    next_steps: list[str] = []

    if not bool(git_state.get("available")):
        blocking_reasons.append("git_not_available")
        next_steps.append("Install Git and reopen the repo.")
    if str(git_state.get("branch") or "") not in {"main", ""}:
        blocking_reasons.append("branch_not_main")
        next_steps.append("git checkout main")
    if not bool(git_state.get("clean", False)):
        warnings.append("working_tree_dirty")
    if str(resolver.get("status") or "") != "ok":
        blocking_reasons.append(str(resolver.get("error") or resolver.get("fallback_reason") or "resolver_failed"))
    if not bool(selected.get("torch_available")):
        blocking_reasons.append("selected_training_python_missing_torch")
        next_steps.append("powershell -ExecutionPolicy Bypass -File scripts\\setup_windows.ps1 -Mode auto")
    if require_cuda and not bool(selected.get("cuda_available")):
        blocking_reasons.append("cuda_required_but_not_available")
        next_steps.append("powershell -ExecutionPolicy Bypass -File scripts\\setup_windows.ps1 -Mode cuda")
    if int(config_sync.get("returncode") or 0) != 0:
        blocking_reasons.append("config_sidecar_check_failed")
        next_steps.append("powershell -ExecutionPolicy Bypass -File scripts\\sync_config_sidecars.ps1")
    if recommended_mode == "blocked":
        blocking_reasons.append("no_healthy_training_env")
        if "powershell -ExecutionPolicy Bypass -File scripts\\setup_windows.ps1 -Mode auto" not in next_steps:
            next_steps.append("powershell -ExecutionPolicy Bypass -File scripts\\setup_windows.ps1 -Mode auto")
    if not bool(bootstrap_complete):
        warnings.append("bootstrap_state_missing_or_incomplete")
        if "powershell -ExecutionPolicy Bypass -File scripts\\setup_windows.ps1 -Mode auto" not in next_steps:
            next_steps.append("powershell -ExecutionPolicy Bypass -File scripts\\setup_windows.ps1 -Mode auto")
    if not bool((env_probes.get("cpu") or {}).get("yaml_available")):
        warnings.append("cpu_fallback_env_missing_pyyaml")
        next_steps.append("powershell -ExecutionPolicy Bypass -File scripts\\setup_cpu_env.ps1")
    if not bool((env_probes.get("cuda") or {}).get("yaml_available")) and bool((env_probes.get("cuda") or {}).get("exists")):
        warnings.append("cuda_env_missing_pyyaml")
        next_steps.append("powershell -ExecutionPolicy Bypass -File scripts\\setup_cuda_env.ps1")
    if str(readiness_live.get("status") or "") != "ready":
        warnings.append("service_readiness_unavailable_or_not_running")
    if not bool(runtime_paths.get("dashboard", {}).get("exists")):
        warnings.append("dashboard_not_built")
        next_steps.append("powershell -ExecutionPolicy Bypass -File scripts\\run_dashboard.ps1")
    if not bool(runtime_paths.get("registry", {}).get("exists")):
        warnings.append("registry_snapshot_missing")
    if not bool(runtime_paths.get("ops_ui", {}).get("exists")):
        warnings.append("ops_ui_state_missing")
    if not next_steps:
        next_steps = [
            "powershell -ExecutionPolicy Bypass -File scripts\\run_p22.ps1 -Quick",
            "powershell -ExecutionPolicy Bypass -File scripts\\run_ops_ui.ps1",
        ]

    status = "blocked" if blocking_reasons else ("warning" if warnings else "ready")
    payload: dict[str, Any] = {
        "schema": "p58_doctor_report_v1",
        "generated_at": now_iso(),
        "repo_root": str(root),
        "requested_mode": str(requested_mode or "auto"),
        "recommended_mode": recommended_mode,
        "ready_for_continuation": status != "blocked",
        "status": status,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "next_steps": next_steps,
        "git": git_state,
        "py_launcher": py_launcher,
        "system_python": system_python,
        "nvidia_smi": detect_nvidia_smi(root),
        "bootstrap_state_path": str(latest_bootstrap_state_path(root)) if latest_bootstrap_state_path(root).exists() else "",
        "bootstrap_state": bootstrap_state if isinstance(bootstrap_state, dict) else {},
        "bootstrap_complete": bootstrap_complete,
        "resolver": resolver,
        "env_probes": env_probes,
        "config_sidecar_check": config_sync,
        "runtime_paths": runtime_paths,
        "live_readiness": readiness_live,
        "training_capability": {
            "cpu_fallback": {
                "available": bool((env_probes.get("cpu") or {}).get("ok")) and bool((env_probes.get("cpu") or {}).get("torch_available")),
                "python": str((env_probes.get("cpu") or {}).get("python") or ""),
                "warnings": list((env_probes.get("cpu") or {}).get("warnings") or []),
            },
            "cuda_mainline": {
                "available": bool((env_probes.get("cuda") or {}).get("ok")) and bool((env_probes.get("cuda") or {}).get("cuda_available")),
                "python": str((env_probes.get("cuda") or {}).get("python") or ""),
                "warnings": list((env_probes.get("cuda") or {}).get("warnings") or []),
            },
        },
    }

    stamp = now_stamp()
    json_path = output_root / f"doctor_{stamp}.json"
    md_path = output_root / f"doctor_{stamp}.md"
    payload["json_path"] = str(json_path.resolve())
    payload["md_path"] = str(md_path.resolve())
    write_json(json_path, payload)
    md_text = _build_md(payload)
    md_path.write_text(md_text, encoding="utf-8")
    write_json(output_root / "latest_doctor.json", payload)
    (output_root / "latest_doctor.md").write_text(md_text, encoding="utf-8")

    if queue_attention_on_block and status == "blocked":
        attention = open_attention_item(
            severity="block",
            category="environment",
            title="Windows bootstrap/doctor blocked continuation",
            summary="The latest doctor report found blocking environment issues, so the mainline workflow should stop until the machine is repaired.",
            blocking_stage="environment_doctor",
            attempted_actions=["run_doctor", "resolve_training_python", "check_config_sidecars"],
            recommended_options=[
                {"label": "run_setup_auto", "description": "Repair or create the standard Windows envs with setup_windows.ps1."},
                {"label": "review_doctor_report", "description": "Open the doctor report and address the listed blockers."},
            ],
            recommended_default="run_setup_auto",
            required_human_input=["Confirm whether environment changes are allowed on this machine and repair the blocking issues."],
            artifact_refs=[str(json_path.resolve()), str(md_path.resolve())],
        )
        payload["attention_item_ref"] = str(attention.get("item_md_path") or attention.get("attention_id") or "")
        write_json(json_path, payload)
        md_text = _build_md(payload)
        md_path.write_text(md_text, encoding="utf-8")
        write_json(output_root / "latest_doctor.json", payload)
        (output_root / "latest_doctor.md").write_text(md_text, encoding="utf-8")

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P58 Windows environment doctor")
    parser.add_argument("--mode", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--explicit-python", default="")
    parser.add_argument("--explicit-env", default="")
    parser.add_argument("--out-root", default="docs/artifacts/p58")
    parser.add_argument("--queue-attention-on-block", action="store_true")
    parser.add_argument("--skip-config-check", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_doctor(
        repo_root=resolve_repo_root(),
        requested_mode=str(args.mode or "auto"),
        explicit_python=str(args.explicit_python or ""),
        explicit_env=str(args.explicit_env or ""),
        out_root=str(args.out_root or "docs/artifacts/p58"),
        queue_attention_on_block=bool(args.queue_attention_on_block),
        skip_config_check=bool(args.skip_config_check),
        base_url=str(args.base_url or "http://127.0.0.1:12346"),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if str(payload.get("status") or "") != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
    if recommended_mode == "blocked":
        if bool(selected.get("torch_available")) and bool(selected.get("cuda_available")):
            recommended_mode = "cuda_mainline"
        elif bool(selected.get("torch_available")):
            recommended_mode = "cpu_safe"
