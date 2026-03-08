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


def resolve_repo_root() -> Path:
    preferred = Path("D:/MYFILES/BalatroAI")
    if preferred.exists():
        return preferred.resolve()
    return Path(__file__).resolve().parents[2]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bootstrap_artifacts_root(repo_root: Path | None = None) -> Path:
    root = repo_root.resolve() if isinstance(repo_root, Path) else resolve_repo_root()
    return (root / "docs" / "artifacts" / "p58" / "bootstrap").resolve()


def latest_bootstrap_state_path(repo_root: Path | None = None) -> Path:
    return (bootstrap_artifacts_root(repo_root) / "latest_bootstrap_state.json").resolve()


def latest_bootstrap_md_path(repo_root: Path | None = None) -> Path:
    return (bootstrap_artifacts_root(repo_root) / "latest_bootstrap_state.md").resolve()


def python_from_env_dir(env_path: str | Path) -> Path:
    candidate = Path(env_path)
    if candidate.is_file():
        return candidate.resolve()
    if os.name == "nt":
        return (candidate / "Scripts" / "python.exe").resolve()
    return (candidate / "bin" / "python").resolve()


def normalize_python_path(value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_file():
        return candidate.resolve()
    if candidate.suffix.lower() == ".exe" or candidate.name.lower().startswith("python"):
        return candidate.resolve()
    return python_from_env_dir(candidate)


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


def _env_dir_for_python(candidate: Path) -> Path | None:
    parent = candidate.parent
    if parent.name.lower() in {"scripts", "bin"}:
        return parent.parent.resolve()
    return None


def _repo_env_role(env_dir: Path | None, repo_root: Path) -> str:
    if env_dir is None:
        return "external"
    try:
        if env_dir.resolve() == (repo_root / ".venv_trainer_cuda").resolve():
            return "cuda"
        if env_dir.resolve() == (repo_root / ".venv_trainer").resolve():
            return "cpu"
    except Exception:
        return "external"
    return "external"


def probe_python_interpreter(
    candidate: str | Path,
    *,
    repo_root: Path | None = None,
    label: str = "",
    timeout_sec: int = 60,
) -> dict[str, Any]:
    root = repo_root.resolve() if isinstance(repo_root, Path) else resolve_repo_root()
    python_path = normalize_python_path(candidate)
    env_dir = _env_dir_for_python(python_path)
    env_name = env_dir.name if isinstance(env_dir, Path) else ""
    bootstrap_role = _repo_env_role(env_dir, root)
    payload: dict[str, Any] = {
        "label": str(label or python_path.name),
        "python": str(python_path),
        "exists": python_path.exists(),
        "ok": False,
        "python_version": "",
        "env_dir": str(env_dir) if isinstance(env_dir, Path) else "",
        "env_name": env_name,
        "bootstrap_role": bootstrap_role,
        "is_repo_env": bootstrap_role in {"cpu", "cuda"},
        "torch_available": False,
        "torch_version": None,
        "cuda_available": False,
        "device_count": 0,
        "device_name": None,
        "yaml_available": False,
        "yaml_version": None,
        "pip_version": None,
        "health_status": "missing" if not python_path.exists() else "unknown",
        "warnings": [],
        "error": "",
    }
    if not python_path.exists():
        payload["error"] = "python_not_found"
        return payload

    probe_code = (
        "import json, sys\n"
        "data = {'python': sys.executable, 'python_version': sys.version.replace('\\n', ' ').strip(), 'prefix': sys.prefix, 'base_prefix': getattr(sys, 'base_prefix', sys.prefix)}\n"
        "try:\n"
        "  import yaml\n"
        "  data['yaml_available'] = True\n"
        "  data['yaml_version'] = str(getattr(yaml, '__version__', ''))\n"
        "except Exception as exc:\n"
        "  data['yaml_available'] = False\n"
        "  data['yaml_version'] = None\n"
        "  data['yaml_error'] = repr(exc)\n"
        "try:\n"
        "  import torch\n"
        "  cuda = bool(torch.cuda.is_available())\n"
        "  data.update({'torch_available': True, 'torch_version': str(torch.__version__), 'cuda_available': cuda, 'device_count': int(torch.cuda.device_count()) if cuda else 0, 'device_name': str(torch.cuda.get_device_name(0)) if cuda and int(torch.cuda.device_count()) > 0 else None})\n"
        "except Exception as exc:\n"
        "  data.update({'torch_available': False, 'torch_version': None, 'cuda_available': False, 'device_count': 0, 'device_name': None, 'torch_error': repr(exc)})\n"
        "try:\n"
        "  import pip\n"
        "  data['pip_version'] = str(getattr(pip, '__version__', ''))\n"
        "except Exception as exc:\n"
        "  data['pip_version'] = None\n"
        "  data['pip_error'] = repr(exc)\n"
        "print(json.dumps(data, ensure_ascii=False))\n"
    )
    result = _run_command([str(python_path), "-c", probe_code], cwd=root, timeout_sec=timeout_sec)
    if int(result.get("returncode") or 0) != 0:
        payload["error"] = str(result.get("stderr") or result.get("stdout") or "probe_failed").strip()
        payload["health_status"] = "failed"
        return payload
    try:
        probe = json.loads(str(result.get("stdout") or "").strip())
    except Exception as exc:
        payload["error"] = f"invalid_probe_output:{exc!r}"
        payload["health_status"] = "failed"
        return payload

    payload.update(
        {
            "ok": True,
            "python": str(probe.get("python") or python_path),
            "python_version": str(probe.get("python_version") or ""),
            "torch_available": bool(probe.get("torch_available")),
            "torch_version": probe.get("torch_version"),
            "cuda_available": bool(probe.get("cuda_available")),
            "device_count": int(probe.get("device_count") or 0),
            "device_name": probe.get("device_name"),
            "yaml_available": bool(probe.get("yaml_available")),
            "yaml_version": probe.get("yaml_version"),
            "pip_version": probe.get("pip_version"),
            "error": str(probe.get("torch_error") or probe.get("yaml_error") or ""),
        }
    )
    warnings: list[str] = []
    if not bool(payload.get("yaml_available")):
        warnings.append("pyyaml_missing")
    if not bool(payload.get("torch_available")):
        warnings.append("torch_missing")
    if bootstrap_role == "cuda" and not bool(payload.get("cuda_available")):
        warnings.append("cuda_unavailable")
    payload["health_status"] = "ready" if not warnings else ("degraded" if bool(payload.get("ok")) else "failed")
    payload["warnings"] = warnings
    payload["env_type"] = "cuda" if bool(payload.get("cuda_available")) else ("cpu" if bool(payload.get("torch_available")) else "unknown")
    return payload


def detect_nvidia_smi(repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root.resolve() if isinstance(repo_root, Path) else resolve_repo_root()
    result = _run_command(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        cwd=root,
        timeout_sec=30,
    )
    if int(result.get("returncode") or 0) != 0:
        return {"status": "missing", "gpus": [], "error": str(result.get("stderr") or result.get("stdout") or "").strip()}
    gpus: list[dict[str, Any]] = []
    for line in str(result.get("stdout") or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        gpus.append({"name": parts[0], "driver_version": parts[1], "memory_total": parts[2]})
    return {"status": "ok", "gpus": gpus, "error": ""}


def load_bootstrap_state(repo_root: Path | None = None) -> dict[str, Any]:
    path = latest_bootstrap_state_path(repo_root)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def render_bootstrap_state_md(payload: dict[str, Any]) -> str:
    envs = payload.get("envs") if isinstance(payload.get("envs"), dict) else {}
    lines = [
        "# P58 Windows Bootstrap State",
        "",
        f"- generated_at: `{payload.get('generated_at') or ''}`",
        f"- repo_root: `{payload.get('repo_root') or ''}`",
        f"- mode_requested: `{payload.get('mode_requested') or ''}`",
        f"- mode_resolved: `{payload.get('mode_resolved') or ''}`",
        f"- recommended_mode: `{payload.get('recommended_mode') or ''}`",
        f"- bootstrap_complete: `{payload.get('bootstrap_complete')}`",
        f"- selected_training_python: `{payload.get('selected_training_python') or ''}`",
        f"- bootstrap_state_json: `{payload.get('json_path') or ''}`",
        "",
        "## Environment Probes",
        "",
        "| Role | Python | Torch | PyYAML | CUDA | Health |",
        "|---|---|---|---|---|---|",
    ]
    for role in ("cpu", "cuda"):
        row = envs.get(role) if isinstance(envs.get(role), dict) else {}
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
    next_commands = payload.get("next_commands") if isinstance(payload.get("next_commands"), list) else []
    if next_commands:
        lines += ["", "## Recommended Next Commands", ""]
        for command in next_commands:
            lines.append(f"- `{command}`")
    refs = payload.get("refs") if isinstance(payload.get("refs"), dict) else {}
    if refs:
        lines += ["", "## Refs", ""]
        for key, value in refs.items():
            if value:
                lines.append(f"- {key}: `{value}`")
    return "\n".join(lines).rstrip() + "\n"


def build_bootstrap_state(
    *,
    repo_root: Path | None = None,
    run_id: str = "",
    mode_requested: str = "auto",
    mode_resolved: str = "cpu",
    cpu_env: str = "",
    cuda_env: str = "",
    selected_python: str = "",
    force_recreate: bool = False,
    skip_smoke: bool = False,
    config_sync_report_path: str = "",
    smoke_summary_path: str = "",
    notes: list[str] | None = None,
) -> dict[str, Any]:
    root = repo_root.resolve() if isinstance(repo_root, Path) else resolve_repo_root()
    cpu_path = cpu_env or str((root / ".venv_trainer").resolve())
    cuda_path = cuda_env or str((root / ".venv_trainer_cuda").resolve())
    cpu_probe = probe_python_interpreter(cpu_path, repo_root=root, label="bootstrap_cpu")
    cuda_probe = probe_python_interpreter(cuda_path, repo_root=root, label="bootstrap_cuda")
    selected_probe = (
        probe_python_interpreter(selected_python, repo_root=root, label="selected_training_python")
        if str(selected_python or "").strip()
        else (cuda_probe if str(mode_resolved).lower().startswith("cuda") else cpu_probe)
    )
    recommended_mode = "blocked"
    if bool(cuda_probe.get("ok")) and bool(cuda_probe.get("torch_available")) and bool(cuda_probe.get("cuda_available")):
        recommended_mode = "cuda_mainline"
    elif bool(cpu_probe.get("ok")) and bool(cpu_probe.get("torch_available")):
        recommended_mode = "cpu_safe"
    elif bool(selected_probe.get("ok")) and bool(selected_probe.get("torch_available")):
        recommended_mode = "cpu_safe"
    bootstrap_complete = bool(
        selected_probe.get("ok")
        and selected_probe.get("torch_available")
        and selected_probe.get("yaml_available")
        and (recommended_mode != "cuda_mainline" or selected_probe.get("cuda_available"))
    )
    return {
        "schema": "p58_bootstrap_state_v1",
        "generated_at": now_iso(),
        "run_id": str(run_id or f"bootstrap-{now_stamp()}"),
        "repo_root": str(root),
        "mode_requested": str(mode_requested or "auto"),
        "mode_resolved": str(mode_resolved or "cpu"),
        "recommended_mode": recommended_mode,
        "bootstrap_complete": bootstrap_complete,
        "selected_training_python": str(selected_probe.get("python") or ""),
        "selected_env_name": str(selected_probe.get("env_name") or ""),
        "selected_env_source": str(selected_probe.get("label") or ""),
        "force_recreate": bool(force_recreate),
        "skip_smoke": bool(skip_smoke),
        "envs": {"cpu": cpu_probe, "cuda": cuda_probe, "selected": selected_probe},
        "nvidia_smi": detect_nvidia_smi(root),
        "refs": {
            "config_sync_report_path": str(config_sync_report_path or ""),
            "smoke_summary_path": str(smoke_summary_path or ""),
        },
        "notes": [str(item) for item in (notes or []) if str(item).strip()],
        "next_commands": [
            "powershell -ExecutionPolicy Bypass -File scripts\\doctor.ps1",
            "powershell -ExecutionPolicy Bypass -File scripts\\run_p22.ps1 -Quick",
            "powershell -ExecutionPolicy Bypass -File scripts\\run_ops_ui.ps1",
        ],
    }


def write_bootstrap_state(payload: dict[str, Any], *, repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root.resolve() if isinstance(repo_root, Path) else resolve_repo_root()
    out_root = bootstrap_artifacts_root(root)
    run_id = str(payload.get("run_id") or f"bootstrap-{now_stamp()}")
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "bootstrap_state.json"
    md_path = run_dir / "bootstrap_state.md"
    payload = dict(payload)
    payload["json_path"] = str(json_path.resolve())
    payload["md_path"] = str(md_path.resolve())
    write_json(json_path, payload)
    md_text = render_bootstrap_state_md(payload)
    md_path.write_text(md_text, encoding="utf-8")
    write_json(latest_bootstrap_state_path(root), payload)
    latest_bootstrap_md_path(root).write_text(md_text, encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P58 bootstrap environment helpers")
    parser.add_argument("--probe-python", default="")
    parser.add_argument("--write-state", action="store_true")
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--mode-requested", default="auto")
    parser.add_argument("--mode-resolved", default="cpu")
    parser.add_argument("--cpu-env", default="")
    parser.add_argument("--cuda-env", default="")
    parser.add_argument("--selected-python", default="")
    parser.add_argument("--config-sync-report", default="")
    parser.add_argument("--smoke-summary", default="")
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument("--force-recreate", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.repo_root).resolve() if str(args.repo_root).strip() else resolve_repo_root()
    if str(args.probe_python).strip():
        payload = probe_python_interpreter(args.probe_python, repo_root=root, label="cli_probe")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if bool(payload.get("ok")) else 1
    if args.write_state:
        payload = build_bootstrap_state(
            repo_root=root,
            run_id=str(args.run_id or ""),
            mode_requested=str(args.mode_requested or "auto"),
            mode_resolved=str(args.mode_resolved or "cpu"),
            cpu_env=str(args.cpu_env or ""),
            cuda_env=str(args.cuda_env or ""),
            selected_python=str(args.selected_python or ""),
            force_recreate=bool(args.force_recreate),
            skip_smoke=bool(args.skip_smoke),
            config_sync_report_path=str(args.config_sync_report or ""),
            smoke_summary_path=str(args.smoke_summary or ""),
            notes=[str(item) for item in args.note or []],
        )
        payload = write_bootstrap_state(payload, repo_root=root)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if bool(payload.get("bootstrap_complete")) else 1
    payload = load_bootstrap_state(root)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload else 1


if __name__ == "__main__":
    raise SystemExit(main())
