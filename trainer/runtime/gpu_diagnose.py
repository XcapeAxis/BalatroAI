from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trainer.runtime.python_resolver import resolve_training_python


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
        }
    except Exception as exc:
        return {
            "returncode": 1,
            "stdout": "",
            "stderr": repr(exc),
        }


def _probe_profile(selected_python: str, *, profile_name: str) -> dict[str, Any]:
    code = (
        "import json, sys\n"
        "from trainer.runtime.device_profile import load_device_profile\n"
        "payload = load_device_profile(profile_name=sys.argv[1])\n"
        "print(json.dumps(payload, ensure_ascii=False))\n"
    )
    result = _run_command([selected_python, "-c", code, profile_name], cwd=_repo_root())
    if int(result.get("returncode") or 0) != 0:
        return {"status": "failed", "error": str(result.get("stderr") or result.get("stdout") or "").strip()}
    try:
        return {
            "status": "ok",
            "payload": json.loads(str(result.get("stdout") or "").strip()),
        }
    except Exception as exc:
        return {"status": "failed", "error": f"invalid_profile_payload:{exc!r}"}


def _query_nvidia_smi() -> dict[str, Any]:
    result = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        cwd=_repo_root(),
        timeout_sec=30,
    )
    if int(result.get("returncode") or 0) != 0:
        return {
            "status": "failed",
            "error": str(result.get("stderr") or result.get("stdout") or "").strip(),
        }
    rows: list[dict[str, Any]] = []
    for line in str(result.get("stdout") or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        rows.append(
            {
                "name": parts[0],
                "driver_version": parts[1],
                "memory_total_mb": int(parts[2]),
                "memory_used_mb": int(parts[3]),
                "utilization_gpu_pct": int(parts[4]),
            }
        )
    return {
        "status": "ok",
        "gpus": rows,
    }


def diagnose_gpu(
    *,
    profile_name: str = "single_gpu_mainline",
    explicit_python: str = "",
    explicit_env: str = "",
    require_cuda: bool = False,
) -> dict[str, Any]:
    resolver = resolve_training_python(
        repo_root=_repo_root(),
        explicit_python=explicit_python,
        explicit_env=explicit_env,
        require_cuda=require_cuda,
    )
    selected = resolver.get("selected") if isinstance(resolver.get("selected"), dict) else {}
    selected_python = str(selected.get("python") or "").strip()
    profile_probe = _probe_profile(selected_python, profile_name=profile_name) if selected_python else {"status": "failed", "error": "no_selected_python"}
    nvidia = _query_nvidia_smi()

    hints: list[str] = []
    if str(resolver.get("status") or "") != "ok":
        hints.append(f"resolver_failed:{resolver.get('error') or resolver.get('fallback_reason') or 'unknown'}")
    if not bool(selected.get("torch_available")):
        hints.append("selected_python_missing_torch")
    if not bool(selected.get("cuda_available")):
        hints.append("selected_python_is_not_cuda_enabled")
    if str(selected.get("label") or "") != ".venv_trainer_cuda":
        hints.append("resolver_did_not_choose_cuda_env")
    if str(profile_probe.get("status") or "") != "ok":
        hints.append(f"profile_probe_failed:{profile_probe.get('error') or 'unknown'}")
    else:
        payload = profile_probe.get("payload") if isinstance(profile_probe.get("payload"), dict) else {}
        resolved = payload.get("resolved") if isinstance(payload.get("resolved"), dict) else {}
        learner_device = str(resolved.get("learner_device") or "")
        if bool(selected.get("cuda_available")) and not learner_device.startswith("cuda"):
            hints.append("profile_resolved_learner_device_is_not_cuda")
    if str(nvidia.get("status") or "") != "ok":
        hints.append("nvidia_smi_unavailable")

    return {
        "schema": "p50_gpu_diagnose_v1",
        "generated_at": _now_iso(),
        "repo_root": str(_repo_root()),
        "profile_name": profile_name,
        "resolver": resolver,
        "selected_python": selected_python,
        "selected_env_type": str(selected.get("env_type") or ""),
        "profile_probe": profile_probe,
        "nvidia_smi": nvidia,
        "hints": hints,
        "status": "ok" if not hints or (bool(selected.get("cuda_available")) and str(profile_probe.get("status") or "") == "ok") else "warning",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose the active CUDA training environment and runtime profile.")
    parser.add_argument("--profile", default="single_gpu_mainline")
    parser.add_argument("--explicit-python", default="")
    parser.add_argument("--explicit-env", default="")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = diagnose_gpu(
        profile_name=str(args.profile or "single_gpu_mainline"),
        explicit_python=str(args.explicit_python or ""),
        explicit_env=str(args.explicit_env or ""),
        require_cuda=bool(args.require_cuda),
    )
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = (_repo_root() / out_path).resolve()
        _write_json(out_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if bool(((payload.get("resolver") or {}).get("selected") or {}).get("cuda_available")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
