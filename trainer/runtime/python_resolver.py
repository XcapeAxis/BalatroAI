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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _python_from_env_path(env_path: str | Path) -> Path:
    candidate = Path(env_path)
    if candidate.is_file():
        return candidate.resolve()
    if os.name == "nt":
        return (candidate / "Scripts" / "python.exe").resolve()
    return (candidate / "bin" / "python").resolve()


def _normalize_python_path(value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.suffix.lower() == ".exe" or candidate.name.lower().startswith("python"):
        return candidate.resolve()
    return _python_from_env_path(candidate)


def _probe_python(candidate: Path, *, label: str, priority: str, timeout_sec: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "label": label,
        "priority": priority,
        "candidate": str(candidate),
        "python": str(candidate),
        "exists": candidate.exists(),
        "ok": False,
        "torch_available": False,
        "torch_version": None,
        "cuda_available": False,
        "device_count": 0,
        "device_name": None,
        "env_type": "missing" if not candidate.exists() else "unknown",
        "error": "",
    }
    if not candidate.exists():
        payload["error"] = "python_not_found"
        return payload

    probe_code = (
        "import json, sys\n"
        "data = {'python': sys.executable, 'python_version': sys.version.replace('\\n', ' ').strip()}\n"
        "try:\n"
        "    import torch\n"
        "    cuda = bool(torch.cuda.is_available())\n"
        "    data.update({\n"
        "        'torch_available': True,\n"
        "        'torch_version': str(torch.__version__),\n"
        "        'cuda_available': cuda,\n"
        "        'device_count': int(torch.cuda.device_count()) if cuda else 0,\n"
        "        'device_name': str(torch.cuda.get_device_name(0)) if cuda and int(torch.cuda.device_count()) > 0 else None,\n"
        "    })\n"
        "except Exception as exc:\n"
        "    data.update({'torch_available': False, 'torch_version': None, 'cuda_available': False, 'device_count': 0, 'device_name': None, 'torch_error': repr(exc)})\n"
        "print(json.dumps(data))\n"
    )
    try:
        proc = subprocess.run(
            [str(candidate), "-c", probe_code],
            cwd=str(_repo_root()),
            text=True,
            capture_output=True,
            timeout=max(5, int(timeout_sec)),
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        payload["error"] = "probe_timeout"
        return payload
    except Exception as exc:
        payload["error"] = repr(exc)
        return payload

    if int(proc.returncode) != 0:
        payload["error"] = (proc.stderr or proc.stdout or f"probe_returncode:{proc.returncode}").strip()
        return payload

    try:
        result = json.loads((proc.stdout or "").strip())
    except Exception as exc:
        payload["error"] = f"invalid_probe_output:{exc!r}"
        return payload

    payload.update(
        {
            "python": str(result.get("python") or candidate),
            "python_version": str(result.get("python_version") or ""),
            "ok": True,
            "torch_available": bool(result.get("torch_available")),
            "torch_version": result.get("torch_version"),
            "cuda_available": bool(result.get("cuda_available")),
            "device_count": int(result.get("device_count") or 0),
            "device_name": result.get("device_name"),
            "env_type": "cuda" if bool(result.get("cuda_available")) else ("cpu" if bool(result.get("torch_available")) else "unknown"),
            "error": str(result.get("torch_error") or ""),
        }
    )
    return payload


def _append_candidate(
    candidates: list[dict[str, Any]],
    seen: set[str],
    *,
    label: str,
    priority: str,
    candidate: str | Path | None,
) -> None:
    if candidate is None:
        return
    text = str(candidate).strip()
    if not text:
        return
    try:
        normalized = str(_normalize_python_path(text))
    except Exception:
        normalized = str(Path(text))
    key = normalized.lower()
    if key in seen:
        return
    seen.add(key)
    candidates.append(
        {
            "label": label,
            "priority": priority,
            "python": normalized,
        }
    )


def resolve_training_python(
    *,
    repo_root: str | Path | None = None,
    explicit_python: str = "",
    explicit_env: str = "",
    prefer_cuda: bool = True,
    require_cuda: bool = False,
    allow_cpu_fallback: bool = True,
    timeout_sec: int = 60,
) -> dict[str, Any]:
    root = Path(repo_root).resolve() if repo_root else _repo_root()
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    explicit_python = str(explicit_python or os.environ.get("BALATRO_TRAIN_PYTHON") or "").strip()
    explicit_env = str(explicit_env or os.environ.get("BALATRO_TRAIN_ENV") or "").strip()

    _append_candidate(candidates, seen, label="explicit_python", priority="explicit_python", candidate=explicit_python)
    if explicit_env:
        _append_candidate(
            candidates,
            seen,
            label="explicit_env",
            priority="explicit_env",
            candidate=_python_from_env_path(explicit_env),
        )
    _append_candidate(
        candidates,
        seen,
        label=".venv_trainer_cuda",
        priority="venv_cuda",
        candidate=root / ".venv_trainer_cuda" / "Scripts" / "python.exe",
    )
    _append_candidate(
        candidates,
        seen,
        label=".venv_trainer",
        priority="venv_cpu",
        candidate=root / ".venv_trainer" / "Scripts" / "python.exe",
    )
    _append_candidate(candidates, seen, label="current_sys_executable", priority="current", candidate=sys.executable)

    probes = [
        _probe_python(Path(candidate["python"]), label=str(candidate["label"]), priority=str(candidate["priority"]), timeout_sec=timeout_sec)
        for candidate in candidates
    ]

    selected: dict[str, Any] | None = None
    selection_reason = ""
    fallback_reason = ""

    def _first(predicate) -> dict[str, Any] | None:
        for probe in probes:
            if predicate(probe):
                return probe
        return None

    if explicit_python or explicit_env:
        explicit_selection = _first(lambda probe: probe["priority"] in {"explicit_python", "explicit_env"} and bool(probe["ok"]))
        if explicit_selection is not None:
            if require_cuda and not bool(explicit_selection.get("cuda_available")):
                fallback_reason = "explicit_python_not_cuda_capable"
            else:
                selected = explicit_selection
                selection_reason = str(explicit_selection.get("priority") or "explicit")
        else:
            fallback_reason = "explicit_python_probe_failed"

    if selected is None and prefer_cuda:
        selected = _first(lambda probe: bool(probe.get("ok")) and bool(probe.get("cuda_available")))
        if selected is not None:
            selection_reason = "preferred_cuda_env"

    if selected is None and allow_cpu_fallback:
        selected = _first(lambda probe: bool(probe.get("ok")) and bool(probe.get("torch_available")))
        if selected is not None:
            selection_reason = "cpu_fallback_env"
            if not fallback_reason:
                fallback_reason = "cuda_env_unavailable"

    if selected is None:
        selected = _first(lambda probe: bool(probe.get("ok")))
        if selected is not None:
            selection_reason = "python_fallback_without_torch"
            if not fallback_reason:
                fallback_reason = "torch_unavailable_in_known_envs"

    status = "ok"
    error = ""
    if selected is None:
        status = "failed"
        error = "no_viable_python_found"
        selected = {
            "label": "",
            "priority": "",
            "python": "",
            "ok": False,
            "torch_available": False,
            "torch_version": None,
            "cuda_available": False,
            "device_count": 0,
            "device_name": None,
            "env_type": "missing",
            "error": error,
        }
    elif require_cuda and not bool(selected.get("cuda_available")):
        status = "failed"
        error = "cuda_required_but_not_available"

    return {
        "schema": "p50_python_resolver_v1",
        "generated_at": _now_iso(),
        "repo_root": str(root),
        "requested": {
            "explicit_python": explicit_python,
            "explicit_env": explicit_env,
            "prefer_cuda": bool(prefer_cuda),
            "require_cuda": bool(require_cuda),
            "allow_cpu_fallback": bool(allow_cpu_fallback),
        },
        "selected": selected,
        "selection_reason": selection_reason,
        "fallback_reason": fallback_reason,
        "status": status,
        "error": error,
        "candidates": probes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve the preferred training python for CUDA-first workloads.")
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--explicit-python", default="")
    parser.add_argument("--explicit-env", default="")
    parser.add_argument("--timeout-sec", type=int, default=60)
    parser.add_argument("--no-prefer-cuda", action="store_true")
    parser.add_argument("--no-cpu-fallback", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--emit", choices=("json", "path"), default="json")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = resolve_training_python(
        repo_root=args.repo_root or None,
        explicit_python=args.explicit_python,
        explicit_env=args.explicit_env,
        prefer_cuda=not bool(args.no_prefer_cuda),
        require_cuda=bool(args.require_cuda),
        allow_cpu_fallback=not bool(args.no_cpu_fallback),
        timeout_sec=max(5, int(args.timeout_sec)),
    )
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = (_repo_root() / out_path).resolve()
        _write_json(out_path, payload)
    if args.emit == "path":
        print(str((payload.get("selected") or {}).get("python") or ""))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if str(payload.get("status") or "") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
