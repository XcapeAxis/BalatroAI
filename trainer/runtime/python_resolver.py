from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from trainer.runtime.bootstrap_env import (
    load_bootstrap_state,
    normalize_python_path,
    now_iso,
    probe_python_interpreter,
    python_from_env_dir,
    resolve_repo_root,
    write_json,
)


def _candidate_repo_roots(root: Path) -> list[Path]:
    roots: list[Path] = [root.resolve()]
    fallback_root = resolve_repo_root().resolve()
    if all(existing.resolve() != fallback_root for existing in roots):
        roots.append(fallback_root)
    return roots


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
        normalized = str(normalize_python_path(text))
    except Exception:
        normalized = str(Path(text))
    key = normalized.lower()
    if key in seen:
        return
    seen.add(key)
    candidates.append({"label": label, "priority": priority, "python": normalized})


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
    root = Path(repo_root).resolve() if repo_root else resolve_repo_root()
    explicit_python = str(explicit_python or os.environ.get("BALATRO_TRAIN_PYTHON") or "").strip()
    explicit_env = str(explicit_env or os.environ.get("BALATRO_TRAIN_ENV") or "").strip()
    bootstrap_roots = _candidate_repo_roots(root)
    bootstrap_sources: list[dict[str, Any]] = []
    for source_root in bootstrap_roots:
        state = load_bootstrap_state(source_root)
        envs = state.get("envs") if isinstance(state.get("envs"), dict) else {}
        selected_python = str(state.get("selected_training_python") or "").strip()
        recommended_mode = str(state.get("recommended_mode") or "").strip()
        bootstrap_sources.append(
            {
                "root": source_root,
                "state": state if isinstance(state, dict) else {},
                "envs": envs,
                "selected_training_python": selected_python,
                "recommended_mode": recommended_mode,
            }
        )
    bootstrap_primary = next(
        (
            source
            for source in bootstrap_sources
            if source.get("selected_training_python") or source.get("envs")
        ),
        (bootstrap_sources[0] if bootstrap_sources else {"root": root, "state": {}, "envs": {}, "selected_training_python": "", "recommended_mode": ""}),
    )
    bootstrap_state = bootstrap_primary.get("state") if isinstance(bootstrap_primary.get("state"), dict) else {}
    bootstrap_envs = bootstrap_primary.get("envs") if isinstance(bootstrap_primary.get("envs"), dict) else {}
    bootstrap_selected = str(bootstrap_primary.get("selected_training_python") or "").strip()
    bootstrap_mode = str(bootstrap_primary.get("recommended_mode") or "").strip()
    bootstrap_source_root = Path(bootstrap_primary.get("root") or root).resolve()

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    _append_candidate(candidates, seen, label="explicit_python", priority="explicit_python", candidate=explicit_python)
    if explicit_env:
        _append_candidate(candidates, seen, label="explicit_env", priority="explicit_env", candidate=python_from_env_dir(explicit_env))
    for source in bootstrap_sources:
        source_root = Path(source.get("root") or root).resolve()
        source_key = "root" if source_root == root else source_root.name
        source_envs = source.get("envs") if isinstance(source.get("envs"), dict) else {}
        source_selected = str(source.get("selected_training_python") or "").strip()
        cpu_bootstrap = (source_envs.get("cpu") or {}).get("python") if isinstance(source_envs.get("cpu"), dict) else ""
        cuda_bootstrap = (source_envs.get("cuda") or {}).get("python") if isinstance(source_envs.get("cuda"), dict) else ""

        _append_candidate(candidates, seen, label=f"bootstrap_selected[{source_key}]", priority="bootstrap_selected", candidate=source_selected)
        if prefer_cuda:
            _append_candidate(candidates, seen, label=f"bootstrap_cuda[{source_key}]", priority="bootstrap_cuda", candidate=cuda_bootstrap)
            _append_candidate(candidates, seen, label=f".venv_trainer_cuda[{source_key}]", priority="venv_cuda", candidate=source_root / ".venv_trainer_cuda" / "Scripts" / "python.exe")
            _append_candidate(candidates, seen, label=f"bootstrap_cpu[{source_key}]", priority="bootstrap_cpu", candidate=cpu_bootstrap)
            _append_candidate(candidates, seen, label=f".venv_trainer[{source_key}]", priority="venv_cpu", candidate=source_root / ".venv_trainer" / "Scripts" / "python.exe")
        else:
            _append_candidate(candidates, seen, label=f"bootstrap_cpu[{source_key}]", priority="bootstrap_cpu", candidate=cpu_bootstrap)
            _append_candidate(candidates, seen, label=f".venv_trainer[{source_key}]", priority="venv_cpu", candidate=source_root / ".venv_trainer" / "Scripts" / "python.exe")
            _append_candidate(candidates, seen, label=f"bootstrap_cuda[{source_key}]", priority="bootstrap_cuda", candidate=cuda_bootstrap)
            _append_candidate(candidates, seen, label=f".venv_trainer_cuda[{source_key}]", priority="venv_cuda", candidate=source_root / ".venv_trainer_cuda" / "Scripts" / "python.exe")
    _append_candidate(candidates, seen, label="current_sys_executable", priority="current", candidate=sys.executable)

    probes = [
        probe_python_interpreter(Path(candidate["python"]), repo_root=root, label=f"{candidate['label']}::{candidate['priority']}", timeout_sec=timeout_sec)
        for candidate in candidates
    ]
    for probe, candidate in zip(probes, candidates):
        probe["label"] = str(candidate["label"])
        probe["priority"] = str(candidate["priority"])
        bootstrap_role = str(probe.get("bootstrap_role") or "").strip().lower()
        bootstrap_probe = bootstrap_envs.get(bootstrap_role) if bootstrap_role in {"cpu", "cuda"} and isinstance(bootstrap_envs.get(bootstrap_role), dict) else {}
        same_python = str(probe.get("python") or "").strip().lower() == str((bootstrap_probe or {}).get("python") or "").strip().lower()
        if (
            same_python
            and bool(probe.get("ok"))
            and not bool(probe.get("torch_available"))
            and bool((bootstrap_probe or {}).get("ok"))
            and bool((bootstrap_probe or {}).get("torch_available"))
        ):
            live_error = str(probe.get("error") or "").strip()
            probe["torch_available"] = bool((bootstrap_probe or {}).get("torch_available"))
            probe["torch_version"] = (bootstrap_probe or {}).get("torch_version")
            probe["cuda_available"] = bool((bootstrap_probe or {}).get("cuda_available"))
            probe["device_count"] = int((bootstrap_probe or {}).get("device_count") or 0)
            probe["device_name"] = (bootstrap_probe or {}).get("device_name")
            probe["env_type"] = str((bootstrap_probe or {}).get("env_type") or probe.get("env_type") or "")
            probe["health_status"] = str((bootstrap_probe or {}).get("health_status") or probe.get("health_status") or "")
            warnings = [str(item) for item in (probe.get("warnings") or []) if str(item).strip() and str(item) != "torch_missing" and str(item) != "cuda_unavailable"]
            warnings.append("torch_probe_fallback_bootstrap_state")
            if live_error:
                warnings.append("live_torch_probe_timeout")
                probe["live_probe_error"] = live_error
            probe["warnings"] = warnings
            probe["error"] = ""

    selected: dict[str, Any] | None = None
    selection_reason = ""
    fallback_reason = ""
    warnings: list[str] = []

    def _first(predicate) -> dict[str, Any] | None:
        for probe in probes:
            if predicate(probe):
                return probe
        return None

    if explicit_python or explicit_env:
        explicit_selection = _first(lambda probe: probe["priority"] in {"explicit_python", "explicit_env"} and bool(probe.get("ok")))
        if explicit_selection is not None:
            if require_cuda and not bool(explicit_selection.get("cuda_available")):
                fallback_reason = "explicit_python_not_cuda_capable"
            else:
                selected = explicit_selection
                selection_reason = str(explicit_selection.get("priority") or "explicit")
        else:
            fallback_reason = "explicit_python_probe_failed"

    if selected is None and prefer_cuda:
        selected = _first(lambda probe: bool(probe.get("ok")) and bool(probe.get("cuda_available")) and bool(probe.get("yaml_available")))
        if selected is not None:
            selection_reason = "preferred_cuda_env_with_yaml"
        if selected is None:
            selected = _first(lambda probe: bool(probe.get("ok")) and bool(probe.get("cuda_available")))
            if selected is not None:
                selection_reason = "preferred_cuda_env"

    if selected is None and allow_cpu_fallback:
        selected = _first(lambda probe: bool(probe.get("ok")) and bool(probe.get("torch_available")) and bool(probe.get("yaml_available")))
        if selected is not None:
            selection_reason = "cpu_fallback_env_with_yaml"
            fallback_reason = fallback_reason or "cuda_env_unavailable"
        if selected is None:
            selected = _first(lambda probe: bool(probe.get("ok")) and bool(probe.get("torch_available")))
            if selected is not None:
                selection_reason = "cpu_fallback_env"
                fallback_reason = fallback_reason or "cuda_env_unavailable"

    if selected is None:
        selected = _first(lambda probe: bool(probe.get("ok")))
        if selected is not None:
            selection_reason = "python_fallback_without_torch"
            fallback_reason = fallback_reason or "torch_unavailable_in_known_envs"

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
            "yaml_available": False,
            "yaml_version": None,
            "env_name": "",
            "env_dir": "",
            "health_status": "failed",
            "warnings": [],
            "error": error,
        }
    elif require_cuda and not bool(selected.get("cuda_available")):
        status = "failed"
        error = "cuda_required_but_not_available"

    if not bool(selected.get("yaml_available")):
        warnings.append("selected_env_missing_pyyaml")
    if not bool(selected.get("torch_available")):
        warnings.append("selected_env_missing_torch")
    if str(bootstrap_mode or ""):
        warnings.append(f"bootstrap_recommended_mode:{bootstrap_mode}")

    return {
        "schema": "p50_python_resolver_v2",
        "generated_at": now_iso(),
        "repo_root": str(root),
        "bootstrap_source_root": str(bootstrap_source_root),
        "requested": {
            "explicit_python": explicit_python,
            "explicit_env": explicit_env,
            "prefer_cuda": bool(prefer_cuda),
            "require_cuda": bool(require_cuda),
            "allow_cpu_fallback": bool(allow_cpu_fallback),
        },
        "bootstrap_state_path": str((bootstrap_source_root / "docs" / "artifacts" / "p58" / "bootstrap" / "latest_bootstrap_state.json").resolve()),
        "bootstrap_recommended_mode": bootstrap_mode,
        "selected": selected,
        "selection_reason": selection_reason,
        "fallback_reason": fallback_reason,
        "status": status,
        "error": error,
        "warnings": warnings,
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
            out_path = (resolve_repo_root() / out_path).resolve()
        write_json(out_path, payload)
    if args.emit == "path":
        print(str((payload.get("selected") or {}).get("python") or ""))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if str(payload.get("status") or "") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
