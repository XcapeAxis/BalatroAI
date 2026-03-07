from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from trainer.runtime.python_resolver import resolve_training_python


PRESET_PROFILES: dict[str, dict[str, Any]] = {
    "gpu_debug_small": {
        "device_profile": "single_gpu_mainline",
        "runtime": {
            "batch_size": 32,
            "micro_batch_size": 16,
            "grad_accum_steps": 2,
            "num_dataloader_workers": 1,
            "pin_memory": True,
            "amp_enabled": True,
            "bf16_enabled": False,
            "max_gpu_memory_mb": 6144,
            "oom_fallback_policy": "cpu_fallback",
        },
    },
    "single_gpu_mainline": {
        "device_profile": "single_gpu_mainline",
        "runtime": {
            "batch_size": 128,
            "micro_batch_size": 128,
            "grad_accum_steps": 1,
            "num_dataloader_workers": 2,
            "pin_memory": True,
            "amp_enabled": True,
            "bf16_enabled": True,
            "max_gpu_memory_mb": 11264,
            "oom_fallback_policy": "reduce_batch",
        },
    },
    "single_gpu_nightly_balanced": {
        "device_profile": "single_gpu_mainline",
        "runtime": {
            "batch_size": 192,
            "micro_batch_size": 96,
            "grad_accum_steps": 2,
            "num_dataloader_workers": 3,
            "pin_memory": True,
            "amp_enabled": True,
            "bf16_enabled": True,
            "max_gpu_memory_mb": 11264,
            "oom_fallback_policy": "reduce_batch",
        },
    },
    "single_gpu_nightly_aggressive": {
        "device_profile": "single_gpu_mainline",
        "runtime": {
            "batch_size": 256,
            "micro_batch_size": 128,
            "grad_accum_steps": 2,
            "num_dataloader_workers": 4,
            "pin_memory": True,
            "amp_enabled": True,
            "bf16_enabled": True,
            "max_gpu_memory_mb": 11776,
            "oom_fallback_policy": "reduce_batch",
        },
    },
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is not None:
            payload = yaml.safe_load(text)
        else:
            sidecar = path.with_suffix(".json")
            if not sidecar.exists():
                raise RuntimeError(f"PyYAML unavailable and no sidecar JSON for: {path}")
            payload = json.loads(sidecar.read_text(encoding="utf-8-sig"))
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"config must be mapping: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _run_command(command: list[str], *, cwd: Path, timeout_sec: int) -> dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=max(30, int(timeout_sec)),
            encoding="utf-8",
            errors="replace",
        )
        return {
            "returncode": int(proc.returncode),
            "stdout": str(proc.stdout or ""),
            "stderr": str(proc.stderr or ""),
            "elapsed_sec": time.time() - started,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": 124,
            "stdout": str(exc.stdout or ""),
            "stderr": str(exc.stderr or ""),
            "elapsed_sec": time.time() - started,
        }


def _load_progress_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as fp:
        for line in fp:
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _latest_progress_with_throughput(path: Path, *, run_id: str, component: str) -> dict[str, Any]:
    rows = [
        row
        for row in _load_progress_rows(path)
        if str(row.get("schema") or "") == "p49_progress_event_v1"
        and str(row.get("run_id") or "") == run_id
        and str(row.get("component") or "") == component
    ]
    with_throughput = [row for row in rows if row.get("throughput") is not None]
    if with_throughput:
        return with_throughput[-1]
    return rows[-1] if rows else {}


def _parse_summary(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _build_component_config(base_config: Path, *, profile_name: str, run_out_dir: Path, component: str) -> Path:
    payload = _read_yaml_or_json(base_config)
    preset = PRESET_PROFILES[profile_name]
    runtime_block = payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {}
    runtime_block = dict(runtime_block)
    runtime_block["device_profile"] = str(preset.get("device_profile") or "single_gpu_mainline")
    runtime_block.update(preset.get("runtime") if isinstance(preset.get("runtime"), dict) else {})
    payload["runtime"] = runtime_block
    payload["device_profile"] = str(preset.get("device_profile") or "single_gpu_mainline")
    if component == "p45":
        arena_compare = payload.get("arena_compare") if isinstance(payload.get("arena_compare"), dict) else {}
        arena_compare = dict(arena_compare)
        arena_compare["enabled"] = False
        payload["arena_compare"] = arena_compare
        planning = payload.get("planning") if isinstance(payload.get("planning"), dict) else {}
        planning = dict(planning)
        planning["enabled"] = False
        payload["planning"] = planning
    tmp_path = run_out_dir / "tmp" / f"{component}_{profile_name}.json"
    _write_json(tmp_path, payload)
    return tmp_path


def _benchmark_component(
    *,
    python_exe: str,
    profile_name: str,
    component: str,
    out_dir: Path,
) -> dict[str, Any]:
    repo_root = _repo_root()
    config_path = (
        repo_root / "configs/experiments/p42_rl_smoke.yaml"
        if component == "p42"
        else repo_root / "configs/experiments/p45_world_model_smoke.yaml"
    )
    run_out_dir = out_dir / "runs" / component / profile_name
    run_out_dir.mkdir(parents=True, exist_ok=True)
    temp_config = _build_component_config(config_path, profile_name=profile_name, run_out_dir=out_dir, component=component)
    run_id = f"{component}-{profile_name}"
    command = [
        python_exe,
        "-B",
        "-m",
        ("trainer.rl.ppo_lite" if component == "p42" else "trainer.world_model.train"),
        "--config",
        str(temp_config),
        "--out-dir",
        str(run_out_dir),
        "--run-id",
        run_id,
        "--quick",
    ]
    result = _run_command(command, cwd=repo_root, timeout_sec=(1800 if component == "p45" else 1200))
    summary = _parse_summary(str(result.get("stdout") or ""))
    progress_path = Path(str(summary.get("progress_unified_jsonl") or ""))
    if not progress_path.is_absolute():
        progress_path = (repo_root / progress_path).resolve()
    event = _latest_progress_with_throughput(
        progress_path,
        run_id=str(summary.get("run_id") or run_id),
        component=("p42_rl" if component == "p42" else "p45_wm"),
    )
    throughput = event.get("throughput")
    gpu_mem_mb = event.get("gpu_mem_mb")
    learner_device = str(event.get("learner_device") or "")
    learner_updates_per_sec = None
    rollout_steps_per_sec = None
    metrics_path = Path(str(summary.get("metrics") or summary.get("metrics_json") or ""))
    if metrics_path and not metrics_path.is_absolute():
        metrics_path = (repo_root / metrics_path).resolve()
    metrics = {}
    if metrics_path and metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
        except Exception:
            metrics = {}
    if component == "p42":
        seed_results = summary.get("seed_results") if isinstance(summary.get("seed_results"), list) else []
        learner_values = []
        rollout_values = []
        for row in seed_results:
            if not isinstance(row, dict):
                continue
            row_metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            learner_values.append(float(row_metrics.get("learner_updates_per_sec") or 0.0))
            rollout_values.append(float(row_metrics.get("rollout_steps_per_sec") or 0.0))
        if learner_values:
            learner_updates_per_sec = sum(learner_values) / max(1, len(learner_values))
        if rollout_values:
            rollout_steps_per_sec = sum(rollout_values) / max(1, len(rollout_values))
        if throughput is None:
            throughput = learner_updates_per_sec
    elif component == "p45":
        learner_updates_per_sec = throughput
    return {
        "component": component,
        "profile_name": profile_name,
        "status": ("ok" if int(result.get("returncode") or 0) == 0 else "failed"),
        "returncode": int(result.get("returncode") or 0),
        "elapsed_sec": float(result.get("elapsed_sec") or 0.0),
        "throughput": throughput,
        "learner_updates_per_sec": learner_updates_per_sec,
        "rollout_steps_per_sec": rollout_steps_per_sec,
        "gpu_mem_mb": gpu_mem_mb,
        "learner_device": learner_device,
        "oom_restart_count": (metrics.get("oom_restart_count") if isinstance(metrics, dict) else None),
        "summary": summary,
        "metrics": metrics,
        "stdout_tail": str(result.get("stdout") or "").splitlines()[-8:],
        "stderr_tail": str(result.get("stderr") or "").splitlines()[-8:],
    }


def _recommend_profiles(rows: list[dict[str, Any]]) -> dict[str, Any]:
    recommendations: dict[str, Any] = {}
    by_profile: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_profile.setdefault(str(row.get("profile_name") or ""), []).append(row)
    for profile_name, profile_rows in by_profile.items():
        both_ok = len(profile_rows) >= 2 and all(str(row.get("status") or "") == "ok" and str(row.get("learner_device") or "").startswith("cuda") for row in profile_rows)
        if both_ok:
            recommendations[profile_name] = {
                "profile_name": profile_name,
                "preset": PRESET_PROFILES[profile_name],
                "mean_throughput": sum(float(row.get("throughput") or 0.0) for row in profile_rows) / max(1, len(profile_rows)),
                "max_gpu_mem_mb": max(float(row.get("gpu_mem_mb") or 0.0) for row in profile_rows),
            }
    return recommendations


def run_benchmark(*, out_dir: Path, profiles: list[str]) -> dict[str, Any]:
    resolver = resolve_training_python(repo_root=_repo_root(), require_cuda=True)
    selected = resolver.get("selected") if isinstance(resolver.get("selected"), dict) else {}
    python_exe = str(selected.get("python") or "").strip()
    if not python_exe or not bool(selected.get("cuda_available")):
        raise RuntimeError("CUDA training python unavailable for benchmark")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for profile_name in profiles:
        for component in ("p42", "p45"):
            rows.append(_benchmark_component(python_exe=python_exe, profile_name=profile_name, component=component, out_dir=out_dir))

    csv_path = out_dir / "benchmark_matrix.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "component",
                "profile_name",
                "status",
                "returncode",
                "elapsed_sec",
                "throughput",
                "learner_updates_per_sec",
                "rollout_steps_per_sec",
                "gpu_mem_mb",
                "learner_device",
                "oom_restart_count",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "component": row.get("component"),
                    "profile_name": row.get("profile_name"),
                    "status": row.get("status"),
                    "returncode": row.get("returncode"),
                    "elapsed_sec": row.get("elapsed_sec"),
                    "throughput": row.get("throughput"),
                    "learner_updates_per_sec": row.get("learner_updates_per_sec"),
                    "rollout_steps_per_sec": row.get("rollout_steps_per_sec"),
                    "gpu_mem_mb": row.get("gpu_mem_mb"),
                    "learner_device": row.get("learner_device"),
                    "oom_restart_count": row.get("oom_restart_count"),
                }
            )

    recommendations = _recommend_profiles(rows)
    summary = {
        "schema": "p50_gpu_benchmark_summary_v1",
        "generated_at": _now_iso(),
        "resolver": resolver,
        "profiles": profiles,
        "rows": rows,
        "recommended_profiles": recommendations,
        "benchmark_matrix_csv": str(csv_path.resolve()),
    }
    _write_json(out_dir / "benchmark_summary.json", summary)
    _write_json(out_dir / "recommended_profiles.json", recommendations)
    lines = [
        "# P50 GPU Benchmark Summary",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Training python: `{python_exe}`",
        "",
        "| component | profile | status | throughput | gpu_mem_mb | learner_device |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {component} | {profile_name} | {status} | {throughput} | {gpu_mem_mb} | {learner_device} |".format(
                component=row.get("component"),
                profile_name=row.get("profile_name"),
                status=row.get("status"),
                throughput=("%.4f" % float(row.get("throughput") or 0.0)),
                gpu_mem_mb=("%.1f" % float(row.get("gpu_mem_mb") or 0.0)),
                learner_device=row.get("learner_device"),
            )
        )
    _write_markdown(out_dir / "benchmark_summary.md", lines)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run small CUDA benchmark sweeps for P42/P45 training profiles.")
    parser.add_argument("--out-dir", default="docs/artifacts/p50/benchmarks/latest")
    parser.add_argument("--profiles", default="gpu_debug_small,single_gpu_mainline,single_gpu_nightly_balanced,single_gpu_nightly_aggressive")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (_repo_root() / out_dir).resolve()
    profiles = [token.strip() for token in str(args.profiles or "").split(",") if token.strip()]
    if not profiles:
        profiles = list(PRESET_PROFILES.keys())
    summary = run_benchmark(out_dir=out_dir, profiles=profiles)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
