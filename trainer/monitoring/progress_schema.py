from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_torch():
    try:
        import torch
    except Exception:
        return None
    return torch


def get_gpu_mem_mb(device: Any = None) -> float | None:
    torch = _require_torch()
    if torch is None or not bool(torch.cuda.is_available()):
        return None
    try:
        if device is None:
            idx = 0
        else:
            token = str(device)
            idx = int(token.split(":")[1]) if ":" in token else 0
        samples = [
            float(torch.cuda.memory_allocated(idx)),
            float(torch.cuda.memory_reserved(idx)),
        ]
        try:
            samples.append(float(torch.cuda.max_memory_allocated(idx)))
            samples.append(float(torch.cuda.max_memory_reserved(idx)))
        except Exception:
            pass
        return float(max(samples) / (1024.0 * 1024.0))
    except Exception:
        return None


def build_progress_event(
    *,
    run_id: str,
    component: str,
    phase: str,
    status: str,
    step: int | str | None = None,
    epoch_or_iter: int | str | None = None,
    seed: str | None = None,
    metrics: dict[str, Any] | None = None,
    device_profile: dict[str, Any] | str | None = None,
    learner_device: str = "",
    rollout_device: str = "",
    throughput: float | None = None,
    eta_sec: float | None = None,
    warning: str = "",
    gpu_mem_mb: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "p49_progress_event_v1",
        "timestamp": _now_iso(),
        "run_id": str(run_id),
        "component": str(component),
        "phase": str(phase),
        "step": step,
        "epoch_or_iter": epoch_or_iter,
        "seed": seed,
        "metrics": metrics or {},
        "device_profile": device_profile,
        "learner_device": str(learner_device or ""),
        "rollout_device": str(rollout_device or ""),
        "gpu_mem_mb": gpu_mem_mb,
        "throughput": throughput,
        "eta_sec": eta_sec,
        "warning": str(warning or ""),
        "status": str(status),
    }
    if extra:
        payload.update(extra)
    return payload


def append_progress_event(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
