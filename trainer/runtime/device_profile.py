from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency in local env
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


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
        raise ValueError(f"profile config must be mapping: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _maybe_torch():
    try:
        import torch
    except Exception:
        return None
    return torch


def detect_torch_environment() -> dict[str, Any]:
    torch = _maybe_torch()
    if torch is None:
        return {
            "torch_available": False,
            "torch_version": None,
            "cuda_available": False,
            "device_count": 0,
            "device_name": None,
            "bf16_supported": False,
            "matmul_precision": None,
        }
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    device_name = None
    if cuda_available and device_count > 0:
        try:
            device_name = str(torch.cuda.get_device_name(0))
        except Exception:
            device_name = None
    bf16_supported = False
    if cuda_available:
        try:
            bf16_supported = bool(torch.cuda.is_bf16_supported())
        except Exception:
            bf16_supported = False
    try:
        matmul_precision = str(torch.get_float32_matmul_precision())
    except Exception:
        matmul_precision = None
    return {
        "torch_available": True,
        "torch_version": str(torch.__version__),
        "cuda_available": cuda_available,
        "device_count": device_count,
        "device_name": device_name,
        "bf16_supported": bf16_supported,
        "matmul_precision": matmul_precision,
    }


@dataclass(frozen=True)
class DeviceProfile:
    profile_name: str = "single_gpu_mainline"
    device: str = "auto"
    gpu_id: int = 0
    amp_enabled: bool = True
    bf16_enabled: bool = False
    grad_accum_steps: int = 1
    batch_size: int = 128
    micro_batch_size: int = 128
    num_dataloader_workers: int = 2
    pin_memory: bool = True
    prefetch_factor: int = 2
    rollout_device: str = "cpu"
    learner_device: str = "cuda"
    max_gpu_memory_mb: int = 0
    oom_fallback_policy: str = "reduce_batch"
    torch_compile_enabled: bool = False
    matmul_precision: str = "high"
    checkpoint_interval: int = 1
    log_interval: int = 1

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "DeviceProfile":
        payload = raw if isinstance(raw, dict) else {}
        return cls(
            profile_name=str(payload.get("profile_name") or payload.get("name") or "single_gpu_mainline"),
            device=str(payload.get("device") or "auto"),
            gpu_id=max(0, _safe_int(payload.get("gpu_id"), 0)),
            amp_enabled=_safe_bool(payload.get("amp_enabled"), True),
            bf16_enabled=_safe_bool(payload.get("bf16_enabled"), False),
            grad_accum_steps=max(1, _safe_int(payload.get("grad_accum_steps"), 1)),
            batch_size=max(1, _safe_int(payload.get("batch_size"), 128)),
            micro_batch_size=max(1, _safe_int(payload.get("micro_batch_size"), _safe_int(payload.get("batch_size"), 128))),
            num_dataloader_workers=max(0, _safe_int(payload.get("num_dataloader_workers"), 2)),
            pin_memory=_safe_bool(payload.get("pin_memory"), True),
            prefetch_factor=max(1, _safe_int(payload.get("prefetch_factor"), 2)),
            rollout_device=str(payload.get("rollout_device") or "cpu"),
            learner_device=str(payload.get("learner_device") or "cuda"),
            max_gpu_memory_mb=max(0, _safe_int(payload.get("max_gpu_memory_mb"), 0)),
            oom_fallback_policy=str(payload.get("oom_fallback_policy") or "reduce_batch"),
            torch_compile_enabled=_safe_bool(payload.get("torch_compile_enabled"), False),
            matmul_precision=str(payload.get("matmul_precision") or "high"),
            checkpoint_interval=max(1, _safe_int(payload.get("checkpoint_interval"), 1)),
            log_interval=max(1, _safe_int(payload.get("log_interval"), 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_device_token(token: str, *, env: dict[str, Any], gpu_id: int) -> tuple[str, list[str]]:
    requested = str(token or "auto").strip().lower() or "auto"
    warnings: list[str] = []
    if requested == "auto":
        requested = "cuda" if bool(env.get("cuda_available")) else "cpu"
    if requested == "cuda":
        if not bool(env.get("cuda_available")):
            warnings.append("cuda_requested_but_unavailable")
            return "cpu", warnings
        if int(env.get("device_count") or 0) <= int(gpu_id):
            warnings.append(f"gpu_id_out_of_range:{gpu_id}")
            return "cpu", warnings
        return f"cuda:{gpu_id}", warnings
    if requested.startswith("cuda:"):
        if not bool(env.get("cuda_available")):
            warnings.append("cuda_requested_but_unavailable")
            return "cpu", warnings
        return requested, warnings
    return "cpu", warnings


def load_device_profile(
    *,
    profile_name: str = "single_gpu_mainline",
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repo_root = _repo_root()
    defaults_path = repo_root / "configs" / "runtime" / "runtime_defaults.yaml"
    profiles_path = repo_root / "configs" / "runtime" / "device_profiles.yaml"
    if config_path is not None:
        profiles_path = Path(config_path)
        if not profiles_path.is_absolute():
            profiles_path = (repo_root / profiles_path).resolve()
    defaults_payload = _read_yaml_or_json(defaults_path)
    profiles_payload = _read_yaml_or_json(profiles_path)
    profile_block = profiles_payload.get("profiles") if isinstance(profiles_payload.get("profiles"), dict) else {}
    if str(profile_name) not in profile_block:
        raise KeyError(f"unknown device profile: {profile_name}")

    merged = {}
    merged.update(defaults_payload.get("defaults") if isinstance(defaults_payload.get("defaults"), dict) else {})
    merged.update(profile_block.get(str(profile_name)) if isinstance(profile_block.get(str(profile_name)), dict) else {})
    if overrides:
        merged.update({key: value for key, value in overrides.items() if value is not None})
    merged["profile_name"] = str(profile_name)

    profile = DeviceProfile.from_mapping(merged)
    env = detect_torch_environment()
    learner_device, learner_warnings = _resolve_device_token(profile.learner_device or profile.device, env=env, gpu_id=profile.gpu_id)
    rollout_device, rollout_warnings = _resolve_device_token(profile.rollout_device, env=env, gpu_id=profile.gpu_id)
    use_cuda = learner_device.startswith("cuda")
    amp_enabled = bool(profile.amp_enabled and use_cuda)
    bf16_enabled = bool(profile.bf16_enabled and use_cuda and bool(env.get("bf16_supported")))
    warnings = [*learner_warnings, *rollout_warnings]
    if bool(profile.bf16_enabled) and not bf16_enabled:
        warnings.append("bf16_requested_but_unavailable")

    return {
        "profile_name": profile.profile_name,
        "requested": profile.to_dict(),
        "resolved": {
            **profile.to_dict(),
            "device": learner_device if profile.device != "cpu" else "cpu",
            "learner_device": learner_device,
            "rollout_device": rollout_device,
            "amp_enabled": amp_enabled,
            "bf16_enabled": bf16_enabled,
            "pin_memory": bool(profile.pin_memory and use_cuda),
            "torch_compile_enabled": bool(profile.torch_compile_enabled and use_cuda),
            "matmul_precision": str(profile.matmul_precision or env.get("matmul_precision") or "high"),
        },
        "environment": env,
        "warnings": warnings,
        "config_path": str(profiles_path),
        "defaults_path": str(defaults_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve and export a runtime device profile.")
    parser.add_argument("--profile", default="single_gpu_mainline")
    parser.add_argument("--config", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--gpu-id", type=int, default=-1)
    parser.add_argument("--amp", default="")
    parser.add_argument("--bf16", default="")
    parser.add_argument("--batch-size", type=int, default=-1)
    parser.add_argument("--micro-batch-size", type=int, default=-1)
    parser.add_argument("--grad-accum-steps", type=int, default=-1)
    parser.add_argument("--rollout-device", default="")
    parser.add_argument("--learner-device", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    overrides: dict[str, Any] = {}
    if str(args.device).strip():
        overrides["device"] = str(args.device).strip()
    if int(args.gpu_id) >= 0:
        overrides["gpu_id"] = int(args.gpu_id)
    if str(args.amp).strip():
        overrides["amp_enabled"] = _safe_bool(args.amp, True)
    if str(args.bf16).strip():
        overrides["bf16_enabled"] = _safe_bool(args.bf16, False)
    if int(args.batch_size) > 0:
        overrides["batch_size"] = int(args.batch_size)
    if int(args.micro_batch_size) > 0:
        overrides["micro_batch_size"] = int(args.micro_batch_size)
    if int(args.grad_accum_steps) > 0:
        overrides["grad_accum_steps"] = int(args.grad_accum_steps)
    if str(args.rollout_device).strip():
        overrides["rollout_device"] = str(args.rollout_device).strip()
    if str(args.learner_device).strip():
        overrides["learner_device"] = str(args.learner_device).strip()
    payload = load_device_profile(
        profile_name=str(args.profile or "single_gpu_mainline"),
        config_path=(str(args.config).strip() or None),
        overrides=overrides,
    )
    out_path = Path(str(args.out).strip()) if str(args.out).strip() else (_repo_root() / "docs" / "artifacts" / "p49" / f"device_profile_{_now_stamp()}.json").resolve()
    if not out_path.is_absolute():
        out_path = (_repo_root() / out_path).resolve()
    _write_json(out_path, payload)
    print(json.dumps({"profile": payload["resolved"]["profile_name"], "out": str(out_path), "learner_device": payload["resolved"]["learner_device"], "rollout_device": payload["resolved"]["rollout_device"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
