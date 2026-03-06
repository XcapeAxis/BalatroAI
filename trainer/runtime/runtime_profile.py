from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trainer.runtime.device_profile import load_device_profile


@dataclass(frozen=True)
class RuntimeProfile:
    component: str
    profile_name: str
    resolved_profile: dict[str, Any]
    config_overrides: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "profile_name": self.profile_name,
            "resolved_profile": self.resolved_profile,
            "config_overrides": self.config_overrides,
        }


def _extract_runtime_block(config: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    payload = config if isinstance(config, dict) else {}
    runtime_block = payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {}
    profile_name = str(
        runtime_block.get("device_profile")
        or payload.get("device_profile")
        or payload.get("runtime_profile")
        or "single_gpu_mainline"
    )
    overrides = dict(runtime_block)
    overrides.pop("device_profile", None)
    return profile_name, overrides


def load_runtime_profile(
    *,
    config: dict[str, Any] | None = None,
    component: str,
    profile_name: str = "",
    profile_config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> RuntimeProfile:
    config_profile_name, config_overrides = _extract_runtime_block(config)
    merged_overrides = {}
    merged_overrides.update(config_overrides)
    if overrides:
        merged_overrides.update({key: value for key, value in overrides.items() if value is not None})
    chosen_profile = str(profile_name or config_profile_name or "single_gpu_mainline")
    resolved = load_device_profile(
        profile_name=chosen_profile,
        config_path=profile_config_path,
        overrides=merged_overrides,
    )
    return RuntimeProfile(
        component=str(component),
        profile_name=chosen_profile,
        resolved_profile=resolved,
        config_overrides=merged_overrides,
    )
