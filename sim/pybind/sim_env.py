import os
from typing import Any

from sim.core.engine import SimEnv


class SimEnvBackend:
    def __init__(self, seed: str = "AAAAAAA", fail_fast: bool | None = None):
        env_raw = os.getenv("SIM_FAIL_FAST")
        resolved_fail_fast = fail_fast
        if resolved_fail_fast is None and env_raw is not None:
            env_flag = str(env_raw).strip().lower()
            resolved_fail_fast = env_flag in {"1", "true", "yes", "on"}
        self._env = SimEnv(seed=seed, fail_fast=resolved_fail_fast)
        self._seed = seed

    def reset(self, seed: str | None = None, from_snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
        if from_snapshot is not None:
            return self._env.reset(from_snapshot=from_snapshot)
        if seed is None:
            seed = self._seed
        self._seed = seed
        return self._env.reset(seed=seed)

    def get_state(self) -> dict[str, Any]:
        return self._env.get_state()

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        return self._env.step(action)

    def health(self) -> bool:
        return True

    def close(self) -> None:
        return None


def create_backend(seed: str = "AAAAAAA") -> SimEnvBackend:
    return SimEnvBackend(seed=seed)
