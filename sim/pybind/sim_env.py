from typing import Any

from sim.core.engine import SimEnv


class SimEnvBackend:
    def __init__(self, seed: str = "AAAAAAA"):
        self._env = SimEnv(seed=seed)
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
