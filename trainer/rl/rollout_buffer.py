from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RolloutRecord:
    obs: list[float]
    action: int
    reward: float
    done: bool
    legal_action_ids: list[int]
    logprob: float | None = None
    value: float | None = None
    episode_idx: int = 0
    seed: str = ""


class RolloutBuffer:
    def __init__(self) -> None:
        self.records: list[RolloutRecord] = []
        self.returns: list[float] = []

    def __len__(self) -> int:
        return len(self.records)

    def add(
        self,
        *,
        obs: list[float],
        action: int,
        reward: float,
        done: bool,
        legal_action_ids: list[int],
        logprob: float | None = None,
        value: float | None = None,
        episode_idx: int = 0,
        seed: str = "",
    ) -> None:
        self.records.append(
            RolloutRecord(
                obs=[float(x) for x in obs],
                action=int(action),
                reward=float(reward),
                done=bool(done),
                legal_action_ids=[int(x) for x in legal_action_ids],
                logprob=(float(logprob) if logprob is not None else None),
                value=(float(value) if value is not None else None),
                episode_idx=int(episode_idx),
                seed=str(seed),
            )
        )

    def compute_returns(self, gamma: float = 0.99) -> list[float]:
        g = float(gamma)
        running = 0.0
        out = [0.0] * len(self.records)
        for idx in range(len(self.records) - 1, -1, -1):
            row = self.records[idx]
            if row.done:
                running = 0.0
            running = float(row.reward) + (g * running)
            out[idx] = float(running)
        self.returns = out
        return list(out)

    def to_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for idx, row in enumerate(self.records):
            rows.append(
                {
                    "step_idx": idx,
                    "episode_idx": row.episode_idx,
                    "seed": row.seed,
                    "obs": list(row.obs),
                    "action": int(row.action),
                    "reward": float(row.reward),
                    "done": bool(row.done),
                    "legal_action_ids": list(row.legal_action_ids),
                    "logprob": row.logprob,
                    "value": row.value,
                    "return": self.returns[idx] if idx < len(self.returns) else None,
                }
            )
        return rows

    def to_tensors(self, *, torch_mod, device: Any | None = None) -> dict[str, Any]:
        if not self.records:
            raise ValueError("rollout buffer is empty")
        obs = torch_mod.tensor([row.obs for row in self.records], dtype=torch_mod.float32, device=device)
        actions = torch_mod.tensor([row.action for row in self.records], dtype=torch_mod.long, device=device)
        rewards = torch_mod.tensor([row.reward for row in self.records], dtype=torch_mod.float32, device=device)
        dones = torch_mod.tensor([1.0 if row.done else 0.0 for row in self.records], dtype=torch_mod.float32, device=device)
        returns = (
            torch_mod.tensor(self.returns, dtype=torch_mod.float32, device=device)
            if self.returns
            else rewards.clone()
        )
        logprobs = torch_mod.tensor(
            [float(row.logprob) if row.logprob is not None else 0.0 for row in self.records],
            dtype=torch_mod.float32,
            device=device,
        )
        values = torch_mod.tensor(
            [float(row.value) if row.value is not None else 0.0 for row in self.records],
            dtype=torch_mod.float32,
            device=device,
        )
        return {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "returns": returns,
            "logprobs": logprobs,
            "values": values,
        }

