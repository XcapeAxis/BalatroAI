from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trainer.models.rl_policy import RLPolicy
from trainer.models.rl_value import RLValue
from trainer.rl.env import BalatroEnv
from trainer.rl.selfplay import run_selfplay


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
        from torch.distributions import Categorical
    except Exception as exc:
        raise RuntimeError("PyTorch is required for trainer.rl.ppo_skeleton") from exc
    return torch, F, Categorical


class TorchRolloutPolicy:
    def __init__(self, *, policy_model, value_model, torch_mod, categorical_cls, device) -> None:
        self.policy_model = policy_model
        self.value_model = value_model
        self.torch = torch_mod
        self.Categorical = categorical_cls
        self.device = device

    def act(self, obs: dict[str, Any]) -> tuple[int, float | None, float | None]:
        legal_ids = [int(x) for x in (obs.get("legal_action_ids") or [])]
        if not legal_ids:
            return 0, None, None
        x = self.torch.tensor([list(obs.get("vector") or [])], dtype=self.torch.float32, device=self.device)
        with self.torch.no_grad():
            logits = self.policy_model(x)
            value = self.value_model(x)

        masked = logits.clone()
        mask = self.torch.full_like(masked, -1e9)
        for aid in legal_ids:
            if 0 <= aid < masked.shape[-1]:
                mask[0, aid] = 0.0
        masked = masked + mask
        dist = self.Categorical(logits=masked)
        action = int(dist.sample().item())
        logprob = float(dist.log_prob(self.torch.tensor(action, device=self.device)).item())
        value_f = float(value.squeeze(-1).item() if value.ndim > 1 else value.item())
        return action, logprob, value_f


def run_ppo_skeleton(
    *,
    episodes: int,
    seed: str,
    gamma: float = 0.99,
    lr: float = 1e-3,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    backend: str = "sim",
    max_steps_per_episode: int = 320,
    reward_mode: str = "score_delta",
    out_root: str | Path = "docs/artifacts/p37/selfplay",
    run_id: str = "",
    out_dir: str | Path | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    torch, F, Categorical = _require_torch()
    seed_token = str(seed or "AAAAAAA")
    random.seed(seed_token)
    torch.manual_seed(int.from_bytes(seed_token.encode("utf-8"), "little", signed=False) % (2**31 - 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = BalatroEnv(
        backend=backend,
        seed=seed_token,
        reward_mode=reward_mode,
        max_steps_per_episode=max_steps_per_episode,
        auto_advance=True,
    )
    policy = RLPolicy(obs_dim=env.obs_dim, action_dim=env.action_dim).to(device)
    value = RLValue(obs_dim=env.obs_dim).to(device)
    actor = TorchRolloutPolicy(
        policy_model=policy,
        value_model=value,
        torch_mod=torch,
        categorical_cls=Categorical,
        device=device,
    )

    run_token = str(run_id or _now_stamp())
    try:
        selfplay = run_selfplay(
            policy=actor,
            env=env,
            episodes=max(1, int(episodes)),
            seed=seed_token,
            gamma=float(gamma),
            run_id=run_token,
            out_root=out_root,
            out_dir=out_dir,
            max_steps_per_episode=max_steps_per_episode,
        )
    finally:
        env.close()

    buffer = selfplay["buffer"]
    if len(buffer) <= 0:
        raise RuntimeError("selfplay produced an empty rollout buffer")
    tensors = buffer.to_tensors(torch_mod=torch, device=device)
    obs = tensors["obs"]
    actions = tensors["actions"]
    returns = tensors["returns"]

    logits = policy(obs)
    log_probs = torch.log_softmax(logits, dim=-1)
    chosen_logp = log_probs.gather(dim=1, index=actions.unsqueeze(-1)).squeeze(-1)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()

    values = value(obs)
    advantages = (returns - values.detach())
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)

    policy_loss = -(chosen_logp * advantages).mean()
    value_loss = F.mse_loss(values, returns)
    total_loss = policy_loss + (float(value_coef) * value_loss) - (float(entropy_coef) * entropy)

    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value.parameters()),
        lr=float(lr),
    )
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    optimizer.step()

    summary = selfplay.get("summary") if isinstance(selfplay.get("summary"), dict) else {}
    episode_rows = selfplay.get("episode_rows") if isinstance(selfplay.get("episode_rows"), list) else []
    avg_reward = float(summary.get("avg_reward") or 0.0)
    std_reward = float(summary.get("std_reward") or 0.0)
    best_reward = float(summary.get("best_episode_reward") or 0.0)
    avg_episode_length = float(summary.get("avg_episode_length") or 0.0)

    run_dir = Path(str(selfplay.get("run_dir"))).resolve()
    metrics = {
        "schema": "p37_ppo_skeleton_metrics_v1",
        "status": "ok",
        "generated_at": _now_iso(),
        "algo": "ppo_skeleton_pg_step",
        "seed": seed_token,
        "episodes": int(episodes),
        "steps": int(len(buffer)),
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "best_episode_reward": best_reward,
        "episode_length": avg_episode_length,
        "loss": float(total_loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.item()),
        "run_dir": str(run_dir),
    }

    _write_json(run_dir / "metrics.json", metrics)
    _write_json(
        run_dir / "model_manifest.json",
        {
            "schema": "p37_ppo_skeleton_manifest_v1",
            "generated_at": _now_iso(),
            "policy_state_dict": str(run_dir / "policy.pt"),
            "value_state_dict": str(run_dir / "value.pt"),
        },
    )
    torch.save({"state_dict": policy.state_dict()}, run_dir / "policy.pt")
    torch.save({"state_dict": value.state_dict()}, run_dir / "value.pt")

    result = {
        "status": "ok",
        "run_dir": str(run_dir),
        "metrics": metrics,
        "summary": summary,
        "episodes": episode_rows,
    }
    if not quiet:
        print(json.dumps(result, ensure_ascii=False))
    return result


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="P37 PPO skeleton (single update, self-play loop).")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", default="AAAAAAA")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--backend", choices=["sim", "real"], default="sim")
    p.add_argument("--max-steps-per-episode", type=int, default=320)
    p.add_argument("--reward-mode", choices=["score_delta", "episode_total_score"], default="score_delta")
    p.add_argument("--out-root", default="docs/artifacts/p37/selfplay")
    p.add_argument("--run-id", default="")
    p.add_argument("--out-dir", default="")
    p.add_argument("--quiet", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    result = run_ppo_skeleton(
        episodes=max(1, int(args.episodes)),
        seed=str(args.seed),
        gamma=float(args.gamma),
        lr=float(args.lr),
        entropy_coef=float(args.entropy_coef),
        value_coef=float(args.value_coef),
        backend=str(args.backend),
        max_steps_per_episode=max(1, int(args.max_steps_per_episode)),
        reward_mode=str(args.reward_mode),
        out_root=str(args.out_root),
        run_id=str(args.run_id),
        out_dir=(str(args.out_dir) if str(args.out_dir).strip() else None),
        quiet=bool(args.quiet),
    )
    if args.quiet:
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

