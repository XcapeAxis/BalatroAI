from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import math
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from trainer.closed_loop.replay_manifest import build_seeds_payload
from trainer.rl.checkpointing import save_torch_checkpoint, write_manifest
from trainer.rl.ppo_config import PPOConfig
from trainer.rl.rollout_collector import RolloutPolicy, run_rollout_collection


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is not None:
            obj = yaml.safe_load(text)
        else:
            try:
                obj = json.loads(text)
            except Exception:
                sidecar = path.with_suffix(".json")
                if sidecar.exists():
                    obj = json.loads(sidecar.read_text(encoding="utf-8-sig"))
                else:
                    raise RuntimeError(f"PyYAML unavailable and no sidecar JSON for: {path}")
    else:
        obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError(f"config must be mapping: {path}")
    return obj


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
        from torch.distributions import Categorical
    except Exception as exc:  # pragma: no cover - runtime guarded
        raise RuntimeError("PyTorch is required for trainer.rl.ppo_lite") from exc
    return torch, F, Categorical


def _compute_gae(
    *,
    rewards: list[float],
    dones: list[bool],
    values: list[float],
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    n = len(rewards)
    if n <= 0:
        return [], []
    adv = [0.0] * n
    last_adv = 0.0
    next_value = 0.0
    for i in range(n - 1, -1, -1):
        not_done = 0.0 if bool(dones[i]) else 1.0
        delta = float(rewards[i]) + float(gamma) * next_value * not_done - float(values[i])
        last_adv = delta + float(gamma) * float(gae_lambda) * not_done * last_adv
        adv[i] = float(last_adv)
        next_value = float(values[i])
    returns = [float(a + v) for a, v in zip(adv, values)]
    return adv, returns


def _build_masks_tensor(
    *,
    torch_mod,
    legal_action_ids: list[list[int]],
    action_dim: int,
    device,
):
    masks = torch_mod.zeros((len(legal_action_ids), int(action_dim)), dtype=torch_mod.float32, device=device)
    for row_idx, ids in enumerate(legal_action_ids):
        if not ids:
            masks[row_idx, 0] = 1.0
            continue
        for aid in ids:
            if 0 <= int(aid) < int(action_dim):
                masks[row_idx, int(aid)] = 1.0
    return masks


def _masked_logits(logits, masks, torch_mod):
    neg = torch_mod.full_like(logits, -1e9)
    return torch_mod.where(masks > 0.0, logits, neg)


class TorchMaskedRolloutPolicy(RolloutPolicy):
    def __init__(self, *, model: Any, torch_mod, categorical_cls, device, seed: str) -> None:
        self.model = model
        self.torch = torch_mod
        self.Categorical = categorical_cls
        self.device = device
        self.rng = random.Random(str(seed))

    def act(
        self,
        *,
        obs: dict[str, Any],
        action_mask: list[int],
        info: dict[str, Any],
    ) -> tuple[int, float | None, float | None, dict[str, Any] | None]:
        requested_legal_ids = [idx for idx, flag in enumerate(action_mask) if int(flag) > 0]
        if not requested_legal_ids:
            return 0, 0.0, 0.0, {"policy": "ppo_lite", "fallback": "no_legal_actions"}
        with self.torch.no_grad():
            x = self.torch.tensor([list(obs.get("vector") or [])], dtype=self.torch.float32, device=self.device)
            logits, values = self.model(x)
            action_dim = int(logits.shape[-1])
            legal_ids = [aid for aid in requested_legal_ids if 0 <= int(aid) < action_dim]
            if not legal_ids:
                return 0, 0.0, 0.0, {"policy": "ppo_lite", "fallback": "no_legal_actions_in_head"}
            mask_tensor = self.torch.zeros_like(logits)
            mask_tensor.fill_(-1e9)
            for aid in legal_ids:
                mask_tensor[0, int(aid)] = 0.0
            dist = self.Categorical(logits=logits + mask_tensor)
            action = int(dist.sample().item())
            logp = float(dist.log_prob(self.torch.tensor(action, device=self.device)).item())
            value_pred = float(values.squeeze(-1).item() if values.ndim > 1 else values.item())
        return action, logp, value_pred, {"policy": "ppo_lite"}


def _train_one_seed(
    *,
    seed: str,
    seed_index: int,
    seed_total: int,
    cfg: PPOConfig,
    run_dir: Path,
    progress_path: Path,
    warnings_log_path: Path,
) -> dict[str, Any]:
    torch, F, Categorical = _require_torch()
    from trainer.rl.policy_value_model import PolicyValueModel

    seed_int = int.from_bytes(str(seed).encode("utf-8"), "little", signed=False) % (2**31 - 1)
    random.seed(seed_int)
    torch.manual_seed(seed_int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe_dim = 48
    probe_action_dim = 128
    model = PolicyValueModel(obs_dim=probe_dim, action_dim=probe_action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.train.lr))

    seed_dir = run_dir / "seed_runs" / f"seed_{seed_index:03d}_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = seed_dir / "best.pt"
    last_ckpt = seed_dir / "last.pt"
    rollout_root = seed_dir / "rollout_updates"

    update_rewards: list[float] = []
    update_invalid_rates: list[float] = []
    update_losses: list[float] = []
    update_kls: list[float] = []
    best_reward = -1e18
    status = "ok"
    warn_lines: list[str] = []

    for update_idx in range(1, int(cfg.train.max_updates) + 1):
        rollout_policy = TorchMaskedRolloutPolicy(
            model=model,
            torch_mod=torch,
            categorical_cls=Categorical,
            device=device,
            seed=f"{seed}-update-{update_idx}",
        )
        rollout_summary = run_rollout_collection(
            policy=rollout_policy,
            policy_id="ppo_lite",
            seeds=[str(seed)],
            episodes_per_seed=int(cfg.rollout.episodes_per_seed),
            max_steps_per_episode=int(cfg.rollout.max_steps_per_episode),
            total_steps_cap=int(cfg.rollout.total_steps_cap),
            early_stop_invalid_rate=float(cfg.rollout.early_stop_invalid_rate),
            min_steps_before_early_stop=int(cfg.rollout.min_steps_before_early_stop),
            run_id=f"{seed}-u{update_idx:03d}",
            out_dir=rollout_root / f"update_{update_idx:03d}",
            backend=str(cfg.env.backend),
            reward_config=cfg.env.reward,
            include_steps_in_result=True,
        )
        steps = rollout_summary.get("steps") if isinstance(rollout_summary.get("steps"), list) else []
        if not steps:
            status = "stub"
            warn_lines.append(f"update_{update_idx}:empty_rollout_steps")
            break

        obs_list = [list(row.get("obs_vector") or []) for row in steps if isinstance(row, dict)]
        action_list = [_safe_int(row.get("action"), 0) for row in steps if isinstance(row, dict)]
        reward_list = [_safe_float(row.get("reward"), 0.0) for row in steps if isinstance(row, dict)]
        done_list = [
            bool(row.get("terminated") or row.get("truncated"))
            for row in steps
            if isinstance(row, dict)
        ]
        old_logprob_list = [_safe_float(row.get("action_logprob"), 0.0) for row in steps if isinstance(row, dict)]
        old_value_list = [_safe_float(row.get("value_pred"), 0.0) for row in steps if isinstance(row, dict)]
        legal_ids_list = [
            [int(x) for x in (row.get("legal_action_ids") if isinstance(row.get("legal_action_ids"), list) else [])]
            for row in steps
            if isinstance(row, dict)
        ]
        if not obs_list:
            status = "stub"
            warn_lines.append(f"update_{update_idx}:empty_obs_list")
            break

        obs_dim = len(obs_list[0])
        action_dim = max(
            model.action_dim,
            max([max(ids) + 1 for ids in legal_ids_list if ids] + [1]),
        )
        if obs_dim != model.obs_dim or action_dim != model.action_dim:
            model = PolicyValueModel(obs_dim=obs_dim, action_dim=action_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.train.lr))

        obs_t = torch.tensor(obs_list, dtype=torch.float32, device=device)
        actions_t = torch.tensor(action_list, dtype=torch.long, device=device)
        rewards_t = torch.tensor(reward_list, dtype=torch.float32, device=device)
        old_logprob_t = torch.tensor(old_logprob_list, dtype=torch.float32, device=device)
        old_values_t = torch.tensor(old_value_list, dtype=torch.float32, device=device)
        masks_t = _build_masks_tensor(
            torch_mod=torch,
            legal_action_ids=legal_ids_list,
            action_dim=int(model.action_dim),
            device=device,
        )

        advantages, returns = _compute_gae(
            rewards=reward_list,
            dones=done_list,
            values=old_value_list,
            gamma=float(cfg.train.gamma),
            gae_lambda=float(cfg.train.gae_lambda),
        )
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=device)
        if bool(cfg.train.normalize_advantage) and adv_t.numel() > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-6)

        n = obs_t.shape[0]
        batch_size = max(1, min(int(cfg.train.minibatch_size), n))
        last_total_loss = 0.0
        last_kl = 0.0
        for _epoch in range(int(cfg.train.ppo_epochs)):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                logits, values = model(obs_t[idx])
                masked = _masked_logits(logits, masks_t[idx], torch)
                dist = Categorical(logits=masked)
                new_logprob = dist.log_prob(actions_t[idx])
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_logprob - old_logprob_t[idx])
                unclipped = ratio * adv_t[idx]
                clipped = torch.clamp(ratio, 1.0 - float(cfg.train.clip_range), 1.0 + float(cfg.train.clip_range)) * adv_t[idx]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(values, ret_t[idx])
                total_loss = policy_loss + float(cfg.train.value_coef) * value_loss - float(cfg.train.entropy_coef) * entropy
                if bool(cfg.train.nan_fail_fast) and (not torch.isfinite(total_loss)):
                    status = "failed"
                    warn_lines.append(f"update_{update_idx}:nan_loss")
                    break
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                if float(cfg.train.grad_clip_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip_norm))
                optimizer.step()
                approx_kl = torch.mean(old_logprob_t[idx] - new_logprob).item()
                last_total_loss = float(total_loss.item())
                last_kl = float(approx_kl)
            if status == "failed":
                break
        if status == "failed":
            break

        step_count = int(len(steps))
        invalid_count = sum(1 for row in steps if isinstance(row, dict) and bool(row.get("invalid_action")))
        invalid_rate = (float(invalid_count) / float(step_count)) if step_count > 0 else 0.0
        mean_reward = float(statistics.mean(reward_list)) if reward_list else 0.0
        update_rewards.append(mean_reward)
        update_invalid_rates.append(invalid_rate)
        update_losses.append(last_total_loss)
        update_kls.append(last_kl)
        if invalid_rate > float(cfg.train.invalid_action_warn_threshold):
            warn_lines.append(
                f"update_{update_idx}:invalid_action_rate={invalid_rate:.4f}>threshold={float(cfg.train.invalid_action_warn_threshold):.4f}"
            )
        if abs(last_kl) > float(cfg.train.max_kl_warn):
            warn_lines.append(f"update_{update_idx}:approx_kl={last_kl:.4f}>max_kl_warn={float(cfg.train.max_kl_warn):.4f}")

        checkpoint_payload = {
            "schema": "p42_rl_checkpoint_v1",
            "generated_at": _now_iso(),
            "seed": str(seed),
            "update_index": int(update_idx),
            "model": model.snapshot(),
            "optimizer_state": optimizer.state_dict(),
            "train_cfg": cfg.to_dict(),
            "metrics": {
                "mean_reward": mean_reward,
                "invalid_action_rate": invalid_rate,
                "loss": last_total_loss,
                "approx_kl": last_kl,
            },
        }
        save_torch_checkpoint(last_ckpt, checkpoint_payload)
        if mean_reward >= best_reward:
            best_reward = mean_reward
            save_torch_checkpoint(best_ckpt, checkpoint_payload)

        _append_jsonl(
            progress_path,
            {
                "schema": "p42_rl_progress_v1",
                "ts": _now_iso(),
                "seed": str(seed),
                "seed_index": int(seed_index),
                "seed_total": int(seed_total),
                "update": int(update_idx),
                "status": status,
                "mean_reward": mean_reward,
                "invalid_action_rate": invalid_rate,
                "loss": last_total_loss,
                "approx_kl": last_kl,
                "rollout_steps": step_count,
                "rollout_dir": str(rollout_root / f"update_{update_idx:03d}"),
            },
        )

    if warn_lines:
        with warnings_log_path.open("a", encoding="utf-8", newline="\n") as fp:
            for line in warn_lines:
                fp.write(line + "\n")

    metrics = {
        "schema": "p42_rl_seed_metrics_v1",
        "generated_at": _now_iso(),
        "seed": str(seed),
        "status": status,
        "updates_completed": len(update_rewards),
        "mean_reward": float(statistics.mean(update_rewards)) if update_rewards else 0.0,
        "mean_invalid_action_rate": float(statistics.mean(update_invalid_rates)) if update_invalid_rates else 0.0,
        "mean_loss": float(statistics.mean(update_losses)) if update_losses else 0.0,
        "mean_approx_kl": float(statistics.mean(update_kls)) if update_kls else 0.0,
        "best_update_reward": float(max(update_rewards)) if update_rewards else 0.0,
        "best_checkpoint": str(best_ckpt if best_ckpt.exists() else ""),
        "last_checkpoint": str(last_ckpt if last_ckpt.exists() else ""),
    }
    _write_json(seed_dir / "metrics.json", metrics)
    return {
        "seed": str(seed),
        "status": status,
        "run_dir": str(seed_dir),
        "metrics": metrics,
        "best_checkpoint": str(best_ckpt if best_ckpt.exists() else ""),
        "last_checkpoint": str(last_ckpt if last_ckpt.exists() else ""),
    }


def run_ppo_lite_training(
    *,
    config_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
    out_dir: str | Path | None = None,
    run_id: str = "",
    quick: bool = False,
    dry_run: bool = False,
    seeds_override: list[str] | None = None,
) -> dict[str, Any]:
    cfg_raw: dict[str, Any]
    if isinstance(config, dict):
        cfg_raw = dict(config)
    elif config_path is not None:
        cfg_raw = _read_yaml_or_json(Path(config_path))
    else:
        cfg_raw = {}
    cfg = PPOConfig.from_mapping(cfg_raw)
    if quick:
        cfg.rollout.episodes_per_seed = min(cfg.rollout.episodes_per_seed, 1)
        cfg.rollout.max_steps_per_episode = min(cfg.rollout.max_steps_per_episode, 80)
        cfg.rollout.total_steps_cap = min(cfg.rollout.total_steps_cap, 400)
        cfg.train.max_updates = min(cfg.train.max_updates, 2)
        cfg.train.ppo_epochs = min(cfg.train.ppo_epochs, 1)
        cfg.train.minibatch_size = min(cfg.train.minibatch_size, 64)

    seeds = [str(s).strip() for s in (seeds_override if seeds_override else cfg.seeds) if str(s).strip()]
    if not seeds:
        seeds = ["AAAAAAA", "BBBBBBB"]
    if quick and len(seeds) > 2:
        seeds = seeds[:2]

    run_token = str(run_id or cfg.run_id or _now_stamp())
    run_dir = Path(out_dir).resolve() if out_dir else (Path(cfg.output_artifacts_root).resolve() / run_token)
    run_dir.mkdir(parents=True, exist_ok=True)

    seeds_payload = build_seeds_payload(seeds, seed_policy_version="p42.rl_ppo_lite")
    _write_json(run_dir / "seeds_used.json", seeds_payload)
    _write_json(run_dir / "reward_config.json", cfg.env.reward if isinstance(cfg.env.reward, dict) else {})
    progress_path = run_dir / "progress.jsonl"
    warnings_log_path = run_dir / "warnings.log"

    if dry_run:
        manifest = {
            "schema": "p42_rl_train_manifest_v1",
            "generated_at": _now_iso(),
            "run_id": run_token,
            "status": "dry_run",
            "run_dir": str(run_dir),
            "config": cfg.to_dict(),
            "seed_results": [],
            "best_checkpoint": "",
        }
        write_manifest(run_dir / "train_manifest.json", manifest)
        _write_json(
            run_dir / "metrics.json",
            {
                "schema": "p42_rl_train_metrics_v1",
                "generated_at": _now_iso(),
                "run_id": run_token,
                "status": "dry_run",
                "seed_count": len(seeds),
                "ok_seed_count": 0,
            },
        )
        (run_dir / "best_checkpoint.txt").write_text("\n", encoding="utf-8")
        return {
            "status": "dry_run",
            "run_id": run_token,
            "run_dir": str(run_dir),
            "train_manifest": str(run_dir / "train_manifest.json"),
            "metrics": str(run_dir / "metrics.json"),
            "best_checkpoint": "",
            "seeds_used": str(run_dir / "seeds_used.json"),
            "reward_config": str(run_dir / "reward_config.json"),
            "warnings_log": str(warnings_log_path),
        }

    try:
        _require_torch()
    except Exception as exc:
        stub_checkpoint = run_dir / "best_checkpoint_stub.json"
        _write_json(
            stub_checkpoint,
            {
                "schema": "p42_rl_checkpoint_stub_v1",
                "generated_at": _now_iso(),
                "run_id": run_token,
                "reason": "torch_missing",
                "note": "fallback artifact for non-torch environments",
                "config": cfg.to_dict(),
            },
        )
        with warnings_log_path.open("a", encoding="utf-8", newline="\n") as fp:
            fp.write(f"torch_missing:{exc}\n")
        manifest = {
            "schema": "p42_rl_train_manifest_v1",
            "generated_at": _now_iso(),
            "run_id": run_token,
            "status": "stub",
            "run_dir": str(run_dir),
            "config": cfg.to_dict(),
            "seed_results": [],
            "best_checkpoint": str(stub_checkpoint),
            "reason": "torch_missing",
        }
        write_manifest(run_dir / "train_manifest.json", manifest)
        _write_json(
            run_dir / "metrics.json",
            {
                "schema": "p42_rl_train_metrics_v1",
                "generated_at": _now_iso(),
                "run_id": run_token,
                "status": "stub",
                "seed_count": len(seeds),
                "ok_seed_count": 0,
                "reason": "torch_missing",
                "candidate_checkpoint": str(stub_checkpoint),
            },
        )
        (run_dir / "best_checkpoint.txt").write_text(str(stub_checkpoint) + "\n", encoding="utf-8")
        return {
            "status": "stub",
            "run_id": run_token,
            "run_dir": str(run_dir),
            "train_manifest": str(run_dir / "train_manifest.json"),
            "metrics": str(run_dir / "metrics.json"),
            "best_checkpoint": str(stub_checkpoint),
            "seeds_used": str(run_dir / "seeds_used.json"),
            "reward_config": str(run_dir / "reward_config.json"),
            "warnings_log": str(warnings_log_path),
        }

    seed_results: list[dict[str, Any]] = []
    for idx, seed in enumerate(seeds, start=1):
        seed_result = _train_one_seed(
            seed=seed,
            seed_index=idx,
            seed_total=len(seeds),
            cfg=cfg,
            run_dir=run_dir,
            progress_path=progress_path,
            warnings_log_path=warnings_log_path,
        )
        seed_results.append(seed_result)

    ok_rows = [row for row in seed_results if str(row.get("status")) == "ok"]
    if ok_rows:
        best_row = sorted(ok_rows, key=lambda row: float(((row.get("metrics") or {}).get("best_update_reward") or 0.0)), reverse=True)[0]
        best_checkpoint = str(best_row.get("best_checkpoint") or "")
        status = "ok"
    else:
        best_row = seed_results[0] if seed_results else {}
        best_checkpoint = str(best_row.get("best_checkpoint") or "")
        status = "stub" if seed_results else "failed"

    metrics_payload = {
        "schema": "p42_rl_train_metrics_v1",
        "generated_at": _now_iso(),
        "run_id": run_token,
        "status": status,
        "seed_count": len(seeds),
        "ok_seed_count": len(ok_rows),
        "mean_reward": (
            float(statistics.mean([float(((r.get("metrics") or {}).get("mean_reward") or 0.0)) for r in ok_rows]))
            if ok_rows
            else 0.0
        ),
        "invalid_action_rate": (
            float(statistics.mean([float(((r.get("metrics") or {}).get("mean_invalid_action_rate") or 0.0)) for r in ok_rows]))
            if ok_rows
            else 0.0
        ),
        "final_loss": (
            float(statistics.mean([float(((r.get("metrics") or {}).get("mean_loss") or 0.0)) for r in ok_rows]))
            if ok_rows
            else 0.0
        ),
        "candidate_checkpoint": best_checkpoint,
    }
    manifest = {
        "schema": "p42_rl_train_manifest_v1",
        "generated_at": _now_iso(),
        "run_id": run_token,
        "status": status,
        "run_dir": str(run_dir),
        "config": cfg.to_dict(),
        "seed_results": seed_results,
        "best_checkpoint": best_checkpoint,
        "paths": {
            "metrics": str(run_dir / "metrics.json"),
            "progress_jsonl": str(progress_path),
            "warnings_log": str(warnings_log_path),
            "seeds_used": str(run_dir / "seeds_used.json"),
            "reward_config": str(run_dir / "reward_config.json"),
        },
    }
    _write_json(run_dir / "metrics.json", metrics_payload)
    write_manifest(run_dir / "train_manifest.json", manifest)
    (run_dir / "best_checkpoint.txt").write_text(str(best_checkpoint).strip() + "\n", encoding="utf-8")

    return {
        "status": status,
        "run_id": run_token,
        "run_dir": str(run_dir),
        "train_manifest": str(run_dir / "train_manifest.json"),
        "metrics": str(run_dir / "metrics.json"),
        "best_checkpoint": str(best_checkpoint),
        "seeds_used": str(run_dir / "seeds_used.json"),
        "reward_config": str(run_dir / "reward_config.json"),
        "warnings_log": str(warnings_log_path),
        "seed_results": seed_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P42 PPO-lite RL trainer.")
    parser.add_argument("--config", default="configs/experiments/p42_rl_smoke.yaml")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--seeds", default="", help="Optional comma-separated seed override")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_override = [s.strip() for s in str(args.seeds).split(",") if s.strip()]
    summary = run_ppo_lite_training(
        config_path=args.config,
        out_dir=(args.out_dir if str(args.out_dir).strip() else None),
        run_id=str(args.run_id or ""),
        quick=bool(args.quick),
        dry_run=bool(args.dry_run),
        seeds_override=seed_override or None,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if str(summary.get("status")) in {"ok", "stub", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
