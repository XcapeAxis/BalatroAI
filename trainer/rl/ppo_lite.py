from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import hashlib
import json
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
from trainer.rl.curriculum_rl import load_curriculum_scheduler
from trainer.rl.diagnostics import run_diagnostics
from trainer.rl.distributed_rollout import run_distributed_rollout
from trainer.rl.eval_multi_seed import run_multi_seed_evaluation
from trainer.rl.ppo_config import PPOConfig


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(repo_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is not None:
            payload = yaml.safe_load(text)
        else:
            try:
                payload = json.loads(text)
            except Exception:
                sidecar = path.with_suffix(".json")
                if sidecar.exists():
                    payload = json.loads(sidecar.read_text(encoding="utf-8-sig"))
                else:
                    raise RuntimeError(f"PyYAML unavailable and no sidecar JSON for: {path}")
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"config must be mapping: {path}")
    return payload


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


def _cpu_model_snapshot(model: Any) -> dict[str, Any]:
    snapshot = model.snapshot()
    try:
        torch, _, _ = _require_torch()
    except Exception:
        return snapshot
    state_dict = snapshot.get("state_dict")
    if isinstance(state_dict, dict):
        snapshot["state_dict"] = {
            key: value.detach().cpu() if hasattr(value, "detach") else value
            for key, value in state_dict.items()
        }
    return snapshot


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
    for idx in range(n - 1, -1, -1):
        not_done = 0.0 if bool(dones[idx]) else 1.0
        delta = float(rewards[idx]) + float(gamma) * next_value * not_done - float(values[idx])
        last_adv = delta + float(gamma) * float(gae_lambda) * not_done * last_adv
        adv[idx] = float(last_adv)
        next_value = float(values[idx])
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
        for action_id in ids:
            if 0 <= int(action_id) < int(action_dim):
                masks[row_idx, int(action_id)] = 1.0
    return masks


def _masked_logits(logits, masks, torch_mod):
    neg = torch_mod.full_like(logits, -1e9)
    return torch_mod.where(masks > 0.0, logits, neg)


def _resolve_env_dimensions(cfg: PPOConfig) -> tuple[int, int]:
    from trainer.rl.env_adapter import RLEnvAdapter

    adapter = RLEnvAdapter(
        backend=str(cfg.env.backend),
        seed=str(cfg.seeds[0] if cfg.seeds else "AAAAAAA"),
        timeout_sec=float(cfg.env.timeout_sec),
        max_steps_per_episode=max(1, int(cfg.env.max_steps_per_episode)),
        max_auto_steps=max(1, int(cfg.env.max_auto_steps)),
        max_ante=max(0, int(cfg.env.max_ante)),
        auto_advance=bool(cfg.env.auto_advance),
        reward_config=cfg.env.reward,
    )
    try:
        return int(adapter.obs_dim), int(adapter.action_dim)
    finally:
        adapter.close()


def _ppo_update_from_steps(
    *,
    model: Any,
    optimizer: Any,
    steps: list[dict[str, Any]],
    cfg: PPOConfig,
    device: Any,
) -> tuple[str, dict[str, Any]]:
    torch, F, Categorical = _require_torch()

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
        return "stub", {}

    obs_t = torch.tensor(obs_list, dtype=torch.float32, device=device)
    actions_t = torch.tensor(action_list, dtype=torch.long, device=device)
    old_logprob_t = torch.tensor(old_logprob_list, dtype=torch.float32, device=device)

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

    masks_t = _build_masks_tensor(
        torch_mod=torch,
        legal_action_ids=legal_ids_list,
        action_dim=int(model.action_dim),
        device=device,
    )

    n = obs_t.shape[0]
    batch_size = max(1, min(int(cfg.train.minibatch_size), n))
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    approx_kls: list[float] = []

    for _epoch in range(int(cfg.train.ppo_epochs)):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            logits, values = model(obs_t[idx])
            values = values.squeeze(-1)
            masked = _masked_logits(logits, masks_t[idx], torch)
            dist = Categorical(logits=masked)
            new_logprob = dist.log_prob(actions_t[idx])
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_logprob - old_logprob_t[idx])
            unclipped = ratio * adv_t[idx]
            clipped = torch.clamp(
                ratio,
                1.0 - float(cfg.train.clip_range),
                1.0 + float(cfg.train.clip_range),
            ) * adv_t[idx]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, ret_t[idx])
            total_loss = (
                policy_loss
                + float(cfg.train.value_coef) * value_loss
                - float(cfg.train.entropy_coef) * entropy
            )
            if bool(cfg.train.nan_fail_fast) and (not torch.isfinite(total_loss)):
                return "failed", {"reason": "nan_loss"}
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if float(cfg.train.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip_norm))
            optimizer.step()
            approx_kl = torch.mean(old_logprob_t[idx] - new_logprob).item()
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.item()))
            approx_kls.append(float(approx_kl))

    return "ok", {
        "policy_loss": float(statistics.mean(policy_losses)) if policy_losses else 0.0,
        "value_loss": float(statistics.mean(value_losses)) if value_losses else 0.0,
        "entropy": float(statistics.mean(entropies)) if entropies else 0.0,
        "kl_divergence": float(statistics.mean(approx_kls)) if approx_kls else 0.0,
        "return_mean": float(statistics.mean(returns)) if returns else 0.0,
    }


def _train_one_seed(
    *,
    seed: str,
    seed_index: int,
    seed_total: int,
    cfg: PPOConfig,
    cfg_raw: dict[str, Any],
    scheduler: Any,
    run_dir: Path,
    progress_path: Path,
    warnings_log_path: Path,
    curriculum_applied_path: Path,
) -> dict[str, Any]:
    torch, _, _ = _require_torch()
    from trainer.rl.policy_value_model import PolicyValueModel

    seed_int = int(hashlib.sha256(str(seed).encode("utf-8")).hexdigest()[:8], 16)
    torch.manual_seed(seed_int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim, action_dim = _resolve_env_dimensions(cfg)
    model = PolicyValueModel(obs_dim=obs_dim, action_dim=action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.train.lr))

    seed_dir = run_dir / "seed_runs" / f"seed_{seed_index:03d}_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = seed_dir / "best.pt"
    last_ckpt = seed_dir / "last.pt"
    rollout_root = seed_dir / "rollout_updates"

    rewards_hist: list[float] = []
    scores_hist: list[float] = []
    invalid_hist: list[float] = []
    policy_loss_hist: list[float] = []
    value_loss_hist: list[float] = []
    entropy_hist: list[float] = []
    kl_hist: list[float] = []
    rollout_buffers: list[str] = []
    rollout_manifests: list[str] = []
    best_reward = -1e18
    status = "ok"
    warnings: list[str] = []

    for update_idx in range(1, int(cfg.train.max_updates) + 1):
        applied_cfg_dict, stage_payload = scheduler.apply_to_config(cfg_raw, training_iteration=update_idx)
        applied_cfg = PPOConfig.from_mapping(applied_cfg_dict)
        _append_jsonl(
            curriculum_applied_path,
            {
                "schema": "p44_curriculum_applied_v1",
                "ts": _now_iso(),
                "seed": str(seed),
                **stage_payload,
            },
        )

        distributed_seeds = list(applied_cfg.distributed.seeds) if applied_cfg.distributed.seeds else [str(seed)]
        rollout_summary = run_distributed_rollout(
            policy_snapshot=_cpu_model_snapshot(model),
            policy_id="ppo_lite",
            num_workers=int(applied_cfg.distributed.num_workers),
            seeds=distributed_seeds,
            episodes_per_worker=int(applied_cfg.distributed.episodes_per_worker),
            max_steps_per_episode=int(applied_cfg.distributed.max_steps_per_episode),
            total_steps_cap=int(applied_cfg.rollout.total_steps_cap),
            run_id=f"{seed}-u{update_idx:03d}",
            out_dir=rollout_root / f"update_{update_idx:03d}",
            backend=str(applied_cfg.env.backend),
            reward_config=applied_cfg.env.reward,
            env_config={
                "timeout_sec": float(applied_cfg.env.timeout_sec),
                "max_steps_per_episode": int(applied_cfg.env.max_steps_per_episode),
                "max_auto_steps": int(applied_cfg.env.max_auto_steps),
                "max_ante": int(applied_cfg.env.max_ante),
                "auto_advance": bool(applied_cfg.env.auto_advance),
            },
            include_steps_in_result=True,
        )
        rollout_buffers.append(str(rollout_summary.get("rollout_buffer_jsonl") or ""))
        rollout_manifests.append(str(rollout_summary.get("rollout_manifest") or ""))
        steps = rollout_summary.get("steps") if isinstance(rollout_summary.get("steps"), list) else []
        if not steps:
            status = "stub"
            warnings.append(f"update_{update_idx}:empty_rollout_steps")
            break

        update_status, update_metrics = _ppo_update_from_steps(
            model=model,
            optimizer=optimizer,
            steps=steps,
            cfg=applied_cfg,
            device=device,
        )
        if update_status != "ok":
            status = update_status
            warnings.append(
                f"update_{update_idx}:{str(update_metrics.get('reason') or 'ppo_update_failed')}"
            )
            break

        reward_values = [_safe_float(row.get("reward"), 0.0) for row in steps]
        rollout_stats_payload = _read_json(Path(str(rollout_summary.get("rollout_stats_json") or ""))) or {}
        mean_reward = float(statistics.mean(reward_values)) if reward_values else 0.0
        mean_score = _safe_float(rollout_stats_payload.get("mean_score"), mean_reward)
        invalid_rate = _safe_float(rollout_summary.get("invalid_action_rate"), 0.0)

        rewards_hist.append(mean_reward)
        scores_hist.append(mean_score)
        invalid_hist.append(invalid_rate)
        policy_loss_hist.append(_safe_float(update_metrics.get("policy_loss"), 0.0))
        value_loss_hist.append(_safe_float(update_metrics.get("value_loss"), 0.0))
        entropy_hist.append(_safe_float(update_metrics.get("entropy"), 0.0))
        kl_hist.append(_safe_float(update_metrics.get("kl_divergence"), 0.0))

        if invalid_rate > float(cfg.train.invalid_action_warn_threshold):
            warnings.append(
                "update_{idx}:invalid_action_rate={value:.4f}>threshold={thr:.4f}".format(
                    idx=update_idx,
                    value=invalid_rate,
                    thr=float(cfg.train.invalid_action_warn_threshold),
                )
            )
        if abs(_safe_float(update_metrics.get("kl_divergence"), 0.0)) > float(cfg.train.max_kl_warn):
            warnings.append(
                "update_{idx}:kl_divergence={value:.4f}>max_kl_warn={thr:.4f}".format(
                    idx=update_idx,
                    value=_safe_float(update_metrics.get("kl_divergence"), 0.0),
                    thr=float(cfg.train.max_kl_warn),
                )
            )

        checkpoint_payload = {
            "schema": "p44_rl_checkpoint_v1",
            "generated_at": _now_iso(),
            "seed": str(seed),
            "update_index": int(update_idx),
            "model": _cpu_model_snapshot(model),
            "optimizer_state": optimizer.state_dict(),
            "train_cfg": applied_cfg.to_dict(),
            "curriculum_stage": stage_payload,
            "metrics": {
                "mean_reward": mean_reward,
                "mean_score": mean_score,
                "invalid_action_rate": invalid_rate,
                "policy_loss": _safe_float(update_metrics.get("policy_loss"), 0.0),
                "value_loss": _safe_float(update_metrics.get("value_loss"), 0.0),
                "entropy": _safe_float(update_metrics.get("entropy"), 0.0),
                "kl_divergence": _safe_float(update_metrics.get("kl_divergence"), 0.0),
            },
        }
        save_torch_checkpoint(last_ckpt, checkpoint_payload)
        if mean_reward >= best_reward:
            best_reward = mean_reward
            save_torch_checkpoint(best_ckpt, checkpoint_payload)

        _append_jsonl(
            progress_path,
            {
                "schema": "p44_rl_progress_v1",
                "ts": _now_iso(),
                "seed": str(seed),
                "seed_index": int(seed_index),
                "seed_total": int(seed_total),
                "update": int(update_idx),
                "status": status,
                "curriculum_stage": str(stage_payload.get("stage") or ""),
                "phase_index": int(stage_payload.get("phase_index") or 0),
                "mean_reward": mean_reward,
                "mean_score": mean_score,
                "policy_loss": _safe_float(update_metrics.get("policy_loss"), 0.0),
                "value_loss": _safe_float(update_metrics.get("value_loss"), 0.0),
                "entropy": _safe_float(update_metrics.get("entropy"), 0.0),
                "kl_divergence": _safe_float(update_metrics.get("kl_divergence"), 0.0),
                "invalid_action_rate": invalid_rate,
                "rollout_steps": int(rollout_summary.get("step_count") or 0),
                "rollout_dir": str(rollout_summary.get("run_dir") or ""),
            },
        )

    if warnings:
        with warnings_log_path.open("a", encoding="utf-8", newline="\n") as fp:
            for line in warnings:
                fp.write(line + "\n")

    metrics = {
        "schema": "p44_rl_seed_metrics_v1",
        "generated_at": _now_iso(),
        "seed": str(seed),
        "status": status,
        "updates_completed": len(rewards_hist),
        "mean_reward": float(statistics.mean(rewards_hist)) if rewards_hist else 0.0,
        "mean_score": float(statistics.mean(scores_hist)) if scores_hist else 0.0,
        "mean_invalid_action_rate": float(statistics.mean(invalid_hist)) if invalid_hist else 0.0,
        "policy_loss": float(statistics.mean(policy_loss_hist)) if policy_loss_hist else 0.0,
        "value_loss": float(statistics.mean(value_loss_hist)) if value_loss_hist else 0.0,
        "entropy": float(statistics.mean(entropy_hist)) if entropy_hist else 0.0,
        "kl_divergence": float(statistics.mean(kl_hist)) if kl_hist else 0.0,
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
        "rollout_buffers": rollout_buffers,
        "rollout_manifests": rollout_manifests,
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


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
    repo_root = _resolve_repo_root()
    if isinstance(config, dict):
        cfg_raw = dict(config)
    elif config_path is not None:
        cfg_raw = _read_yaml_or_json(Path(config_path))
    else:
        cfg_raw = {}
    cfg = PPOConfig.from_mapping(cfg_raw)

    if quick:
        cfg.distributed.num_workers = min(cfg.distributed.num_workers, 2)
        cfg.distributed.episodes_per_worker = min(cfg.distributed.episodes_per_worker, 2)
        cfg.distributed.max_steps_per_episode = min(cfg.distributed.max_steps_per_episode, 80)
        cfg.rollout.total_steps_cap = min(cfg.rollout.total_steps_cap, 320)
        cfg.train.max_updates = min(cfg.train.max_updates, 2)
        cfg.train.ppo_epochs = min(cfg.train.ppo_epochs, 1)
        cfg.train.minibatch_size = min(cfg.train.minibatch_size, 64)
        cfg.evaluation.seeds = cfg.evaluation.seeds[:2]
        cfg.evaluation.episodes_per_seed = min(cfg.evaluation.episodes_per_seed, 1)
        cfg.evaluation.max_steps_per_episode = min(cfg.evaluation.max_steps_per_episode, 80)

    seeds = [str(s).strip() for s in (seeds_override if seeds_override else cfg.seeds) if str(s).strip()]
    if not seeds:
        seeds = ["AAAAAAA", "BBBBBBB"]
    if quick and len(seeds) > 2:
        seeds = seeds[:2]
    cfg.seeds = seeds
    cfg_raw = cfg.to_dict()
    cfg_raw["seeds"] = list(seeds)

    run_token = str(run_id or cfg.run_id or _now_stamp())
    run_dir = _resolve_path(repo_root, out_dir) if out_dir else (_resolve_path(repo_root, cfg.output_artifacts_root) / run_token)
    run_dir.mkdir(parents=True, exist_ok=True)

    seeds_payload = build_seeds_payload(seeds, seed_policy_version="p44.rl_ppo_lite")
    _write_json(run_dir / "seeds_used.json", seeds_payload)
    _write_json(run_dir / "reward_config.json", cfg.env.reward if isinstance(cfg.env.reward, dict) else {})
    progress_path = run_dir / "progress.jsonl"
    warnings_log_path = run_dir / "warnings.log"

    scheduler = load_curriculum_scheduler(config_path=cfg.curriculum_config or None)
    curriculum_plan = scheduler.plan_payload(seeds=seeds)
    curriculum_plan_path = run_dir / "curriculum_plan.json"
    curriculum_applied_path = run_dir / "curriculum_applied.jsonl"
    _write_json(curriculum_plan_path, curriculum_plan)
    if not curriculum_applied_path.exists():
        curriculum_applied_path.write_text("", encoding="utf-8")

    evaluation_out_dir = _resolve_path(repo_root, cfg.evaluation.out_root) / run_token
    diagnostics_out_dir = _resolve_path(repo_root, cfg.diagnostics.out_root) / run_token

    if dry_run:
        manifest = {
            "schema": "p44_rl_train_manifest_v1",
            "generated_at": _now_iso(),
            "run_id": run_token,
            "status": "dry_run",
            "run_dir": str(run_dir),
            "config": cfg.to_dict(),
            "curriculum_plan": curriculum_plan,
            "seed_results": [],
            "best_checkpoint": "",
        }
        write_manifest(run_dir / "train_manifest.json", manifest)
        _write_json(
            run_dir / "metrics.json",
            {
                "schema": "p44_rl_train_metrics_v1",
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
            "curriculum_plan": str(curriculum_plan_path),
            "curriculum_applied": str(curriculum_applied_path),
        }

    try:
        _require_torch()
    except Exception as exc:
        stub_checkpoint = run_dir / "best_checkpoint_stub.json"
        _write_json(
            stub_checkpoint,
            {
                "schema": "p44_rl_checkpoint_stub_v1",
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
            "schema": "p44_rl_train_manifest_v1",
            "generated_at": _now_iso(),
            "run_id": run_token,
            "status": "stub",
            "run_dir": str(run_dir),
            "config": cfg.to_dict(),
            "curriculum_plan": curriculum_plan,
            "seed_results": [],
            "best_checkpoint": str(stub_checkpoint),
            "reason": "torch_missing",
        }
        write_manifest(run_dir / "train_manifest.json", manifest)
        _write_json(
            run_dir / "metrics.json",
            {
                "schema": "p44_rl_train_metrics_v1",
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
            "curriculum_plan": str(curriculum_plan_path),
            "curriculum_applied": str(curriculum_applied_path),
        }

    seed_results: list[dict[str, Any]] = []
    for idx, seed in enumerate(seeds, start=1):
        seed_result = _train_one_seed(
            seed=seed,
            seed_index=idx,
            seed_total=len(seeds),
            cfg=cfg,
            cfg_raw=cfg_raw,
            scheduler=scheduler,
            run_dir=run_dir,
            progress_path=progress_path,
            warnings_log_path=warnings_log_path,
            curriculum_applied_path=curriculum_applied_path,
        )
        seed_results.append(seed_result)

    ok_rows = [row for row in seed_results if str(row.get("status")) == "ok"]
    checkpoint_candidates = [
        str(row.get("best_checkpoint") or "")
        for row in ok_rows
        if str(row.get("best_checkpoint") or "").strip()
    ]
    if checkpoint_candidates:
        eval_summary = run_multi_seed_evaluation(
            checkpoint_paths=checkpoint_candidates,
            seeds=cfg.evaluation.seeds,
            episodes_per_seed=int(cfg.evaluation.episodes_per_seed),
            max_steps_per_episode=int(cfg.evaluation.max_steps_per_episode),
            backend=str(cfg.env.backend),
            reward_config=cfg.env.reward,
            env_config={
                "timeout_sec": float(cfg.env.timeout_sec),
                "max_steps_per_episode": int(cfg.env.max_steps_per_episode),
                "max_auto_steps": int(cfg.env.max_auto_steps),
                "max_ante": int(cfg.env.max_ante),
                "auto_advance": bool(cfg.env.auto_advance),
            },
            greedy=bool(cfg.evaluation.greedy),
            run_id=run_token,
            out_dir=evaluation_out_dir,
        )
    else:
        eval_summary = {
            "status": "stub",
            "run_id": run_token,
            "run_dir": str(evaluation_out_dir),
            "seed_results_json": "",
            "best_checkpoint": "",
            "checkpoint_results": [],
        }

    best_checkpoint = str(eval_summary.get("best_checkpoint") or "")
    if not best_checkpoint and checkpoint_candidates:
        best_checkpoint = checkpoint_candidates[0]
    status = "ok" if ok_rows else "stub"
    selected_eval_row = (
        (eval_summary.get("checkpoint_results") or [])[0]
        if isinstance(eval_summary.get("checkpoint_results"), list) and (eval_summary.get("checkpoint_results") or [])
        else {}
    )

    rollout_buffers = [
        str(path)
        for row in seed_results
        for path in (row.get("rollout_buffers") or [])
        if str(path).strip()
    ]
    diagnostics_summary = run_diagnostics(
        progress_jsonl=progress_path,
        rollout_buffers=rollout_buffers,
        out_dir=diagnostics_out_dir,
        action_topk=int(cfg.diagnostics.action_topk),
    )

    metrics_payload = {
        "schema": "p44_rl_train_metrics_v1",
        "generated_at": _now_iso(),
        "run_id": run_token,
        "status": status,
        "seed_count": len(seeds),
        "ok_seed_count": len(ok_rows),
        "mean_reward": (
            float(statistics.mean([_safe_float((row.get("metrics") or {}).get("mean_reward"), 0.0) for row in ok_rows]))
            if ok_rows
            else 0.0
        ),
        "mean_score": (
            float(statistics.mean([_safe_float((row.get("metrics") or {}).get("mean_score"), 0.0) for row in ok_rows]))
            if ok_rows
            else 0.0
        ),
        "policy_loss": (
            float(statistics.mean([_safe_float((row.get("metrics") or {}).get("policy_loss"), 0.0) for row in ok_rows]))
            if ok_rows
            else 0.0
        ),
        "value_loss": (
            float(statistics.mean([_safe_float((row.get("metrics") or {}).get("value_loss"), 0.0) for row in ok_rows]))
            if ok_rows
            else 0.0
        ),
        "entropy": (
            float(statistics.mean([_safe_float((row.get("metrics") or {}).get("entropy"), 0.0) for row in ok_rows]))
            if ok_rows
            else 0.0
        ),
        "kl_divergence": (
            float(statistics.mean([_safe_float((row.get("metrics") or {}).get("kl_divergence"), 0.0) for row in ok_rows]))
            if ok_rows
            else 0.0
        ),
        "invalid_action_rate": (
            float(
                statistics.mean(
                    [_safe_float((row.get("metrics") or {}).get("mean_invalid_action_rate"), 0.0) for row in ok_rows]
                )
            )
            if ok_rows
            else 0.0
        ),
        "eval_mean_score": _safe_float(selected_eval_row.get("mean_score"), 0.0),
        "eval_std_score": _safe_float(selected_eval_row.get("std_score"), 0.0),
        "candidate_checkpoint": best_checkpoint,
    }
    manifest = {
        "schema": "p44_rl_train_manifest_v1",
        "generated_at": _now_iso(),
        "run_id": run_token,
        "status": status,
        "run_dir": str(run_dir),
        "config": cfg.to_dict(),
        "curriculum_plan": curriculum_plan,
        "paths": {
            "metrics": str(run_dir / "metrics.json"),
            "progress_jsonl": str(progress_path),
            "warnings_log": str(warnings_log_path),
            "seeds_used": str(run_dir / "seeds_used.json"),
            "reward_config": str(run_dir / "reward_config.json"),
            "curriculum_plan": str(curriculum_plan_path),
            "curriculum_applied": str(curriculum_applied_path),
            "multi_seed_eval": str(eval_summary.get("seed_results_json") or ""),
            "diagnostics_json": str(diagnostics_summary.get("diagnostics_json") or ""),
            "diagnostics_report_md": str(diagnostics_summary.get("diagnostics_report_md") or ""),
        },
        "seed_results": seed_results,
        "best_checkpoint": best_checkpoint,
        "multi_seed_eval": eval_summary,
        "diagnostics": diagnostics_summary,
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
        "curriculum_plan": str(curriculum_plan_path),
        "curriculum_applied": str(curriculum_applied_path),
        "eval_seed_results": str(eval_summary.get("seed_results_json") or ""),
        "diagnostics_json": str(diagnostics_summary.get("diagnostics_json") or ""),
        "diagnostics_report_md": str(diagnostics_summary.get("diagnostics_report_md") or ""),
        "seed_results": seed_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P44 PPO-lite RL trainer.")
    parser.add_argument("--config", default="configs/experiments/p42_rl_smoke.yaml")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--seeds", default="", help="Optional comma-separated seed override")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_override = [seed.strip() for seed in str(args.seeds).split(",") if seed.strip()]
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
