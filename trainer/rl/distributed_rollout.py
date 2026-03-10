from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import copy
import json
import math
import multiprocessing as mp
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trainer.monitoring.progress_schema import append_progress_event as append_unified_progress_event
from trainer.monitoring.progress_schema import build_progress_event, get_gpu_mem_mb
from trainer.rl.action_mask import action_mask_density
from trainer.rl.env_adapter import RLEnvAdapter
from trainer.rl.rollout_schema import RolloutStepRecord


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
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


def _require_torch():
    try:
        import torch
        from torch.distributions import Categorical
    except Exception as exc:  # pragma: no cover - runtime guarded
        raise RuntimeError("PyTorch is required for policy-backed distributed rollouts") from exc
    return torch, Categorical


def _snapshot_to_cpu(snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(snapshot, dict):
        return None
    copied = copy.deepcopy(snapshot)
    try:
        torch, _ = _require_torch()
    except Exception:
        return copied
    state_dict = copied.get("state_dict")
    if isinstance(state_dict, dict):
        copied["state_dict"] = {
            key: value.detach().cpu() if hasattr(value, "detach") else value
            for key, value in state_dict.items()
        }
    return copied


def load_policy_snapshot(source: str | Path | dict[str, Any] | None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if source is None:
        return None, {"status": "random_policy", "reason": "source_missing"}
    if isinstance(source, dict):
        if isinstance(source.get("state_dict"), dict):
            return _snapshot_to_cpu(source), {"status": "ok", "source_kind": "snapshot_dict"}
        if isinstance(source.get("model"), dict):
            return _snapshot_to_cpu(source.get("model")), {"status": "ok", "source_kind": "checkpoint_payload"}
        return None, {"status": "stub", "reason": "snapshot_missing_state_dict"}

    path = Path(source).resolve()
    if not path.exists():
        return None, {"status": "stub", "reason": f"checkpoint_missing:{path}"}
    if path.suffix.lower() == ".pt":
        try:
            torch, _ = _require_torch()
            payload = torch.load(path, map_location="cpu")
        except Exception as exc:
            return None, {"status": "stub", "reason": f"checkpoint_load_failed:{exc}"}
        if isinstance(payload, dict) and isinstance(payload.get("model"), dict):
            return _snapshot_to_cpu(payload.get("model")), {"status": "ok", "source_kind": "torch_checkpoint"}
        if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
            return _snapshot_to_cpu(payload), {"status": "ok", "source_kind": "torch_snapshot"}
        return None, {"status": "stub", "reason": "checkpoint_missing_model_snapshot"}
    payload = _read_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("model"), dict):
        return _snapshot_to_cpu(payload.get("model")), {"status": "ok", "source_kind": "json_checkpoint"}
    return None, {"status": "stub", "reason": "json_checkpoint_missing_model"}


def _fit_obs_vector(vector: list[Any], target_dim: int) -> list[float]:
    raw = [float(x) for x in (vector or [])]
    if len(raw) >= int(target_dim):
        return raw[: int(target_dim)]
    return raw + [0.0] * max(0, int(target_dim) - len(raw))


class SnapshotMaskedRolloutPolicy:
    def __init__(
        self,
        *,
        policy_snapshot: dict[str, Any] | None = None,
        seed: str = "AAAAAAA",
        greedy: bool = False,
        requested_device: str = "cpu",
    ) -> None:
        self.seed = str(seed)
        self.greedy = bool(greedy)
        self.rng = random.Random(self.seed)
        self.snapshot = _snapshot_to_cpu(policy_snapshot)
        self.model = None
        self.torch = None
        self.Categorical = None
        self.device = None
        self.load_meta = {"status": "random_policy", "source_kind": "none"}
        self.requested_device = str(requested_device or "cpu")

        if not isinstance(self.snapshot, dict):
            return
        try:
            torch, categorical_cls = _require_torch()
            from trainer.rl.policy_value_model import PolicyValueModel

            obs_dim = max(1, _safe_int(self.snapshot.get("obs_dim"), 48))
            action_dim = max(1, _safe_int(self.snapshot.get("action_dim"), 1275))
            model = PolicyValueModel(obs_dim=obs_dim, action_dim=action_dim)
            model.load_state_dict(self.snapshot.get("state_dict") or {})
            model.eval()
            self.model = model
            self.torch = torch
            self.Categorical = categorical_cls
            resolved_device = "cpu"
            if str(self.requested_device).startswith("cuda") and bool(torch.cuda.is_available()):
                resolved_device = str(self.requested_device)
            self.device = torch.device(resolved_device)
            self.model = self.model.to(self.device)
            self.load_meta = {
                "status": "ok",
                "source_kind": "snapshot_dict",
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "requested_device": self.requested_device,
                "resolved_device": str(self.device),
            }
        except Exception as exc:
            self.model = None
            self.load_meta = {"status": "random_policy", "reason": f"snapshot_init_failed:{exc}"}

    def act(
        self,
        *,
        obs: dict[str, Any],
        action_mask: list[int],
        info: dict[str, Any],
    ) -> tuple[int, float | None, float | None, dict[str, Any]]:
        legal_ids = [idx for idx, flag in enumerate(action_mask) if int(flag) > 0]
        if not legal_ids:
            return 0, 0.0, 0.0, {"policy": "snapshot_masked", "fallback": "no_legal_actions"}
        if self.model is None or self.torch is None or self.Categorical is None:
            action = int(self.rng.choice(legal_ids))
            return action, None, None, {"policy": "random_legal", **self.load_meta}

        obs_dim = max(1, int(getattr(self.model, "obs_dim", len(obs.get("vector") or []))))
        vector = _fit_obs_vector(list(obs.get("vector") or []), obs_dim)
        with self.torch.no_grad():
            x = self.torch.tensor([vector], dtype=self.torch.float32, device=self.device)
            logits, values = self.model(x)
            action_dim = int(logits.shape[-1])
            clipped_legal = [aid for aid in legal_ids if 0 <= int(aid) < action_dim]
            if not clipped_legal:
                action = int(legal_ids[0])
                return action, 0.0, float(values.squeeze(-1).item()), {
                    "policy": "snapshot_masked",
                    "fallback": "head_too_small",
                    **self.load_meta,
                }
            mask = self.torch.full_like(logits, -1e9)
            for aid in clipped_legal:
                mask[0, int(aid)] = 0.0
            masked_logits = logits + mask
            if self.greedy:
                action = int(self.torch.argmax(masked_logits, dim=-1).item())
                dist = self.Categorical(logits=masked_logits)
            else:
                dist = self.Categorical(logits=masked_logits)
                action = int(dist.sample().item())
            logprob = float(dist.log_prob(self.torch.tensor(action, device=self.device)).item())
            value_pred = float(values.squeeze(-1).item() if values.ndim > 1 else values.item())
        return action, logprob, value_pred, {"policy": "snapshot_masked", **self.load_meta}


def _worker_run(spec: dict[str, Any]) -> None:
    worker_id = max(1, _safe_int(spec.get("worker_id"), 1))
    worker_dir = Path(str(spec.get("worker_dir"))).resolve()
    worker_dir.mkdir(parents=True, exist_ok=True)
    rollout_steps_path = worker_dir / "rollout_steps.jsonl"
    worker_summary_path = worker_dir / "worker_summary.json"

    worker_seeds = [str(seed).strip() for seed in (spec.get("seeds") or []) if str(seed).strip()]
    if not worker_seeds:
        worker_seeds = [f"worker{worker_id:02d}"]

    snapshot, snapshot_meta = load_policy_snapshot(spec.get("policy_snapshot"))
    policy = SnapshotMaskedRolloutPolicy(
        policy_snapshot=snapshot,
        seed=f"worker-{worker_id}-{worker_seeds[0]}",
        greedy=bool(spec.get("greedy")),
        requested_device=str(spec.get("rollout_device") or "cpu"),
    )

    env_kwargs = spec.get("env") if isinstance(spec.get("env"), dict) else {}
    reward_cfg = spec.get("reward_config") if isinstance(spec.get("reward_config"), dict) else {}
    max_steps_per_episode = max(1, _safe_int(spec.get("max_steps_per_episode"), 120))
    episodes_per_worker = max(1, _safe_int(spec.get("episodes_per_worker"), 1))
    worker_step_cap = max(0, _safe_int(spec.get("worker_step_cap"), 0))
    preserve_seed_identity = bool(spec.get("preserve_seed_identity", False))

    warnings: list[str] = []
    episodes: list[dict[str, Any]] = []
    step_count = 0
    invalid_steps = 0
    status = "ok"

    adapter = RLEnvAdapter(
        backend=str(spec.get("backend") or "sim"),
        seed=worker_seeds[0],
        timeout_sec=_safe_float(env_kwargs.get("timeout_sec"), 8.0),
        max_steps_per_episode=max(1, _safe_int(env_kwargs.get("max_steps_per_episode"), max_steps_per_episode)),
        max_auto_steps=max(1, _safe_int(env_kwargs.get("max_auto_steps"), 8)),
        max_ante=max(0, _safe_int(env_kwargs.get("max_ante"), 0)),
        auto_advance=bool(env_kwargs.get("auto_advance", True)),
        reward_config=reward_cfg,
    )
    try:
        for episode_index in range(1, episodes_per_worker + 1):
            if worker_step_cap > 0 and step_count >= worker_step_cap:
                break
            base_seed = worker_seeds[(episode_index - 1) % len(worker_seeds)]
            episode_token = f"{base_seed}-w{worker_id:02d}-ep{episode_index:04d}"
            reset_seed = base_seed if preserve_seed_identity else episode_token
            try:
                obs, info = adapter.reset(seed=reset_seed)
            except Exception as exc:
                warnings.append(f"reset_failed:{episode_token}:{exc}")
                status = "degraded"
                continue

            episode_reward = 0.0
            episode_invalid = 0
            episode_steps = 0
            final_score = 0.0
            for step_id in range(max_steps_per_episode):
                if worker_step_cap > 0 and step_count >= worker_step_cap:
                    break
                action_mask = adapter.get_action_mask(obs, info)
                legal_ids = [idx for idx, flag in enumerate(action_mask) if int(flag) > 0]
                action, logprob, value_pred, policy_meta = policy.act(obs=obs, action_mask=action_mask, info=info)
                try:
                    next_obs, reward, terminated, truncated, step_info = adapter.step(action)
                except Exception as exc:
                    warnings.append(f"step_failed:{episode_token}:{step_id}:{exc}")
                    status = "degraded"
                    break
                step_record = RolloutStepRecord(
                    seed=base_seed,
                    episode_id=episode_token,
                    step_id=int(step_id),
                    obs_vector=[float(x) for x in (obs.get("vector") or [])],
                    action=int(action),
                    action_logprob=logprob,
                    value_pred=value_pred,
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    action_mask_density=float(action_mask_density(action_mask)),
                    action_mask_legal_count=int(sum(1 for x in action_mask if int(x) > 0)),
                    legal_action_ids=legal_ids,
                    invalid_action=bool(step_info.get("invalid_action")),
                    phase=str(step_info.get("phase") or ""),
                    score_delta=float(step_info.get("score_delta") or 0.0),
                    ante=_safe_int(step_info.get("ante"), 0),
                    round_num=_safe_int(step_info.get("round"), 0),
                    info_summary={
                        "worker_id": int(worker_id),
                        "action_applied": step_info.get("action_applied"),
                        "action_resolution": step_info.get("action_resolution"),
                        "mask_density": step_info.get("mask_density"),
                        "policy_meta": policy_meta if isinstance(policy_meta, dict) else {},
                    },
                ).to_dict()
                _append_jsonl(rollout_steps_path, step_record)

                step_count += 1
                episode_steps += 1
                episode_reward += float(reward)
                if bool(step_record.get("invalid_action")):
                    invalid_steps += 1
                    episode_invalid += 1
                final_score = _safe_float(
                    ((step_info.get("episode_metrics_partial") or {}).get("score")),
                    final_score,
                )
                obs, info = next_obs, step_info
                if terminated or truncated:
                    break

            episodes.append(
                {
                    "episode_id": episode_token,
                    "seed": base_seed,
                    "reward": float(episode_reward),
                    "episode_length": int(episode_steps),
                    "final_score": float(final_score),
                    "invalid_action_rate": float(episode_invalid) / float(max(1, episode_steps)),
                }
            )
    finally:
        adapter.close()

    rewards = [float(row.get("reward") or 0.0) for row in episodes]
    lengths = [int(row.get("episode_length") or 0) for row in episodes]
    scores = [float(row.get("final_score") or 0.0) for row in episodes]
    summary = {
        "schema": "p44_rollout_worker_summary_v1",
        "generated_at": _now_iso(),
        "worker_id": int(worker_id),
        "status": status if step_count > 0 else "stub",
        "backend": str(spec.get("backend") or "sim"),
        "policy_id": str(spec.get("policy_id") or "snapshot_masked"),
        "snapshot_meta": snapshot_meta,
        "policy_meta": policy.load_meta,
        "seeds": worker_seeds,
        "episodes_requested": episodes_per_worker,
        "episodes_completed": len(episodes),
        "step_count": int(step_count),
        "invalid_action_count": int(invalid_steps),
        "invalid_action_rate": float(invalid_steps) / float(max(1, step_count)),
        "avg_reward": float(statistics.mean(rewards)) if rewards else 0.0,
        "avg_episode_length": float(statistics.mean(lengths)) if lengths else 0.0,
        "mean_score": float(statistics.mean(scores)) if scores else 0.0,
        "warnings": warnings,
        "paths": {
            "rollout_steps_jsonl": str(rollout_steps_path),
            "worker_summary_json": str(worker_summary_path),
        },
        "runtime": {
            "rollout_device_requested": str(spec.get("rollout_device") or "cpu"),
            "rollout_device_resolved": str(policy.load_meta.get("resolved_device") or "cpu"),
            "preserve_seed_identity": bool(preserve_seed_identity),
        },
        "episodes": episodes,
    }
    _write_json(worker_summary_path, summary)


def _partition_worker_seeds(seed_list: list[str], num_workers: int) -> list[list[str]]:
    if not seed_list:
        seed_list = ["AAAAAAA"]
    parts = [[] for _ in range(max(1, num_workers))]
    for idx, seed in enumerate(seed_list):
        parts[idx % len(parts)].append(str(seed))
    for idx, part in enumerate(parts):
        if not part:
            part.append(str(seed_list[idx % len(seed_list)]))
    return parts


def run_distributed_rollout(
    *,
    policy_snapshot: str | Path | dict[str, Any] | None = None,
    policy_id: str = "snapshot_masked",
    num_workers: int = 2,
    seeds: list[str] | None = None,
    episodes_per_worker: int = 4,
    max_steps_per_episode: int = 120,
    total_steps_cap: int = 0,
    run_id: str = "",
    out_root: str | Path = "docs/artifacts/p44/rollouts",
    out_dir: str | Path | None = None,
    backend: str = "sim",
    reward_config: dict[str, Any] | None = None,
    env_config: dict[str, Any] | None = None,
    greedy: bool = False,
    include_steps_in_result: bool = False,
    rollout_device: str = "cpu",
    runtime_profile: dict[str, Any] | None = None,
    progress_path: str | Path | None = None,
    preserve_seed_identity: bool = False,
) -> dict[str, Any]:
    started = time.time()
    seed_list = [str(seed).strip() for seed in (seeds or ["AAAAAAA", "BBBBBBB"]) if str(seed).strip()]
    if not seed_list:
        seed_list = ["AAAAAAA", "BBBBBBB"]
    run_token = str(run_id or _now_stamp())
    run_dir = Path(out_dir).resolve() if out_dir else (Path(out_root).resolve() / run_token)
    run_dir.mkdir(parents=True, exist_ok=True)

    worker_count = max(1, int(num_workers))
    worker_dirs = [run_dir / f"worker_{idx:02d}" for idx in range(1, worker_count + 1)]
    worker_seed_parts = _partition_worker_seeds(seed_list, worker_count)
    worker_step_cap = 0
    if int(total_steps_cap) > 0:
        worker_step_cap = max(1, int(math.ceil(float(total_steps_cap) / float(worker_count))))

    frozen_snapshot = None
    if isinstance(policy_snapshot, dict):
        frozen_snapshot = _snapshot_to_cpu(policy_snapshot)
    else:
        frozen_snapshot = policy_snapshot

    specs = []
    for idx in range(1, worker_count + 1):
        specs.append(
            {
                "worker_id": idx,
                "worker_dir": str(worker_dirs[idx - 1]),
                "policy_snapshot": frozen_snapshot,
                "policy_id": str(policy_id),
                "backend": str(backend),
                "env": dict(env_config or {}),
                "reward_config": dict(reward_config or {}),
                "seeds": list(worker_seed_parts[idx - 1]),
                "episodes_per_worker": int(episodes_per_worker),
                "max_steps_per_episode": int(max_steps_per_episode),
                "worker_step_cap": int(worker_step_cap),
                "greedy": bool(greedy),
                "rollout_device": str(rollout_device or "cpu"),
                "preserve_seed_identity": bool(preserve_seed_identity),
            }
        )

    if worker_count == 1:
        _worker_run(specs[0])
    else:
        ctx = mp.get_context("spawn")
        processes: list[mp.Process] = []
        for spec in specs:
            proc = ctx.Process(target=_worker_run, args=(spec,))
            proc.start()
            processes.append(proc)
        for proc in processes:
            proc.join()

    rollout_buffer_path = run_dir / "rollout_buffer.jsonl"
    worker_stats_path = run_dir / "worker_stats.json"
    rollout_stats_path = run_dir / "rollout_stats.json"
    rollout_manifest_path = run_dir / "rollout_manifest.json"

    worker_stats: list[dict[str, Any]] = []
    aggregated_steps: list[dict[str, Any]] = []
    if rollout_buffer_path.exists():
        rollout_buffer_path.unlink()
    for worker_dir in worker_dirs:
        summary = _read_json(worker_dir / "worker_summary.json")
        if isinstance(summary, dict):
            worker_stats.append(summary)
        step_rows = _read_jsonl(worker_dir / "rollout_steps.jsonl")
        for row in step_rows:
            _append_jsonl(rollout_buffer_path, row)
        if include_steps_in_result:
            aggregated_steps.extend(step_rows)

    all_steps = aggregated_steps if include_steps_in_result else _read_jsonl(rollout_buffer_path)
    rewards = [_safe_float(row.get("reward"), 0.0) for row in all_steps]
    episode_ids = {str(row.get("episode_id") or "") for row in all_steps if str(row.get("episode_id") or "").strip()}
    invalid_steps = sum(1 for row in all_steps if bool(row.get("invalid_action")))
    action_counts: dict[str, int] = {}
    episode_lengths: dict[str, int] = {}
    episode_rewards: dict[str, float] = {}
    episode_scores: list[float] = []
    for worker_summary in worker_stats:
        if not isinstance(worker_summary, dict):
            continue
        for episode in worker_summary.get("episodes") if isinstance(worker_summary.get("episodes"), list) else []:
            if not isinstance(episode, dict):
                continue
            episode_scores.append(_safe_float(episode.get("final_score"), 0.0))
    for row in all_steps:
        episode_id = str(row.get("episode_id") or "")
        if episode_id:
            episode_lengths[episode_id] = int(episode_lengths.get(episode_id, 0)) + 1
            episode_rewards[episode_id] = float(episode_rewards.get(episode_id, 0.0)) + _safe_float(row.get("reward"), 0.0)
        action_key = str(row.get("action") or 0)
        action_counts[action_key] = int(action_counts.get(action_key, 0)) + 1

    step_count = len(all_steps)
    episode_length_values = list(episode_lengths.values())
    episode_reward_values = list(episode_rewards.values())
    worker_failures = [
        row for row in worker_stats if str(row.get("status") or "").lower() not in {"ok", "degraded"}
    ]
    stats = {
        "schema": "p44_rollout_stats_v1",
        "generated_at": _now_iso(),
        "run_id": run_token,
        "status": "ok" if step_count > 0 else "stub",
        "policy_id": str(policy_id),
        "worker_count": int(worker_count),
        "seed_count": len(seed_list),
        "seeds": seed_list,
        "episodes_per_worker": int(episodes_per_worker),
        "max_steps_per_episode": int(max_steps_per_episode),
        "step_count": int(step_count),
        "episode_count": len(episode_ids),
        "avg_reward": float(statistics.mean(rewards)) if rewards else 0.0,
        "avg_episode_reward": float(statistics.mean(episode_reward_values)) if episode_reward_values else 0.0,
        "avg_episode_length": float(statistics.mean(episode_length_values)) if episode_length_values else 0.0,
        "mean_score": float(statistics.mean(episode_scores)) if episode_scores else 0.0,
        "std_score": float(statistics.pstdev(episode_scores)) if len(episode_scores) > 1 else 0.0,
        "invalid_action_count": int(invalid_steps),
        "invalid_action_rate": float(invalid_steps) / float(max(1, step_count)),
        "worker_failure_count": len(worker_failures),
        "elapsed_sec": time.time() - started,
        "rollout_steps_per_sec": float(step_count) / max(0.001, (time.time() - started)),
        "action_frequency": {
            key: action_counts[key]
            for key in sorted(action_counts.keys(), key=lambda token: (-action_counts[token], token))[:16]
        },
    }
    manifest = {
        "schema": "p44_rollout_manifest_v1",
        "generated_at": _now_iso(),
        "run_id": run_token,
        "run_dir": str(run_dir),
        "policy_id": str(policy_id),
        "backend": str(backend),
        "config": {
            "num_workers": int(worker_count),
            "seeds": seed_list,
            "episodes_per_worker": int(episodes_per_worker),
            "max_steps_per_episode": int(max_steps_per_episode),
            "total_steps_cap": int(total_steps_cap),
            "greedy": bool(greedy),
            "rollout_device": str(rollout_device or "cpu"),
            "preserve_seed_identity": bool(preserve_seed_identity),
        },
        "paths": {
            "rollout_buffer_jsonl": str(rollout_buffer_path),
            "rollout_stats_json": str(rollout_stats_path),
            "worker_stats_json": str(worker_stats_path),
            "rollout_manifest_json": str(rollout_manifest_path),
        },
        "workers": [
            {
                "worker_id": int(spec.get("worker_id") or 0),
                "worker_dir": str(spec.get("worker_dir") or ""),
                "seeds": list(spec.get("seeds") or []),
            }
            for spec in specs
        ],
        "runtime_profile": runtime_profile if isinstance(runtime_profile, dict) else {},
    }
    _write_json(worker_stats_path, worker_stats)
    _write_json(rollout_stats_path, stats)
    _write_json(rollout_manifest_path, manifest)

    if progress_path:
        append_unified_progress_event(
            progress_path,
            build_progress_event(
                run_id=run_token,
                component="p44_rollout",
                phase="rollout",
                status=str(stats.get("status") or "ok"),
                step=int(step_count),
                epoch_or_iter=None,
                seed=",".join(seed_list),
                metrics={
                    "step_count": int(step_count),
                    "episode_count": int(len(episode_ids)),
                    "invalid_action_rate": float(stats.get("invalid_action_rate") or 0.0),
                    "worker_failure_count": int(stats.get("worker_failure_count") or 0),
                },
                device_profile=runtime_profile if isinstance(runtime_profile, dict) else {},
                learner_device=str(((runtime_profile or {}).get("resolved") or {}).get("learner_device") or ""),
                rollout_device=str(rollout_device or "cpu"),
                throughput=float(stats.get("rollout_steps_per_sec") or 0.0),
                gpu_mem_mb=get_gpu_mem_mb(None),
            ),
        )

    result = {
        "status": str(stats.get("status")),
        "run_id": run_token,
        "run_dir": str(run_dir),
        "rollout_manifest": str(rollout_manifest_path),
        "rollout_stats_json": str(rollout_stats_path),
        "worker_stats_json": str(worker_stats_path),
        "rollout_buffer_jsonl": str(rollout_buffer_path),
        "step_count": int(step_count),
        "episode_count": int(len(episode_ids)),
        "invalid_action_rate": float(stats.get("invalid_action_rate") or 0.0),
        "elapsed_sec": float(stats.get("elapsed_sec") or 0.0),
        "rollout_steps_per_sec": float(stats.get("rollout_steps_per_sec") or 0.0),
    }
    if include_steps_in_result:
        result["steps"] = all_steps
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P44 distributed rollout workers.")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--out-root", default="docs/artifacts/p44/rollouts")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seeds", default="AAAAAAA,BBBBBBB")
    parser.add_argument("--episodes-per-worker", type=int, default=2)
    parser.add_argument("--max-steps-per-episode", type=int, default=80)
    parser.add_argument("--total-steps-cap", type=int, default=0)
    parser.add_argument("--backend", default="sim")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--greedy", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [token.strip() for token in str(args.seeds).split(",") if token.strip()]
    summary = run_distributed_rollout(
        policy_snapshot=(str(args.checkpoint) if str(args.checkpoint).strip() else None),
        num_workers=max(1, int(args.num_workers)),
        seeds=seeds,
        episodes_per_worker=max(1, int(args.episodes_per_worker)),
        max_steps_per_episode=max(1, int(args.max_steps_per_episode)),
        total_steps_cap=max(0, int(args.total_steps_cap)),
        run_id=str(args.run_id or ""),
        out_root=str(args.out_root),
        out_dir=(str(args.out_dir) if str(args.out_dir).strip() else None),
        backend=str(args.backend),
        greedy=bool(args.greedy),
        include_steps_in_result=False,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if str(summary.get("status")) in {"ok", "stub"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
