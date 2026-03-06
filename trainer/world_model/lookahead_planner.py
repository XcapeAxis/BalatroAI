from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from trainer import action_space, action_space_shop
from trainer.closed_loop.replay_manifest import write_json, write_markdown
from trainer.policy_arena.policy_adapter import phase_from_obs
from trainer.world_model.candidate_actions import capture_sample_state, generate_candidate_actions
from trainer.world_model.model import load_world_model_from_checkpoint
from trainer.world_model.planning_hook import _vectorize_obs_state
from trainer.world_model.schema import action_token_from_parts, make_sample_id, stable_hash_int


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - runtime guarded
        raise RuntimeError("PyTorch is required for trainer.world_model.lookahead_planner") from exc
    return torch


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


def _parse_csv(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for part in str(text or "").split(","):
        token = part.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


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
        raise ValueError(f"config must be mapping: {path}")
    return payload


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _find_latest_world_model_checkpoint(repo_root: Path) -> str:
    root = (repo_root / "docs/artifacts/p45/wm_train").resolve()
    if not root.exists():
        return ""
    candidates = sorted(root.glob("**/best.pt"), key=lambda path: str(path))
    return str(candidates[-1].resolve()) if candidates else ""


def _resolve_checkpoint(repo_root: Path, cfg: dict[str, Any], override: str = "") -> str:
    if str(override or "").strip():
        return str((Path(override) if Path(override).is_absolute() else (repo_root / override).resolve()))
    wm_cfg = cfg.get("world_model") if isinstance(cfg.get("world_model"), dict) else {}
    raw = str(wm_cfg.get("checkpoint") or cfg.get("world_model_checkpoint") or "").strip()
    if raw:
        return str((Path(raw) if Path(raw).is_absolute() else (repo_root / raw).resolve()))
    return _find_latest_world_model_checkpoint(repo_root)


class WorldModelLookaheadPlanner:
    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        horizon: int = 1,
        gamma: float = 0.95,
        uncertainty_penalty: float = 0.5,
        reward_weight: float = 1.0,
        score_weight: float = 0.5,
        value_weight: float = 0.15,
        terminal_bonus: float = 0.0,
        branching_strategy: str = "repeat_action",
    ) -> None:
        self.checkpoint_path = str(checkpoint_path or "")
        self.horizon = max(1, int(horizon))
        self.gamma = float(gamma)
        self.uncertainty_penalty = float(uncertainty_penalty)
        self.reward_weight = float(reward_weight)
        self.score_weight = float(score_weight)
        self.value_weight = float(value_weight)
        self.terminal_bonus = float(terminal_bonus)
        self.branching_strategy = str(branching_strategy or "repeat_action")
        self.available = bool(self.checkpoint_path) and Path(self.checkpoint_path).exists()
        self._model = None
        self._model_cfg = None
        if self.available:
            self._model, self._model_cfg, _payload = load_world_model_from_checkpoint(self.checkpoint_path, device="cpu")

    def describe(self) -> dict[str, Any]:
        return {
            "world_model_assist": bool(self.available),
            "assist_mode": "rerank",
            "checkpoint_path": self.checkpoint_path,
            "horizon": int(self.horizon),
            "gamma": float(self.gamma),
            "uncertainty_penalty": float(self.uncertainty_penalty),
            "reward_weight": float(self.reward_weight),
            "score_weight": float(self.score_weight),
            "value_weight": float(self.value_weight),
            "terminal_bonus": float(self.terminal_bonus),
            "branching_strategy": self.branching_strategy,
        }

    def _normalize_candidate(self, obs: dict[str, Any], candidate: dict[str, Any], fallback_rank: int) -> dict[str, Any]:
        phase = phase_from_obs(obs)
        action = dict(candidate.get("action") or {}) if isinstance(candidate.get("action"), dict) else {}
        action_token = str(candidate.get("action_token") or "")
        if not action_token:
            action_numeric = _safe_int(action.get("id"), -1) if action.get("id") is not None else None
            action_token = action_token_from_parts(
                phase=phase,
                action_type=str(action.get("action_type") or "OTHER"),
                action_payload=action,
                numeric_action=(action_numeric if action_numeric is not None and action_numeric >= 0 else None),
            )
        payload = dict(candidate)
        payload["action"] = action
        payload["action_token"] = action_token
        payload["candidate_id"] = str(candidate.get("candidate_id") or make_sample_id([phase, action_token, fallback_rank]))
        payload["source_rank"] = _safe_int(candidate.get("source_rank"), fallback_rank)
        payload["source_score"] = _safe_float(candidate.get("source_score"), 0.0)
        payload["source"] = str(candidate.get("source") or "unknown")
        payload["legal"] = bool(candidate.get("legal", True))
        return payload

    def _rollout_candidate(self, obs: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
        if not self.available or self._model is None or self._model_cfg is None:
            return {
                "candidate_id": str(candidate.get("candidate_id") or ""),
                "predicted_return": 0.0,
                "uncertainty_score": 0.0,
                "wm_score": float(candidate.get("source_score") or 0.0),
                "horizon_used": 0,
                "fallback_happened": True,
                "rollout_trace_summary": [],
            }
        torch = _require_torch()
        action_vocab_size = int(self._model_cfg.action_vocab_size)
        obs_vector = _vectorize_obs_state(
            obs,
            action_dim=max(action_space.max_actions(), action_space_shop.max_actions()),
        )
        action_token = str(candidate.get("action_token") or "")
        action_id = stable_hash_int(action_token, action_vocab_size)
        obs_t = torch.tensor([list(obs_vector)], dtype=torch.float32)
        action_tensor = torch.tensor([int(action_id)], dtype=torch.long)

        discounted_return = 0.0
        uncertainty_values: list[float] = []
        trace_rows: list[dict[str, Any]] = []
        with torch.no_grad():
            z_curr = self._model.encode(obs_t)
            for step_idx in range(1, self.horizon + 1):
                joint = self._model.joint(z_curr, action_tensor)
                z_next = self._model.transition(joint)
                reward_pred = float(self._model.reward_head(joint).squeeze(-1).detach().cpu().item())
                score_pred = float(self._model.score_head(joint).squeeze(-1).detach().cpu().item())
                resource_pred = list(self._model.resource_head(joint).detach().cpu().tolist()[0])
                uncertainty = float(self._model.uncertainty_head(joint).abs().squeeze(-1).detach().cpu().item())
                next_state_proxy = float(z_next.mean(dim=-1).detach().cpu().item())
                value_term = float(self.value_weight) * next_state_proxy if step_idx == self.horizon else 0.0
                step_return = (
                    float(self.reward_weight) * reward_pred
                    + float(self.score_weight) * score_pred
                    + value_term
                    + float(self.terminal_bonus if step_idx == self.horizon else 0.0)
                )
                discounted_return += (float(self.gamma) ** (step_idx - 1)) * step_return
                uncertainty_values.append(max(0.0, uncertainty))
                trace_rows.append(
                    {
                        "step_idx": int(step_idx),
                        "reward_pred": reward_pred,
                        "score_delta_pred": score_pred,
                        "resource_delta_pred": resource_pred,
                        "uncertainty_score": max(0.0, uncertainty),
                        "next_state_proxy": next_state_proxy,
                        "branching_strategy": self.branching_strategy,
                    }
                )
                z_curr = z_next

        mean_uncertainty = sum(uncertainty_values) / max(1, len(uncertainty_values))
        wm_score = float(discounted_return) - float(self.uncertainty_penalty) * float(mean_uncertainty)
        return {
            "candidate_id": str(candidate.get("candidate_id") or ""),
            "predicted_return": float(discounted_return),
            "uncertainty_score": float(mean_uncertainty),
            "wm_score": float(wm_score),
            "horizon_used": int(self.horizon),
            "fallback_happened": False,
            "rollout_trace_summary": trace_rows,
        }

    def rerank_candidates(
        self,
        *,
        obs: dict[str, Any],
        candidates: list[dict[str, Any]],
        baseline_action: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        normalized_candidates = [self._normalize_candidate(obs, candidate, idx + 1) for idx, candidate in enumerate(candidates)]
        for candidate in normalized_candidates:
            row = dict(candidate)
            row.update(self._rollout_candidate(obs, candidate))
            rows.append(row)
        if rows:
            rows.sort(
                key=lambda item: (
                    float(item.get("wm_score") or float("-inf")),
                    float(item.get("source_score") or float("-inf")),
                    -float(item.get("uncertainty_score") or 0.0),
                ),
                reverse=True,
            )
        baseline_token = ""
        if isinstance(baseline_action, dict):
            baseline_token = str(
                baseline_action.get("action_token")
                or baseline_action.get("candidate_id")
                or baseline_action.get("action_type")
                or ""
            )
        before = normalized_candidates[0] if normalized_candidates else {}
        after = rows[0] if rows else {}
        stats = {
            "candidate_count": len(candidates),
            "horizon": int(self.horizon),
            "average_predicted_return": sum(_safe_float(row.get("predicted_return"), 0.0) for row in rows) / max(1, len(rows)),
            "average_uncertainty": sum(_safe_float(row.get("uncertainty_score"), 0.0) for row in rows) / max(1, len(rows)),
            "chosen_action_before": str(before.get("action_token") or before.get("candidate_id") or baseline_token),
            "chosen_action_after": str(after.get("action_token") or after.get("candidate_id") or ""),
            "fallback_happened": not bool(rows) or any(bool(row.get("fallback_happened")) for row in rows),
            "world_model_checkpoint": self.checkpoint_path,
            "assist_mode": "rerank",
            "uncertainty_penalty": float(self.uncertainty_penalty),
            "branching_strategy": self.branching_strategy,
        }
        return {"ranked_candidates": rows, "planner_stats": stats}


def _planner_stats_markdown(stats: dict[str, Any], ranked: list[dict[str, Any]]) -> list[str]:
    lines = [
        f"# P47 Lookahead Planner ({stats.get('run_id')})",
        "",
        f"- checkpoint: `{stats.get('world_model_checkpoint')}`",
        f"- candidate_count: {int(stats.get('candidate_count') or 0)}",
        f"- horizon: {int(stats.get('horizon') or 0)}",
        f"- average_predicted_return: {float(stats.get('average_predicted_return') or 0.0):.6f}",
        f"- average_uncertainty: {float(stats.get('average_uncertainty') or 0.0):.6f}",
        f"- chosen_action_before: `{stats.get('chosen_action_before')}`",
        f"- chosen_action_after: `{stats.get('chosen_action_after')}`",
        f"- fallback_happened: `{str(bool(stats.get('fallback_happened'))).lower()}`",
        "",
        "## Top Candidates",
    ]
    for row in ranked[:5]:
        lines.append(
            "- {candidate} wm_score={wm:.6f} predicted_return={ret:.6f} uncertainty={unc:.6f} source={source}".format(
                candidate=row.get("action_token"),
                wm=float(row.get("wm_score") or 0.0),
                ret=float(row.get("predicted_return") or 0.0),
                unc=float(row.get("uncertainty_score") or 0.0),
                source=row.get("source"),
            )
        )
    return lines


def run_planner_smoke(
    *,
    config_path: str | Path = "configs/experiments/p47_wm_search_smoke.yaml",
    out_dir: str | Path | None = None,
    run_id: str = "",
    quick: bool = False,
    checkpoint_override: str = "",
    source_override: str = "",
    model_path_override: str = "",
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = (repo_root / cfg_path).resolve()
    cfg = _read_yaml_or_json(cfg_path) if cfg_path.exists() else {}
    output_cfg = cfg.get("output") if isinstance(cfg.get("output"), dict) else {}
    planner_cfg = cfg.get("planner") if isinstance(cfg.get("planner"), dict) else {}
    candidate_cfg = cfg.get("candidate_actions") if isinstance(cfg.get("candidate_actions"), dict) else {}
    run_name = str(run_id or output_cfg.get("run_id") or _now_stamp())
    output_root = (
        (repo_root / str(output_cfg.get("artifacts_root") or "docs/artifacts/p47/lookahead")).resolve()
        if out_dir is None
        else (Path(out_dir) if Path(out_dir).is_absolute() else (repo_root / out_dir).resolve())
    )
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = _resolve_checkpoint(repo_root, cfg, override=checkpoint_override)
    if not checkpoint_path or not Path(checkpoint_path).exists():
        raise FileNotFoundError("world model checkpoint unavailable for P47 lookahead planner")

    seed = str(candidate_cfg.get("seed") or "AAAAAAA")
    if quick:
        top_k = min(max(2, _safe_int(candidate_cfg.get("top_k"), 4)), 4)
        horizon = min(max(1, _safe_int(planner_cfg.get("horizon"), 1)), 2)
    else:
        top_k = max(1, _safe_int(candidate_cfg.get("top_k"), 4))
        horizon = max(1, _safe_int(planner_cfg.get("horizon"), 1))
    state = capture_sample_state(seed=seed, target_phase=str(candidate_cfg.get("target_phase") or "SELECTING_HAND"))
    source = str(source_override or candidate_cfg.get("source") or "heuristic_candidates")
    model_path = str(model_path_override or candidate_cfg.get("model_path") or "")
    candidate_payload = generate_candidate_actions(
        obs=state,
        source=source,
        top_k=top_k,
        legal_actions=None,
        seed=seed,
        model_path=model_path,
        search_max_branch=_safe_int(candidate_cfg.get("search_max_branch"), 80),
        search_max_depth=_safe_int(candidate_cfg.get("search_max_depth"), 2),
        search_time_budget_ms=_safe_float(candidate_cfg.get("search_time_budget_ms"), 15.0),
    )

    planner = WorldModelLookaheadPlanner(
        checkpoint_path=checkpoint_path,
        horizon=horizon,
        gamma=_safe_float(planner_cfg.get("gamma"), 0.95),
        uncertainty_penalty=_safe_float(planner_cfg.get("uncertainty_penalty"), 0.5),
        reward_weight=_safe_float(planner_cfg.get("reward_weight"), 1.0),
        score_weight=_safe_float(planner_cfg.get("score_weight"), 0.5),
        value_weight=_safe_float(planner_cfg.get("value_weight"), 0.15),
        terminal_bonus=_safe_float(planner_cfg.get("terminal_bonus"), 0.0),
        branching_strategy=str(planner_cfg.get("branching_strategy") or "repeat_action"),
    )
    reranked = planner.rerank_candidates(obs=state, candidates=list(candidate_payload.get("candidates") or []))
    ranked = list(reranked.get("ranked_candidates") or [])
    planner_stats = dict(reranked.get("planner_stats") or {})
    planner_stats["schema"] = "p47_lookahead_planner_stats_v1"
    planner_stats["generated_at"] = _now_iso()
    planner_stats["run_id"] = run_name
    planner_stats["candidate_source"] = str(candidate_payload.get("source") or source)
    planner_stats["phase"] = phase_from_obs(state)
    planner_stats["top_k"] = int(top_k)
    planner_stats["wm_checkpoint"] = checkpoint_path
    planner_stats["candidate_artifact"] = str(run_dir / "candidate_snapshot.json")
    planner_stats["planner_trace_jsonl"] = str(run_dir / "planner_trace.jsonl")

    trace_path = run_dir / "planner_trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()
    for row in ranked:
        trace_row = {
            "schema": "p47_planner_trace_row_v1",
            "generated_at": _now_iso(),
            "run_id": run_name,
            "phase": phase_from_obs(state),
            "candidate_id": str(row.get("candidate_id") or ""),
            "action_token": str(row.get("action_token") or ""),
            "source": str(row.get("source") or ""),
            "source_rank": _safe_int(row.get("source_rank"), 0),
            "source_score": _safe_float(row.get("source_score"), 0.0),
            "predicted_return": _safe_float(row.get("predicted_return"), 0.0),
            "uncertainty_score": _safe_float(row.get("uncertainty_score"), 0.0),
            "wm_score": _safe_float(row.get("wm_score"), 0.0),
            "horizon_used": _safe_int(row.get("horizon_used"), 0),
            "rollout_trace_summary": row.get("rollout_trace_summary") if isinstance(row.get("rollout_trace_summary"), list) else [],
        }
        _append_jsonl(trace_path, trace_row)

    write_json(run_dir / "candidate_snapshot.json", candidate_payload)
    write_json(run_dir / "planner_stats.json", planner_stats)
    write_markdown(run_dir / "planner_stats.md", _planner_stats_markdown(planner_stats, ranked))
    summary = {
        "status": "ok" if ranked else "stub",
        "run_id": run_name,
        "run_dir": str(run_dir),
        "planner_stats_json": str(run_dir / "planner_stats.json"),
        "planner_stats_md": str(run_dir / "planner_stats.md"),
        "planner_trace_jsonl": str(trace_path),
        "candidate_snapshot_json": str(run_dir / "candidate_snapshot.json"),
        "top_action_before": str(planner_stats.get("chosen_action_before") or ""),
        "top_action_after": str(planner_stats.get("chosen_action_after") or ""),
        "world_model_checkpoint": checkpoint_path,
        "candidate_source": str(candidate_payload.get("source") or source),
    }
    write_json(run_dir / "planner_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P47 uncertainty-aware world-model lookahead planner.")
    parser.add_argument("--config", default="configs/experiments/p47_wm_search_smoke.yaml")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--source", default="")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--sources", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_planner_smoke(
        config_path=args.config,
        out_dir=(args.out_dir if str(args.out_dir).strip() else None),
        run_id=str(args.run_id or ""),
        quick=bool(args.quick),
        checkpoint_override=str(args.checkpoint or ""),
        source_override=str(args.source or (_parse_csv(args.sources)[0] if _parse_csv(args.sources) else "")),
        model_path_override=str(args.model_path or ""),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if str(summary.get("status")) in {"ok", "stub"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
