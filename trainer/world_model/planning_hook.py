from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from trainer import action_space, action_space_shop
from trainer.features import extract_features
from trainer.features_shop import extract_shop_features
from trainer.policy_arena.policy_adapter import normalize_action, phase_from_obs
from trainer.world_model.model import load_world_model_from_checkpoint
from trainer.world_model.schema import action_token_from_parts, stable_hash_int


PHASE_TO_ID = {
    "BLIND_SELECT": 0,
    "SELECTING_HAND": 1,
    "ROUND_EVAL": 2,
    "SHOP": 3,
    "SMODS_BOOSTER_OPENED": 4,
    "MENU": 5,
    "GAME_OVER": 6,
    "OTHER": 7,
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - runtime guarded
        raise RuntimeError("PyTorch is required for trainer.world_model.planning_hook") from exc
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


def _vectorize_obs_state(state: dict[str, Any], *, action_dim: int) -> list[float]:
    phase = str(state.get("state") or "OTHER")
    phase_id = PHASE_TO_ID.get(phase, PHASE_TO_ID["OTHER"])
    if phase == "SELECTING_HAND":
        hand_cards = ((state.get("hand") or {}).get("cards") or []) if isinstance(state.get("hand"), dict) else []
        hand_size = min(len(hand_cards), action_space.MAX_HAND)
        legal_ids = action_space.legal_action_ids(hand_size) if hand_size > 0 else []
    elif phase in action_space_shop.SHOP_PHASES:
        legal_ids = action_space_shop.legal_action_ids(state)
    else:
        legal_ids = []

    feat = extract_features(state)
    shop_feat = extract_shop_features(state)
    context = list(feat.get("context") or [0.0] * 12)
    if len(context) < 12:
        context = (context + [0.0] * 12)[:12]
    norm_context = [
        float(context[0]) / 5.0,
        float(context[1]) / 5.0,
        float(context[2]) / 5000.0,
        float(context[3]) / 100.0,
        float(context[4]) / 20.0,
        float(context[5]) / 5000.0,
        float(context[6]) / 5000.0,
        float(context[7]) / 5000.0,
        float(context[8]) / 10.0,
        float(context[9]) / 4.0,
        float(context[10]) / 8.0,
        float(context[11]) / 20.0,
    ]
    shop_context = list(shop_feat.get("shop_context") or [0.0] * 16)
    if len(shop_context) < 16:
        shop_context = (shop_context + [0.0] * 16)[:16]
    rank_ids = list(feat.get("card_rank_ids") or [])
    chip_hint = list(feat.get("card_chip_hint") or [])
    enh_flags = list(feat.get("card_has_enhancement") or [])
    edt_flags = list(feat.get("card_has_edition") or [])
    seal_flags = list(feat.get("card_has_seal") or [])
    hand_size = int(feat.get("hand_size") or 0)
    divisor = max(1, hand_size)
    rank_mean = (sum(float(x) for x in rank_ids[: action_space.MAX_HAND]) / divisor) / 14.0
    rank_max = (max([float(x) for x in rank_ids[: action_space.MAX_HAND]] or [0.0])) / 14.0
    chip_mean = (sum(float(x) for x in chip_hint[: action_space.MAX_HAND]) / divisor) / 20.0
    enh_mean = (sum(float(x) for x in enh_flags[: action_space.MAX_HAND]) / divisor)
    edt_mean = (sum(float(x) for x in edt_flags[: action_space.MAX_HAND]) / divisor)
    seal_mean = (sum(float(x) for x in seal_flags[: action_space.MAX_HAND]) / divisor)
    phase_onehot = [0.0] * len(PHASE_TO_ID)
    phase_onehot[min(phase_id, len(phase_onehot) - 1)] = 1.0
    misc = [
        float(hand_size) / float(max(1, action_space.MAX_HAND)),
        float(len(legal_ids)) / float(max(1, action_dim)),
        float(state.get("ante_num") or 0.0) / 8.0,
        float(state.get("round_num") or 0.0) / 20.0,
        float((state.get("round") or {}).get("hands_left") or 0.0) / 5.0,
        float((state.get("round") or {}).get("discards_left") or 0.0) / 5.0,
    ]
    vector = [
        *norm_context,
        *shop_context[:16],
        rank_mean,
        rank_max,
        chip_mean,
        enh_mean,
        edt_mean,
        seal_mean,
        *phase_onehot,
        *misc,
    ]
    if len(vector) < 48:
        vector.extend([0.0] * (48 - len(vector)))
    return vector[:48]


def _action_from_legal_entry(entry: dict[str, Any], *, phase: str) -> dict[str, Any]:
    if isinstance(entry.get("action"), dict):
        return normalize_action(dict(entry.get("action") or {}), phase=phase)
    if str(entry.get("action_type") or "").strip():
        return normalize_action(dict(entry), phase=phase)
    if entry.get("id") is not None:
        return {"action_type": f"ACTION_{int(entry.get('id'))}", "id": int(entry.get("id"))}
    return {"action_type": "WAIT"}


class WorldModelPlanningHook:
    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        assist_mode: str = "one_step_heuristic",
        weight: float = 0.35,
        uncertainty_penalty: float = 0.5,
    ) -> None:
        self.checkpoint_path = str(checkpoint_path or "")
        self.assist_mode = str(assist_mode or "one_step_heuristic")
        self.weight = float(weight)
        self.uncertainty_penalty = float(uncertainty_penalty)
        self.available = bool(self.checkpoint_path) and Path(self.checkpoint_path).exists()
        self._model = None
        self._action_vocab_size = 4096
        self._obs_dim = 48
        if self.available:
            self._model, model_cfg, _payload = load_world_model_from_checkpoint(self.checkpoint_path, device="cpu")
            self._action_vocab_size = int(model_cfg.action_vocab_size)
            self._obs_dim = int(model_cfg.input_dim)

    def describe(self) -> dict[str, Any]:
        return {
            "world_model_assist": bool(self.available),
            "checkpoint_path": self.checkpoint_path,
            "assist_mode": self.assist_mode,
            "weight": float(self.weight),
            "uncertainty_penalty": float(self.uncertainty_penalty),
        }

    def score_candidates(
        self,
        *,
        obs: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        fallback_action: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.available or self._model is None or not legal_actions:
            return []

        torch = _require_torch()
        phase = phase_from_obs(obs)
        obs_vector = _vectorize_obs_state(obs, action_dim=max(action_space.max_actions(), action_space_shop.max_actions()))
        action_rows: list[dict[str, Any]] = []
        fallback_token = ""
        if isinstance(fallback_action, dict):
            fallback_token = action_token_from_parts(
                phase=phase,
                action_type=str(fallback_action.get("action_type") or "OTHER"),
                action_payload=fallback_action,
                numeric_action=(int(fallback_action.get("id")) if fallback_action.get("id") is not None else None),
            )
        for entry in legal_actions:
            if not isinstance(entry, dict):
                continue
            action = _action_from_legal_entry(entry, phase=phase)
            token = action_token_from_parts(
                phase=phase,
                action_type=str(action.get("action_type") or "OTHER"),
                action_payload=action,
                numeric_action=(int(entry.get("id")) if entry.get("id") is not None else None),
            )
            action_rows.append(
                {
                    "action": action,
                    "action_token": token,
                    "action_id": stable_hash_int(token, self._action_vocab_size),
                }
            )
        if not action_rows:
            return []

        with torch.no_grad():
            obs_t = torch.tensor([list(obs_vector) for _ in action_rows], dtype=torch.float32)
            action_ids = torch.tensor([int(row.get("action_id") or 0) for row in action_rows], dtype=torch.long)
            outputs = self._model(obs_t, action_ids)
            reward_pred = outputs["reward_pred"].detach().cpu().tolist()
            score_pred = outputs["score_pred"].detach().cpu().tolist()
            next_state_proxy = outputs["next_state_proxy"].detach().cpu().tolist()
            uncertainty_pred = outputs["uncertainty_pred"].detach().cpu().tolist()
            resource_pred = outputs["resource_pred"].detach().cpu().tolist()

        ranked: list[dict[str, Any]] = []
        for idx, row in enumerate(action_rows):
            resource = list(resource_pred[idx] or [])
            money_proxy = _safe_float(resource[1] if len(resource) > 1 else 0.0, 0.0)
            hand_proxy = _safe_float(resource[3] if len(resource) > 3 else 0.0, 0.0)
            discard_proxy = _safe_float(resource[4] if len(resource) > 4 else 0.0, 0.0)
            uncertainty = max(0.0, _safe_float(uncertainty_pred[idx], 0.0))
            wm_score = (
                _safe_float(reward_pred[idx], 0.0)
                + 0.5 * _safe_float(score_pred[idx], 0.0)
                + 0.1 * money_proxy
                + 0.05 * hand_proxy
                + 0.05 * discard_proxy
                + 0.1 * _safe_float(next_state_proxy[idx], 0.0)
            )
            stabilized = float(self.weight) * (wm_score / (1.0 + float(self.uncertainty_penalty) * uncertainty))
            if fallback_token and str(row.get("action_token") or "") == fallback_token:
                stabilized += 0.05
            ranked.append(
                {
                    "action": row.get("action") or {},
                    "action_token": str(row.get("action_token") or ""),
                    "wm_score": wm_score,
                    "effective_score": stabilized,
                    "uncertainty": uncertainty,
                    "reward_pred": _safe_float(reward_pred[idx], 0.0),
                    "score_pred": _safe_float(score_pred[idx], 0.0),
                }
            )
        ranked.sort(key=lambda item: (float(item.get("effective_score") or 0.0), -float(item.get("uncertainty") or 0.0)), reverse=True)
        return ranked


def run_world_model_assist_compare(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    out_dir: str | Path | None = None,
    run_id: str = "",
    quick: bool = False,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = (repo_root / cfg_path).resolve()
    cfg = _read_yaml_or_json(cfg_path)
    arena_cfg = cfg.get("arena_compare") if isinstance(cfg.get("arena_compare"), dict) else {}
    planning_cfg = cfg.get("planning") if isinstance(cfg.get("planning"), dict) else {}
    chosen_run_id = str(run_id or run_id or cfg.get("run_id") or "wm_assist")
    out_root = (
        (repo_root / str(arena_cfg.get("output_artifacts_root") or "docs/artifacts/p45/wm_assist_compare")).resolve()
        if out_dir is None
        else (Path(out_dir) if Path(out_dir).is_absolute() else (repo_root / out_dir).resolve())
    )
    run_dir = out_root / (chosen_run_id if chosen_run_id else "wm_assist_compare")
    run_dir.mkdir(parents=True, exist_ok=True)

    seeds = arena_cfg.get("seeds") if isinstance(arena_cfg.get("seeds"), list) else ["AAAAAAA", "BBBBBBB"]
    seeds = [str(seed).strip() for seed in seeds if str(seed).strip()]
    if quick and len(seeds) > 2:
        seeds = seeds[:2]
    episodes_per_seed = max(1, int(arena_cfg.get("episodes_per_seed") or 1))
    max_steps = max(1, int(arena_cfg.get("max_steps") or 120))
    if quick:
        episodes_per_seed = min(1, episodes_per_seed)
        max_steps = min(max_steps, 120)

    cmd = [
        sys.executable,
        "-B",
        "-m",
        "trainer.policy_arena.arena_runner",
        "--out-dir",
        str(run_dir),
        "--run-id",
        chosen_run_id or "wm_assist_compare",
        "--backend",
        "sim",
        "--mode",
        "long_episode",
        "--policies",
        ",".join(
            [
                str(arena_cfg.get("champion_policy") or "heuristic_baseline"),
                str(arena_cfg.get("candidate_policy") or "heuristic_wm_assist"),
            ]
        ),
        "--seeds",
        ",".join(seeds),
        "--episodes-per-seed",
        str(episodes_per_seed),
        "--max-steps",
        str(max_steps),
        "--world-model-checkpoint",
        str(Path(checkpoint_path).resolve()),
        "--world-model-assist-mode",
        str(planning_cfg.get("assist_mode") or "one_step_heuristic"),
        "--world-model-weight",
        str(float(planning_cfg.get("weight") or 0.35)),
        "--world-model-uncertainty-penalty",
        str(float(planning_cfg.get("uncertainty_penalty") or 0.5)),
    ]
    if quick:
        cmd.append("--quick")

    proc = subprocess.run(cmd, cwd=str(repo_root), text=True, capture_output=True)
    stdout_lines = [line for line in str(proc.stdout or "").splitlines() if line.strip()]
    parsed = {}
    if stdout_lines:
        try:
            parsed = json.loads(stdout_lines[-1])
        except Exception:
            parsed = {}
    summary = {
        "schema": "p45_world_model_assist_compare_v1",
        "generated_at": now_iso(),
        "status": "ok" if int(proc.returncode) == 0 else "failed",
        "returncode": int(proc.returncode),
        "command": cmd,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "world_model_assist": True,
        "assist_mode": str(planning_cfg.get("assist_mode") or "one_step_heuristic"),
        "weight": float(planning_cfg.get("weight") or 0.35),
        "uncertainty_penalty": float(planning_cfg.get("uncertainty_penalty") or 0.5),
        "arena_summary": parsed if isinstance(parsed, dict) else {},
        "stdout_tail": stdout_lines[-20:],
        "stderr_tail": str(proc.stderr or "").splitlines()[-20:],
    }
    summary_path = run_dir / "assist_compare_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "status": str(summary.get("status") or "failed"),
        "run_dir": str(run_dir),
        "assist_compare_summary_json": str(summary_path),
        "summary": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a P45 world-model-assisted policy arena compare.")
    parser.add_argument("--config", default="configs/experiments/p45_world_model_smoke.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_world_model_assist_compare(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        out_dir=(args.out_dir if str(args.out_dir).strip() else None),
        run_id=str(args.run_id or ""),
        quick=bool(args.quick),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if str(summary.get("status")) == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
