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

from trainer.rl.env_adapter import RLEnvAdapter


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_env_smoke(
    *,
    seed: str,
    max_steps: int,
    out_root: str | Path,
) -> dict[str, Any]:
    rng = random.Random(str(seed))
    adapter = RLEnvAdapter(
        backend="sim",
        seed=str(seed),
        max_steps_per_episode=max(16, int(max_steps)),
        reward_config={
            "score_delta_weight": 1.0,
            "survival_bonus": 0.01,
            "terminal_win_bonus": 0.5,
            "terminal_loss_penalty": 0.5,
            "invalid_action_penalty": 0.1,
            "clip_abs": 2.0,
            "invalid_action_mode": "fallback_random_legal",
        },
    )

    steps: list[dict[str, Any]] = []
    terminated = False
    truncated = False
    try:
        obs, info = adapter.reset(seed=seed)
        for step_idx in range(max(1, int(max_steps))):
            mask = adapter.get_action_mask(obs, info)
            legal_ids = [idx for idx, val in enumerate(mask) if int(val) > 0]
            if legal_ids:
                action = int(rng.choice(legal_ids))
                if step_idx == 0:
                    action = max(legal_ids) + 3  # inject one invalid request to validate fallback accounting
            else:
                action = 0
            next_obs, reward, terminated, truncated, step_info = adapter.step(action)
            steps.append(
                {
                    "step_idx": int(step_idx),
                    "action_requested": int(action),
                    "action_applied": int(step_info.get("action_applied") or 0),
                    "reward": float(reward),
                    "phase": str(step_info.get("phase") or ""),
                    "score_delta": float(step_info.get("score_delta") or 0.0),
                    "invalid_action": bool(step_info.get("invalid_action")),
                    "mask_density": float(step_info.get("mask_density") or 0.0),
                }
            )
            obs, info = next_obs, step_info
            if terminated or truncated:
                break
    finally:
        adapter.close()

    out_dir = Path(out_root).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"env_smoke_{_now_stamp()}.json"
    invalid_count = sum(1 for row in steps if bool(row.get("invalid_action")))
    payload = {
        "schema": "p42_env_smoke_v1",
        "generated_at": _now_iso(),
        "seed": str(seed),
        "step_count": len(steps),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "invalid_action_count": int(invalid_count),
        "invalid_action_rate": (float(invalid_count) / max(1, len(steps))),
        "steps": steps,
    }
    _write_json(out_path, payload)
    return {
        "status": "ok",
        "env_smoke_json": str(out_path),
        "step_count": len(steps),
        "invalid_action_rate": payload["invalid_action_rate"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P42 smoke test for RL env adapter.")
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--out-root", default="docs/artifacts/p42")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_env_smoke(
        seed=str(args.seed),
        max_steps=max(1, int(args.max_steps)),
        out_root=args.out_root,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

