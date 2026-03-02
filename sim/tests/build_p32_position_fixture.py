from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from sim.core.canonical import to_canonical_state
from sim.core.engine import SimEnv
from sim.core.hashing import (
    state_hash_full,
    state_hash_hand_core,
    state_hash_p14_real_action_observed_core,
    state_hash_p32_real_action_position_observed_core,
)
from sim.core.score_observed import compute_score_observed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic P32 position-action replay fixture.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", default="P32POS01")
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _starter_state(seed: str) -> dict[str, Any]:
    env = SimEnv(seed=seed)
    env.reset(seed=seed)
    env.step({"action_type": "SELECT", "index": 0})
    env._state["jokers"] = [  # noqa: SLF001 - deterministic fixture seeding
        {"joker_id": "j_blueprint", "key": "j_blueprint"},
        {"joker_id": "j_brainstorm", "key": "j_brainstorm"},
        {"joker_id": "j_joker", "key": "j_joker"},
    ]
    env._state["state"] = "SELECTING_HAND"  # noqa: SLF001
    env._state["round"]["hands_left"] = max(1, int((env._state.get("round") or {}).get("hands_left") or 1))  # noqa: SLF001
    return copy.deepcopy(env._state)  # noqa: SLF001


def _build_actions(start_state: dict[str, Any]) -> list[dict[str, Any]]:
    hand_cards = (start_state.get("hand") or {}).get("cards") if isinstance(start_state.get("hand"), dict) else []
    hand_count = len(hand_cards or [])
    if hand_count < 2:
        raise RuntimeError("starter hand is too small for reorder fixture")

    reorder = list(range(hand_count))
    reorder[0], reorder[1] = reorder[1], reorder[0]
    swap_j = [0, 2]
    replay_stub = {"enabled": False, "source": "", "outcomes": []}

    return [
        {"schema_version": "action_v1", "phase": "SELECTING_HAND", "action_type": "REORDER_HAND", "permutation": reorder, "rng_replay": replay_stub},
        {"schema_version": "action_v1", "phase": "SELECTING_HAND", "action_type": "SWAP_HAND_CARDS", "i": 0, "j": hand_count - 1, "rng_replay": replay_stub},
        {"schema_version": "action_v1", "phase": "SELECTING_HAND", "action_type": "REORDER_JOKERS", "permutation": [2, 0, 1], "rng_replay": replay_stub},
        {"schema_version": "action_v1", "phase": "SELECTING_HAND", "action_type": "SWAP_JOKERS", "i": swap_j[0], "j": swap_j[1], "rng_replay": replay_stub},
        {"schema_version": "action_v1", "phase": "SELECTING_HAND", "action_type": "PLAY", "indices": [0], "rng_replay": replay_stub},
    ]


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_raw = _starter_state(args.seed)
    start_snapshot = to_canonical_state(start_raw, rng_mode="native", seed=args.seed, rng_cursor=0, rng_events=[])
    actions = _build_actions(start_raw)

    replay = SimEnv(seed=args.seed)
    replay.reset(from_snapshot=start_snapshot)
    oracle_rows: list[dict[str, Any]] = []

    state = replay.get_state()
    for step_id, action in enumerate(actions):
        action_to_run = dict(action)
        action_to_run["phase"] = str(state.get("state") or action_to_run.get("phase") or "UNKNOWN")
        before = copy.deepcopy(state)
        next_state, reward, done, info = replay.step(action_to_run)
        observed = compute_score_observed(before, next_state)

        canonical = to_canonical_state(next_state, rng_mode="native", seed=args.seed, rng_cursor=0, rng_events=[])
        canonical_obs = dict(canonical)
        canonical_obs["score_observed"] = dict(observed)
        canonical_obs["rng_replay"] = {"enabled": False, "source": "", "outcomes": []}

        oracle_rows.append(
            {
                "schema_version": "trace_v1",
                "step_id": step_id,
                "phase": str(canonical.get("phase") or "UNKNOWN"),
                "action": action_to_run,
                "state_hash_full": state_hash_full(canonical),
                "state_hash_hand_core": state_hash_hand_core(canonical),
                "state_hash_p14_real_action_observed_core": state_hash_p14_real_action_observed_core(canonical_obs),
                "state_hash_p32_real_action_position_observed_core": state_hash_p32_real_action_position_observed_core(canonical_obs),
                "reward": float(reward),
                "done": bool(done),
                "score_observed": observed,
                "rng_replay": {"enabled": False, "source": "", "outcomes": []},
                "info": {"source": "p32_position_fixture", "engine_info": info},
                "canonical_state_snapshot": canonical,
            }
        )
        state = next_state

    _write_json(out_dir / "oracle_start_snapshot_real.json", start_snapshot)
    _write_jsonl(out_dir / "action_trace_real.jsonl", actions)
    _write_jsonl(out_dir / "oracle_trace_real.jsonl", oracle_rows)
    _write_json(
        out_dir / "manifest.json",
        {
            "fixture_type": "p32_position_actions",
            "seed": args.seed,
            "actions_count": len(actions),
            "action_types": [str(a.get("action_type") or "").upper() for a in actions],
            "out_dir": str(out_dir),
        },
    )

    print(json.dumps({"out_dir": str(out_dir), "actions_count": len(actions)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
