from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from sim.core.canonical import to_canonical_state
from sim.core.engine import SimEnv
from trainer.real_observer import build_observation
from trainer.utils import timestamp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build synthetic session jsonl with position actions for P32 round-trip.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", default="P32SYN01")
    return parser.parse_args()


def _initial_state(seed: str) -> dict[str, Any]:
    env = SimEnv(seed=seed)
    env.reset(seed=seed)
    env.step({"action_type": "SELECT", "index": 0})
    env._state["jokers"] = [  # noqa: SLF001 - synthetic fixture setup
        {"joker_id": "j_blueprint", "key": "j_blueprint"},
        {"joker_id": "j_brainstorm", "key": "j_brainstorm"},
        {"joker_id": "j_joker", "key": "j_joker"},
    ]
    env._state["state"] = "SELECTING_HAND"  # noqa: SLF001
    return copy.deepcopy(env._state)  # noqa: SLF001


def _action_trace_from_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    hand_cards = (state.get("hand") or {}).get("cards") if isinstance(state.get("hand"), dict) else []
    hand_count = len(hand_cards or [])
    if hand_count < 2:
        raise RuntimeError("starter hand too small for synthetic position session")
    perm = list(range(hand_count))
    perm[0], perm[1] = perm[1], perm[0]
    return [
        {"schema_version": "action_v1", "phase": "SELECTING_HAND", "action_type": "REORDER_HAND", "permutation": perm},
        {"schema_version": "action_v1", "phase": "SELECTING_HAND", "action_type": "SWAP_HAND_CARDS", "i": 0, "j": hand_count - 1},
        {"schema_version": "action_v1", "phase": "SELECTING_HAND", "action_type": "REORDER_JOKERS", "permutation": [2, 0, 1]},
        {"schema_version": "action_v1", "phase": "SELECTING_HAND", "action_type": "SWAP_JOKERS", "i": 0, "j": 2},
        {"schema_version": "action_v1", "phase": "SELECTING_HAND", "action_type": "PLAY", "indices": [0]},
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = _parse_args()
    out_path = Path(args.out)
    start = _initial_state(args.seed)
    actions = _action_trace_from_state(start)
    start_snapshot = to_canonical_state(start, rng_mode="native", seed=args.seed, rng_cursor=0, rng_events=[])

    env = SimEnv(seed=args.seed)
    env.reset(from_snapshot=start_snapshot)

    rows: list[dict[str, Any]] = []
    for step_idx, action in enumerate(actions):
        before = copy.deepcopy(env.get_state())
        obs_before = build_observation(before)
        action_use = dict(action)
        action_use["phase"] = str(before.get("state") or action_use.get("phase") or "UNKNOWN")
        after, reward, done, info = env.step(action_use)
        obs_after = build_observation(after)

        row = {
            "ts": timestamp(),
            "step_idx": step_idx,
            "base_url": "synthetic://sim",
            "mode": "synthetic_execute",
            "phase": str(before.get("state") or "UNKNOWN"),
            "gamestate_min": obs_before,
            "gamestate_min_before": obs_before,
            "gamestate_min_after": obs_after,
            "action_sent": action_use,
            "action_result": {"reward": float(reward), "done": bool(done), "info": info},
            "state_changed": True,
            "errors": [],
            "gamestate_raw_before": before,
            "gamestate_raw_after": after,
        }
        rows.append(row)

    _write_jsonl(out_path, rows)
    summary = {
        "schema": "p32_synthetic_session_v1",
        "seed": args.seed,
        "steps": len(rows),
        "out": str(out_path),
        "action_types": [str((r.get("action_sent") or {}).get("action_type") or "") for r in rows],
    }
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
