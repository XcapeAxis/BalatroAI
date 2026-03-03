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

from sim.core.engine import SimEnv
from trainer.real_observer import build_observation
from trainer.utils import timestamp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build synthetic real-session style jsonl for P37.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", default="P37SYN01")
    parser.add_argument(
        "--scenario",
        choices=["explicit_position", "inferred_position", "shop_consumable"],
        required=True,
    )
    return parser.parse_args()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _starter_env(seed: str, scenario: str) -> SimEnv:
    env = SimEnv(seed=seed)
    env.reset(seed=seed)
    env.step({"action_type": "SELECT", "index": 0})
    if scenario in {"explicit_position", "inferred_position"}:
        env._state["jokers"] = [  # noqa: SLF001 - synthetic fixture setup
            {"joker_id": "j_blueprint", "key": "j_blueprint"},
            {"joker_id": "j_brainstorm", "key": "j_brainstorm"},
            {"joker_id": "j_joker", "key": "j_joker"},
        ]
    if scenario == "shop_consumable":
        env._state["state"] = "SHOP"  # noqa: SLF001
        env._state["money"] = 20  # noqa: SLF001
        env._state["round"]["reroll_cost"] = 5  # noqa: SLF001
        env._state["consumables"] = {  # noqa: SLF001
            "count": 1,
            "limit": 2,
            "highlighted_limit": 1,
            "cards": [{"key": "c_death", "label": "Death", "set": "TAROT"}],
        }
        env._state["packs"] = {  # noqa: SLF001
            "count": 1,
            "limit": 1,
            "highlighted_limit": 1,
            "cards": [{"key": "p_arcana_mega_1", "label": "Arcana Pack", "set": "PACK", "cost": {"buy": 4.0}, "slot_index": 0}],
        }
        env._state["shop"] = {  # noqa: SLF001
            "count": 1,
            "limit": 2,
            "highlighted_limit": 1,
            "cards": [{"key": "j_joker", "label": "Joker", "set": "JOKER", "cost": {"buy": 2.0}, "slot_index": 0}],
        }
    return env


def _scenario_actions(scenario: str) -> list[dict[str, Any]]:
    if scenario in {"explicit_position", "inferred_position"}:
        return [
            {"schema_version": "action_v1", "phase": "SELECTING_HAND", "action_type": "MOVE_HAND_CARD", "src_index": 0, "dst_index": 3, "index_base": 0, "params": {"index_base": 0}},
            {"schema_version": "action_v1", "phase": "SELECTING_HAND", "action_type": "MOVE_JOKER", "src_index": 0, "dst_index": 2, "index_base": 0, "params": {"index_base": 0}},
            {"schema_version": "action_v1", "phase": "SELECTING_HAND", "action_type": "PLAY", "indices": [0]},
        ]
    return [
        {
            "schema_version": "action_v1",
            "phase": "SHOP",
            "action_type": "CONSUMABLE_USE",
            "kind": "tarot",
            "key": "c_death",
            "params": {"consumable_index": 0, "hand_indices": [0, 1], "target_side": "left", "index_base": 0},
        },
        {"schema_version": "action_v1", "phase": "SHOP", "action_type": "SHOP_REROLL"},
        {"schema_version": "action_v1", "phase": "SHOP", "action_type": "SHOP_BUY", "params": {"pack_index": 0, "index_base": 0}},
        {"schema_version": "action_v1", "phase": "SMODS_BOOSTER_OPENED", "action_type": "PACK_OPEN", "params": {"choice_index": 0, "index_base": 0}},
    ]


def main() -> int:
    args = _parse_args()
    out_path = Path(args.out)
    env = _starter_env(args.seed, args.scenario)
    actions = _scenario_actions(args.scenario)

    rows: list[dict[str, Any]] = []
    for step_idx, action in enumerate(actions):
        before = copy.deepcopy(env.get_state())
        action_use = dict(action)
        action_use["phase"] = str(before.get("state") or action_use.get("phase") or "UNKNOWN")
        after, reward, done, info = env.step(action_use)
        obs_before = build_observation(before)
        obs_after = build_observation(after)

        row: dict[str, Any] = {
            "ts": timestamp(),
            "step_idx": step_idx,
            "base_url": "synthetic://sim",
            "mode": "synthetic_execute",
            "phase": str(before.get("state") or "UNKNOWN"),
            "gamestate_min": obs_before,
            "gamestate_min_before": obs_before,
            "gamestate_min_after": obs_after,
            "action_result": {"reward": float(reward), "done": bool(done), "info": info},
            "state_changed": True,
            "errors": [],
            "gamestate_raw_before": before,
            "gamestate_raw_after": after,
        }
        if args.scenario == "inferred_position":
            row["action_sent"] = None
            row["action_normalized"] = None
        else:
            row["action_sent"] = action_use
            row["action_normalized"] = action_use
        rows.append(row)

    _write_jsonl(out_path, rows)
    summary = {
        "schema": "p37_synthetic_session_v1",
        "seed": args.seed,
        "scenario": args.scenario,
        "steps": len(rows),
        "out": str(out_path),
        "action_types": [str(a.get("action_type") or "") for a in actions],
    }
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
