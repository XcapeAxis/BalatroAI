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
    state_hash_p37_action_fidelity_core,
)
from sim.core.score_observed import compute_score_observed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic P37 action-fidelity fixture.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", default="P37FIX01")
    parser.add_argument(
        "--scenario",
        choices=["hand_move_play", "joker_move_play", "shop_ops"],
        required=True,
    )
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _starter_env(seed: str, scenario: str) -> SimEnv:
    env = SimEnv(seed=seed)
    env.reset(seed=seed)
    env.step({"action_type": "SELECT", "index": 0})
    if scenario in {"joker_move_play", "shop_ops"}:
        env._state["jokers"] = [  # noqa: SLF001 - deterministic fixture seeding
            {"joker_id": "j_blueprint", "key": "j_blueprint"},
            {"joker_id": "j_brainstorm", "key": "j_brainstorm"},
            {"joker_id": "j_joker", "key": "j_joker"},
        ]
    if scenario == "shop_ops":
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


def _scenario_actions(state: dict[str, Any], scenario: str) -> list[dict[str, Any]]:
    replay_stub = {"enabled": False, "source": "", "outcomes": []}
    if scenario == "hand_move_play":
        hand_cards = (state.get("hand") or {}).get("cards") if isinstance(state.get("hand"), dict) else []
        if not hand_cards or len(hand_cards) < 4:
            raise RuntimeError("hand_move_play requires hand size >= 4")
        return [
            {
                "schema_version": "action_v1",
                "phase": "SELECTING_HAND",
                "action_type": "MOVE_HAND_CARD",
                "src_index": 0,
                "dst_index": 3,
                "index_base": 0,
                "params": {"index_base": 0},
                "rng_replay": replay_stub,
            },
            {"schema_version": "action_v1", "phase": "SELECTING_HAND", "action_type": "PLAY", "indices": [0], "rng_replay": replay_stub},
        ]
    if scenario == "joker_move_play":
        return [
            {
                "schema_version": "action_v1",
                "phase": "SELECTING_HAND",
                "action_type": "MOVE_JOKER",
                "src_index": 0,
                "dst_index": 2,
                "index_base": 0,
                "params": {"index_base": 0},
                "rng_replay": replay_stub,
            },
            {"schema_version": "action_v1", "phase": "SELECTING_HAND", "action_type": "PLAY", "indices": [0], "rng_replay": replay_stub},
        ]
    return [
        {
            "schema_version": "action_v1",
            "phase": "SHOP",
            "action_type": "CONSUMABLE_USE",
            "kind": "tarot",
            "key": "c_death",
            "params": {"consumable_index": 0, "hand_indices": [0, 1], "target_side": "left", "index_base": 0},
            "rng_replay": replay_stub,
        },
        {"schema_version": "action_v1", "phase": "SHOP", "action_type": "SHOP_REROLL", "rng_replay": replay_stub},
        {
            "schema_version": "action_v1",
            "phase": "SHOP",
            "action_type": "SHOP_BUY",
            "params": {"pack_index": 0, "index_base": 0},
            "rng_replay": replay_stub,
        },
        {
            "schema_version": "action_v1",
            "phase": "SMODS_BOOSTER_OPENED",
            "action_type": "PACK_OPEN",
            "params": {"choice_index": 0, "index_base": 0},
            "rng_replay": replay_stub,
        },
    ]


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = _starter_env(args.seed, args.scenario)
    start_raw = copy.deepcopy(env.get_state())
    start_snapshot = to_canonical_state(start_raw, rng_mode="native", seed=args.seed, rng_cursor=0, rng_events=[])
    actions = _scenario_actions(start_raw, args.scenario)

    replay = SimEnv(seed=args.seed)
    replay.reset(from_snapshot=start_snapshot)
    rows: list[dict[str, Any]] = []
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
        canonical_obs["rng_replay"] = dict(action_to_run.get("rng_replay") or {"enabled": False, "source": "", "outcomes": []})
        canonical_obs["_last_action_type"] = str(action_to_run.get("action_type") or "").upper()

        rows.append(
            {
                "schema_version": "trace_v1",
                "step_id": step_id,
                "phase": str(canonical.get("phase") or "UNKNOWN"),
                "action": action_to_run,
                "state_hash_full": state_hash_full(canonical),
                "state_hash_hand_core": state_hash_hand_core(canonical),
                "state_hash_p14_real_action_observed_core": state_hash_p14_real_action_observed_core(canonical_obs),
                "state_hash_p32_real_action_position_observed_core": state_hash_p32_real_action_position_observed_core(canonical_obs),
                "state_hash_p37_action_fidelity_core": state_hash_p37_action_fidelity_core(canonical_obs),
                "reward": float(reward),
                "done": bool(done),
                "score_observed": observed,
                "rng_replay": dict(action_to_run.get("rng_replay") or {"enabled": False, "source": "", "outcomes": []}),
                "info": {"source": "p37_action_fixture", "scenario": args.scenario, "engine_info": info},
                "canonical_state_snapshot": canonical,
            }
        )
        state = next_state

    _write_json(out_dir / "oracle_start_snapshot_real.json", start_snapshot)
    _write_jsonl(out_dir / "action_trace_real.jsonl", actions)
    _write_jsonl(out_dir / "oracle_trace_real.jsonl", rows)
    _write_json(
        out_dir / "manifest.json",
        {
            "fixture_type": "p37_action_fidelity",
            "scenario": args.scenario,
            "seed": args.seed,
            "actions_count": len(actions),
            "action_types": [str(a.get("action_type") or "").upper() for a in actions],
            "out_dir": str(out_dir),
        },
    )
    print(json.dumps({"out_dir": str(out_dir), "scenario": args.scenario, "actions_count": len(actions)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
