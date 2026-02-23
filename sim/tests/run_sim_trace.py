if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from pathlib import Path
from typing import Any

from sim.core.canonical import to_canonical_state
from sim.core.hashing import (
    state_hash_economy_core,
    state_hash_full,
    state_hash_hand_core,
    state_hash_p0_hand_score_core,
    state_hash_p0_hand_score_observed_core,
    state_hash_p1_hand_score_observed_core,
    state_hash_p2_hand_score_observed_core,
    state_hash_p2b_hand_score_observed_core,
    state_hash_p3_hand_score_observed_core,
    state_hash_p4_consumable_observed_core,
    state_hash_p5_modifier_observed_core,
    state_hash_p5_voucher_pack_observed_core,
    state_hash_p7_stateful_observed_core,
    state_hash_p8_rng_observed_core,
    state_hash_p8_shop_observed_core,
    state_hash_p9_episode_observed_core,
    state_hash_rng_events_core,
    state_hash_score_core,
    state_hash_zones_core,
    state_hash_zones_counts_core,
)
from sim.core.score_observed import compute_score_observed
from sim.core.validate import validate_action, validate_trace_line
from sim.oracle.extract_rng_events import extract_rng_events
from sim.score.expected_basic import compute_expected_for_action
from sim.pybind.sim_env import SimEnvBackend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simulator trace from an action trace file.")
    parser.add_argument("--action-trace", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument("--stop-on-done", action="store_true")
    return parser.parse_args()


def load_action_trace(path: str | Path) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as fp:
        for line_no, line in enumerate(fp, start=1):
            text = line.strip()
            if not text:
                continue
            obj = json.loads(text)
            if isinstance(obj, dict):
                validate_action(obj)
            actions.append(obj)
    return actions


def phase_default_action(state: dict[str, Any], seed: str) -> dict[str, Any]:
    phase = str(state.get("state") or "UNKNOWN")
    if phase == "BLIND_SELECT":
        return {"action_type": "SELECT", "index": 0}
    if phase == "SELECTING_HAND":
        hand = (state.get("hand") or {}).get("cards") or []
        round_info = state.get("round") or {}
        hands_left = int(round_info.get("hands_left") or 0)
        discards_left = int(round_info.get("discards_left") or 0)
        if hands_left <= 0 and discards_left > 0 and hand:
            return {"action_type": "DISCARD", "indices": [0]}
        if hands_left <= 0 and discards_left <= 0:
            return {"action_type": "WAIT"}
        if hand:
            return {"action_type": "PLAY", "indices": [0]}
        return {"action_type": "WAIT"}
    if phase == "ROUND_EVAL":
        return {"action_type": "CASH_OUT"}
    if phase == "SHOP":
        return {"action_type": "NEXT_ROUND"}
    if phase in {"MENU", "GAME_OVER"}:
        return {"action_type": "START", "seed": seed}
    return {"action_type": "WAIT"}


def main() -> int:
    args = parse_args()

    actions = load_action_trace(args.action_trace)
    if not actions:
        print("ERROR: empty action trace")
        return 2

    actions_count = len(actions)
    states_written = 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = SimEnvBackend(seed=args.seed)
    state = env.reset(seed=args.seed)
    rng_cursor = 0

    with out_path.open("w", encoding="utf-8-sig", newline="\n") as fp:
        for step_id, action in enumerate(actions):
            action_input = dict(action)
            executed_action = dict(action_input)
            overridden = False

            try:
                next_state, reward, done, info = env.step(executed_action)
            except Exception as exc:
                fallback = phase_default_action(state, args.seed)
                fallback["_fallback_reason"] = str(exc)
                executed_action = fallback
                overridden = True
                next_state, reward, done, info = env.step(executed_action)

            events = extract_rng_events(state, next_state)
            rng_cursor += len(events)
            canonical = to_canonical_state(
                next_state,
                rng_mode="native",
                seed=args.seed,
                rng_cursor=rng_cursor,
                rng_events=events,
            )
            score_observed = compute_score_observed(state, next_state)
            computed_expected = compute_expected_for_action(state, executed_action)
            rng_replay = dict(executed_action.get("rng_replay")) if isinstance(executed_action.get("rng_replay"), dict) else {"enabled": False, "source": "", "outcomes": []}
            canonical_with_observed = dict(canonical)
            canonical_with_observed["score_observed"] = dict(score_observed)
            canonical_with_observed["rng_replay"] = dict(rng_replay)

            include_snapshot = (
                step_id == 0
                or step_id == len(actions) - 1
                or done
                or (args.snapshot_every > 0 and (step_id + 1) % args.snapshot_every == 0)
            )

            trace_line: dict[str, Any] = {
                "schema_version": "trace_v1",
                "step_id": step_id,
                "phase": str(canonical.get("phase") or "UNKNOWN"),
                "action": executed_action,
                "state_hash_full": state_hash_full(canonical),
                "state_hash_hand_core": state_hash_hand_core(canonical),
                "state_hash_score_core": state_hash_score_core(canonical),
                "state_hash_p0_hand_score_core": state_hash_p0_hand_score_core(canonical),
                "state_hash_p0_hand_score_observed_core": state_hash_p0_hand_score_observed_core(canonical_with_observed),
                "state_hash_p1_hand_score_observed_core": state_hash_p1_hand_score_observed_core(canonical_with_observed),
                "state_hash_p2_hand_score_observed_core": state_hash_p2_hand_score_observed_core(canonical_with_observed),
                "state_hash_p2b_hand_score_observed_core": state_hash_p2b_hand_score_observed_core(canonical_with_observed),
                "state_hash_p3_hand_score_observed_core": state_hash_p3_hand_score_observed_core(canonical_with_observed),
                "state_hash_p4_consumable_observed_core": state_hash_p4_consumable_observed_core(canonical_with_observed),
                "state_hash_p5_modifier_observed_core": state_hash_p5_modifier_observed_core(canonical_with_observed),
                "state_hash_p5_voucher_pack_observed_core": state_hash_p5_voucher_pack_observed_core(canonical_with_observed),
                "state_hash_p7_stateful_observed_core": state_hash_p7_stateful_observed_core(canonical_with_observed),
                "state_hash_p8_shop_observed_core": state_hash_p8_shop_observed_core(canonical_with_observed),
                "state_hash_p8_rng_observed_core": state_hash_p8_rng_observed_core(canonical_with_observed),
                "state_hash_p9_episode_observed_core": state_hash_p9_episode_observed_core(canonical_with_observed),
                "state_hash_zones_core": state_hash_zones_core(canonical),
                "state_hash_zones_counts_core": state_hash_zones_counts_core(canonical),
                "state_hash_economy_core": state_hash_economy_core(canonical),
                "state_hash_rng_events_core": state_hash_rng_events_core(canonical),
                "reward": float(reward),
                "done": bool(done),
                "score_observed": score_observed,
                "rng_replay": rng_replay,
                "computed_expected": computed_expected,
                "info": {
                    "source": "sim",
                    "overridden": overridden,
                    "engine_info": info,
                    "input_action": action_input,
                },
            }
            if include_snapshot:
                trace_line["canonical_state_snapshot"] = canonical

            validate_trace_line(trace_line)
            fp.write(json.dumps(trace_line, ensure_ascii=False) + "\n")
            states_written += 1

            state = next_state
            if done and args.stop_on_done:
                break

    env.close()
    print(f"trace_contract sim: actions={actions_count}, sim_states={states_written}")
    if states_written != actions_count:
        print("ERROR: trace contract violation in sim trace (states != actions)")
        return 1

    print(f"wrote sim trace: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
