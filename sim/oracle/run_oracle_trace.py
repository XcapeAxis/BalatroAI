if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import time
from pathlib import Path
from typing import Any

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
from sim.oracle.canonicalize_real import canonicalize_real_state
from sim.oracle.extract_rng_events import extract_rng_events
from sim.score.expected_basic import compute_expected_for_action
from trainer.env_client import ConnectionError, RPCError, _call_method, get_state, health


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run oracle trace against balatrobot and export canonical hashes.")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--action-trace", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--wait-sleep", type=float, default=0.05)
    parser.add_argument("--no-phase-default-on-error", action="store_true", help="Disable default phase fallback when action fails.")
    parser.add_argument("--strict-phase", action="store_true", help="Disable automatic phase-default fallback.")
    parser.add_argument("--stop-on-done", action="store_true")
    return parser.parse_args()


def load_action_trace(path: str | Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as fp:
        for line_no, line in enumerate(fp, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at action trace line {line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Action line {line_no} must be object")
            validate_action(item)
            out.append(item)
    return out


def phase_default_action(state: dict[str, Any], seed: str) -> dict[str, Any]:
    phase = str(state.get("state") or "UNKNOWN")
    if phase == "BLIND_SELECT":
        return {"schema_version": "action_v1", "phase": phase, "action_type": "SELECT", "index": 0}
    if phase == "SELECTING_HAND":
        hand = (state.get("hand") or {}).get("cards") or []
        round_info = state.get("round") or {}
        hands_left = int(round_info.get("hands_left") or 0)
        discards_left = int(round_info.get("discards_left") or 0)
        if hands_left <= 0 and discards_left > 0 and hand:
            return {"schema_version": "action_v1", "phase": phase, "action_type": "DISCARD", "indices": [0]}
        if hands_left <= 0 and discards_left <= 0:
            return {"schema_version": "action_v1", "phase": phase, "action_type": "WAIT"}
        if hand:
            return {"schema_version": "action_v1", "phase": phase, "action_type": "PLAY", "indices": [0]}
        return {"schema_version": "action_v1", "phase": phase, "action_type": "WAIT"}
    if phase == "ROUND_EVAL":
        return {"schema_version": "action_v1", "phase": phase, "action_type": "CASH_OUT"}
    if phase == "SHOP":
        return {"schema_version": "action_v1", "phase": phase, "action_type": "NEXT_ROUND"}
    if phase in {"MENU", "GAME_OVER"}:
        return {"schema_version": "action_v1", "phase": phase, "action_type": "START", "seed": seed}
    return {"schema_version": "action_v1", "phase": phase, "action_type": "WAIT"}


def _index_offset(action: dict[str, Any]) -> int:
    params = action.get("params")
    if not isinstance(params, dict):
        return 0
    try:
        value = int(params.get("index_base", 0))
    except Exception:
        return 0
    return 1 if value == 1 else 0


def _action_indices_for_rpc(action: dict[str, Any]) -> list[int]:
    offset = _index_offset(action)
    indices = [int(i) for i in (action.get("indices") or [])]
    return [i + offset for i in indices]


def apply_action(base_url: str, action: dict[str, Any], timeout_sec: float, wait_sleep: float, seed: str) -> dict[str, Any]:
    action_type = str(action.get("action_type") or "WAIT").upper()

    if action_type == "PLAY":
        _call_method(base_url, "play", {"cards": _action_indices_for_rpc(action)}, timeout=timeout_sec)
    elif action_type == "DISCARD":
        _call_method(base_url, "discard", {"cards": _action_indices_for_rpc(action)}, timeout=timeout_sec)
    elif action_type == "SELECT":
        _call_method(base_url, "select", {"index": int(action.get("index", 0))}, timeout=timeout_sec)
    elif action_type == "CASH_OUT":
        _call_method(base_url, "cash_out", {}, timeout=timeout_sec)
    elif action_type == "NEXT_ROUND":
        _call_method(base_url, "next_round", {}, timeout=timeout_sec)
    elif action_type == "START":
        start_seed = str(action.get("seed") or seed)
        params = {"deck": "RED", "stake": "WHITE", "seed": start_seed}
        _call_method(base_url, "start", params, timeout=timeout_sec)
    elif action_type == "MENU":
        _call_method(base_url, "menu", {}, timeout=timeout_sec)
    elif action_type == "SKIP":
        _call_method(base_url, "skip", {}, timeout=timeout_sec)
    elif action_type == "REROLL":
        _call_method(base_url, "reroll", {}, timeout=timeout_sec)
    elif action_type == "BUY":
        params = action.get("params") if isinstance(action.get("params"), dict) else {}
        _call_method(base_url, "buy", params, timeout=timeout_sec)
    elif action_type == "PACK":
        params = action.get("params") if isinstance(action.get("params"), dict) else {}
        _call_method(base_url, "pack", params, timeout=timeout_sec)
    elif action_type == "SELL":
        params = action.get("params") if isinstance(action.get("params"), dict) else {}
        _call_method(base_url, "sell", params, timeout=timeout_sec)
    elif action_type == "USE":
        params = action.get("params") if isinstance(action.get("params"), dict) else {}
        _call_method(base_url, "use", params, timeout=timeout_sec)
    elif action_type == "WAIT":
        time.sleep(max(0.0, float(action.get("sleep") or wait_sleep)))
    else:
        raise ValueError(f"unsupported action_type: {action_type}")

    return get_state(base_url, timeout=timeout_sec)


def _round_chips(state: dict[str, Any]) -> float:
    return float((state.get("round") or {}).get("chips") or 0.0)


def main() -> int:
    args = parse_args()

    if not health(args.base_url):
        print(f"ERROR: oracle base_url unhealthy: {args.base_url}")
        return 2

    actions = load_action_trace(args.action_trace)
    if not actions:
        print("ERROR: action trace is empty")
        return 2

    actions_count = len(actions)
    states_written = 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    state = get_state(args.base_url, timeout=args.timeout_sec)
    if str(state.get("state") or "") in {"MENU", "GAME_OVER"}:
        _call_method(
            args.base_url,
            "start",
            {"deck": "RED", "stake": "WHITE", "seed": args.seed},
            timeout=args.timeout_sec,
        )
        state = get_state(args.base_url, timeout=args.timeout_sec)

    rng_cursor = 0

    with out_path.open("w", encoding="utf-8-sig", newline="\n") as fp:
        for step_id, action in enumerate(actions):
            try:
                action_input = dict(action)
                executed_action = dict(action_input)
                overridden = False

                try:
                    next_state = apply_action(
                        args.base_url,
                        executed_action,
                        timeout_sec=args.timeout_sec,
                        wait_sleep=args.wait_sleep,
                        seed=args.seed,
                    )
                except (RPCError, ConnectionError, ValueError) as exc:
                    if args.strict_phase or args.no_phase_default_on_error:
                        print(f"ERROR: step={step_id} action failed: {exc}")
                        return 1
                    fallback = phase_default_action(state, args.seed)
                    fallback["_fallback_reason"] = str(exc)
                    executed_action = fallback
                    overridden = True
                    next_state = apply_action(
                        args.base_url,
                        executed_action,
                        timeout_sec=args.timeout_sec,
                        wait_sleep=args.wait_sleep,
                        seed=args.seed,
                    )

                events = extract_rng_events(state, next_state)
                rng_cursor += len(events)
                canonical = canonicalize_real_state(
                    next_state,
                    seed=args.seed,
                    rng_events=events,
                    rng_cursor=rng_cursor,
                )

                done = bool((canonical.get("flags") or {}).get("done") or False)
                reward = _round_chips(next_state) - _round_chips(state)
                score_observed = compute_score_observed(state, next_state)
                computed_expected = compute_expected_for_action(state, executed_action)
                rng_replay = {
                    "enabled": True,
                    "source": "oracle_events",
                    "outcomes": list(events),
                }
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
                    "done": done,
                    "score_observed": score_observed,
                    "rng_replay": rng_replay,
                    "computed_expected": computed_expected,
                    "info": {
                        "source": "oracle",
                        "overridden": overridden,
                        "base_url": args.base_url,
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
            except Exception as exc:
                print(f"ERROR: step={step_id} unexpected: {exc}")
                return 1

    print(f"trace_contract oracle: actions={actions_count}, oracle_states={states_written}")
    if states_written != actions_count:
        print("ERROR: trace contract violation in oracle trace (states != actions)")
        return 1

    print(f"wrote oracle trace: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
