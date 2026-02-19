from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from pathlib import Path
from typing import Any

from sim.oracle.generate_p0_trace import (
    TraceBuilder,
    _find_indices_for_target_keys,
    canonical_snapshot,
    detect_index_base,
    extract_hand_cards,
    hard_reset_fixture,
    prepare_selecting_hand,
    state_phase,
)
from trainer.env_client import ConnectionError, RPCError, _call_method, get_state, health

P2_TARGETS: dict[str, dict[str, Any]] = {
    "p2_01_joker_flat_mult": {
        "joker_key": "j_joker",
        "play_card_keys": ["D_2"],
        "jokers_expected": [{"key": "j_joker", "kind": "flat_mult", "mult_add": 4.0}],
    },
    "p2_02_greedy_diamond_mult": {
        "joker_key": "j_greedy_joker",
        "play_card_keys": ["D_A"],
        "jokers_expected": [{"key": "j_greedy_joker", "kind": "suit_scoring_mult", "suit": "D", "mult_add_per_card": 3.0}],
    },
    "p2_03_lusty_heart_mult": {
        "joker_key": "j_lusty_joker",
        "play_card_keys": ["H_A"],
        "jokers_expected": [{"key": "j_lusty_joker", "kind": "suit_scoring_mult", "suit": "H", "mult_add_per_card": 3.0}],
    },
    "p2_04_wrathful_spade_mult": {
        "joker_key": "j_wrathful_joker",
        "play_card_keys": ["S_A"],
        "jokers_expected": [{"key": "j_wrathful_joker", "kind": "suit_scoring_mult", "suit": "S", "mult_add_per_card": 3.0}],
    },
    "p2_05_gluttonous_club_mult": {
        "joker_key": "j_gluttenous_joker",
        "play_card_keys": ["C_A"],
        "jokers_expected": [{"key": "j_gluttenous_joker", "kind": "suit_scoring_mult", "suit": "C", "mult_add_per_card": 3.0}],
    },
    "p2_06_banner_discards_chips": {
        "joker_key": "j_banner",
        "play_card_keys": ["H_9"],
        "jokers_expected": [{"key": "j_banner", "kind": "remaining_discards_chips", "chips_add_per_discard": 30.0}],
    },
    "p2_07_odd_todd_chips": {
        "joker_key": "j_odd_todd",
        "play_card_keys": ["D_9"],
        "jokers_expected": [{"key": "j_odd_todd", "kind": "odd_scoring_chips", "chips_add_per_card": 31.0}],
    },
    "p2_08_even_steven_mult": {
        "joker_key": "j_even_steven",
        "play_card_keys": ["S_8"],
        "jokers_expected": [{"key": "j_even_steven", "kind": "even_scoring_mult", "mult_add_per_card": 4.0}],
    },
    "p2_09_fibonacci_mult": {
        "joker_key": "j_fibonacci",
        "play_card_keys": ["C_8"],
        "jokers_expected": [{"key": "j_fibonacci", "kind": "rank_set_scoring_mult", "ranks": ["A", "2", "3", "5", "8"], "mult_add_per_card": 8.0}],
    },
    "p2_10_scary_face_chips": {
        "joker_key": "j_scary_face",
        "play_card_keys": ["H_K"],
        "jokers_expected": [{"key": "j_scary_face", "kind": "face_scoring_chips", "chips_add_per_card": 30.0}],
    },
    "p2_11_photograph_xmult": {
        "joker_key": "j_photograph",
        "play_card_keys": ["S_Q"],
        "jokers_expected": [{"key": "j_photograph", "kind": "first_face_xmult", "mult_scale": 2.0}],
    },
    "p2_12_baron_held_kings_xmult": {
        "joker_key": "j_baron",
        "hold_card_keys": ["H_K", "S_K"],
        "play_card_keys": ["D_2"],
        "jokers_expected": [{"key": "j_baron", "kind": "held_rank_xmult", "rank": "K", "mult_scale_per_card": 1.5}],
    },
}


def _classify_connection_failure_text(text: str) -> str:
    low = str(text or "").lower()
    if any(token in low for token in ("10061", "connection refused", "actively refused", "refused")):
        return "connection refused"
    if any(token in low for token in ("timed out", "timeout", "read timed out", "connect timeout")):
        return "timeout"
    return "health check failed"


def _probe_connection_failure(base_url: str, timeout_sec: float) -> str | None:
    if health(base_url):
        return None
    timeout = max(1.0, min(float(timeout_sec), 3.0))
    reasons: list[str] = []
    for method in ("health", "gamestate"):
        try:
            _call_method(base_url, method, timeout=timeout)
            return None
        except Exception as exc:
            reasons.append(str(exc))
    return _classify_connection_failure_text(" | ".join(reasons))


def _save_start_state(base_url: str, out_dir: Path, target: str, timeout_sec: float) -> str | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / f"oracle_start_state_{target}.jkr"
    try:
        _call_method(base_url, "save", {"path": str(state_path)}, timeout=timeout_sec)
        return str(state_path)
    except Exception:
        return None


def _attach_expected_context(builder: TraceBuilder, expected_ctx: dict[str, Any]) -> None:
    for action in builder.action_trace:
        if str(action.get("action_type") or "").upper() != "PLAY":
            continue
        cur = action.get("expected_context") if isinstance(action.get("expected_context"), dict) else {}
        merged = dict(cur)
        merged.update(expected_ctx)
        action["expected_context"] = merged


def _state_joker_keys(state: dict[str, Any]) -> list[str]:
    raw = state.get("jokers")
    cards: list[dict[str, Any]] = []
    if isinstance(raw, dict):
        src = raw.get("cards")
        if isinstance(src, list):
            cards = [x for x in src if isinstance(x, dict)]
    elif isinstance(raw, list):
        cards = [x for x in raw if isinstance(x, dict)]
    out: list[str] = []
    for card in cards:
        key = str(card.get("key") or "").strip().lower()
        if key:
            out.append(key)
    return out


def _run_single_fixture(builder: TraceBuilder, cfg: dict[str, Any]) -> tuple[bool, dict[str, Any], str | None]:
    joker_key = str(cfg.get("joker_key") or "").strip().lower()
    if not joker_key:
        return False, {}, "missing_joker_key"

    play_keys = [str(x).strip().upper() for x in (cfg.get("play_card_keys") or []) if str(x).strip()]
    hold_keys = [str(x).strip().upper() for x in (cfg.get("hold_card_keys") or []) if str(x).strip()]
    if not play_keys:
        return False, {}, "missing_play_card_keys"

    try:
        _call_method(builder.base_url, "add", {"key": joker_key}, timeout=builder.timeout_sec)
    except Exception as exc:
        return False, {}, f"add_joker_failed:{exc}"

    for card_key in hold_keys + play_keys:
        try:
            _call_method(builder.base_url, "add", {"key": card_key}, timeout=builder.timeout_sec)
        except Exception as exc:
            return False, {}, f"add_play_card_failed:{card_key}:{exc}"

    builder.state = get_state(builder.base_url, timeout=builder.timeout_sec)

    hand_cards = extract_hand_cards(builder.state)
    play_indices = _find_indices_for_target_keys(hand_cards, play_keys)
    if play_indices is None:
        return False, {}, "play_indices_not_found"

    builder.start_snapshot_override = json.loads(json.dumps(builder.state, ensure_ascii=False))
    if builder.start_state_save_path:
        try:
            _call_method(builder.base_url, "save", {"path": builder.start_state_save_path}, timeout=builder.timeout_sec)
        except Exception:
            pass

    builder.step("PLAY", indices=play_indices)

    jokers_expected = [dict(x) for x in (cfg.get("jokers_expected") or []) if isinstance(x, dict)]
    _attach_expected_context(builder, {"jokers": jokers_expected})

    hit_info = {
        "joker_key": joker_key,
        "joker_keys_in_state": _state_joker_keys(builder.start_snapshot_override),
        "play_keys": play_keys,
        "hold_keys": hold_keys,
        "play_indices": play_indices,
        "observed_delta": float(((builder.state.get("round") or {}).get("chips") or 0.0)),
    }
    return True, hit_info, None


def generate_trace(
    *,
    base_url: str,
    target: str,
    max_steps: int,
    seed: str,
    timeout_sec: float,
    wait_sleep: float,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    if target not in P2_TARGETS:
        raise ValueError(f"unsupported P2 target: {target}")

    connection_reason = _probe_connection_failure(base_url, timeout_sec)
    if connection_reason is not None:
        return {
            "success": False,
            "target": target,
            "steps_used": 0,
            "final_phase": "UNKNOWN",
            "hit_info": {},
            "index_base": 0,
            "failure_reason": connection_reason,
            "start_snapshot": None,
            "action_trace": [],
            "oracle_start_state_path": None,
        }

    try:
        hard_reset_fixture(base_url, seed, timeout_sec, wait_sleep)
        index_base, index_probe = detect_index_base(base_url, seed, timeout_sec, wait_sleep)
        state = prepare_selecting_hand(base_url, seed, timeout_sec, wait_sleep)
    except Exception as exc:
        return {
            "success": False,
            "target": target,
            "steps_used": 0,
            "final_phase": "UNKNOWN",
            "hit_info": {},
            "index_base": 0,
            "failure_reason": _classify_connection_failure_text(str(exc)),
            "start_snapshot": None,
            "action_trace": [],
            "oracle_start_state_path": None,
        }

    oracle_start_state_path: str | None = None
    if out_dir is not None:
        oracle_start_state_path = _save_start_state(base_url, out_dir, target, timeout_sec)

    builder = TraceBuilder(
        base_url=base_url,
        state=state,
        seed=seed,
        index_base=index_base,
        timeout_sec=timeout_sec,
        wait_sleep=wait_sleep,
        max_steps=max_steps,
        start_state_save_path=oracle_start_state_path,
    )

    cfg = P2_TARGETS[target]

    try:
        success, hit_info, failure_reason = _run_single_fixture(builder, cfg)
    except (RPCError, ConnectionError, RuntimeError) as exc:
        success, hit_info, failure_reason = False, {}, str(exc)

    start_snapshot = canonical_snapshot(builder.state, seed=seed)
    if isinstance(builder.start_snapshot_override, dict):
        start_snapshot = canonical_snapshot(builder.start_snapshot_override, seed=seed)

    meta = start_snapshot.setdefault("_meta", {})
    if isinstance(meta, dict):
        meta["index_base_detected"] = int(index_base)
        meta["index_probe"] = dict(index_probe)
        meta["p2_target"] = target
        meta["jokers_active"] = [dict(x) for x in (cfg.get("jokers_expected") or []) if isinstance(x, dict)]

    result = {
        "success": bool(success),
        "target": target,
        "steps_used": builder.steps_used,
        "final_phase": state_phase(builder.state),
        "hit_info": hit_info,
        "index_base": int(index_base),
        "failure_reason": failure_reason,
        "start_snapshot": start_snapshot,
        "action_trace": builder.action_trace,
        "oracle_start_state_path": oracle_start_state_path,
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        snap_path = out_dir / f"oracle_start_snapshot_{target}.json"
        action_path = out_dir / f"action_trace_{target}.jsonl"
        snap_path.write_text(json.dumps(start_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        with action_path.open("w", encoding="utf-8-sig", newline="\n") as fp:
            for action in builder.action_trace:
                fp.write(json.dumps(action, ensure_ascii=False) + "\n")

        result["artifact_paths"] = {
            "oracle_start_snapshot": str(snap_path),
            "action_trace": str(action_path),
            "oracle_start_state": oracle_start_state_path,
        }

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate adaptive oracle action trace for one P2 joker smoke target.")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--target", required=True, choices=sorted(P2_TARGETS.keys()))
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--wait-sleep", type=float, default=0.05)
    parser.add_argument("--out-dir")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else None

    result = generate_trace(
        base_url=args.base_url,
        target=args.target,
        max_steps=int(args.max_steps),
        seed=args.seed,
        timeout_sec=float(args.timeout_sec),
        wait_sleep=float(args.wait_sleep),
        out_dir=out_dir,
    )

    print(
        "target={target} success={success} steps_used={steps_used} final_phase={final_phase} hit_info={hit_info}".format(
            target=result.get("target"),
            success=result.get("success"),
            steps_used=result.get("steps_used"),
            final_phase=result.get("final_phase"),
            hit_info=json.dumps(result.get("hit_info") or {}, ensure_ascii=False),
        )
    )

    if out_dir is None:
        print(json.dumps(result, ensure_ascii=False))

    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
