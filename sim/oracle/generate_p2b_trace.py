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
    _try_synthesize_target_hand,
    advance_phase,
    canonical_snapshot,
    detect_index_base,
    extract_hand_cards,
    hard_reset_fixture,
    prepare_selecting_hand,
    state_phase,
)
from trainer.env_client import ConnectionError, RPCError, _call_method, get_state, health

P2B_TARGETS: dict[str, dict[str, Any]] = {
    "p2b_01_stack_flatmult_scaryface": {
        "group": "stacked",
        "jokers": [
            {"key": "j_joker", "kind": "flat_mult", "mult_add": 4.0},
            {"key": "j_scary_face", "kind": "face_scoring_chips", "chips_add_per_card": 30.0},
        ],
        "play_card_keys": ["H_K"],
    },
    "p2b_02_stack_suitmult_flatmult": {
        "group": "stacked",
        "jokers": [
            {"key": "j_greedy_joker", "kind": "suit_scoring_mult", "suit": "D", "mult_add_per_card": 3.0},
            {"key": "j_joker", "kind": "flat_mult", "mult_add": 4.0},
        ],
        "play_card_keys": ["D_A"],
    },
    "p2b_03_stack_banner_flatmult": {
        "group": "stacked",
        "jokers": [
            {"key": "j_banner", "kind": "remaining_discards_chips", "chips_add_per_discard": 30.0},
            {"key": "j_joker", "kind": "flat_mult", "mult_add": 4.0},
        ],
        "play_card_keys": ["H_9"],
    },
    "p2b_04_stack_fibonacci_flatmult": {
        "group": "stacked",
        "jokers": [
            {"key": "j_fibonacci", "kind": "rank_set_scoring_mult", "ranks": ["A", "2", "3", "5", "8"], "mult_add_per_card": 8.0},
            {"key": "j_joker", "kind": "flat_mult", "mult_add": 4.0},
        ],
        "play_card_keys": ["C_8"],
    },
    "p2b_05_stack_odd_even_two_pair": {
        "group": "stacked",
        "jokers": [
            {"key": "j_odd_todd", "kind": "odd_scoring_chips", "chips_add_per_card": 31.0},
            {"key": "j_even_steven", "kind": "even_scoring_mult", "mult_add_per_card": 4.0},
        ],
        "play_card_keys": ["D_9", "C_9", "S_8", "H_8"],
    },
    "p2b_06_stack_photograph_scaryface": {
        "group": "stacked",
        "jokers": [
            {"key": "j_photograph", "kind": "first_face_xmult", "mult_scale": 2.0},
            {"key": "j_scary_face", "kind": "face_scoring_chips", "chips_add_per_card": 30.0},
        ],
        "play_card_keys": ["S_Q"],
    },
    "p2b_07_planet_straight_plus_flatmult": {
        "group": "joker_planet",
        "planet": {"card_key": "c_saturn", "levels": 1, "hand_type": "STRAIGHT"},
        "synth_target": "p0_01_straight",
        "synth_hand_type": "STRAIGHT",
        "jokers": [{"key": "j_joker", "kind": "flat_mult", "mult_add": 4.0}],
    },
    "p2b_08_planet_flush_plus_lusty": {
        "group": "joker_planet",
        "planet": {"card_key": "c_jupiter", "levels": 1, "hand_type": "FLUSH"},
        "synth_target": "p0_02_flush",
        "synth_hand_type": "FLUSH",
        "jokers": [{"key": "j_lusty_joker", "kind": "suit_scoring_mult", "suit": "H", "mult_add_per_card": 3.0}],
    },
    "p2b_09_planet_full_house_plus_flatmult": {
        "group": "joker_planet",
        "planet": {"card_key": "c_earth", "levels": 1, "hand_type": "FULL_HOUSE"},
        "synth_target": "p0_03_full_house",
        "synth_hand_type": "FULL_HOUSE",
        "jokers": [{"key": "j_joker", "kind": "flat_mult", "mult_add": 4.0}],
    },
    "p2b_10_planet_four_kind_plus_scaryface": {
        "group": "joker_planet",
        "planet": {"card_key": "c_mars", "levels": 1, "hand_type": "FOUR_OF_A_KIND"},
        "synth_target": "p0_04_four_kind",
        "synth_hand_type": "FOUR_OF_A_KIND",
        "jokers": [{"key": "j_scary_face", "kind": "face_scoring_chips", "chips_add_per_card": 30.0}],
    },
    "p2b_11_planet_straight_flush_plus_fibonacci": {
        "group": "joker_planet",
        "planet": {"card_key": "c_neptune", "levels": 1, "hand_type": "STRAIGHT_FLUSH"},
        "synth_target": "p0_05_straight_flush",
        "synth_hand_type": "STRAIGHT_FLUSH",
        "jokers": [{"key": "j_fibonacci", "kind": "rank_set_scoring_mult", "ranks": ["A", "2", "3", "5", "8"], "mult_add_per_card": 8.0}],
    },
    "p2b_12_planet_straight_plus_banner": {
        "group": "joker_planet",
        "planet": {"card_key": "c_saturn", "levels": 1, "hand_type": "STRAIGHT"},
        "synth_target": "p0_01_straight",
        "synth_hand_type": "STRAIGHT",
        "jokers": [{"key": "j_banner", "kind": "remaining_discards_chips", "chips_add_per_discard": 30.0}],
    },
    "p2b_13_order_photograph_face_first": {
        "group": "order_selection",
        "jokers": [{"key": "j_photograph", "kind": "first_face_xmult", "mult_scale": 2.0}],
        "play_card_keys": ["H_Q", "S_2"],
    },
    "p2b_14_order_photograph_face_last": {
        "group": "order_selection",
        "jokers": [{"key": "j_photograph", "kind": "first_face_xmult", "mult_scale": 2.0}],
        "play_card_keys": ["S_2", "H_Q"],
    },
    "p2b_15_selection_pair_vs_high": {
        "group": "order_selection",
        "jokers": [{"key": "j_scary_face", "kind": "face_scoring_chips", "chips_add_per_card": 30.0}],
        "play_card_keys": ["D_K", "S_K", "C_2", "H_3"],
    },
    "p2b_16_resource_banner_discards4": {
        "group": "resource_sensitive",
        "resource_discards_left": 3,
        "jokers": [{"key": "j_banner", "kind": "remaining_discards_chips", "chips_add_per_discard": 30.0}],
        "play_card_keys": ["H_9"],
    },
    "p2b_17_resource_banner_discards3": {
        "group": "resource_sensitive",
        "resource_discards_left": 2,
        "jokers": [{"key": "j_banner", "kind": "remaining_discards_chips", "chips_add_per_discard": 30.0}],
        "play_card_keys": ["H_9"],
    },
    "p2b_18_resource_banner_discards2_plus_even": {
        "group": "resource_sensitive",
        "resource_discards_left": 1,
        "jokers": [
            {"key": "j_banner", "kind": "remaining_discards_chips", "chips_add_per_discard": 30.0},
            {"key": "j_even_steven", "kind": "even_scoring_mult", "mult_add_per_card": 4.0},
        ],
        "play_card_keys": ["S_8"],
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


def _save_state_if_possible(base_url: str, path: str | None, timeout_sec: float) -> None:
    if not path:
        return
    try:
        _call_method(base_url, "save", {"path": path}, timeout=timeout_sec)
    except Exception:
        pass


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


def _apply_planet(base_url: str, planet_key: str, timeout_sec: float) -> tuple[bool, str]:
    try:
        _call_method(base_url, "add", {"key": planet_key}, timeout=timeout_sec)
    except Exception as exc:
        return False, f"add_failed:{exc}"

    state = get_state(base_url, timeout=timeout_sec)
    cards = ((state.get("consumables") or {}).get("cards") or []) if isinstance(state.get("consumables"), dict) else []
    if not cards:
        return False, "no_consumable_after_add"

    try:
        _call_method(base_url, "use", {"consumable": 0}, timeout=timeout_sec)
        return True, "planet_used"
    except Exception as exc:
        return False, f"use_failed:{exc}"


def _add_jokers(base_url: str, jokers: list[dict[str, Any]], timeout_sec: float) -> tuple[bool, str | None]:
    for spec in jokers:
        key = str(spec.get("key") or "").strip().lower()
        if not key:
            return False, "missing_joker_key"
        try:
            _call_method(base_url, "add", {"key": key}, timeout=timeout_sec)
        except Exception as exc:
            return False, f"add_joker_failed:{key}:{exc}"
    return True, None


def _ensure_selecting_hand(builder: TraceBuilder) -> bool:
    for _ in range(60):
        phase = state_phase(builder.state)
        if phase == "SELECTING_HAND":
            return True
        advance_phase(builder)
    return state_phase(builder.state) == "SELECTING_HAND"


def _force_discards_left(base_url: str, discards_left: int, timeout_sec: float) -> tuple[bool, str | None]:
    try:
        _call_method(base_url, "set", {"discards_left": int(discards_left)}, timeout=timeout_sec)
        return True, "set_direct"
    except Exception:
        pass

    state = get_state(base_url, timeout=timeout_sec)
    now_left = int((state.get("round") or {}).get("discards_left") or 0)
    if now_left < discards_left:
        return False, f"cannot_increase_discards_from_{now_left}_to_{discards_left}"

    consume = now_left - discards_left
    for _ in range(consume):
        hand = extract_hand_cards(state)
        if not hand:
            return False, "empty_hand_while_consuming_discards"
        try:
            _call_method(base_url, "discard", {"cards": [0]}, timeout=timeout_sec)
        except Exception as exc:
            return False, f"discard_consume_failed:{exc}"
        state = get_state(base_url, timeout=timeout_sec)
        phase = state_phase(state)
        if phase != "SELECTING_HAND":
            return False, f"left_selecting_hand_during_resource_setup:{phase}"
    return True, "consume_actions"


def _run_direct_play_fixture(builder: TraceBuilder, cfg: dict[str, Any]) -> tuple[bool, dict[str, Any], str | None]:
    play_keys = [str(x).strip().upper() for x in (cfg.get("play_card_keys") or []) if str(x).strip()]
    hold_keys = [str(x).strip().upper() for x in (cfg.get("hold_card_keys") or []) if str(x).strip()]
    jokers_expected = [dict(x) for x in (cfg.get("jokers") or []) if isinstance(x, dict)]
    if not play_keys:
        return False, {}, "missing_play_card_keys"

    if not _ensure_selecting_hand(builder):
        return False, {}, "failed_to_reach_selecting_hand"

    if "resource_discards_left" in cfg:
        ok, note = _force_discards_left(builder.base_url, int(cfg.get("resource_discards_left") or 0), builder.timeout_sec)
        if not ok:
            return False, {}, f"resource_setup_failed:{note}"
        builder.state = get_state(builder.base_url, timeout=builder.timeout_sec)

    ok, err = _add_jokers(builder.base_url, jokers_expected, builder.timeout_sec)
    if not ok:
        return False, {}, err

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
    _save_state_if_possible(builder.base_url, builder.start_state_save_path, builder.timeout_sec)

    builder.step("PLAY", indices=play_indices)

    expected_ctx: dict[str, Any] = {"jokers": jokers_expected}
    _attach_expected_context(builder, expected_ctx)

    hit_info = {
        "group": str(cfg.get("group") or ""),
        "play_keys": play_keys,
        "hold_keys": hold_keys,
        "play_indices": play_indices,
        "jokers": [x.get("key") for x in jokers_expected],
        "joker_keys_in_state": _state_joker_keys(builder.start_snapshot_override),
        "observed_delta": float(((builder.state.get("round") or {}).get("chips") or 0.0)),
    }
    return True, hit_info, None


def _run_planet_combo_fixture(builder: TraceBuilder, cfg: dict[str, Any]) -> tuple[bool, dict[str, Any], str | None]:
    jokers_expected = [dict(x) for x in (cfg.get("jokers") or []) if isinstance(x, dict)]
    planet = cfg.get("planet") if isinstance(cfg.get("planet"), dict) else {}
    planet_key = str(planet.get("card_key") or "").strip().lower()
    target = str(cfg.get("synth_target") or "")
    hand_type = str(cfg.get("synth_hand_type") or "")
    if not planet_key or not target or not hand_type:
        return False, {}, "missing_planet_fixture_fields"

    if not _ensure_selecting_hand(builder):
        return False, {}, "failed_to_reach_selecting_hand"

    ok, err = _add_jokers(builder.base_url, jokers_expected, builder.timeout_sec)
    if not ok:
        return False, {}, err

    planet_ok, planet_note = _apply_planet(builder.base_url, planet_key, builder.timeout_sec)
    builder.state = get_state(builder.base_url, timeout=builder.timeout_sec)

    success, hit_info, reason = _try_synthesize_target_hand(builder, target, hand_type)
    if not success:
        return False, hit_info, reason or f"synth_failed:{target}"

    expected_ctx = {
        "jokers": jokers_expected,
        "planet": {
            "card_key": planet_key,
            "levels": int(planet.get("levels") or 1),
            "applied": bool(planet_ok),
            "note": planet_note,
        },
    }
    _attach_expected_context(builder, expected_ctx)

    info = dict(hit_info or {})
    info.update(
        {
            "group": str(cfg.get("group") or ""),
            "planet_key": planet_key,
            "planet_applied": bool(planet_ok),
            "planet_note": planet_note,
            "jokers": [x.get("key") for x in jokers_expected],
        }
    )
    return True, info, None


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
    if target not in P2B_TARGETS:
        raise ValueError(f"unsupported P2b target: {target}")

    cfg = P2B_TARGETS[target]
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

    try:
        if "planet" in cfg:
            success, hit_info, failure_reason = _run_planet_combo_fixture(builder, cfg)
        else:
            success, hit_info, failure_reason = _run_direct_play_fixture(builder, cfg)
    except (RPCError, ConnectionError, RuntimeError) as exc:
        success, hit_info, failure_reason = False, {}, str(exc)

    start_snapshot = canonical_snapshot(builder.state, seed=seed)
    if isinstance(builder.start_snapshot_override, dict):
        start_snapshot = canonical_snapshot(builder.start_snapshot_override, seed=seed)

    meta = start_snapshot.setdefault("_meta", {})
    if isinstance(meta, dict):
        meta["index_base_detected"] = int(index_base)
        meta["index_probe"] = dict(index_probe)
        meta["p2b_target"] = target
        meta["p2b_group"] = str(cfg.get("group") or "")
        meta["jokers_active"] = [dict(x) for x in (cfg.get("jokers") or []) if isinstance(x, dict)]
        if isinstance(cfg.get("planet"), dict):
            meta["planet_context"] = dict(cfg.get("planet") or {})

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
    parser = argparse.ArgumentParser(description="Generate adaptive oracle action trace for one P2b joker interaction target.")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--target", required=True, choices=sorted(P2B_TARGETS.keys()))
    parser.add_argument("--max-steps", type=int, default=120)
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
