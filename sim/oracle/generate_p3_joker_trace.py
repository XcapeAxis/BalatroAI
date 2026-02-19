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
    advance_phase,
    canonical_snapshot,
    detect_index_base,
    extract_hand_cards,
    hard_reset_fixture,
    prepare_selecting_hand,
    state_phase,
)
from sim.oracle.p3_joker_classifier import SUPPORTED_TEMPLATES, build_and_write
from trainer.env_client import ConnectionError, RPCError, _call_method, get_state, health


def _classify_connection_failure_text(text: str) -> str:
    low = str(text or "").lower()
    if any(token in low for token in ("10061", "connection refused", "actively refused", "refused")):
        return "connection refused"
    if any(token in low for token in ("timed out", "timeout", "read timed out", "connect timeout")):
        return "timeout"
    return "health check failed"


def _state_context_summary(state: dict[str, Any]) -> str:
    phase = state_phase(state)
    round_info = state.get("round") or {}
    hands_left = int(round_info.get("hands_left") or 0)
    discards_left = int(round_info.get("discards_left") or 0)
    hand = extract_hand_cards(state)
    brief = []
    for c in hand[:8]:
        brief.append(f"{c.get('rank')}-{c.get('suit')}")
    return f"final_phase={phase}; hands_left={hands_left}; discards_left={discards_left}; hand={brief}"


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


def _ensure_selecting_hand(builder: TraceBuilder) -> bool:
    for _ in range(60):
        if state_phase(builder.state) == "SELECTING_HAND":
            return True
        advance_phase(builder)
    return state_phase(builder.state) == "SELECTING_HAND"


def _attach_expected_context(builder: TraceBuilder, expected_ctx: dict[str, Any]) -> None:
    for action in builder.action_trace:
        if str(action.get("action_type") or "").upper() != "PLAY":
            continue
        cur = action.get("expected_context") if isinstance(action.get("expected_context"), dict) else {}
        merged = dict(cur)
        merged.update(expected_ctx)
        action["expected_context"] = merged


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
        if state_phase(state) != "SELECTING_HAND":
            return False, f"left_selecting_hand_during_resource_setup:{state_phase(state)}"
    return True, "consume_actions"


def _entry_to_joker_spec(entry: dict[str, Any]) -> dict[str, Any]:
    template = str(entry.get("template") or "")
    params = entry.get("params") if isinstance(entry.get("params"), dict) else {}
    key = str(entry.get("joker_key") or "")

    if template == "flat_mult":
        return {"key": key, "kind": "flat_mult", "mult_add": float(params.get("mult_add") or 0.0)}
    if template == "suit_mult_per_scoring_card":
        return {
            "key": key,
            "kind": "suit_scoring_mult",
            "suit": str(params.get("suit") or ""),
            "mult_add_per_card": float(params.get("mult_add_per_card") or 0.0),
        }
    if template == "suit_chips_per_scoring_card":
        return {
            "key": key,
            "kind": "suit_scoring_chips",
            "suit": str(params.get("suit") or ""),
            "chips_add_per_card": float(params.get("chips_add_per_card") or 0.0),
        }
    if template == "face_chips_per_scoring_card":
        return {"key": key, "kind": "face_scoring_chips", "chips_add_per_card": float(params.get("chips_add_per_card") or 0.0)}
    if template == "face_mult_per_scoring_card":
        return {"key": key, "kind": "face_scoring_mult", "mult_add_per_card": float(params.get("mult_add_per_card") or 0.0)}
    if template == "odd_chips_per_scoring_card":
        return {"key": key, "kind": "odd_scoring_chips", "chips_add_per_card": float(params.get("chips_add_per_card") or 0.0)}
    if template == "even_mult_per_scoring_card":
        return {"key": key, "kind": "even_scoring_mult", "mult_add_per_card": float(params.get("mult_add_per_card") or 0.0)}
    if template == "fibonacci_mult_per_scoring_card":
        return {
            "key": key,
            "kind": "rank_set_scoring_mult",
            "ranks": ["A", "2", "3", "5", "8"],
            "mult_add_per_card": float(params.get("mult_add_per_card") or 0.0),
        }
    if template == "banner_discards_to_chips":
        return {"key": key, "kind": "remaining_discards_chips", "chips_add_per_discard": float(params.get("chips_add_per_discard") or 0.0)}
    if template == "photograph_first_face_xmult":
        return {"key": key, "kind": "first_face_xmult", "mult_scale": float(params.get("mult_scale") or 1.0)}
    if template == "baron_held_kings_xmult":
        return {
            "key": key,
            "kind": "held_rank_xmult",
            "rank": "K",
            "mult_scale_per_card": float(params.get("mult_scale_per_card") or 1.0),
        }
    if template == "hand_contains_mult_add":
        return {
            "key": key,
            "kind": "hand_type_mult_add",
            "hand_type": str(params.get("hand_type") or ""),
            "mult_add": float(params.get("mult_add") or 0.0),
        }
    if template == "hand_contains_chips_add":
        return {
            "key": key,
            "kind": "hand_type_chips_add",
            "hand_type": str(params.get("hand_type") or ""),
            "chips_add": float(params.get("chips_add") or 0.0),
        }
    if template == "hand_contains_xmult":
        return {
            "key": key,
            "kind": "hand_type_xmult",
            "hand_type": str(params.get("hand_type") or ""),
            "mult_scale": float(params.get("mult_scale") or 1.0),
        }
    if template == "rank_set_chips_mult_per_scoring_card":
        return {
            "key": key,
            "kind": "rank_set_scoring_chips_mult",
            "ranks": [str(x) for x in (params.get("ranks") or [])],
            "chips_add_per_card": float(params.get("chips_add_per_card") or 0.0),
            "mult_add_per_card": float(params.get("mult_add_per_card") or 0.0),
        }
    if template == "held_rank_mult_add":
        return {
            "key": key,
            "kind": "held_rank_mult_add",
            "rank": str(params.get("rank") or ""),
            "mult_add_per_card": float(params.get("mult_add_per_card") or 0.0),
        }
    if template == "held_lowest_rank_mult_add":
        return {
            "key": key,
            "kind": "held_lowest_rank_mult_add",
            "scale": float(params.get("scale") or 2.0),
        }
    if template == "all_held_suits_xmult":
        return {
            "key": key,
            "kind": "all_held_suits_xmult",
            "allowed_suits": [str(x) for x in (params.get("allowed_suits") or [])],
            "mult_scale": float(params.get("mult_scale") or 1.0),
        }
    if template == "scoring_club_and_other_xmult":
        return {
            "key": key,
            "kind": "scoring_has_suit_and_other_xmult",
            "required_suit": str(params.get("required_suit") or "C"),
            "mult_scale": float(params.get("mult_scale") or 1.0),
        }
    if template == "scoring_all_suits_xmult":
        return {
            "key": key,
            "kind": "scoring_has_all_suits_xmult",
            "required_suits": [str(x) for x in (params.get("required_suits") or [])],
            "mult_scale": float(params.get("mult_scale") or 1.0),
        }
    if template == "discards_zero_mult_add":
        return {
            "key": key,
            "kind": "discards_left_eq_mult_add",
            "discards_left": int(params.get("discards_left") or 0),
            "mult_add": float(params.get("mult_add") or 0.0),
        }
    if template == "final_hand_xmult":
        return {
            "key": key,
            "kind": "hands_left_eq_xmult",
            "hands_left": int(params.get("hands_left") or 1),
            "mult_scale": float(params.get("mult_scale") or 1.0),
        }
    if template == "hand_size_lte_mult_add":
        return {
            "key": key,
            "kind": "hand_size_lte_mult_add",
            "max_cards": int(params.get("max_cards") or 3),
            "mult_add": float(params.get("mult_add") or 0.0),
        }
    if template == "rank_set_xmult_per_scoring_card":
        return {
            "key": key,
            "kind": "rank_set_scoring_xmult",
            "ranks": [str(x) for x in (params.get("ranks") or [])],
            "mult_scale_per_card": float(params.get("mult_scale_per_card") or 1.0),
        }
    return {"key": key}



def _candidate_play_key_sets(template: str, params: dict[str, Any]) -> tuple[list[list[str]], list[str], int | None]:
    hold_keys: list[str] = []
    resource_discards_left: int | None = None

    hand_type_keys = {
        "PAIR": ["H_A", "D_A"],
        "TWO_PAIR": ["H_A", "D_A", "S_K", "C_K"],
        "THREE_OF_A_KIND": ["H_A", "D_A", "S_A"],
        "STRAIGHT": ["H_5", "D_6", "S_7", "C_8", "H_9"],
        "FLUSH": ["H_2", "H_5", "H_8", "H_J", "H_K"],
        "FOUR_OF_A_KIND": ["H_A", "D_A", "S_A", "C_A", "H_2"],
    }

    if template == "flat_mult":
        return [["H_2"], ["D_2"], ["S_2"]], hold_keys, resource_discards_left
    if template in {"suit_mult_per_scoring_card", "suit_chips_per_scoring_card"}:
        suit = str(params.get("suit") or "H").upper()[:1]
        return [[f"{suit}_A"], [f"{suit}_K"], [f"{suit}_9"]], hold_keys, resource_discards_left
    if template in {"face_chips_per_scoring_card", "face_mult_per_scoring_card"}:
        return [["H_K"], ["S_Q"], ["D_J"]], hold_keys, resource_discards_left
    if template == "odd_chips_per_scoring_card":
        return [["D_9"], ["C_7"], ["S_5"]], hold_keys, resource_discards_left
    if template == "even_mult_per_scoring_card":
        return [["S_8"], ["C_6"], ["D_4"]], hold_keys, resource_discards_left
    if template == "fibonacci_mult_per_scoring_card":
        return [["C_8"], ["H_5"], ["D_3"]], hold_keys, resource_discards_left
    if template == "banner_discards_to_chips":
        resource_discards_left = 2
        return [["H_9"], ["D_8"], ["S_7"]], hold_keys, resource_discards_left
    if template == "photograph_first_face_xmult":
        return [["H_Q"], ["S_K"], ["D_J"]], hold_keys, resource_discards_left
    if template == "baron_held_kings_xmult":
        hold_keys = ["H_K", "S_K"]
        return [["D_2"], ["C_2"], ["H_3"]], hold_keys, resource_discards_left

    if template in {"hand_contains_mult_add", "hand_contains_chips_add", "hand_contains_xmult"}:
        hand_type = str(params.get("hand_type") or "").upper()
        return [[x for x in hand_type_keys.get(hand_type, ["H_2"])]], hold_keys, resource_discards_left

    if template == "rank_set_chips_mult_per_scoring_card":
        key_map = {"A": "A", "10": "T", "4": "4", "K": "K", "Q": "Q", "J": "J"}
        keys = [f"H_{key_map.get(str(r).upper(), str(r).upper())}" for r in (params.get("ranks") or []) if str(r).upper() in key_map]
        return [keys or ["H_A"]], hold_keys, resource_discards_left

    if template == "held_rank_mult_add":
        rank = str(params.get("rank") or "Q").upper()
        rk = "T" if rank == "10" else rank
        hold_keys = [f"H_{rk}", f"S_{rk}"]
        return [["D_2"]], hold_keys, resource_discards_left

    if template == "held_lowest_rank_mult_add":
        hold_keys = ["H_2", "S_3"]
        return [["D_K"]], hold_keys, resource_discards_left

    if template == "all_held_suits_xmult":
        hold_keys = ["S_2", "C_3", "S_4", "C_5"]
        return [["H_A"]], hold_keys, resource_discards_left

    if template == "scoring_club_and_other_xmult":
        return [["C_A", "H_K"]], hold_keys, resource_discards_left

    if template == "scoring_all_suits_xmult":
        return [["D_A", "C_K", "H_Q", "S_J"]], hold_keys, resource_discards_left

    if template == "discards_zero_mult_add":
        resource_discards_left = 0
        return [["H_A"]], hold_keys, resource_discards_left

    if template == "final_hand_xmult":
        return [["H_A"]], hold_keys, resource_discards_left

    if template == "hand_size_lte_mult_add":
        return [["H_A", "D_A", "S_A"]], hold_keys, resource_discards_left

    if template == "rank_set_xmult_per_scoring_card":
        return [["H_K", "D_Q"]], hold_keys, resource_discards_left

    return [["H_2"]], hold_keys, resource_discards_left



def _find_fallback_indices(hand_cards: list[dict[str, Any]], template: str, params: dict[str, Any]) -> list[int] | None:
    if not hand_cards:
        return None

    if template in {"suit_mult_per_scoring_card", "suit_chips_per_scoring_card"}:
        suit = str(params.get("suit") or "").upper()[:1]
        for c in hand_cards:
            if str(c.get("suit") or "") == suit:
                return [int(c.get("idx") or 0)]

    if template in {"face_chips_per_scoring_card", "face_mult_per_scoring_card", "photograph_first_face_xmult"}:
        for c in hand_cards:
            if str(c.get("rank") or "") in {"K", "Q", "J"}:
                return [int(c.get("idx") or 0)]

    if template == "odd_chips_per_scoring_card":
        for c in hand_cards:
            if str(c.get("rank") or "") in {"A", "9", "7", "5", "3"}:
                return [int(c.get("idx") or 0)]

    if template == "even_mult_per_scoring_card":
        for c in hand_cards:
            if str(c.get("rank") or "") in {"10", "8", "6", "4", "2"}:
                return [int(c.get("idx") or 0)]

    if template == "fibonacci_mult_per_scoring_card":
        for c in hand_cards:
            if str(c.get("rank") or "") in {"A", "2", "3", "5", "8"}:
                return [int(c.get("idx") or 0)]

    if template in {"baron_held_kings_xmult", "held_rank_mult_add"}:
        rank = str(params.get("rank") or "K").upper()
        for c in hand_cards:
            if str(c.get("rank") or "") != rank:
                return [int(c.get("idx") or 0)]

    if template in {"hand_contains_mult_add", "hand_contains_chips_add", "hand_contains_xmult"}:
        target_hand_type = str(params.get("hand_type") or "").upper()
        if target_hand_type:
            combo = find_target_combo(hand_cards, target_hand_type)
            if combo:
                return combo

    if template == "hand_size_lte_mult_add":
        n = min(3, len(hand_cards))
        if n > 0:
            return list(range(n))

    return [int(hand_cards[0].get("idx") or 0)]



def load_supported_entries(project_root: Path) -> list[dict[str, Any]]:
    summary = build_and_write(project_root)
    map_path = Path(summary["map_path"])
    mapping = json.loads(map_path.read_text(encoding="utf-8"))
    entries = [x for x in mapping if isinstance(x, dict) and str(x.get("template") or "") in SUPPORTED_TEMPLATES]
    dedup: dict[str, dict[str, Any]] = {}
    for e in entries:
        t = str(e.get("target") or "")
        if t and t not in dedup:
            dedup[t] = e
    return [dedup[k] for k in sorted(dedup.keys())]


def select_entries(project_root: Path, targets_csv: str | None, limit: int | None) -> list[dict[str, Any]]:
    entries = load_supported_entries(project_root)
    by_target = {str(e.get("target") or ""): e for e in entries}

    if targets_csv:
        selected: list[dict[str, Any]] = []
        for raw in str(targets_csv).split(","):
            t = raw.strip()
            if not t:
                continue
            if t in by_target:
                selected.append(by_target[t])
        entries = selected

    if limit is not None and limit > 0:
        entries = entries[: int(limit)]
    return entries


def generate_one_trace(
    *,
    base_url: str,
    entry: dict[str, Any],
    max_steps: int,
    seed: str,
    timeout_sec: float,
    wait_sleep: float,
    max_attempts: int,
    out_dir: Path | None,
) -> dict[str, Any]:
    target = str(entry.get("target") or "")
    template = str(entry.get("template") or "")

    connection_reason = _probe_connection_failure(base_url, timeout_sec)
    if connection_reason is not None:
        return {
            "success": False,
            "target": target,
            "template": template,
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
            "template": template,
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

    if template not in SUPPORTED_TEMPLATES:
        return {
            "success": False,
            "target": target,
            "template": template,
            "steps_used": 0,
            "final_phase": state_phase(state),
            "hit_info": {},
            "index_base": int(index_base),
            "failure_reason": f"unsupported_template:{template}",
            "start_snapshot": canonical_snapshot(state, seed=seed),
            "action_trace": [],
            "oracle_start_state_path": oracle_start_state_path,
        }

    try:
        if not _ensure_selecting_hand(builder):
            raise RuntimeError("failed_to_reach_selecting_hand")

        params = entry.get("params") if isinstance(entry.get("params"), dict) else {}
        play_key_sets, hold_keys, resource_discards_left = _candidate_play_key_sets(template, params)
        joker_spec = _entry_to_joker_spec(entry)

        if resource_discards_left is not None:
            ok, note = _force_discards_left(base_url, int(resource_discards_left), timeout_sec)
            if not ok:
                raise RuntimeError(f"resource_setup_failed:{note}")
            builder.state = get_state(base_url, timeout=timeout_sec)

        try:
            _call_method(base_url, "add", {"key": str(entry.get("joker_key") or "")}, timeout=timeout_sec)
        except Exception as exc:
            raise RuntimeError(f"add_joker_failed:{exc}")

        found_indices: list[int] | None = None
        chosen_keys: list[str] | None = None

        for attempt in range(max(1, int(max_attempts))):
            keys = play_key_sets[attempt % len(play_key_sets)]
            for hold in hold_keys + keys:
                try:
                    _call_method(base_url, "add", {"key": hold}, timeout=timeout_sec)
                except Exception:
                    pass
            builder.state = get_state(base_url, timeout=timeout_sec)

            hand_cards = extract_hand_cards(builder.state)
            idx = _find_indices_for_target_keys(hand_cards, keys)
            if idx is None:
                idx = _find_fallback_indices(hand_cards, template, params)
            if idx:
                found_indices = idx
                chosen_keys = keys
                break

        if found_indices is None:
            raise RuntimeError(f"play_indices_not_found; {_state_context_summary(builder.state)}")

        builder.start_snapshot_override = json.loads(json.dumps(builder.state, ensure_ascii=False))
        _save_state_if_possible(builder.base_url, builder.start_state_save_path, builder.timeout_sec)
        builder.step("PLAY", indices=found_indices)

        expected_ctx = {
            "jokers": [joker_spec],
            "p3_template": template,
        }
        _attach_expected_context(builder, expected_ctx)

        success = True
        hit_info = {
            "template": template,
            "joker_key": entry.get("joker_key"),
            "play_indices": found_indices,
            "play_keys": chosen_keys,
            "observed_delta": float(((builder.state.get("round") or {}).get("chips") or 0.0)),
        }
        failure_reason = None
    except (RPCError, ConnectionError, RuntimeError) as exc:
        success = False
        hit_info = {}
        failure_reason = f"{exc}; {_state_context_summary(builder.state)}"

    start_snapshot = canonical_snapshot(builder.state, seed=seed)
    if isinstance(builder.start_snapshot_override, dict):
        start_snapshot = canonical_snapshot(builder.start_snapshot_override, seed=seed)

    meta = start_snapshot.setdefault("_meta", {})
    if isinstance(meta, dict):
        meta["index_base_detected"] = int(index_base)
        meta["index_probe"] = dict(index_probe)
        meta["p3_target"] = target
        meta["p3_template"] = template
        meta["p3_joker_key"] = entry.get("joker_key")

    result = {
        "success": bool(success),
        "target": target,
        "template": template,
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


def generate_many(
    *,
    base_url: str,
    out_dir: Path,
    seed: str,
    targets_csv: str | None,
    limit: int | None,
    resume: bool,
    max_steps: int,
    timeout_sec: float,
    wait_sleep: float,
    max_attempts: int,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parent.parent.parent
    entries = select_entries(project_root, targets_csv, limit)

    rows: list[dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for entry in entries:
        target = str(entry.get("target") or "")
        snap_path = out_dir / f"oracle_start_snapshot_{target}.json"
        action_path = out_dir / f"action_trace_{target}.jsonl"

        row = {
            "target": target,
            "template": entry.get("template"),
            "joker_key": entry.get("joker_key"),
            "status": "fail",
            "steps_used": 0,
            "failure_reason": None,
            "artifacts": {
                "oracle_start_snapshot": str(snap_path),
                "action_trace": str(action_path),
            },
        }

        if resume and snap_path.exists() and action_path.exists():
            row["status"] = "skipped"
            row["failure_reason"] = "resume: artifacts already exist"
            rows.append(row)
            print(f"[P3-gen] {target} | skipped")
            continue

        result = generate_one_trace(
            base_url=base_url,
            entry=entry,
            max_steps=max_steps,
            seed=seed,
            timeout_sec=timeout_sec,
            wait_sleep=wait_sleep,
            max_attempts=max_attempts,
            out_dir=out_dir,
        )

        row["steps_used"] = int(result.get("steps_used") or 0)
        row["failure_reason"] = result.get("failure_reason")
        row["status"] = "success" if bool(result.get("success")) else "gen_fail"
        rows.append(row)
        print(f"[P3-gen] {target} | {row['status']} | steps={row['steps_used']} | {row['failure_reason'] or ''}")

    summary = {
        "total": len(rows),
        "success": sum(1 for r in rows if r.get("status") == "success"),
        "gen_fail": sum(1 for r in rows if r.get("status") == "gen_fail"),
        "skipped": sum(1 for r in rows if r.get("status") == "skipped"),
        "results": rows,
    }
    (out_dir / "report_generate_p3.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate P3 joker fixtures (start snapshot + action trace) from template map.")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--out-dir", default="sim/tests/fixtures_runtime/oracle_p3_jokers_gen")
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--targets", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--wait-sleep", type=float, default=0.05)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--list-targets", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent.parent
    entries = select_entries(project_root, args.targets, args.limit if int(args.limit) > 0 else None)

    if args.list_targets:
        for e in entries:
            print(f"{e.get('target')} template={e.get('template')} key={e.get('joker_key')}")
        return 0

    out_dir = (project_root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    summary = generate_many(
        base_url=args.base_url,
        out_dir=out_dir,
        seed=args.seed,
        targets_csv=args.targets,
        limit=(int(args.limit) if int(args.limit) > 0 else None),
        resume=bool(args.resume),
        max_steps=int(args.max_steps),
        timeout_sec=float(args.timeout_sec),
        wait_sleep=float(args.wait_sleep),
        max_attempts=int(args.max_attempts),
    )
    print(json.dumps({"total": summary["total"], "success": summary["success"], "gen_fail": summary["gen_fail"], "skipped": summary["skipped"]}, ensure_ascii=False))
    return 0 if summary["gen_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
