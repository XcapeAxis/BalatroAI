from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from pathlib import Path
from typing import Any

from sim.oracle.generate_p0_trace import (
    TraceBuilder,
    _try_synthesize_target_hand,
    canonical_snapshot,
    detect_index_base,
    hard_reset_fixture,
    prepare_selecting_hand,
    state_phase,
)
from trainer.env_client import ConnectionError, RPCError, _call_method, get_state, health


P1_TARGETS: dict[str, dict[str, Any]] = {
    "p1_01_planet_straight_saturn": {
        "kind": "planet",
        "planet_key": "c_saturn",
        "p0_target": "p0_01_straight",
        "hand_type": "STRAIGHT",
    },
    "p1_02_planet_flush_jupiter": {
        "kind": "planet",
        "planet_key": "c_jupiter",
        "p0_target": "p0_02_flush",
        "hand_type": "FLUSH",
    },
    "p1_03_bonus_card_high": {
        "kind": "modifier",
        "play_card": {"key": "H_A", "enhancement": "BONUS"},
    },
    "p1_04_mult_card_high": {
        "kind": "modifier",
        "play_card": {"key": "S_K", "enhancement": "MULT"},
    },
    "p1_05_foil_card_high": {
        "kind": "modifier",
        "play_card": {"key": "D_Q", "edition": "FOIL"},
    },
    "p1_06_holo_card_high": {
        "kind": "modifier",
        "play_card": {"key": "C_J", "edition": "HOLO"},
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


def _modifier_match(card: dict[str, Any], expected: dict[str, Any]) -> bool:
    if str(card.get("key") or "").upper() != str(expected.get("key") or "").upper():
        return False
    mod = card.get("modifier") if isinstance(card.get("modifier"), dict) else {}
    for field in ("enhancement", "edition", "seal"):
        exp = expected.get(field)
        if exp is None:
            continue
        cur = mod.get(field)
        if str(cur or "").upper() != str(exp).upper():
            return False
    return True


def _find_modifier_card_index(state: dict[str, Any], expected: dict[str, Any]) -> int | None:
    hand = (state.get("hand") or {}).get("cards") or []
    for idx, card in enumerate(hand):
        if isinstance(card, dict) and _modifier_match(card, expected):
            return idx
    return None


def _attach_expected_context(builder: TraceBuilder, context: dict[str, Any]) -> None:
    for action in builder.action_trace:
        if str(action.get("action_type") or "").upper() == "PLAY":
            action["expected_context"] = context


def _run_planet_fixture(builder: TraceBuilder, cfg: dict[str, Any]) -> tuple[bool, dict[str, Any], str | None]:
    planet_key = str(cfg.get("planet_key") or "")
    ok, note = _apply_planet(builder.base_url, planet_key, builder.timeout_sec)
    builder.state = get_state(builder.base_url, timeout=builder.timeout_sec)

    # Synthesize target hand and play once.
    success, hit_info, reason = _try_synthesize_target_hand(
        builder,
        str(cfg.get("p0_target") or ""),
        str(cfg.get("hand_type") or ""),
    )

    _attach_expected_context(
        builder,
        {
            "planet": {
                "card_key": planet_key,
                "levels": 1,
                "applied": bool(ok),
                "note": note,
            }
        },
    )

    hit_info = dict(hit_info or {})
    hit_info["planet_applied"] = bool(ok)
    hit_info["planet_note"] = note
    if not success and reason is None:
        reason = f"planet_fixture_failed:{note}"
    return success, hit_info, reason


def _run_modifier_fixture(builder: TraceBuilder, cfg: dict[str, Any]) -> tuple[bool, dict[str, Any], str | None]:
    play_card = dict(cfg.get("play_card") or {})
    if not play_card.get("key"):
        return False, {}, "missing_play_card_key"

    try:
        _call_method(builder.base_url, "add", play_card, timeout=builder.timeout_sec)
    except Exception as exc:
        return False, {}, f"add_modifier_card_failed:{exc}"

    builder.state = get_state(builder.base_url, timeout=builder.timeout_sec)
    idx = _find_modifier_card_index(builder.state, play_card)
    if idx is None:
        return False, {}, "modifier_card_not_found_in_hand"

    builder.start_snapshot_override = json.loads(json.dumps(builder.state, ensure_ascii=False))
    if builder.start_state_save_path:
        try:
            _call_method(builder.base_url, "save", {"path": builder.start_state_save_path}, timeout=builder.timeout_sec)
        except Exception:
            pass

    builder.step("PLAY", indices=[idx])

    _attach_expected_context(
        builder,
        {
            "modifier": {
                "card_key": str(play_card.get("key") or ""),
                "enhancement": play_card.get("enhancement"),
                "edition": play_card.get("edition"),
                "seal": play_card.get("seal"),
                "applied": True,
            }
        },
    )

    return True, {"modifier_play_index": idx, "modifier_card": play_card}, None


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
    if target not in P1_TARGETS:
        raise ValueError(f"unsupported P1 target: {target}")

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

    cfg = P1_TARGETS[target]
    kind = str(cfg.get("kind") or "")

    try:
        if kind == "planet":
            success, hit_info, failure_reason = _run_planet_fixture(builder, cfg)
        elif kind == "modifier":
            success, hit_info, failure_reason = _run_modifier_fixture(builder, cfg)
        else:
            success, hit_info, failure_reason = False, {}, "unknown_kind"
    except (RPCError, ConnectionError, RuntimeError) as exc:
        success, hit_info, failure_reason = False, {}, str(exc)

    start_snapshot = canonical_snapshot(builder.state, seed=seed)
    if isinstance(builder.start_snapshot_override, dict):
        start_snapshot = canonical_snapshot(builder.start_snapshot_override, seed=seed)

    meta = start_snapshot.setdefault("_meta", {})
    if isinstance(meta, dict):
        meta["index_base_detected"] = int(index_base)
        meta["index_probe"] = dict(index_probe)
        meta["p1_target"] = target
        meta["p1_kind"] = kind

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
    parser = argparse.ArgumentParser(description="Generate adaptive oracle action trace for one P1 target.")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--target", required=True, choices=sorted(P1_TARGETS.keys()))
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
