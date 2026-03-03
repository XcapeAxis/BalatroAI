from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from trainer.actions.replay import normalize_high_level_action
from sim.core.hashing import (
    state_hash_full,
    state_hash_hand_core,
    state_hash_p14_real_action_observed_core,
    state_hash_p32_real_action_position_observed_core,
    state_hash_p37_action_fidelity_core,
)
from sim.core.score_observed import compute_score_observed
from sim.oracle.canonicalize_real import canonicalize_real_state
from sim.oracle.extract_rng_outcomes import extract_rng_outcomes


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid json at line {line_no}: {exc}") from exc
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert real session trace into fixture-like artifacts.")
    parser.add_argument("--in", dest="inp", required=True, help="Input session jsonl.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", default="AAAAAAA")
    return parser.parse_args()


def _state_min_hash(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _raw_before(row: dict[str, Any]) -> dict[str, Any] | None:
    raw = row.get("gamestate_raw_before")
    if isinstance(raw, dict):
        return raw
    raw_alt = row.get("gamestate_raw")
    if isinstance(raw_alt, dict):
        return raw_alt
    return None


def _raw_after(row: dict[str, Any]) -> dict[str, Any] | None:
    raw = row.get("gamestate_raw_after")
    if isinstance(raw, dict):
        return raw
    raw_alt = row.get("gamestate_raw")
    if isinstance(raw_alt, dict):
        return raw_alt
    return None


def _rng_replay(
    row: dict[str, Any],
    *,
    raw_before: dict[str, Any] | None = None,
    raw_after: dict[str, Any] | None = None,
    action: dict[str, Any] | None = None,
    step_id: int = 0,
) -> dict[str, Any]:
    outcomes = row.get("outcome_tokens")
    if not isinstance(outcomes, list):
        outcomes = []
    if not outcomes and isinstance(raw_before, dict) and isinstance(raw_after, dict):
        try:
            inferred = extract_rng_outcomes(raw_before, raw_after, action=action, step_id=int(step_id))
            outcomes = [dict(x) for x in inferred if isinstance(x, dict)]
        except Exception:
            outcomes = []
    return {
        "enabled": len(outcomes) > 0,
        "source": "real_recording",
        "outcomes": outcomes,
    }


def _phase_for_action(canonical_before: dict[str, Any], row: dict[str, Any]) -> str:
    phase = str(canonical_before.get("phase") or row.get("phase") or "UNKNOWN")
    return phase if phase else "UNKNOWN"


def _ordered_hand_tokens(canonical_state: dict[str, Any]) -> list[str]:
    zones = canonical_state.get("zones") if isinstance(canonical_state.get("zones"), dict) else {}
    hand = zones.get("hand") if isinstance(zones.get("hand"), list) else []
    tokens: list[str] = []
    for idx, card in enumerate(hand):
        if not isinstance(card, dict):
            continue
        uid = str(card.get("uid") or "").strip()
        if uid:
            tokens.append(uid)
            continue
        key = str(card.get("key") or "").strip().lower()
        tokens.append(f"{key}@{idx}")
    return tokens


def _ordered_joker_tokens(canonical_state: dict[str, Any]) -> list[str]:
    jokers = canonical_state.get("jokers") if isinstance(canonical_state.get("jokers"), list) else []
    counts: dict[str, int] = {}
    tokens: list[str] = []
    for item in jokers:
        if isinstance(item, dict):
            key = str(item.get("joker_id") or item.get("id") or item.get("key") or "").strip().lower()
        else:
            key = str(item).strip().lower()
        if not key:
            key = "joker"
        seen = int(counts.get(key) or 0)
        counts[key] = seen + 1
        tokens.append(f"{key}#{seen}")
    return tokens


def _is_position_only_change(
    *,
    canonical_before: dict[str, Any],
    canonical_after: dict[str, Any],
    raw_before: dict[str, Any],
    raw_after: dict[str, Any],
) -> bool:
    before_round = canonical_before.get("round") if isinstance(canonical_before.get("round"), dict) else {}
    after_round = canonical_after.get("round") if isinstance(canonical_after.get("round"), dict) else {}
    before_econ = canonical_before.get("economy") if isinstance(canonical_before.get("economy"), dict) else {}
    after_econ = canonical_after.get("economy") if isinstance(canonical_after.get("economy"), dict) else {}
    observed = compute_score_observed(raw_before, raw_after)

    delta = float(observed.get("delta") or 0.0)
    if abs(delta) > 1e-9:
        return False
    if int(before_round.get("hands_left") or 0) != int(after_round.get("hands_left") or 0):
        return False
    if int(before_round.get("discards_left") or 0) != int(after_round.get("discards_left") or 0):
        return False
    if str(before_round.get("blind") or "").strip().lower() != str(after_round.get("blind") or "").strip().lower():
        return False
    if float(before_econ.get("money") or 0.0) != float(after_econ.get("money") or 0.0):
        return False
    return True


def _compute_move_sequence(before: list[str], after: list[str]) -> list[tuple[int, int]] | None:
    if not before or not after or len(before) != len(after):
        return None
    if sorted(before) != sorted(after):
        return None
    if before == after:
        return []

    working = list(before)
    moves: list[tuple[int, int]] = []
    for dst, token in enumerate(after):
        if working[dst] == token:
            continue
        try:
            src = working.index(token, dst + 1)
        except ValueError:
            return None
        moved = working.pop(src)
        working.insert(dst, moved)
        moves.append((int(src), int(dst)))
    if working != after:
        return None
    return moves


def _apply_move_to_canonical_state(canonical_state: dict[str, Any], *, action_type: str, src: int, dst: int) -> dict[str, Any]:
    next_state = copy.deepcopy(canonical_state)
    if action_type == "MOVE_HAND_CARD":
        zones = next_state.get("zones") if isinstance(next_state.get("zones"), dict) else {}
        hand = zones.get("hand") if isinstance(zones.get("hand"), list) else []
        moved = list(hand)
        card = moved.pop(src)
        moved.insert(dst, card)
        zones["hand"] = moved
        next_state["zones"] = zones
        return next_state

    jokers = next_state.get("jokers") if isinstance(next_state.get("jokers"), list) else []
    moved = list(jokers)
    card = moved.pop(src)
    moved.insert(dst, card)
    next_state["jokers"] = moved
    return next_state


def _infer_position_action_sequence(
    raw_before: dict[str, Any],
    raw_after: dict[str, Any],
    *,
    row: dict[str, Any],
    seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    canonical_before = canonicalize_real_state(raw_before, seed=seed, rng_events=[], rng_cursor=0)
    canonical_after = canonicalize_real_state(raw_after, seed=seed, rng_events=[], rng_cursor=0)
    if not _is_position_only_change(
        canonical_before=canonical_before,
        canonical_after=canonical_after,
        raw_before=raw_before,
        raw_after=raw_after,
    ):
        return None

    hand_before = _ordered_hand_tokens(canonical_before)
    hand_after = _ordered_hand_tokens(canonical_after)
    joker_before = _ordered_joker_tokens(canonical_before)
    joker_after = _ordered_joker_tokens(canonical_after)

    hand_moves = _compute_move_sequence(hand_before, hand_after)
    joker_moves = _compute_move_sequence(joker_before, joker_after)
    if hand_moves is None or joker_moves is None:
        return None
    if not hand_moves and not joker_moves:
        return None

    phase = _phase_for_action(canonical_before, row)
    actions: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    current = copy.deepcopy(canonical_before)

    for src, dst in hand_moves:
        actions.append(
            {
                "schema_version": "action_v1",
                "phase": phase,
                "action_type": "MOVE_HAND_CARD",
                "src_index": int(src),
                "dst_index": int(dst),
                "index_base": 0,
                "params": {"index_base": 0},
            }
        )
        current = _apply_move_to_canonical_state(current, action_type="MOVE_HAND_CARD", src=src, dst=dst)
        states.append(copy.deepcopy(current))

    for src, dst in joker_moves:
        actions.append(
            {
                "schema_version": "action_v1",
                "phase": phase,
                "action_type": "MOVE_JOKER",
                "src_index": int(src),
                "dst_index": int(dst),
                "index_base": 0,
                "params": {"index_base": 0},
            }
        )
        current = _apply_move_to_canonical_state(current, action_type="MOVE_JOKER", src=src, dst=dst)
        states.append(copy.deepcopy(current))

    if _ordered_hand_tokens(current) != hand_after:
        return None
    if _ordered_joker_tokens(current) != joker_after:
        return None
    return actions, states


def _infer_reorder_or_swap(
    *,
    before: list[str],
    after: list[str],
    reorder_action_type: str,
    swap_action_type: str,
) -> dict[str, Any] | None:
    if not before or not after or len(before) != len(after):
        return None
    if before == after:
        return None
    if sorted(before) != sorted(after):
        return None

    changed = [i for i, (lhs, rhs) in enumerate(zip(before, after)) if lhs != rhs]
    if len(changed) == 2:
        i, j = changed
        if before[i] == after[j] and before[j] == after[i]:
            return {"action_type": swap_action_type, "i": int(i), "j": int(j)}

    index_map: dict[str, list[int]] = {}
    for idx, token in enumerate(before):
        index_map.setdefault(token, []).append(idx)

    permutation: list[int] = []
    for token in after:
        candidates = index_map.get(token) or []
        if not candidates:
            return None
        permutation.append(int(candidates.pop(0)))

    if permutation == list(range(len(before))):
        return None
    return {"action_type": reorder_action_type, "permutation": permutation}


def _infer_position_action(raw_before: dict[str, Any], raw_after: dict[str, Any], row: dict[str, Any], seed: str) -> dict[str, Any] | None:
    canonical_before = canonicalize_real_state(raw_before, seed=seed, rng_events=[], rng_cursor=0)
    canonical_after = canonicalize_real_state(raw_after, seed=seed, rng_events=[], rng_cursor=0)
    if not _is_position_only_change(
        canonical_before=canonical_before,
        canonical_after=canonical_after,
        raw_before=raw_before,
        raw_after=raw_after,
    ):
        return None

    hand_before = _ordered_hand_tokens(canonical_before)
    hand_after = _ordered_hand_tokens(canonical_after)
    joker_before = _ordered_joker_tokens(canonical_before)
    joker_after = _ordered_joker_tokens(canonical_after)

    hand_action = _infer_reorder_or_swap(
        before=hand_before,
        after=hand_after,
        reorder_action_type="REORDER_HAND",
        swap_action_type="SWAP_HAND_CARDS",
    )
    joker_action = _infer_reorder_or_swap(
        before=joker_before,
        after=joker_after,
        reorder_action_type="REORDER_JOKERS",
        swap_action_type="SWAP_JOKERS",
    )

    if hand_action and joker_action:
        return None

    inferred = hand_action or joker_action
    if not inferred:
        return None

    return {
        "schema_version": "action_v1",
        "phase": _phase_for_action(canonical_before, row),
        **inferred,
    }


def main() -> int:
    args = _parse_args()
    inp = Path(args.inp)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(inp)
    if not rows:
        raise RuntimeError("empty session trace")

    start_raw = None
    for row in rows:
        raw = _raw_before(row)
        if isinstance(raw, dict):
            start_raw = raw
            break
    if start_raw is None:
        raise RuntimeError("session lacks gamestate_raw_before; rerun recorder with --include-raw or --execute")

    start_snapshot = canonicalize_real_state(start_raw, seed=args.seed, rng_events=[], rng_cursor=0)
    (out_dir / "oracle_start_snapshot_real.json").write_text(
        json.dumps(start_snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    action_trace: list[dict[str, Any]] = []
    state_trace: list[dict[str, Any]] = []
    oracle_trace: list[dict[str, Any]] = []
    skipped_actions: list[dict[str, Any]] = []
    explicit_action_count = 0
    inferred_action_count = 0

    for i, row in enumerate(rows):
        phase = str(row.get("phase") or "UNKNOWN")
        state_min = row.get("gamestate_min")
        state_trace.append(
            {
                "step_id": i,
                "ts": row.get("ts"),
                "phase": phase,
                "projection": state_min,
                "state_hash_real_min": _state_min_hash(state_min),
            }
        )

        raw_before = _raw_before(row)
        raw_after = _raw_after(row)
        if not isinstance(raw_before, dict) or not isinstance(raw_after, dict):
            skipped_actions.append(
                {
                    "step_idx": row.get("step_idx", i),
                    "reason": "missing_raw_before_or_after",
                }
            )
            continue

        external_before = row.get("external_raw_before")
        external_after = row.get("external_raw_after")
        external_inferred_actions = row.get("external_inferred_actions")
        if isinstance(external_before, dict) and isinstance(external_after, dict) and isinstance(external_inferred_actions, list) and external_inferred_actions:
            external_seq = _infer_position_action_sequence(external_before, external_after, row=row, seed=args.seed)
            if external_seq is not None:
                ext_actions_raw, ext_states = external_seq
                ext_score_observed = compute_score_observed(external_before, external_after)
                inferred_action_count += len(ext_actions_raw)
                for ext_frame_idx, (ext_action_raw, ext_state) in enumerate(zip(ext_actions_raw, ext_states)):
                    ext_action = normalize_high_level_action(ext_action_raw, phase=_phase_for_action(ext_state, row))
                    ext_rng = {"enabled": False, "source": "external_inferred_sequence", "outcomes": []}
                    ext_action["rng_replay"] = dict(ext_rng)
                    ext_state_obs = dict(ext_state)
                    ext_state_obs["score_observed"] = dict(ext_score_observed)
                    ext_state_obs["rng_replay"] = dict(ext_rng)
                    ext_state_obs["_last_action_type"] = str(ext_action.get("action_type") or "").upper()
                    action_trace.append(ext_action)
                    oracle_trace.append(
                        {
                            "schema_version": "trace_v1",
                            "step_id": len(action_trace) - 1,
                            "phase": str(ext_state.get("phase") or phase),
                            "action": ext_action,
                            "state_hash_full": state_hash_full(ext_state),
                            "state_hash_hand_core": state_hash_hand_core(ext_state),
                            "state_hash_p14_real_action_observed_core": state_hash_p14_real_action_observed_core(ext_state_obs),
                            "state_hash_p32_real_action_position_observed_core": state_hash_p32_real_action_position_observed_core(ext_state_obs),
                            "state_hash_p37_action_fidelity_core": state_hash_p37_action_fidelity_core(ext_state_obs),
                            "reward": 0.0,
                            "done": False,
                            "score_observed": ext_score_observed,
                            "rng_replay": ext_rng,
                            "canonical_state_snapshot": ext_state,
                            "info": {
                                "source": "real_trace_to_fixture",
                                "session_step_idx": row.get("step_idx", i),
                                "session_frame_idx": int(ext_frame_idx),
                                "mode": row.get("mode"),
                                "action_source": "external_inferred_sequence",
                            },
                        }
                    )

        state_changed = bool(row.get("state_changed", True))
        if not state_changed:
            skipped_actions.append(
                {
                    "step_idx": row.get("step_idx", i),
                    "reason": "no_state_change",
                }
            )
            continue

        score_observed = compute_score_observed(raw_before, raw_after)
        canonical_after = canonicalize_real_state(raw_after, seed=args.seed, rng_events=[], rng_cursor=0)

        action_result = row.get("action_result") if isinstance(row.get("action_result"), dict) else {}
        explicit_action = row.get("action_sent")
        action_source = "action_sent"
        if not isinstance(explicit_action, dict):
            explicit_action = row.get("executed_action")
            action_source = "executed_action"
        rng_replay = _rng_replay(
            row,
            raw_before=raw_before,
            raw_after=raw_after,
            action=explicit_action if isinstance(explicit_action, dict) else None,
            step_id=int(row.get("step_idx", i) or 0),
        )

        action_frames: list[dict[str, Any]] = []
        if isinstance(explicit_action, dict):
            explicit_action_count += 1
            action_for_trace = normalize_high_level_action(explicit_action, phase=_phase_for_action(canonical_after, row))
            if not isinstance(action_for_trace.get("rng_replay"), dict):
                action_for_trace["rng_replay"] = dict(rng_replay)
            action_frames.append(
                {
                    "action_source": action_source,
                    "action": action_for_trace,
                    "canonical_state": canonical_after,
                    "rng_replay": dict(action_for_trace.get("rng_replay") or rng_replay),
                    "reward": float(action_result.get("reward") or 0.0),
                    "done": bool(action_result.get("done") or False),
                }
            )
        else:
            inferred_sequence = _infer_position_action_sequence(raw_before, raw_after, row=row, seed=args.seed)
            if inferred_sequence is not None:
                inferred_actions_raw, inferred_states = inferred_sequence
                inferred_action_count += len(inferred_actions_raw)
                for action_seq_raw, seq_state in zip(inferred_actions_raw, inferred_states):
                    seq_action = normalize_high_level_action(action_seq_raw, phase=_phase_for_action(seq_state, row))
                    seq_action["rng_replay"] = {"enabled": False, "source": "inferred_position_sequence", "outcomes": []}
                    action_frames.append(
                        {
                            "action_source": "inferred_position_sequence",
                            "action": seq_action,
                            "canonical_state": seq_state,
                            "rng_replay": {"enabled": False, "source": "inferred_position_sequence", "outcomes": []},
                            "reward": 0.0,
                            "done": False,
                        }
                    )
            else:
                inferred = _infer_position_action(raw_before, raw_after, row=row, seed=args.seed)
                if isinstance(inferred, dict):
                    inferred_action_count += 1
                    action_for_trace = normalize_high_level_action(inferred, phase=_phase_for_action(canonical_after, row))
                    action_for_trace["rng_replay"] = {"enabled": False, "source": "inferred_position_action", "outcomes": []}
                    action_frames.append(
                        {
                            "action_source": "inferred_position_action",
                            "action": action_for_trace,
                            "canonical_state": canonical_after,
                            "rng_replay": {"enabled": False, "source": "inferred_position_action", "outcomes": []},
                            "reward": 0.0,
                            "done": False,
                        }
                    )
                else:
                    skipped_actions.append(
                        {
                            "step_idx": row.get("step_idx", i),
                            "reason": "missing_action_and_no_inference",
                        }
                    )
                    continue

        for frame_idx, frame in enumerate(action_frames):
            canonical_state = frame["canonical_state"]
            action_for_trace = dict(frame["action"])
            canonical_after_obs = dict(canonical_state)
            canonical_after_obs["score_observed"] = dict(score_observed)
            canonical_after_obs["rng_replay"] = dict(frame["rng_replay"])
            canonical_after_obs["_last_action_type"] = str(action_for_trace.get("action_type") or "").upper()

            action_trace.append(action_for_trace)
            oracle_trace.append(
                {
                    "schema_version": "trace_v1",
                    "step_id": len(action_trace) - 1,
                    "phase": str(canonical_state.get("phase") or phase),
                    "action": action_for_trace,
                    "state_hash_full": state_hash_full(canonical_state),
                    "state_hash_hand_core": state_hash_hand_core(canonical_state),
                    "state_hash_p14_real_action_observed_core": state_hash_p14_real_action_observed_core(canonical_after_obs),
                    "state_hash_p32_real_action_position_observed_core": state_hash_p32_real_action_position_observed_core(canonical_after_obs),
                    "state_hash_p37_action_fidelity_core": state_hash_p37_action_fidelity_core(canonical_after_obs),
                    "reward": float(frame["reward"]),
                    "done": bool(frame["done"]),
                    "score_observed": score_observed,
                    "rng_replay": frame["rng_replay"],
                    "canonical_state_snapshot": canonical_state,
                    "info": {
                        "source": "real_trace_to_fixture",
                        "session_step_idx": row.get("step_idx", i),
                        "session_frame_idx": int(frame_idx),
                        "mode": row.get("mode"),
                        "action_source": frame["action_source"],
                    },
                }
            )

    _write_jsonl(out_dir / "state_trace.jsonl", state_trace)
    _write_jsonl(out_dir / "oracle_trace_real.jsonl", oracle_trace)
    _write_jsonl(out_dir / "action_trace_real.jsonl", action_trace)

    phase_hist: dict[str, int] = {}
    for row in rows:
        key = str(row.get("phase") or "UNKNOWN")
        phase_hist[key] = int(phase_hist.get(key, 0)) + 1

    manifest = {
        "input": str(inp),
        "rows": len(rows),
        "base_url": str(rows[0].get("base_url") or ""),
        "mode": str(rows[0].get("mode") or "shadow"),
        "seed": args.seed,
        "state_trace": str(out_dir / "state_trace.jsonl"),
        "oracle_trace": str(out_dir / "oracle_trace_real.jsonl"),
        "action_trace": str(out_dir / "action_trace_real.jsonl"),
        "actions_count": len(action_trace),
        "explicit_actions_count": int(explicit_action_count),
        "inferred_actions_count": int(inferred_action_count),
        "phase_distribution": phase_hist,
        "skipped_actions": skipped_actions,
        "arm_token_used": bool(rows[0].get("mode") == "execute"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
