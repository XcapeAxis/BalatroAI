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
    canonical_snapshot,
    detect_index_base,
    extract_hand_cards,
    hard_reset_fixture,
    prepare_selecting_hand,
    state_phase,
)
from sim.oracle.p4_consumable_classifier import SUPPORTED_TEMPLATES, build_and_write
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


def _apply_index_base(indices: list[int], index_base: int) -> list[int]:
    return [int(i) + (1 if int(index_base) == 1 else 0) for i in indices]


def _consumable_cards_count(state: dict[str, Any]) -> int:
    cons = state.get("consumables")
    if not isinstance(cons, dict):
        return 0
    cards = cons.get("cards")
    if isinstance(cards, list):
        return len(cards)
    return int(cons.get("count") or 0)


def _resolve_consumable_key(base_url: str, entry: dict[str, Any], timeout_sec: float) -> tuple[str | None, str | None]:
    cands = entry.get("key_candidates") if isinstance(entry.get("key_candidates"), list) else []
    cands = [str(x).strip().lower() for x in cands if str(x).strip()]
    key_guess = str(entry.get("consumable_key") or "").strip().lower()
    if key_guess and key_guess not in cands:
        cands.insert(0, key_guess)

    last_err: str | None = None
    for key in cands:
        try:
            _call_method(base_url, "add", {"key": key}, timeout=timeout_sec)
            return key, None
        except Exception as exc:
            last_err = str(exc)
    return None, last_err


def _use_consumable(
    base_url: str,
    *,
    cards_required: int,
    index_base: int,
    timeout_sec: float,
) -> tuple[bool, str, dict[str, Any]]:
    state = get_state(base_url, timeout=timeout_sec)
    hand_cards = extract_hand_cards(state)
    if cards_required > 0 and not hand_cards:
        return False, "empty_hand_for_use", {"hand_size": 0, "cards_required": cards_required}

    n = min(max(0, int(cards_required)), len(hand_cards))
    local_cards = list(range(n)) if n > 0 else []
    rpc_cards = _apply_index_base(local_cards, index_base)

    attempts: list[dict[str, Any]] = []
    if rpc_cards:
        attempts.append({"consumable": 0, "cards": rpc_cards})
        attempts.append({"consumable": 0, "cards": rpc_cards[:1]})
    attempts.append({"consumable": 0})

    last_err: str | None = None
    for params in attempts:
        try:
            _call_method(base_url, "use", params, timeout=timeout_sec)
            return True, "used", {"use_params": params, "cards_local": local_cards, "cards_rpc": rpc_cards}
        except Exception as exc:
            last_err = str(exc)

    return False, f"use_failed:{last_err}", {"cards_local": local_cards, "cards_rpc": rpc_cards}


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
        cards_required = int((entry.get("params") or {}).get("cards_required") or 0)

        key_used, add_err = _resolve_consumable_key(base_url, entry, timeout_sec)
        if not key_used:
            raise RuntimeError(f"add_consumable_failed:{add_err}")

        before_use = get_state(base_url, timeout=timeout_sec)
        before_count = _consumable_cards_count(before_use)

        used_ok, use_note, use_meta = _use_consumable(
            base_url,
            cards_required=cards_required,
            index_base=index_base,
            timeout_sec=timeout_sec,
        )
        if not used_ok:
            raise RuntimeError(use_note)

        after_use = get_state(base_url, timeout=timeout_sec)
        after_count = _consumable_cards_count(after_use)

        builder.start_snapshot_override = json.loads(json.dumps(after_use, ensure_ascii=False))
        _save_state_if_possible(builder.base_url, builder.start_state_save_path, builder.timeout_sec)
        builder.state = after_use

        # Post-use contract step: keep action trace non-empty while preserving stable observed state.
        builder.step("WAIT", sleep=max(0.01, wait_sleep))

        success = True
        hit_info = {
            "template": template,
            "set_type": entry.get("set_type"),
            "consumable_key": key_used,
            "cards_required": cards_required,
            "consumables_before": before_count,
            "consumables_after": after_count,
            "use_note": use_note,
            "use_meta": use_meta,
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
        meta["p4_target"] = target
        meta["p4_template"] = template
        meta["p4_set_type"] = entry.get("set_type")
        meta["p4_consumable_key"] = entry.get("consumable_key")

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
            "consumable_key": entry.get("consumable_key"),
            "set_type": entry.get("set_type"),
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
            print(f"[P4-gen] {target} | skipped")
            continue

        result = generate_one_trace(
            base_url=base_url,
            entry=entry,
            max_steps=max_steps,
            seed=seed,
            timeout_sec=timeout_sec,
            wait_sleep=wait_sleep,
            out_dir=out_dir,
        )

        row["steps_used"] = int(result.get("steps_used") or 0)
        row["failure_reason"] = result.get("failure_reason")
        row["status"] = "success" if bool(result.get("success")) else "gen_fail"
        rows.append(row)
        print(f"[P4-gen] {target} | {row['status']} | steps={row['steps_used']} | {row['failure_reason'] or ''}")

    summary = {
        "total": len(rows),
        "success": sum(1 for r in rows if r.get("status") == "success"),
        "gen_fail": sum(1 for r in rows if r.get("status") == "gen_fail"),
        "skipped": sum(1 for r in rows if r.get("status") == "skipped"),
        "results": rows,
    }
    (out_dir / "report_generate_p4.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate P4 consumable fixtures (start snapshot + action trace) from template map.")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--out-dir", default="sim/tests/fixtures_runtime/oracle_p4_consumables_gen")
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--targets", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--wait-sleep", type=float, default=0.05)
    parser.add_argument("--list-targets", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent.parent
    entries = select_entries(project_root, args.targets, args.limit if int(args.limit) > 0 else None)

    if args.list_targets:
        for e in entries:
            print(f"{e.get('target')} template={e.get('template')} key={e.get('consumable_key')}")
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
    )
    print(json.dumps({"total": summary["total"], "success": summary["success"], "gen_fail": summary["gen_fail"], "skipped": summary["skipped"]}, ensure_ascii=False))
    return 0 if summary["gen_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
