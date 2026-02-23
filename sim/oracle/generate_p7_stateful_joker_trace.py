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
from sim.oracle.p7_stateful_joker_classifier import SUPPORTED_TEMPLATES, build_and_write
from trainer.env_client import _call_method, get_state, health


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


def _state_context_summary(state: dict[str, Any]) -> str:
    phase = state_phase(state)
    round_info = state.get("round") or {}
    hands_left = int(round_info.get("hands_left") or 0)
    discards_left = int(round_info.get("discards_left") or 0)
    hand = extract_hand_cards(state)
    brief = [f"{c.get('rank')}-{c.get('suit')}" for c in hand[:8]]
    return f"final_phase={phase}; hands_left={hands_left}; discards_left={discards_left}; hand={brief}"


def _expected_context(entry: dict[str, Any], step_counter: int) -> dict[str, Any]:
    joker_key = str(entry.get("joker_key") or "")
    template = str(entry.get("template") or "")
    return {
        "jokers": [
            {
                "key": joker_key,
                "kind": "stateful_noop",
                "template": template,
                "state": {
                    "counter": int(step_counter),
                },
            }
        ],
        "p7_stateful": {
            "template": template,
            "step_counter": int(step_counter),
        },
    }


def _load_mapping(project_root: Path) -> list[dict[str, Any]]:
    summary = build_and_write(project_root)
    map_path = Path(summary["map_path"])
    payload = json.loads(map_path.read_text(encoding="utf-8"))
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return []
    return [x for x in items if isinstance(x, dict) and str(x.get("template") or "") in SUPPORTED_TEMPLATES]


def load_supported_entries(project_root: Path) -> list[dict[str, Any]]:
    entries = _load_mapping(project_root)
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

    try:
        if state_phase(builder.state) != "SELECTING_HAND":
            raise RuntimeError("failed_to_reach_selecting_hand")

        add_joker_ok = True
        add_joker_error: str | None = None
        joker_key = str(entry.get("joker_key") or "")
        if joker_key:
            try:
                _call_method(base_url, "add", {"key": joker_key}, timeout=timeout_sec)
                builder.state = get_state(base_url, timeout=timeout_sec)
            except Exception as exc:
                add_joker_ok = False
                add_joker_error = str(exc)

        # Freeze start snapshot after attempting joker injection.
        builder.start_snapshot_override = json.loads(json.dumps(builder.state, ensure_ascii=False))
        _save_state_if_possible(builder.base_url, builder.start_state_save_path, builder.timeout_sec)

        step_counter = 0
        round_info = builder.state.get("round") or {}
        hand_cards = extract_hand_cards(builder.state)

        if hand_cards and int(round_info.get("discards_left") or 0) > 0:
            builder.step("DISCARD", indices=[int(hand_cards[0].get("idx") or 0)], allow_error=True)
        else:
            builder.step("WAIT", sleep=wait_sleep)
        if builder.action_trace:
            builder.action_trace[-1]["expected_context"] = _expected_context(entry, step_counter)
        step_counter += 1

        # Cross-round node: menu -> start a fresh run deterministically.
        builder.step("MENU", allow_error=True)
        if builder.action_trace:
            builder.action_trace[-1]["expected_context"] = _expected_context(entry, step_counter)
        step_counter += 1

        builder.step("START")
        if builder.action_trace:
            builder.action_trace[-1]["expected_context"] = _expected_context(entry, step_counter)
        step_counter += 1

        # Stabilize at actionable state after start.
        if state_phase(builder.state) == "BLIND_SELECT":
            builder.step("SELECT", index=0, allow_error=True)
            if builder.action_trace:
                builder.action_trace[-1]["expected_context"] = _expected_context(entry, step_counter)

        if len(builder.action_trace) < 2:
            raise RuntimeError("insufficient_actions_for_stateful_trace")

        success = True
        hit_info = {
            "template": template,
            "joker_key": joker_key,
            "add_joker_ok": bool(add_joker_ok),
            "add_joker_error": add_joker_error,
            "actions": [str(a.get("action_type") or "") for a in builder.action_trace],
            "observed_delta": float(((builder.state.get("round") or {}).get("chips") or 0.0)),
        }
        failure_reason = None
    except Exception as exc:
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
        meta["p7_target"] = target
        meta["p7_template"] = template
        meta["p7_joker_key"] = entry.get("joker_key")

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
            print(f"[P7-gen] {target} | skipped")
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
        print(f"[P7-gen] {target} | {row['status']} | steps={row['steps_used']} | {row['failure_reason'] or ''}")

    summary = {
        "total": len(rows),
        "success": sum(1 for r in rows if r.get("status") == "success"),
        "gen_fail": sum(1 for r in rows if r.get("status") == "gen_fail"),
        "skipped": sum(1 for r in rows if r.get("status") == "skipped"),
        "results": rows,
    }
    (out_dir / "report_generate_p7.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate P7 stateful joker fixtures (start snapshot + action trace).")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--out-dir", default="sim/tests/fixtures_runtime/oracle_p7_stateful_gen")
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--targets", default=None)
    parser.add_argument("--targets-file", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--wait-sleep", type=float, default=0.05)
    parser.add_argument("--list-targets", action="store_true")
    return parser.parse_args()


def _read_targets_file(path_value: str | None, project_root: Path) -> str | None:
    if not path_value:
        return None
    p = Path(path_value)
    if not p.is_absolute():
        p = (project_root / p).resolve()
    if not p.exists():
        raise ValueError(f"targets-file not found: {p}")
    targets: list[str] = []
    seen: set[str] = set()
    for raw in p.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line not in seen:
            seen.add(line)
            targets.append(line)
    return ",".join(targets) if targets else None


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent.parent

    targets_csv = args.targets
    if args.targets and args.targets_file:
        print("ERROR: use either --targets or --targets-file, not both")
        return 2
    if args.targets_file:
        try:
            targets_csv = _read_targets_file(args.targets_file, project_root)
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return 2

    entries = select_entries(project_root, targets_csv, args.limit if int(args.limit) > 0 else None)
    if args.list_targets:
        for e in entries:
            print(f"{e.get('target')} template={e.get('template')} key={e.get('joker_key')}")
        return 0

    out_dir = (project_root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    summary = generate_many(
        base_url=args.base_url,
        out_dir=out_dir,
        seed=args.seed,
        targets_csv=targets_csv,
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
