from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from datetime import datetime, timezone
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
from sim.oracle.p11_prob_econ_joker_classifier import SUPPORTED_TEMPLATES, build_and_write
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
    params = entry.get("params") if isinstance(entry.get("params"), dict) else {}
    category = str(params.get("category") or "").strip().lower()
    jokers = params.get("jokers") if isinstance(params.get("jokers"), list) else []
    return {
        "jokers": list(jokers),
        "p11_context": {
            "category": category,
            "template": str(entry.get("template") or ""),
            "target": str(entry.get("target") or ""),
            "step_counter": int(step_counter),
        },
    }


def _load_mapping(project_root: Path) -> list[dict[str, Any]]:
    summary = build_and_write(project_root)
    payload = json.loads(Path(summary["map_path"]).read_text(encoding="utf-8-sig"))
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


def _append_expected_context(builder: TraceBuilder, ctx: dict[str, Any]) -> None:
    if builder.action_trace:
        builder.action_trace[-1]["expected_context"] = dict(ctx)


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
    params = entry.get("params") if isinstance(entry.get("params"), dict) else {}
    category = str(params.get("category") or "").strip().lower()

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

        joker_key = str(entry.get("joker_key") or "")
        add_joker_ok = True
        add_joker_error: str | None = None
        if joker_key:
            try:
                _call_method(base_url, "add", {"key": joker_key}, timeout=timeout_sec)
                builder.state = get_state(base_url, timeout=timeout_sec)
            except Exception as exc:
                add_joker_ok = False
                add_joker_error = str(exc)

        builder.start_snapshot_override = json.loads(json.dumps(builder.state, ensure_ascii=False))
        _save_state_if_possible(builder.base_url, builder.start_state_save_path, builder.timeout_sec)

        def _next_valid_action() -> tuple[str, dict[str, Any]]:
            current = builder.state if isinstance(builder.state, dict) else {}
            phase = state_phase(current)
            round_info = current.get("round") if isinstance(current.get("round"), dict) else {}
            hands_left = int(round_info.get("hands_left") or 0)
            discards_left = int(round_info.get("discards_left") or 0)
            hand_cards = (current.get("hand") or {}).get("cards") or []

            if phase == "SELECTING_HAND":
                if hand_cards and hands_left > 0:
                    return "PLAY", {"indices": [0]}
                if hand_cards and discards_left > 0:
                    return "DISCARD", {"indices": [0]}
                return "WAIT", {"sleep": max(0.01, wait_sleep)}
            if phase == "BLIND_SELECT":
                return "SELECT", {"index": 0}
            if phase == "ROUND_EVAL":
                return "CASH_OUT", {}
            if phase == "SHOP":
                # Keep deterministic and valid: leave shop to continue loop.
                return "NEXT_ROUND", {}
            if phase in {"MENU", "GAME_OVER"}:
                return "START", {}
            return "WAIT", {"sleep": max(0.01, wait_sleep)}

        step_counter = 0
        guard = 0
        while len(builder.action_trace) < 6:
            guard += 1
            if guard > 24:
                raise RuntimeError("unable_to_build_valid_p11_trace_within_guard")
            action_type, kwargs = _next_valid_action()
            ctx = _expected_context(entry, step_counter)
            try:
                if action_type in {"PLAY", "DISCARD"}:
                    builder.step(action_type, indices=list(kwargs.get("indices") or []), allow_error=False)
                elif action_type == "SELECT":
                    builder.step(action_type, index=int(kwargs.get("index") or 0), allow_error=False)
                elif action_type == "WAIT":
                    builder.step(action_type, sleep=float(kwargs.get("sleep") or wait_sleep), allow_error=False)
                else:
                    builder.step(action_type, allow_error=False)
            except Exception as exc:
                raise RuntimeError(
                    f"p11_generate_action_failed action={action_type} phase={state_phase(builder.state)}: {exc}"
                ) from exc
            _append_expected_context(builder, ctx)
            step_counter += 1

        if len(builder.action_trace) < 6:
            raise RuntimeError("insufficient_actions_for_p11_trace")

        success = True
        hit_info = {
            "template": template,
            "category": category,
            "joker_key": joker_key,
            "add_joker_ok": bool(add_joker_ok),
            "add_joker_error": add_joker_error,
            "actions": [str(a.get("action_type") or "") for a in builder.action_trace],
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
        meta["p11_target"] = target
        meta["p11_template"] = template
        meta["p11_joker_key"] = entry.get("joker_key")
        meta["p11_category"] = category

    result = {
        "success": bool(success),
        "target": target,
        "template": template,
        "category": category,
        "joker_key": entry.get("joker_key"),
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
            "category": (entry.get("params") or {}).get("category"),
            "joker_key": entry.get("joker_key"),
            "status": "fail",
            "steps_used": 0,
            "failure_reason": None,
            "artifact_paths": {
                "oracle_start_snapshot": str(snap_path),
                "action_trace": str(action_path),
            },
        }

        if resume and snap_path.exists() and action_path.exists():
            row["status"] = "skipped"
            row["steps_used"] = sum(1 for line in action_path.read_text(encoding="utf-8-sig").splitlines() if line.strip())
            row["failure_reason"] = "resume: artifacts already exist"
            rows.append(row)
            continue

        print(f"[P11] target={target} generating trace...")
        gen = generate_one_trace(
            base_url=base_url,
            entry=entry,
            max_steps=max_steps,
            seed=seed,
            timeout_sec=timeout_sec,
            wait_sleep=wait_sleep,
            out_dir=out_dir,
        )
        row["steps_used"] = int(gen.get("steps_used") or 0)
        row["failure_reason"] = gen.get("failure_reason")
        if bool(gen.get("success")):
            row["status"] = "pass"
            row["artifact_paths"] = dict(gen.get("artifact_paths") or row["artifact_paths"])
        else:
            row["status"] = "gen_fail"
        rows.append(row)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "total": len(rows),
        "passed": sum(1 for r in rows if r["status"] == "pass"),
        "gen_fail": sum(1 for r in rows if r["status"] == "gen_fail"),
        "skipped": sum(1 for r in rows if r["status"] == "skipped"),
        "results": rows,
    }
    return summary


def _write_many(summary: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "generate_p11_report.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate P11 prob/econ joker trace(s).")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--target", default=None)
    parser.add_argument("--targets", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--wait-sleep", type=float, default=0.05)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent.parent
    out_dir = Path(args.out_dir) if args.out_dir else (project_root / "sim" / "tests" / "fixtures_runtime" / "oracle_p11_prob_econ_v1")
    if not out_dir.is_absolute():
        out_dir = (project_root / out_dir).resolve()

    targets_csv = args.targets
    if args.target:
        targets_csv = args.target if not targets_csv else f"{targets_csv},{args.target}"

    summary = generate_many(
        base_url=args.base_url,
        out_dir=out_dir,
        seed=args.seed,
        targets_csv=targets_csv,
        limit=args.limit,
        resume=bool(args.resume),
        max_steps=int(args.max_steps),
        timeout_sec=float(args.timeout_sec),
        wait_sleep=float(args.wait_sleep),
    )
    report_path = _write_many(summary, out_dir)

    print(f"total={summary['total']} pass={summary['passed']} gen_fail={summary['gen_fail']} skipped={summary['skipped']}")
    print(f"report={report_path}")
    return 0 if int(summary["gen_fail"]) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
