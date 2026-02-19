from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from pathlib import Path
from typing import Any

from sim.oracle.generate_p0_trace import canonical_snapshot, extract_hand_cards, state_phase
from sim.oracle.p5_voucher_pack_classifier import SUPPORTED_TEMPLATES, build_and_write
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
    brief = [f"{c.get('rank')}-{c.get('suit')}" for c in hand[:8]]
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


def _to_shop(base_url: str, seed: str, timeout_sec: float) -> dict[str, Any]:
    try:
        _call_method(base_url, "menu", {}, timeout=timeout_sec)
    except Exception:
        pass
    _call_method(base_url, "start", {"deck": "RED", "stake": "WHITE", "seed": seed}, timeout=timeout_sec)
    _call_method(base_url, "select", {"index": 0}, timeout=timeout_sec)
    _call_method(base_url, "set", {"chips": 999999}, timeout=timeout_sec)
    _call_method(base_url, "play", {"cards": [0]}, timeout=timeout_sec)
    _call_method(base_url, "cash_out", {}, timeout=timeout_sec)
    _call_method(base_url, "set", {"money": 100}, timeout=timeout_sec)
    return get_state(base_url, timeout=timeout_sec)


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


def _extract_cards(state: dict[str, Any], field: str) -> list[dict[str, Any]]:
    raw = state.get(field)
    if isinstance(raw, dict):
        cards = raw.get("cards")
        if isinstance(cards, list):
            return [c for c in cards if isinstance(c, dict)]
    return []


def _find_market_index(state: dict[str, Any], field: str, key: str | None) -> int | None:
    cards = _extract_cards(state, field)
    if not cards:
        return None
    if key:
        k = str(key).strip().lower()
        for idx, card in enumerate(cards):
            if str(card.get("key") or "").strip().lower() == k:
                return idx
    return None


def _resolve_opened_pack(base_url: str, timeout_sec: float) -> tuple[bool, str]:
    attempts = [
        {"card": 0},
        {"card": 0, "cards": [0]},
        {"card": 0, "cards": [0, 1]},
        {"card": 1},
        {"skip": True},
    ]
    loops = 0
    last_err = ""
    while loops < 4:
        st = get_state(base_url, timeout=timeout_sec)
        if str(st.get("state") or "") != "SMODS_BOOSTER_OPENED":
            return True, "resolved"
        success = False
        for params in attempts:
            try:
                _call_method(base_url, "pack", params, timeout=timeout_sec)
                success = True
                break
            except Exception as exc:
                last_err = str(exc)
        if not success:
            return False, last_err or "pack resolution failed"
        loops += 1
    st = get_state(base_url, timeout=timeout_sec)
    if str(st.get("state") or "") == "SMODS_BOOSTER_OPENED":
        return False, "pack remained opened after max loops"
    return True, "resolved"


def _free_pack_slot(base_url: str, timeout_sec: float) -> tuple[bool, str]:
    try:
        _call_method(base_url, "buy", {"pack": 0}, timeout=timeout_sec)
    except Exception as exc:
        return False, f"buy(pack=0) failed: {exc}"
    ok, note = _resolve_opened_pack(base_url, timeout_sec)
    if not ok:
        return False, f"free_slot_pack_resolve_failed: {note}"
    try:
        _call_method(base_url, "set", {"money": 100}, timeout=timeout_sec)
    except Exception:
        pass
    return True, "freed"


def _apply_voucher(base_url: str, timeout_sec: float, params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    _call_method(base_url, "set", {"money": 100}, timeout=timeout_sec)
    key_cands = [str(x).strip().lower() for x in (params.get("voucher_key_candidates") or []) if str(x).strip()]
    allow_fallback = bool(params.get("allow_shop_fallback", True))

    added_key = None
    add_errors: list[str] = []
    for key in key_cands:
        try:
            _call_method(base_url, "add", {"key": key}, timeout=timeout_sec)
            added_key = key
            break
        except Exception as exc:
            add_errors.append(f"{key}:{exc}")

    st = get_state(base_url, timeout=timeout_sec)
    buy_idx = _find_market_index(st, "vouchers", added_key)
    if buy_idx is None:
        buy_idx = 0

    if added_key is None and not allow_fallback:
        raise RuntimeError(f"add_voucher_failed:{'; '.join(add_errors)}")

    buy_errors: list[str] = []
    bought_index: int | None = None
    try:
        _call_method(base_url, "buy", {"voucher": int(buy_idx)}, timeout=timeout_sec)
        bought_index = int(buy_idx)
    except Exception as exc:
        buy_errors.append(f"buy({buy_idx}):{exc}")
        if int(buy_idx) != 0 and allow_fallback:
            try:
                _call_method(base_url, "buy", {"voucher": 0}, timeout=timeout_sec)
                bought_index = 0
            except Exception as exc2:
                buy_errors.append(f"buy(0):{exc2}")
        if bought_index is None:
            raise RuntimeError(f"buy_voucher_failed:{'; '.join(buy_errors)}")

    after = get_state(base_url, timeout=timeout_sec)
    used = after.get("used_vouchers")
    if isinstance(used, dict):
        used_keys = sorted(str(k) for k in used.keys())
    elif isinstance(used, list):
        used_keys = [str(x) for x in used]
    else:
        used_keys = []

    meta = {
        "added_key": added_key,
        "buy_index": int(bought_index if bought_index is not None else buy_idx),
        "used_vouchers": used_keys,
        "add_errors": add_errors,
        "buy_errors": buy_errors,
    }
    return after, meta


def _apply_pack(base_url: str, timeout_sec: float, params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    _call_method(base_url, "set", {"money": 100}, timeout=timeout_sec)
    pack_key = str(params.get("pack_key") or "").strip().lower()
    allow_fallback = bool(params.get("allow_shop_fallback", True))

    added = False
    add_error = ""
    if pack_key:
        try:
            _call_method(base_url, "add", {"key": pack_key}, timeout=timeout_sec)
            added = True
        except Exception as exc:
            add_error = str(exc)
            if "shop booster slots are full" in add_error.lower():
                ok, note = _free_pack_slot(base_url, timeout_sec)
                if ok:
                    try:
                        _call_method(base_url, "add", {"key": pack_key}, timeout=timeout_sec)
                        added = True
                        add_error = ""
                    except Exception as exc2:
                        add_error = f"after_free_slot:{exc2}"
                else:
                    add_error = f"{add_error}; {note}"

    st = get_state(base_url, timeout=timeout_sec)
    pack_idx = _find_market_index(st, "packs", pack_key if added else None)
    if pack_idx is None:
        pack_idx = 0

    if not added and not allow_fallback:
        raise RuntimeError(f"add_pack_failed:{add_error}")

    _call_method(base_url, "buy", {"pack": int(pack_idx)}, timeout=timeout_sec)
    ok, note = _resolve_opened_pack(base_url, timeout_sec)
    if not ok:
        raise RuntimeError(f"pack_resolve_failed:{note}")

    after = get_state(base_url, timeout=timeout_sec)
    meta = {
        "pack_key": pack_key,
        "added": added,
        "add_error": add_error,
        "buy_index": int(pack_idx),
        "final_phase": str(after.get("state") or "UNKNOWN"),
    }
    return after, meta


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
    params = dict(entry.get("params") or {})

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
        _to_shop(base_url, seed, timeout_sec)
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

    if template not in SUPPORTED_TEMPLATES:
        state = get_state(base_url, timeout=timeout_sec)
        return {
            "success": False,
            "target": target,
            "template": template,
            "steps_used": 0,
            "final_phase": state_phase(state),
            "hit_info": {},
            "index_base": 0,
            "failure_reason": f"unsupported_template:{template}",
            "start_snapshot": canonical_snapshot(state, seed=seed),
            "action_trace": [],
            "oracle_start_state_path": oracle_start_state_path,
        }

    after_state: dict[str, Any]
    hit_info: dict[str, Any] = {
        "template": template,
        "set_type": entry.get("set_type"),
        "item_key": entry.get("item_key"),
        "item_name": entry.get("item_name"),
    }

    try:
        if template == "voucher_shop_event_observed":
            after_state, meta = _apply_voucher(base_url, timeout_sec, params)
            hit_info.update(meta)
        elif template == "pack_open_pick_first_observed":
            after_state, meta = _apply_pack(base_url, timeout_sec, params)
            hit_info.update(meta)
        else:
            raise RuntimeError(f"unsupported_template_runtime:{template}")

        _save_state_if_possible(base_url, oracle_start_state_path, timeout_sec)

        action_trace = [
            {
                "schema_version": "action_v1",
                "phase": str(after_state.get("state") or "UNKNOWN"),
                "action_type": "WAIT",
                "params": {"index_base": 0, "target": target},
            }
        ]

        start_snapshot = canonical_snapshot(after_state, seed=seed)
        meta = start_snapshot.setdefault("_meta", {})
        if isinstance(meta, dict):
            meta["index_base_detected"] = 0
            meta["index_probe"] = {"method": "fixed", "note": "shop-level fixture uses no hand indices"}
            meta["p5_target"] = target
            meta["p5_template"] = template
            meta["p5_set_type"] = entry.get("set_type")
            meta["p5_item_key"] = entry.get("item_key")

        result = {
            "success": True,
            "target": target,
            "template": template,
            "steps_used": 1,
            "final_phase": state_phase(after_state),
            "hit_info": hit_info,
            "index_base": 0,
            "failure_reason": None,
            "start_snapshot": start_snapshot,
            "action_trace": action_trace,
            "oracle_start_state_path": oracle_start_state_path,
        }
    except (RPCError, ConnectionError, RuntimeError) as exc:
        st = get_state(base_url, timeout=timeout_sec)
        result = {
            "success": False,
            "target": target,
            "template": template,
            "steps_used": 0,
            "final_phase": state_phase(st),
            "hit_info": {},
            "index_base": 0,
            "failure_reason": f"{exc}; {_state_context_summary(st)}",
            "start_snapshot": canonical_snapshot(st, seed=seed),
            "action_trace": [],
            "oracle_start_state_path": oracle_start_state_path,
        }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        snap_path = out_dir / f"oracle_start_snapshot_{target}.json"
        action_path = out_dir / f"action_trace_{target}.jsonl"
        start_snapshot = result.get("start_snapshot")
        if isinstance(start_snapshot, dict):
            snap_path.write_text(json.dumps(start_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        with action_path.open("w", encoding="utf-8-sig", newline="\n") as fp:
            for action in result.get("action_trace") or []:
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
            "item_key": entry.get("item_key"),
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
            print(f"[P5-vp-gen] {target} | skipped")
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
        print(f"[P5-vp-gen] {target} | {row['status']} | steps={row['steps_used']} | {row['failure_reason'] or ''}")

    summary = {
        "total": len(rows),
        "success": sum(1 for r in rows if r.get("status") == "success"),
        "gen_fail": sum(1 for r in rows if r.get("status") == "gen_fail"),
        "skipped": sum(1 for r in rows if r.get("status") == "skipped"),
        "results": rows,
    }
    (out_dir / "report_generate_p5_voucher_pack.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate P5 voucher/pack fixtures (start snapshot + action trace) from template map.")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--out-dir", default="sim/tests/fixtures_runtime/oracle_p5_voucher_pack_gen")
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--targets", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=140)
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
            print(f"{e.get('target')} template={e.get('template')} key={e.get('item_key')}")
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
