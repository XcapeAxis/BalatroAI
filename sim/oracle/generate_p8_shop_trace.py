from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import time
from pathlib import Path
from typing import Any

from sim.oracle.generate_p0_trace import canonical_snapshot, state_phase
from trainer.env_client import ConnectionError, RPCError, _call_method, get_state, health

TARGET_SPECS: list[dict[str, Any]] = [
    {
        "target": "p8_shop_01_enter_shop_wait_reroll",
        "category": "enter_shop",
        "description": "SHOP entry and a reroll.",
        "actions": [
            {"action_type": "WAIT", "params": {}},
            {"action_type": "REROLL", "params": {}},
        ],
    },
    {
        "target": "p8_shop_02_reroll_twice",
        "category": "reroll",
        "description": "Two consecutive rerolls.",
        "actions": [
            {"action_type": "REROLL", "params": {}},
            {"action_type": "REROLL", "params": {}},
        ],
    },
    {
        "target": "p8_shop_03_buy_voucher_then_wait",
        "category": "buy_voucher",
        "description": "Buy voucher from prepared slot.",
        "prepare": "voucher",
        "actions": [
            {"action_type": "BUY", "params": {"voucher": 0}},
            {"action_type": "WAIT", "params": {}},
        ],
    },
    {
        "target": "p8_shop_04_buy_voucher_then_reroll",
        "category": "buy_voucher",
        "description": "Buy voucher then reroll.",
        "prepare": "voucher",
        "actions": [
            {"action_type": "BUY", "params": {"voucher": 0}},
            {"action_type": "REROLL", "params": {}},
        ],
    },
    {
        "target": "p8_shop_05_open_pack_pick_first",
        "category": "open_pack",
        "description": "Buy pack and pick first card.",
        "prepare": "pack",
        "actions": [
            {"action_type": "BUY", "params": {"pack": 0}},
            {"action_type": "PACK", "params": {"card": 0}},
        ],
    },
    {
        "target": "p8_shop_06_open_pack_skip",
        "category": "open_pack",
        "description": "Buy pack and skip pick.",
        "prepare": "pack",
        "actions": [
            {"action_type": "BUY", "params": {"pack": 0}},
            {"action_type": "PACK", "params": {"skip": True}},
        ],
    },
    {
        "target": "p8_shop_07_buy_card_then_wait",
        "category": "buy_joker",
        "description": "Buy shop card then wait.",
        "actions": [
            {"action_type": "BUY", "params": {"card": 0}},
            {"action_type": "WAIT", "params": {}},
        ],
    },
    {
        "target": "p8_shop_08_buy_card_then_reroll",
        "category": "buy_joker",
        "description": "Buy shop card then reroll.",
        "actions": [
            {"action_type": "BUY", "params": {"card": 0}},
            {"action_type": "REROLL", "params": {}},
        ],
    },
    {
        "target": "p8_shop_09_sell_joker_then_wait",
        "category": "sell",
        "description": "Sell prepared joker.",
        "prepare": "joker",
        "actions": [
            {"action_type": "SELL", "params": {"joker": 0}},
            {"action_type": "WAIT", "params": {}},
        ],
    },
    {
        "target": "p8_shop_10_sell_joker_then_reroll",
        "category": "sell",
        "description": "Sell prepared joker then reroll.",
        "prepare": "joker",
        "actions": [
            {"action_type": "SELL", "params": {"joker": 0}},
            {"action_type": "REROLL", "params": {}},
        ],
    },
    {
        "target": "p8_shop_11_voucher_and_pack_combo",
        "category": "combo",
        "description": "Buy voucher and buy pack in one trace.",
        "prepare": "voucher_pack",
        "actions": [
            {"action_type": "BUY", "params": {"voucher": 0}},
            {"action_type": "BUY", "params": {"pack": 0}},
        ],
    },
    {
        "target": "p8_shop_12_reroll_then_buy_card",
        "category": "buy_joker",
        "description": "Reroll before card buy.",
        "actions": [
            {"action_type": "REROLL", "params": {}},
            {"action_type": "BUY", "params": {"card": 0}},
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate P8 shop target trace (start snapshot + action trace).")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--target", required=True)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--wait-sleep", type=float, default=0.05)
    parser.add_argument("--out-dir", default="")
    return parser.parse_args()


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


def _extract_hand_brief(state: dict[str, Any]) -> str:
    hand = ((state.get("hand") or {}).get("cards") or []) if isinstance(state.get("hand"), dict) else []
    out: list[str] = []
    for card in hand[:8]:
        if not isinstance(card, dict):
            continue
        value = card.get("value") if isinstance(card.get("value"), dict) else {}
        rank = str(card.get("rank") or value.get("rank") or "?")
        suit = str(card.get("suit") or value.get("suit") or "?")
        out.append(f"{rank}-{suit}")
    return "[" + ",".join(out) + "]"


def _state_context_summary(state: dict[str, Any]) -> str:
    phase = state_phase(state)
    round_info = state.get("round") or {}
    hands_left = int(round_info.get("hands_left") or 0)
    discards_left = int(round_info.get("discards_left") or 0)
    return f"final_phase={phase}; hands_left={hands_left}; discards_left={discards_left}; hand={_extract_hand_brief(state)}"


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
    _call_method(base_url, "set", {"money": 250}, timeout=timeout_sec)
    return get_state(base_url, timeout=timeout_sec)


def _save_start_state(base_url: str, out_dir: Path, target: str, timeout_sec: float) -> str | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"oracle_start_state_{target}.jkr"
    try:
        _call_method(base_url, "save", {"path": str(path)}, timeout=timeout_sec)
        return str(path)
    except Exception:
        return None


def _save_state_if_possible(base_url: str, path: str | None, timeout_sec: float) -> None:
    if not path:
        return
    try:
        _call_method(base_url, "save", {"path": path}, timeout=timeout_sec)
    except Exception:
        pass


def _find_market_index(state: dict[str, Any], field: str, key_hint: str | None = None) -> int:
    market = state.get(field) if isinstance(state.get(field), dict) else {}
    cards = market.get("cards") if isinstance(market.get("cards"), list) else []
    if key_hint:
        low = str(key_hint).strip().lower()
        for idx, card in enumerate(cards):
            if isinstance(card, dict) and str(card.get("key") or "").strip().lower() == low:
                return idx
    return 0


def _prepare_target(base_url: str, spec: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
    prep = str(spec.get("prepare") or "").strip().lower()
    meta: dict[str, Any] = {"prepare": prep}

    st = get_state(base_url, timeout=timeout_sec)

    if prep in {"voucher", "voucher_pack"}:
        vouchers = ((st.get("vouchers") or {}).get("cards") or []) if isinstance(st.get("vouchers"), dict) else []
        if vouchers:
            meta["voucher_idx"] = 0
            meta["voucher_strategy"] = "shop_existing"
        else:
            try:
                _call_method(base_url, "add", {"key": "v_paint_brush"}, timeout=timeout_sec)
                st = get_state(base_url, timeout=timeout_sec)
                meta["voucher_idx"] = _find_market_index(st, "vouchers", "v_paint_brush")
                meta["voucher_strategy"] = "direct_add"
            except Exception:
                meta["voucher_idx"] = 0
                meta["voucher_strategy"] = "shop_fallback"

    if prep in {"pack", "voucher_pack"}:
        packs = ((st.get("packs") or {}).get("cards") or []) if isinstance(st.get("packs"), dict) else []
        if packs:
            meta["pack_idx"] = 0
            meta["pack_strategy"] = "shop_existing"
        else:
            try:
                _call_method(base_url, "add", {"key": "p_arcana_normal_1"}, timeout=timeout_sec)
                st = get_state(base_url, timeout=timeout_sec)
                meta["pack_idx"] = _find_market_index(st, "packs", "p_arcana_normal_1")
                meta["pack_strategy"] = "direct_add"
            except Exception:
                meta["pack_idx"] = 0
                meta["pack_strategy"] = "shop_fallback"

    if prep == "joker":
        _call_method(base_url, "add", {"key": "j_joker"}, timeout=timeout_sec)
        meta["joker_idx"] = 0

    _call_method(base_url, "set", {"money": 250}, timeout=timeout_sec)
    return meta


def _execute_shop_action(base_url: str, action_type: str, params: dict[str, Any], timeout_sec: float, wait_sleep: float) -> dict[str, Any]:
    at = action_type.upper()
    if at == "WAIT":
        time.sleep(max(0.0, float(wait_sleep)))
    elif at == "REROLL":
        _call_method(base_url, "reroll", {}, timeout=timeout_sec)
    elif at == "NEXT_ROUND":
        _call_method(base_url, "next_round", {}, timeout=timeout_sec)
    elif at == "BUY":
        _call_method(base_url, "buy", params, timeout=timeout_sec)
    elif at == "PACK":
        _call_method(base_url, "pack", params, timeout=timeout_sec)
    elif at == "SELL":
        _call_method(base_url, "sell", params, timeout=timeout_sec)
    elif at == "USE":
        _call_method(base_url, "use", params, timeout=timeout_sec)
    else:
        raise ValueError(f"unsupported action_type: {action_type}")
    return get_state(base_url, timeout=timeout_sec)


def load_supported_entries(_project_root: Path | None = None) -> list[dict[str, Any]]:
    return [dict(x) for x in TARGET_SPECS]


def select_entries(targets_csv: str | None, limit: int | None) -> list[dict[str, Any]]:
    entries = load_supported_entries(None)
    by_target = {str(e.get("target") or ""): e for e in entries}

    if targets_csv:
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in str(targets_csv).split(","):
            t = raw.strip()
            if not t or t in seen:
                continue
            if t in by_target:
                selected.append(by_target[t])
                seen.add(t)
        entries = selected

    if limit is not None and int(limit) > 0:
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

    connection_reason = _probe_connection_failure(base_url, timeout_sec)
    if connection_reason is not None:
        return {
            "success": False,
            "target": target,
            "template": "shop_sequence",
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
        state = _to_shop(base_url, seed, timeout_sec)
    except Exception as exc:
        return {
            "success": False,
            "target": target,
            "template": "shop_sequence",
            "steps_used": 0,
            "final_phase": "UNKNOWN",
            "hit_info": {},
            "index_base": 0,
            "failure_reason": _classify_connection_failure_text(str(exc)),
            "start_snapshot": None,
            "action_trace": [],
            "oracle_start_state_path": None,
        }

    hit_info: dict[str, Any] = {
        "category": entry.get("category"),
        "description": entry.get("description"),
    }

    try:
        prep_meta = _prepare_target(base_url, entry, timeout_sec)
        hit_info.update(prep_meta)
        state = get_state(base_url, timeout=timeout_sec)

        action_trace: list[dict[str, Any]] = []
        steps = 0

        oracle_start_state_path: str | None = None
        if out_dir is not None:
            oracle_start_state_path = _save_start_state(base_url, out_dir, target, timeout_sec)

        start_snapshot = canonical_snapshot(state, seed=seed)

        for spec in (entry.get("actions") or []):
            if steps >= int(max_steps):
                break
            if not isinstance(spec, dict):
                continue
            action_type = str(spec.get("action_type") or "WAIT").upper()
            params = dict(spec.get("params") or {})

            # Resolve indexes from prepare metadata if needed.
            if action_type == "BUY" and "voucher" in params and "voucher_idx" in hit_info:
                params["voucher"] = int(hit_info.get("voucher_idx") or 0)
            if action_type == "BUY" and "pack" in params and "pack_idx" in hit_info:
                params["pack"] = int(hit_info.get("pack_idx") or 0)

            action_trace.append(
                {
                    "schema_version": "action_v1",
                    "phase": state_phase(state),
                    "action_type": action_type,
                    "params": {"index_base": 0, **params},
                }
            )

            state = _execute_shop_action(base_url, action_type, params, timeout_sec, wait_sleep)
            steps += 1

        if len(action_trace) < 2:
            return {
                "success": False,
                "target": target,
                "template": "shop_sequence",
                "steps_used": steps,
                "final_phase": state_phase(state),
                "hit_info": hit_info,
                "index_base": 0,
                "failure_reason": f"insufficient_actions; {_state_context_summary(state)}",
                "start_snapshot": start_snapshot,
                "action_trace": action_trace,
                "oracle_start_state_path": oracle_start_state_path,
            }

        _save_state_if_possible(base_url, oracle_start_state_path, timeout_sec)

        meta = start_snapshot.setdefault("_meta", {})
        if isinstance(meta, dict):
            meta["index_base_detected"] = 0
            meta["index_probe"] = {"method": "fixed", "note": "shop actions do not require hand index base"}
            meta["p8_target"] = target
            meta["p8_category"] = entry.get("category")

        return {
            "success": True,
            "target": target,
            "template": "shop_sequence",
            "steps_used": steps,
            "final_phase": state_phase(state),
            "hit_info": hit_info,
            "index_base": 0,
            "failure_reason": None,
            "start_snapshot": start_snapshot,
            "action_trace": action_trace,
            "oracle_start_state_path": oracle_start_state_path,
        }
    except (RPCError, ConnectionError, ValueError) as exc:
        st = get_state(base_url, timeout=timeout_sec)
        return {
            "success": False,
            "target": target,
            "template": "shop_sequence",
            "steps_used": 0,
            "final_phase": state_phase(st),
            "hit_info": hit_info,
            "index_base": 0,
            "failure_reason": f"{exc}; {_state_context_summary(st)}",
            "start_snapshot": canonical_snapshot(st, seed=seed),
            "action_trace": [],
            "oracle_start_state_path": None,
        }


def main() -> int:
    args = parse_args()
    if not health(args.base_url):
        print(f"base_url unhealthy: {args.base_url}. Start balatrobot serve first.")
        return 2

    entry = next((e for e in TARGET_SPECS if str(e.get("target") or "") == args.target), None)
    if entry is None:
        print(f"ERROR: unknown target: {args.target}")
        print("available targets:")
        for e in TARGET_SPECS:
            print(" -", e.get("target"))
        return 2

    out_dir: Path | None = None
    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = Path(__file__).resolve().parent.parent.parent / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    result = generate_one_trace(
        base_url=args.base_url,
        entry=entry,
        max_steps=int(args.max_steps),
        seed=args.seed,
        timeout_sec=float(args.timeout_sec),
        wait_sleep=float(args.wait_sleep),
        out_dir=out_dir,
    )

    print(
        f"target={result.get('target')} success={result.get('success')} "
        f"steps_used={result.get('steps_used')} final_phase={result.get('final_phase')}"
    )
    print("hit_info=", json.dumps(result.get("hit_info") or {}, ensure_ascii=False))

    if out_dir is not None and result.get("start_snapshot") is not None:
        snapshot_path = out_dir / f"oracle_start_snapshot_{result['target']}.json"
        action_path = out_dir / f"action_trace_{result['target']}.jsonl"
        snapshot_path.write_text(json.dumps(result["start_snapshot"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with action_path.open("w", encoding="utf-8", newline="\n") as fp:
            for action in result.get("action_trace") or []:
                fp.write(json.dumps(action, ensure_ascii=False) + "\n")
        print(f"wrote: {snapshot_path}")
        print(f"wrote: {action_path}")

    return 0 if bool(result.get("success")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
