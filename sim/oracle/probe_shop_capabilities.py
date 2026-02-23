from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from trainer.env_client import _call_method, get_state, health


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe shop/economy RPC capabilities for P8 fixtures.")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--out-json", default="balatro_mechanics/derived/p8_shop_capabilities.json")
    parser.add_argument("--out-log", default="")
    return parser.parse_args()


def _ok(value: Any = None, note: str | None = None) -> dict[str, Any]:
    return {"ok": True, "value": value, "note": note}


def _fail(note: str) -> dict[str, Any]:
    return {"ok": False, "note": note}


def _state_phase(state: dict[str, Any]) -> str:
    return str(state.get("state") or "UNKNOWN")


def _method_probe(base_url: str, method: str, params: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
    try:
        _call_method(base_url, method, params, timeout=timeout_sec)
        return _ok({"params": params})
    except Exception as exc:
        return _fail(str(exc))


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
    _call_method(base_url, "set", {"money": 200}, timeout=timeout_sec)
    return get_state(base_url, timeout=timeout_sec)


def _probe_shop_sequence(base_url: str, seed: str, timeout_sec: float) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "enter_shop": _fail("not_run"),
        "reroll": _fail("not_run"),
        "buy_voucher": _fail("not_run"),
        "buy_pack": _fail("not_run"),
        "open_pack": _fail("not_run"),
        "buy_card": _fail("not_run"),
        "sell_joker": _fail("not_run"),
        "recommended_strategy": {},
    }

    st = _to_shop(base_url, seed, timeout_sec)
    summary["enter_shop"] = _ok({"phase": _state_phase(st)})

    try:
        _call_method(base_url, "reroll", {}, timeout=timeout_sec)
        st = get_state(base_url, timeout=timeout_sec)
        summary["reroll"] = _ok({"phase": _state_phase(st)})
    except Exception as exc:
        summary["reroll"] = _fail(str(exc))

    voucher_idx = 0
    try:
        _call_method(base_url, "add", {"key": "v_paint_brush"}, timeout=timeout_sec)
        st = get_state(base_url, timeout=timeout_sec)
        cards = ((st.get("vouchers") or {}).get("cards") or []) if isinstance(st.get("vouchers"), dict) else []
        voucher_idx = max(0, len(cards) - 1)
        _call_method(base_url, "buy", {"voucher": voucher_idx}, timeout=timeout_sec)
        st2 = get_state(base_url, timeout=timeout_sec)
        summary["buy_voucher"] = _ok({"index": voucher_idx, "phase": _state_phase(st2)})
    except Exception as exc:
        summary["buy_voucher"] = _fail(str(exc))

    pack_idx = 0
    try:
        _call_method(base_url, "add", {"key": "p_arcana_normal_1"}, timeout=timeout_sec)
        st = get_state(base_url, timeout=timeout_sec)
        cards = ((st.get("packs") or {}).get("cards") or []) if isinstance(st.get("packs"), dict) else []
        pack_idx = max(0, len(cards) - 1)
        _call_method(base_url, "buy", {"pack": pack_idx}, timeout=timeout_sec)
        st2 = get_state(base_url, timeout=timeout_sec)
        summary["buy_pack"] = _ok({"index": pack_idx, "phase": _state_phase(st2)})

        resolved = False
        for params in ({"card": 0}, {"card": 0, "cards": [0]}, {"card": 1}, {"skip": True}):
            try:
                _call_method(base_url, "pack", params, timeout=timeout_sec)
                st3 = get_state(base_url, timeout=timeout_sec)
                summary["open_pack"] = _ok({"params": params, "phase": _state_phase(st3)})
                resolved = True
                break
            except Exception:
                continue
        if not resolved:
            summary["open_pack"] = _fail("pack action not accepted for tested params")
    except Exception as exc:
        summary["buy_pack"] = _fail(str(exc))
        if not summary["open_pack"]["ok"]:
            summary["open_pack"] = _fail(str(exc))

    try:
        _call_method(base_url, "buy", {"card": 0}, timeout=timeout_sec)
        st = get_state(base_url, timeout=timeout_sec)
        summary["buy_card"] = _ok({"phase": _state_phase(st)})
    except Exception as exc:
        summary["buy_card"] = _fail(str(exc))

    try:
        _call_method(base_url, "add", {"key": "j_joker"}, timeout=timeout_sec)
        _call_method(base_url, "sell", {"joker": 0}, timeout=timeout_sec)
        st = get_state(base_url, timeout=timeout_sec)
        summary["sell_joker"] = _ok({"phase": _state_phase(st)})
    except Exception as exc:
        summary["sell_joker"] = _fail(str(exc))

    summary["recommended_strategy"] = {
        "voucher": "direct_add" if summary["buy_voucher"]["ok"] else "shop_buy",
        "pack": "direct_add" if summary["buy_pack"]["ok"] and summary["open_pack"]["ok"] else "shop_buy",
        "shop_actions": ["reroll", "buy(voucher)", "buy(pack)+pack", "buy(card)", "sell(joker)"],
    }
    return summary


def main() -> int:
    args = parse_args()

    summary: dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "base_url": args.base_url,
        "seed": args.seed,
        "health": False,
        "rpc_methods": {},
        "shop_sequences": {},
    }

    if not health(args.base_url):
        print(f"base_url unhealthy: {args.base_url}. Start balatrobot serve first.")
        return 2

    summary["health"] = True
    for method, params in {
        "start": {"deck": "RED", "stake": "WHITE", "seed": args.seed},
        "next_round": {},
        "skip": {},
        "select": {"index": 0},
        "play": {"cards": [0]},
        "discard": {"cards": [0]},
        "reroll": {},
        "buy": {"voucher": 0},
        "pack": {"card": 0},
        "sell": {"joker": 0},
        "add": {"key": "j_joker"},
        "set": {"money": 10},
        "use": {"consumable": 0},
    }.items():
        summary["rpc_methods"][method] = _method_probe(args.base_url, method, params, args.timeout_sec)

    summary["shop_sequences"] = _probe_shop_sequence(args.base_url, args.seed, args.timeout_sec)

    txt = json.dumps(summary, ensure_ascii=False, indent=2)
    print(txt)

    out_json = Path(args.out_json)
    if not out_json.is_absolute():
        out_json = Path(__file__).resolve().parent.parent.parent / out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(txt + "\n", encoding="utf-8")
    print(f"wrote: {out_json}")

    out_log = args.out_log.strip()
    if out_log:
        log_path = Path(out_log)
        if not log_path.is_absolute():
            log_path = Path(__file__).resolve().parent.parent.parent / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(txt + "\n", encoding="utf-8")
        print(f"wrote: {log_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
