from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from pathlib import Path
from typing import Any

from trainer.env_client import _call_method, get_state, health


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe balatrobot voucher/booster-pack capabilities (add/set/use/buy/pack).")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--wait-sleep", type=float, default=0.05)
    parser.add_argument("--out-json", default="balatro_mechanics/derived/p5_capabilities.json")
    return parser.parse_args()


def _ok(value: Any = None, note: str | None = None) -> dict[str, Any]:
    return {"ok": True, "value": value, "note": note}


def _fail(note: str) -> dict[str, Any]:
    return {"ok": False, "note": note}


def _state_phase(state: dict[str, Any]) -> str:
    return str(state.get("state") or "UNKNOWN")


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


def _method_probe(base_url: str, method: str, params: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
    try:
        _call_method(base_url, method, params, timeout=timeout_sec)
        return _ok()
    except Exception as exc:
        return _fail(str(exc))


def _free_pack_slot(base_url: str, timeout_sec: float) -> tuple[bool, str]:
    try:
        _call_method(base_url, "buy", {"pack": 0}, timeout=timeout_sec)
    except Exception as exc:
        return False, f"buy(pack=0) failed: {exc}"

    attempts = [
        {"card": 0},
        {"card": 0, "cards": [0]},
        {"card": 0, "cards": [0, 1]},
        {"card": 1},
        {"skip": True},
    ]
    for params in attempts:
        try:
            _call_method(base_url, "pack", params, timeout=timeout_sec)
            return True, f"resolved with {params}"
        except Exception:
            continue
    return False, "pack resolution failed"


def _probe_voucher(base_url: str, seed: str, timeout_sec: float) -> dict[str, Any]:
    state = _to_shop(base_url, seed, timeout_sec)
    result: dict[str, Any] = {
        "direct_add": _fail("not_run"),
        "direct_buy": _fail("not_run"),
        "recommended_strategy": "unsupported",
        "sequence_template": [],
    }

    try:
        _call_method(base_url, "add", {"key": "v_paint_brush"}, timeout=timeout_sec)
        state = get_state(base_url, timeout=timeout_sec)
        vouchers = (state.get("vouchers") or {}) if isinstance(state.get("vouchers"), dict) else {}
        cards = vouchers.get("cards") if isinstance(vouchers.get("cards"), list) else []
        idx = max(0, len(cards) - 1)
        result["direct_add"] = _ok({"added_key": "v_paint_brush", "voucher_count": len(cards), "buy_index": idx})
        try:
            _call_method(base_url, "buy", {"voucher": idx}, timeout=timeout_sec)
            state2 = get_state(base_url, timeout=timeout_sec)
            used = state2.get("used_vouchers")
            if isinstance(used, dict):
                used_keys = sorted(str(k) for k in used.keys())
            elif isinstance(used, list):
                used_keys = [str(x) for x in used]
            else:
                used_keys = []
            result["direct_buy"] = _ok({"phase": _state_phase(state2), "used_vouchers": used_keys})
            result["recommended_strategy"] = "direct_add"
            result["sequence_template"] = ["menu", "start", "select", "set(chips)", "play", "cash_out", "set(money)", "add(voucher_key)", "buy(voucher=index)"]
        except Exception as exc:
            result["direct_buy"] = _fail(str(exc))
            result["recommended_strategy"] = "shop_buy"
            result["sequence_template"] = ["menu", "start", "select", "set(chips)", "play", "cash_out", "set(money)", "buy(voucher=0)"]
    except Exception as exc:
        result["direct_add"] = _fail(str(exc))
        # fallback: try buying current shop voucher only
        try:
            _to_shop(base_url, seed, timeout_sec)
            _call_method(base_url, "buy", {"voucher": 0}, timeout=timeout_sec)
            state2 = get_state(base_url, timeout=timeout_sec)
            result["direct_buy"] = _ok({"phase": _state_phase(state2)})
            result["recommended_strategy"] = "shop_buy"
            result["sequence_template"] = ["menu", "start", "select", "set(chips)", "play", "cash_out", "set(money)", "buy(voucher=0)"]
        except Exception as exc2:
            result["direct_buy"] = _fail(str(exc2))
            result["recommended_strategy"] = "unsupported"
            result["sequence_template"] = []

    return result


def _probe_pack(base_url: str, seed: str, timeout_sec: float) -> dict[str, Any]:
    _to_shop(base_url, seed, timeout_sec)
    result: dict[str, Any] = {
        "direct_add": _fail("not_run"),
        "buy_pack": _fail("not_run"),
        "resolve_pack": _fail("not_run"),
        "recommended_strategy": "unsupported",
        "sequence_template": [],
    }

    # direct add path (requires free booster slot)
    ok_slot, note = _free_pack_slot(base_url, timeout_sec)
    if ok_slot:
        try:
            _call_method(base_url, "set", {"money": 100}, timeout=timeout_sec)
            _call_method(base_url, "add", {"key": "p_arcana_normal_1"}, timeout=timeout_sec)
            state = get_state(base_url, timeout=timeout_sec)
            packs = (state.get("packs") or {}) if isinstance(state.get("packs"), dict) else {}
            cards = packs.get("cards") if isinstance(packs.get("cards"), list) else []
            idx = max(0, len(cards) - 1)
            result["direct_add"] = _ok({"added_key": "p_arcana_normal_1", "pack_count": len(cards), "buy_index": idx, "slot_note": note})

            _call_method(base_url, "buy", {"pack": idx}, timeout=timeout_sec)
            st2 = get_state(base_url, timeout=timeout_sec)
            result["buy_pack"] = _ok({"phase": _state_phase(st2)})

            resolved = False
            last_err = None
            for params in (
                {"card": 0},
                {"card": 0, "cards": [0]},
                {"card": 0, "cards": [0, 1]},
                {"card": 1},
                {"skip": True},
            ):
                try:
                    _call_method(base_url, "pack", params, timeout=timeout_sec)
                    st3 = get_state(base_url, timeout=timeout_sec)
                    result["resolve_pack"] = _ok({"params": params, "phase": _state_phase(st3)})
                    resolved = True
                    break
                except Exception as exc:
                    last_err = str(exc)

            if resolved:
                result["recommended_strategy"] = "direct_add"
                result["sequence_template"] = ["menu", "start", "select", "set(chips)", "play", "cash_out", "set(money)", "buy(pack=0)", "pack(card=0|...) to free slot", "add(pack_key)", "buy(pack=index)", "pack(card=0|cards=[...])"]
            else:
                result["resolve_pack"] = _fail(last_err or "pack resolve failed")
        except Exception as exc:
            result["direct_add"] = _fail(str(exc))

    # fallback shop_buy path
    if not result["buy_pack"].get("ok"):
        try:
            _to_shop(base_url, seed, timeout_sec)
            _call_method(base_url, "buy", {"pack": 0}, timeout=timeout_sec)
            st2 = get_state(base_url, timeout=timeout_sec)
            result["buy_pack"] = _ok({"phase": _state_phase(st2), "mode": "shop_existing_pack"})

            resolved = False
            last_err = None
            for params in (
                {"card": 0},
                {"card": 0, "cards": [0]},
                {"card": 0, "cards": [0, 1]},
                {"card": 1},
                {"skip": True},
            ):
                try:
                    _call_method(base_url, "pack", params, timeout=timeout_sec)
                    st3 = get_state(base_url, timeout=timeout_sec)
                    result["resolve_pack"] = _ok({"params": params, "phase": _state_phase(st3), "mode": "shop_existing_pack"})
                    resolved = True
                    break
                except Exception as exc:
                    last_err = str(exc)
            if resolved:
                result["recommended_strategy"] = "shop_buy"
                result["sequence_template"] = ["menu", "start", "select", "set(chips)", "play", "cash_out", "set(money)", "buy(pack=0)", "pack(card=0|cards=[...])"]
            else:
                result["resolve_pack"] = _fail(last_err or "pack resolve failed")
        except Exception as exc:
            if not result["buy_pack"].get("ok"):
                result["buy_pack"] = _fail(str(exc))

    if result["recommended_strategy"] == "unsupported":
        if result["direct_add"].get("ok") and result["buy_pack"].get("ok") and result["resolve_pack"].get("ok"):
            result["recommended_strategy"] = "direct_add"
        elif result["buy_pack"].get("ok") and result["resolve_pack"].get("ok"):
            result["recommended_strategy"] = "shop_buy"

    return result


def main() -> int:
    args = parse_args()

    summary: dict[str, Any] = {
        "base_url": args.base_url,
        "seed": args.seed,
        "health": False,
        "methods": {},
        "voucher": {},
        "pack": {},
    }

    if not health(args.base_url):
        print(f"base_url unhealthy: {args.base_url}. Start balatrobot serve first.")
        return 2
    summary["health"] = True

    # generic method probes
    method_cases = {
        "add": {"key": "c_empress"},
        "set": {"money": 11},
        "use": {"consumable": 0},
        "buy": {"voucher": 0},
        "pack": {"card": 0},
        "start": {"deck": "RED", "stake": "WHITE", "seed": args.seed},
        "next_round": {},
        "skip": {},
        "select": {"index": 0},
        "play": {"cards": [0]},
        "discard": {"cards": [0]},
    }
    for method, params in method_cases.items():
        try:
            probe = _method_probe(args.base_url, method, params, args.timeout_sec)
            summary["methods"][method] = probe
        except Exception as exc:
            summary["methods"][method] = _fail(str(exc))

    summary["voucher"] = _probe_voucher(args.base_url, args.seed, args.timeout_sec)
    summary["pack"] = _probe_pack(args.base_url, args.seed, args.timeout_sec)

    print("[probe] Voucher/Pack capability summary")
    print(f"  voucher strategy: {summary['voucher'].get('recommended_strategy')}")
    print(f"  pack strategy: {summary['pack'].get('recommended_strategy')}")
    print("")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    out_json = Path(args.out_json)
    if not out_json.is_absolute():
        out_json = Path(__file__).resolve().parent.parent.parent / out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
