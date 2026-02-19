from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from pathlib import Path
from typing import Any

from sim.oracle.generate_p0_trace import prepare_selecting_hand
from trainer.env_client import _call_method, get_state, health


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe balatrobot modifier/sticker capabilities (add/set/use paths).")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--wait-sleep", type=float, default=0.05)
    parser.add_argument("--out-json", default=None)
    return parser.parse_args()


def _ok(value: Any = None, note: str | None = None) -> dict[str, Any]:
    return {"ok": True, "value": value, "note": note}


def _fail(note: str) -> dict[str, Any]:
    return {"ok": False, "note": str(note)}


def _reset_to_selecting_hand(base_url: str, seed: str, timeout_sec: float, wait_sleep: float) -> None:
    try:
        _call_method(base_url, "menu", {}, timeout=timeout_sec)
    except Exception:
        pass
    _call_method(base_url, "start", {"deck": "RED", "stake": "WHITE", "seed": seed}, timeout=timeout_sec)
    prepare_selecting_hand(base_url, seed, timeout_sec, wait_sleep)


def _probe_add(base_url: str, timeout_sec: float, key: str) -> dict[str, Any]:
    try:
        _call_method(base_url, "add", {"key": key}, timeout=timeout_sec)
        state = get_state(base_url, timeout=timeout_sec)
        cons = state.get("consumables") if isinstance(state.get("consumables"), dict) else {}
        cards = cons.get("cards") if isinstance(cons.get("cards"), list) else []
        last_key = None
        if cards and isinstance(cards[-1], dict):
            last_key = cards[-1].get("key")
        return _ok({"consumables_count": len(cards), "last_key": last_key})
    except Exception as exc:
        return _fail(exc)


def _probe_set(base_url: str, timeout_sec: float, params: dict[str, Any]) -> dict[str, Any]:
    try:
        _call_method(base_url, "set", params, timeout=timeout_sec)
        state = get_state(base_url, timeout=timeout_sec)
        return _ok({"phase": state.get("state"), "round": state.get("round")})
    except Exception as exc:
        return _fail(exc)


def _probe_use(base_url: str, timeout_sec: float, add_key: str, use_params: dict[str, Any]) -> dict[str, Any]:
    try:
        _call_method(base_url, "add", {"key": add_key}, timeout=timeout_sec)
    except Exception as exc:
        return _fail(f"add_failed:{exc}")

    try:
        _call_method(base_url, "use", use_params, timeout=timeout_sec)
        state = get_state(base_url, timeout=timeout_sec)
        return _ok(
            {
                "phase": state.get("state"),
                "consumables_count": int(((state.get("consumables") or {}).get("count") or 0) if isinstance(state.get("consumables"), dict) else 0),
            }
        )
    except Exception as exc:
        return _fail(exc)


def main() -> int:
    args = parse_args()

    if not health(args.base_url):
        print(f"base_url unhealthy: {args.base_url}. Start balatrobot serve first.")
        return 2

    out: dict[str, Any] = {
        "base_url": args.base_url,
        "seed": args.seed,
        "health": True,
        "add": {},
        "set": {},
        "use": {},
        "conclusion": {},
    }

    _reset_to_selecting_hand(args.base_url, args.seed, args.timeout_sec, args.wait_sleep)

    out["add"]["tarot_bonus_hierophant"] = _probe_add(args.base_url, args.timeout_sec, "c_heirophant")
    out["add"]["tarot_mult_empress"] = _probe_add(args.base_url, args.timeout_sec, "c_empress")
    out["add"]["tarot_glass_justice"] = _probe_add(args.base_url, args.timeout_sec, "c_justice")
    out["add"]["spectral_red_deja_vu"] = _probe_add(args.base_url, args.timeout_sec, "c_deja_vu")
    out["add"]["spectral_gold_talisman"] = _probe_add(args.base_url, args.timeout_sec, "c_talisman")

    set_cases = {
        "money": {"money": 12},
        "hands_left": {"hands_left": 3},
        "discards_left": {"discards_left": 3},
        "chips": {"chips": 123},
        "modifier_context_attempt": {"modifier": {"enhancement": "BONUS"}},
        "card_edition_attempt": {"cards": {"0": {"modifier": {"edition": "FOIL"}}}},
        "sticker_attempt": {"stickers": {"ETERNAL": True}},
    }
    for name, params in set_cases.items():
        _reset_to_selecting_hand(args.base_url, args.seed, args.timeout_sec, args.wait_sleep)
        out["set"][name] = _probe_set(args.base_url, args.timeout_sec, params)

    _reset_to_selecting_hand(args.base_url, args.seed, args.timeout_sec, args.wait_sleep)
    out["use"]["tarot_with_cards"] = _probe_use(args.base_url, args.timeout_sec, "c_empress", {"consumable": 0, "cards": [0]})

    _reset_to_selecting_hand(args.base_url, args.seed, args.timeout_sec, args.wait_sleep)
    out["use"]["spectral_with_card"] = _probe_use(args.base_url, args.timeout_sec, "c_deja_vu", {"consumable": 0, "cards": [0]})

    _reset_to_selecting_hand(args.base_url, args.seed, args.timeout_sec, args.wait_sleep)
    out["use"]["without_cards"] = _probe_use(args.base_url, args.timeout_sec, "c_talisman", {"consumable": 0})

    add_ok = any(bool(v.get("ok")) for v in out["add"].values())
    use_ok = any(bool(v.get("ok")) for v in out["use"].values())
    set_modifier_ok = bool(out["set"].get("modifier_context_attempt", {}).get("ok"))
    set_sticker_ok = bool(out["set"].get("sticker_attempt", {}).get("ok"))

    out["conclusion"] = {
        "consume_via_add_use_available": add_ok and use_ok,
        "direct_modifier_set_supported": set_modifier_ok,
        "direct_sticker_set_supported": set_sticker_ok,
        "recommended_strategy": (
            "apply tarot/spectral via add+use when possible; fallback to wait_only fixture for non-injectable editions/stickers"
            if add_ok
            else "wait_only fallback (no reliable modifier injection discovered)"
        ),
    }

    print("[probe] Modifier capability summary")
    print(f"  add/use available: {out['conclusion']['consume_via_add_use_available']}")
    print(f"  direct set(modifier): {out['conclusion']['direct_modifier_set_supported']}")
    print(f"  direct set(sticker): {out['conclusion']['direct_sticker_set_supported']}")
    print(f"  strategy: {out['conclusion']['recommended_strategy']}")

    text = json.dumps(out, ensure_ascii=False, indent=2)
    print("\n" + text)

    if args.out_json:
        out_path = Path(args.out_json)
        if not out_path.is_absolute():
            out_path = Path(__file__).resolve().parent.parent.parent / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
