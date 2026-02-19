from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from typing import Any

from sim.oracle.generate_p0_trace import prepare_selecting_hand
from trainer.env_client import ConnectionError, RPCError, _call_method, get_state, health


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe balatrobot consumable capabilities (add/set/use).")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--wait-sleep", type=float, default=0.05)
    return parser.parse_args()


def _ok(value: Any = None, note: str | None = None) -> dict[str, Any]:
    return {"ok": True, "value": value, "note": note}


def _fail(note: str) -> dict[str, Any]:
    return {"ok": False, "note": note}


def _reset_to_selecting_hand(base_url: str, seed: str, timeout_sec: float, wait_sleep: float) -> dict[str, Any]:
    try:
        _call_method(base_url, "menu", {}, timeout=timeout_sec)
    except Exception:
        pass
    _call_method(base_url, "start", {"deck": "RED", "stake": "WHITE", "seed": seed}, timeout=timeout_sec)
    return prepare_selecting_hand(base_url, seed, timeout_sec, wait_sleep)


def _probe_add(base_url: str, timeout_sec: float, key: str) -> dict[str, Any]:
    try:
        _call_method(base_url, "add", {"key": key}, timeout=timeout_sec)
        state = get_state(base_url, timeout=timeout_sec)
        cards = ((state.get("consumables") or {}).get("cards") or []) if isinstance(state.get("consumables"), dict) else []
        last_key = None
        if cards and isinstance(cards[-1], dict):
            last_key = cards[-1].get("key")
        return _ok(value={"consumables_count": len(cards), "last_key": last_key})
    except Exception as exc:
        return _fail(str(exc))


def _probe_set(base_url: str, timeout_sec: float, params: dict[str, Any]) -> dict[str, Any]:
    try:
        _call_method(base_url, "set", params, timeout=timeout_sec)
        state = get_state(base_url, timeout=timeout_sec)
        return _ok(value={"phase": state.get("state"), "round": state.get("round")})
    except Exception as exc:
        return _fail(str(exc))


def _probe_use(base_url: str, timeout_sec: float, add_key: str, use_params: dict[str, Any]) -> dict[str, Any]:
    try:
        _call_method(base_url, "add", {"key": add_key}, timeout=timeout_sec)
    except Exception as exc:
        return _fail(f"add_failed:{exc}")

    try:
        _call_method(base_url, "use", use_params, timeout=timeout_sec)
        state = get_state(base_url, timeout=timeout_sec)
        return _ok(
            value={
                "phase": state.get("state"),
                "consumables_count": int(((state.get("consumables") or {}).get("count") or 0) if isinstance(state.get("consumables"), dict) else 0),
            }
        )
    except Exception as exc:
        return _fail(str(exc))


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
        "notes": [],
    }

    try:
        _reset_to_selecting_hand(args.base_url, args.seed, args.timeout_sec, args.wait_sleep)
    except Exception as exc:
        print(json.dumps({"health": True, "fatal": f"failed to reach SELECTING_HAND: {exc}"}, ensure_ascii=False, indent=2))
        return 1

    # add() probe
    out["add"]["planet:c_saturn"] = _probe_add(args.base_url, args.timeout_sec, "c_saturn")
    out["add"]["tarot:c_magician"] = _probe_add(args.base_url, args.timeout_sec, "c_magician")
    out["add"]["spectral:c_trance"] = _probe_add(args.base_url, args.timeout_sec, "c_trance")

    # set() probe (state-edit hints for fallback)
    set_cases = {
        "money": {"money": 9},
        "hands_left": {"hands_left": 3},
        "discards_left": {"discards_left": 3},
        "chips": {"chips": 123},
        "consumable_limit": {"consumables": {"limit": 3}},
        "hands_level_attempt": {"hands": {"STRAIGHT": {"level": 2}}},
    }
    for name, params in set_cases.items():
        _reset_to_selecting_hand(args.base_url, args.seed, args.timeout_sec, args.wait_sleep)
        out["set"][name] = _probe_set(args.base_url, args.timeout_sec, params)

    # use() probe with common parameter shapes
    _reset_to_selecting_hand(args.base_url, args.seed, args.timeout_sec, args.wait_sleep)
    out["use"]["planet_basic"] = _probe_use(args.base_url, args.timeout_sec, "c_saturn", {"consumable": 0})

    _reset_to_selecting_hand(args.base_url, args.seed, args.timeout_sec, args.wait_sleep)
    out["use"]["tarot_without_cards"] = _probe_use(args.base_url, args.timeout_sec, "c_magician", {"consumable": 0})

    _reset_to_selecting_hand(args.base_url, args.seed, args.timeout_sec, args.wait_sleep)
    out["use"]["tarot_with_cards"] = _probe_use(args.base_url, args.timeout_sec, "c_magician", {"consumable": 0, "cards": [0, 1]})

    _reset_to_selecting_hand(args.base_url, args.seed, args.timeout_sec, args.wait_sleep)
    out["use"]["spectral_with_card"] = _probe_use(args.base_url, args.timeout_sec, "c_trance", {"consumable": 0, "cards": [0]})

    print("[probe] Consumable capability summary")
    for group in ("add", "set", "use"):
        print(f"[{group}]")
        for k, v in out[group].items():
            print(f"  {k}: ok={v.get('ok')} note={v.get('note')}")

    print("\n" + json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
