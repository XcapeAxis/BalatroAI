if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import time
from pathlib import Path

from trainer.env_client import CONFIG_PATH, RPCError, _call_method, get_index_base, get_state, load_config, save_config


def parse_args():
    parser = argparse.ArgumentParser(description="Detect whether card indices are 0-based or 1-based for RPC cards params.")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--out", default=str(CONFIG_PATH), help="Config output path")
    return parser.parse_args()


def _state_name(state):
    return str((state or {}).get("state") or "UNKNOWN")


def _select_blind_index(state) -> int:
    blinds = state.get("blinds") or {}
    for idx, key in enumerate(["small", "big", "boss"]):
        if str((blinds.get(key) or {}).get("status") or "").upper() == "SELECT":
            return idx
    return 0


def ensure_selecting_hand(base_url: str, timeout: float, seed: str) -> dict:
    deadline = time.time() + 30.0
    while time.time() < deadline:
        state = get_state(base_url, timeout=timeout)
        phase = _state_name(state)

        if phase == "SELECTING_HAND":
            cards = (state.get("hand") or {}).get("cards") or []
            if len(cards) > 0:
                return state

        if phase in {"MENU", "GAME_OVER"}:
            _call_method(base_url, "start", {"deck": "RED", "stake": "WHITE", "seed": seed}, timeout=timeout)
            time.sleep(0.1)
            continue

        if phase == "BLIND_SELECT":
            _call_method(base_url, "select", {"index": _select_blind_index(state)}, timeout=timeout)
            time.sleep(0.1)
            continue

        if phase == "ROUND_EVAL":
            _call_method(base_url, "cash_out", {}, timeout=timeout)
            time.sleep(0.1)
            continue

        if phase == "SHOP":
            _call_method(base_url, "next_round", {}, timeout=timeout)
            time.sleep(0.1)
            continue

        time.sleep(0.1)

    raise RuntimeError("Failed to reach SELECTING_HAND with at least one card")


def _looks_like_invalid_index(message: str) -> bool:
    s = message.lower()
    markers = [
        "out of range",
        "invalid index",
        "index",
        "bad argument",
        "invalid params",
    ]
    return any(m in s for m in markers)


def _probe_index_base(base_url: str, timeout: float, seed: str, candidate_base: int) -> tuple[bool, str]:
    state = ensure_selecting_hand(base_url, timeout, seed)
    cards = (state.get("hand") or {}).get("cards") or []
    if not cards:
        return False, "no_cards"

    api_idx = candidate_base

    try:
        _call_method(base_url, "discard", {"cards": [api_idx]}, timeout=timeout)
        return True, "discard_ok"
    except RPCError as exc:
        msg = str(exc)
        if _looks_like_invalid_index(msg):
            return False, f"discard_invalid:{msg}"

    try:
        _call_method(base_url, "play", {"cards": [api_idx]}, timeout=timeout)
        return True, "play_ok"
    except RPCError as exc:
        msg = str(exc)
        if _looks_like_invalid_index(msg):
            return False, f"play_invalid:{msg}"
        return False, f"ambiguous:{msg}"


def main() -> int:
    args = parse_args()

    results = {}
    for base in [0, 1]:
        ok, detail = _probe_index_base(args.base_url, args.timeout_sec, args.seed, base)
        results[base] = {"ok": ok, "detail": detail}

    chosen = None
    if results[0]["ok"] and not results[1]["ok"]:
        chosen = 0
    elif results[1]["ok"] and not results[0]["ok"]:
        chosen = 1
    elif results[0]["ok"] and results[1]["ok"]:
        chosen = 0
    else:
        chosen = get_index_base(force_reload=True)

    cfg = load_config(force_reload=True)
    cfg["index_base"] = int(chosen)
    save_config(cfg)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    out = {
        "base_url": args.base_url,
        "results": results,
        "chosen_index_base": chosen,
        "config_path": str(out_path),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
