from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Any

from trainer import action_space_shop


def _action_priority(action: dict[str, Any]) -> float:
    atype = str(action.get("action_type") or "").upper()
    params = action.get("params") if isinstance(action.get("params"), dict) else {}
    cost = float(params.get("cost") or action.get("cost") or 0.0)

    if atype in {"BUY", "BUY_JOKER", "BUY_VOUCHER", "BUY_CONSUMABLE"}:
        return 80.0 - min(60.0, cost)
    if atype in {"OPEN_PACK", "PACK"}:
        return 60.0
    if atype in {"REROLL", "SHOP_REROLL"}:
        return 45.0 - min(25.0, cost)
    if atype in {"SELL", "SHOP_SELL"}:
        return 35.0
    if atype in {"NEXT_ROUND", "SKIP", "LEAVE_SHOP", "WAIT"}:
        return 10.0
    return 5.0


def generate_shop_candidates(
    state: dict[str, Any],
    *,
    max_candidates: int = 20,
    buy_top_k: int = 6,
    max_reroll: int = 2,
) -> list[dict[str, Any]]:
    legal_ids = action_space_shop.legal_action_ids(state)
    if not legal_ids:
        return [{"action_type": "WAIT", "sleep": 0.01}]

    scored: list[tuple[float, dict[str, Any]]] = []
    buy_count = 0
    reroll_count = 0

    for aid in legal_ids:
        action = action_space_shop.action_from_id(state, int(aid))
        atype = str(action.get("action_type") or "").upper()
        if atype.startswith("BUY"):
            if buy_count >= max(1, int(buy_top_k)):
                continue
            buy_count += 1
        if "REROLL" in atype:
            if reroll_count >= max(0, int(max_reroll)):
                continue
            reroll_count += 1
        scored.append((_action_priority(action), action))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, action in scored:
        key = str(action)
        if key in seen:
            continue
        seen.add(key)
        out.append(action)
        if len(out) >= max(1, int(max_candidates)):
            break

    has_leave = any(str(a.get("action_type") or "").upper() in {"NEXT_ROUND", "SKIP", "LEAVE_SHOP"} for a in out)
    if not has_leave:
        out.append({"action_type": "NEXT_ROUND"})
    return out
