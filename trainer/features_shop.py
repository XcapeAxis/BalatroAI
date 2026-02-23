from __future__ import annotations

from typing import Any

SHOP_CONTEXT_DIM = 16


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _count_cards(container: Any) -> int:
    if isinstance(container, dict):
        cards = container.get("cards")
        if isinstance(cards, list):
            return len(cards)
        return int(container.get("count") or 0)
    if isinstance(container, list):
        return len(container)
    return 0


def extract_shop_features(state: dict[str, Any]) -> dict[str, Any]:
    phase = str(state.get("state") or "UNKNOWN")
    phase_shop = 1.0 if phase == "SHOP" else 0.0
    phase_pack = 1.0 if phase == "SMODS_BOOSTER_OPENED" else 0.0

    round_info = state.get("round") if isinstance(state.get("round"), dict) else {}

    money = _to_float(state.get("money"), 0.0)
    chips = _to_float(round_info.get("chips"), 0.0)
    target = _to_float((state.get("score") or {}).get("target_chips"), 0.0)
    reroll_cost = _to_float(round_info.get("reroll_cost"), 0.0)

    hands_left = _to_float(round_info.get("hands_left"), 0.0)
    discards_left = _to_float(round_info.get("discards_left"), 0.0)

    shop_count = float(_count_cards(state.get("shop")))
    voucher_count = float(_count_cards(state.get("vouchers")))
    pack_count = float(_count_cards(state.get("packs")))
    consumable_count = float(_count_cards(state.get("consumables")))
    joker_count = float(_count_cards(state.get("jokers")))
    pack_choice_count = float(_count_cards(state.get("pack_choices")))

    # Normalize with simple bounded transforms.
    context = [
        phase_shop,
        phase_pack,
        min(1.0, money / 100.0),
        min(1.0, reroll_cost / 20.0),
        min(1.0, hands_left / 10.0),
        min(1.0, discards_left / 10.0),
        min(1.0, chips / 1000.0),
        min(1.0, target / 5000.0),
        min(1.0, shop_count / 8.0),
        min(1.0, voucher_count / 4.0),
        min(1.0, pack_count / 4.0),
        min(1.0, consumable_count / 4.0),
        min(1.0, joker_count / 10.0),
        min(1.0, pack_choice_count / 6.0),
        1.0 if money > 0 else 0.0,
        1.0 if (shop_count + voucher_count + pack_count) > 0 else 0.0,
    ]

    if len(context) != SHOP_CONTEXT_DIM:
        raise RuntimeError(f"shop context dim mismatch: {len(context)}")

    return {
        "shop_context": context,
        "shop_phase": phase,
    }
