from __future__ import annotations

from dataclasses import dataclass

from trainer import action_space_shop


@dataclass
class ShopDecision:
    phase: str
    action_id: int
    action: dict
    reason: str


def _count_cards(container) -> int:
    if isinstance(container, dict):
        cards = container.get("cards")
        if isinstance(cards, list):
            return len(cards)
        return int(container.get("count") or 0)
    if isinstance(container, list):
        return len(container)
    return 0


def choose_shop_action(state: dict) -> ShopDecision:
    phase = str(state.get("state") or "UNKNOWN")
    legal_ids = action_space_shop.legal_action_ids(state)

    # Hard fallback
    if not legal_ids:
        aid = action_space_shop.encode("WAIT", {})
        action = action_space_shop.action_from_id(state, aid)
        return ShopDecision(phase=phase, action_id=aid, action=action, reason="no_legal_fallback_wait")

    def pick(aid: int, reason: str) -> ShopDecision:
        if aid not in legal_ids:
            aid_local = legal_ids[0]
            return ShopDecision(phase=phase, action_id=aid_local, action=action_space_shop.action_from_id(state, aid_local), reason=f"fallback:{reason}")
        return ShopDecision(phase=phase, action_id=aid, action=action_space_shop.action_from_id(state, aid), reason=reason)

    money = float(state.get("money") or 0.0)
    shop_count = _count_cards(state.get("shop"))
    voucher_count = _count_cards(state.get("vouchers"))
    pack_count = _count_cards(state.get("packs"))
    consumable_count = _count_cards(state.get("consumables"))
    joker_count = _count_cards(state.get("jokers"))

    # Phase-specific simple deterministic rules.
    if phase == "SMODS_BOOSTER_OPENED":
        if _count_cards(state.get("pack_choices")) > 0:
            return pick(6, "pack_pick_first")
        return pick(7, "pack_skip")

    if phase == "SHOP":
        if pack_count > 0 and money > 0 and 3 in legal_ids:
            return pick(3, "buy_pack_first")
        if voucher_count > 0 and money > 0 and 4 in legal_ids:
            return pick(4, "buy_voucher")
        if shop_count > 0 and money > 0 and joker_count < 4 and 2 in legal_ids:
            return pick(2, "buy_card")
        if consumable_count > 0 and 8 in legal_ids:
            return pick(8, "use_consumable")
        if money > 0 and 1 in legal_ids and shop_count == 0 and pack_count == 0 and voucher_count == 0:
            return pick(1, "reroll_for_options")
        return pick(0, "next_round")

    return pick(9, "non_shop_wait")
