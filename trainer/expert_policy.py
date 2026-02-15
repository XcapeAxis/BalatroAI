if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
from dataclasses import dataclass

from trainer import action_space

TRANSITION_STATES = {
    "HAND_PLAYED",
    "DRAW_TO_HAND",
    "NEW_ROUND",
    "PLAY_TAROT",
    "TAROT_PACK",
    "PLANET_PACK",
    "SPECTRAL_PACK",
    "STANDARD_PACK",
    "BUFFOON_PACK",
}

RANK_VALUE = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "10": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}


@dataclass
class ExpertDecision:
    phase: str
    action_type: str | None = None
    mask_int: int | None = None
    reason: str = ""
    macro_action: str | None = None
    macro_params: dict | None = None


def _card_rank_suit(card: dict) -> tuple[str, str]:
    value = card.get("value") or {}
    rank = value.get("rank")
    suit = value.get("suit")
    if not rank or not suit:
        key = str(card.get("key") or "")
        if "_" in key:
            s, r = key.split("_", 1)
            suit = suit or s
            rank = rank or r
    rank = str(rank or "")
    suit = str(suit or "")
    return rank, suit


def _score_selected(cards: list[dict], indices: list[int]) -> float:
    selected = [cards[i] for i in indices]
    ranks = []
    suits = []
    values = []
    for card in selected:
        rank, suit = _card_rank_suit(card)
        value = RANK_VALUE.get(rank.upper(), 0)
        values.append(value)
        ranks.append(rank)
        suits.append(suit)

    if not values:
        return -1e9

    score = float(sum(values))
    score += 2.0 * len(values)

    rank_count: dict[str, int] = {}
    for r in ranks:
        rank_count[r] = rank_count.get(r, 0) + 1
    counts = sorted(rank_count.values(), reverse=True)
    if counts and counts[0] >= 5:
        score += 170
    elif counts and counts[0] == 4:
        score += 120
    elif counts and counts[0] == 3:
        score += 70
    elif counts and counts[0] == 2:
        score += 30
    if counts[:2] == [3, 2]:
        score += 55

    suit_count: dict[str, int] = {}
    for s in suits:
        suit_count[s] = suit_count.get(s, 0) + 1
    max_suit = max(suit_count.values())
    if max_suit == len(values) and len(values) >= 3:
        score += 55
    else:
        score += 5 * max_suit

    uniq_vals = sorted(set(values))
    if len(uniq_vals) >= 2:
        gaps = 0
        for i in range(1, len(uniq_vals)):
            gaps += max(0, uniq_vals[i] - uniq_vals[i - 1] - 1)
        score += max(0, 25 - gaps * 8)

    if len(values) == 1 and values[0] <= 7:
        score -= 12

    return score


def _score_discard(cards: list[dict], indices: list[int]) -> float:
    if not indices:
        return 0.0
    selected = [cards[i] for i in indices]
    all_suits = []
    for c in cards:
        _, suit = _card_rank_suit(c)
        all_suits.append(suit)
    suit_count: dict[str, int] = {}
    for s in all_suits:
        suit_count[s] = suit_count.get(s, 0) + 1

    badness = 0.0
    for card in selected:
        rank, suit = _card_rank_suit(card)
        value = RANK_VALUE.get(rank.upper(), 0)
        badness += max(0, 12 - value)
        badness += max(0, 3 - suit_count.get(suit, 0))

    badness += 0.8 * len(indices)
    return badness


def _select_blind_index(state: dict) -> int:
    blinds = state.get("blinds") or {}
    if not isinstance(blinds, dict):
        return 0
    for idx, key in enumerate(["small", "big", "boss"]):
        info = blinds.get(key) or {}
        if str(info.get("status") or "").upper() == "SELECT":
            return idx
    return 0


def choose_action(state: dict, start_seed: str = "AAAAAAA") -> ExpertDecision:
    phase = str(state.get("state") or "UNKNOWN")

    if phase == "SELECTING_HAND":
        cards = (state.get("hand") or {}).get("cards") or []
        hand_size = len(cards)
        if hand_size <= 0:
            return ExpertDecision(phase=phase, macro_action="wait", reason="empty_hand")

        hand_size = min(hand_size, action_space.MAX_HAND)
        discards_left = int((state.get("round") or {}).get("discards_left") or 0)
        hands_left = int((state.get("round") or {}).get("hands_left") or 0)

        best_play = None
        for action_id, atype, mask_int in action_space.action_entries(hand_size, action_space.PLAY):
            idxs = action_space.mask_to_indices(mask_int, hand_size)
            s = _score_selected(cards, idxs)
            if best_play is None or s > best_play[0]:
                best_play = (s, action_id, mask_int, idxs)

        best_discard = None
        if discards_left > 0:
            for action_id, atype, mask_int in action_space.action_entries(hand_size, action_space.DISCARD):
                idxs = action_space.mask_to_indices(mask_int, hand_size)
                s = _score_discard(cards, idxs)
                if best_discard is None or s > best_discard[0]:
                    best_discard = (s, action_id, mask_int, idxs)

        assert best_play is not None
        play_score = best_play[0]
        threshold = 52.0
        force_discard = hands_left <= 0 and discards_left > 0
        choose_discard = force_discard or (
            best_discard is not None and discards_left > 0 and play_score < threshold and best_discard[0] >= 8.0
        )

        if choose_discard and best_discard is not None:
            return ExpertDecision(
                phase=phase,
                action_type=action_space.DISCARD,
                mask_int=best_discard[2],
                reason=f"discard_score={best_discard[0]:.2f}",
            )

        return ExpertDecision(
            phase=phase,
            action_type=action_space.PLAY,
            mask_int=best_play[2],
            reason=f"play_score={best_play[0]:.2f}",
        )

    if phase == "BLIND_SELECT":
        return ExpertDecision(
            phase=phase,
            macro_action="select",
            macro_params={"index": _select_blind_index(state)},
            reason="blind_select",
        )

    if phase == "ROUND_EVAL":
        return ExpertDecision(phase=phase, macro_action="cash_out", macro_params={}, reason="round_eval")

    if phase == "SHOP":
        return ExpertDecision(phase=phase, macro_action="next_round", macro_params={}, reason="shop_next")

    if phase in TRANSITION_STATES:
        return ExpertDecision(phase=phase, macro_action="wait", macro_params={}, reason="transition")

    if phase in {"GAME_OVER", "MENU"}:
        return ExpertDecision(
            phase=phase,
            macro_action="start",
            macro_params={"deck": "RED", "stake": "WHITE", "seed": start_seed},
            reason="restart",
        )

    return ExpertDecision(phase=phase, macro_action="wait", macro_params={}, reason="default_wait")

if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run expert policy on a gamestate JSON file.")
    parser.add_argument("--state-json", required=True, help="Path to gamestate json file.")
    parser.add_argument("--seed", default="AAAAAAA", help="Seed used for start macro suggestion.")
    args = parser.parse_args()

    state = json.loads(Path(args.state_json).read_text(encoding="utf-8"))
    decision = choose_action(state, start_seed=args.seed)
    print(json.dumps(decision.__dict__, ensure_ascii=False, indent=2))

