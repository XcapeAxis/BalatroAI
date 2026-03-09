from __future__ import annotations

from trainer.legal_actions import legal_hand_action_ids_for_state


def _state(*, hands_left: int, discards_left: int, hand_size: int = 5) -> dict:
    return {
        "state": "SELECTING_HAND",
        "hand": {"cards": [{"id": idx} for idx in range(hand_size)]},
        "round": {"hands_left": hands_left, "discards_left": discards_left},
    }


def test_legal_hand_actions_drop_plays_when_no_hands_left() -> None:
    legal_ids = legal_hand_action_ids_for_state(_state(hands_left=0, discards_left=2))
    assert legal_ids
    assert all(action_id >= 31 for action_id in legal_ids)


def test_legal_hand_actions_drop_discards_when_no_discards_left() -> None:
    legal_ids = legal_hand_action_ids_for_state(_state(hands_left=2, discards_left=0))
    assert legal_ids
    assert all(action_id < 31 for action_id in legal_ids)
