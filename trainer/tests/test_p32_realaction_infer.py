from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import copy
import unittest

from sim.core.engine import SimEnv
from trainer.real_trace_to_fixture import _infer_position_action


class RealTraceInferenceTest(unittest.TestCase):
    def test_infer_reorder_hand_from_raw_delta(self) -> None:
        env = SimEnv(seed="P32INFER-HAND")
        env.step({"action_type": "SELECT", "index": 0})
        before = copy.deepcopy(env.get_state())

        hand = (before.get("hand") or {}).get("cards") or []
        permutation = list(range(len(hand)))
        permutation[0], permutation[1] = permutation[1], permutation[0]
        env.step({"action_type": "REORDER_HAND", "permutation": permutation})
        after = copy.deepcopy(env.get_state())

        inferred = _infer_position_action(before, after, row={"phase": "SELECTING_HAND"}, seed="P32INFER-HAND")
        self.assertIsNotNone(inferred)
        self.assertEqual(str((inferred or {}).get("action_type") or ""), "SWAP_HAND_CARDS")

    def test_infer_reorder_jokers_from_raw_delta(self) -> None:
        env = SimEnv(seed="P32INFER-JOKER")
        env.step({"action_type": "SELECT", "index": 0})
        env._state["jokers"] = [  # noqa: SLF001 - test-only setup
            {"joker_id": "j_blueprint", "key": "j_blueprint"},
            {"joker_id": "j_brainstorm", "key": "j_brainstorm"},
            {"joker_id": "j_joker", "key": "j_joker"},
        ]
        before = copy.deepcopy(env.get_state())

        env.step({"action_type": "REORDER_JOKERS", "permutation": [2, 0, 1]})
        after = copy.deepcopy(env.get_state())

        inferred = _infer_position_action(before, after, row={"phase": "SELECTING_HAND"}, seed="P32INFER-JOKER")
        self.assertIsNotNone(inferred)
        self.assertEqual(str((inferred or {}).get("action_type") or ""), "REORDER_JOKERS")


if __name__ == "__main__":
    unittest.main()
