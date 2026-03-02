from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import unittest

from sim.core.canonical import to_canonical_state
from sim.core.engine import SimEnv
from sim.core.hashing import (
    state_hash_p14_real_action_observed_core,
    state_hash_p32_real_action_position_observed_core,
)


def _with_observed(state: dict) -> dict:
    canonical = to_canonical_state(state, rng_mode="native", seed="TESTSEQ", rng_cursor=0, rng_events=[])
    out = dict(canonical)
    out["score_observed"] = {"total": float((state.get("round") or {}).get("chips") or 0.0), "delta": 0.0}
    out["rng_replay"] = {"enabled": False, "source": "", "outcomes": []}
    return out


class ReorderActionTest(unittest.TestCase):
    def test_reorder_hand_updates_order_without_score_change(self) -> None:
        env = SimEnv(seed="TESTSEQ")
        env.step({"action_type": "SELECT", "index": 0})

        before = env._state  # noqa: SLF001 - test-only introspection
        before_keys = [str(c.get("key") or "") for c in (before.get("hand") or {}).get("cards") or []]
        self.assertGreaterEqual(len(before_keys), 2)

        permutation = list(reversed(range(len(before_keys))))
        next_state, reward, _done, _info = env.step({"action_type": "REORDER_HAND", "permutation": permutation})

        after_keys = [str(c.get("key") or "") for c in (next_state.get("hand") or {}).get("cards") or []]
        self.assertEqual(after_keys, [before_keys[idx] for idx in permutation])
        self.assertEqual(float(reward), 0.0)
        self.assertEqual(float((before.get("round") or {}).get("chips") or 0.0), float((next_state.get("round") or {}).get("chips") or 0.0))

    def test_reorder_jokers_and_swap_jokers(self) -> None:
        env = SimEnv(seed="TESTJOKER")
        env.step({"action_type": "SELECT", "index": 0})
        env._state["jokers"] = [  # noqa: SLF001 - test-only setup
            {"joker_id": "j_blueprint", "key": "j_blueprint"},
            {"joker_id": "j_brainstorm", "key": "j_brainstorm"},
            {"joker_id": "j_joker", "key": "j_joker"},
        ]

        state1, reward1, _done1, _info1 = env.step({"action_type": "REORDER_JOKERS", "permutation": [2, 0, 1]})
        keys1 = [str(j.get("joker_id") or j.get("key") or "") for j in state1.get("jokers") or []]
        self.assertEqual(keys1, ["j_joker", "j_blueprint", "j_brainstorm"])
        self.assertEqual(float(reward1), 0.0)

        state2, reward2, _done2, _info2 = env.step({"action_type": "SWAP_JOKERS", "i": 0, "j": 2})
        keys2 = [str(j.get("joker_id") or j.get("key") or "") for j in state2.get("jokers") or []]
        self.assertEqual(keys2, ["j_brainstorm", "j_blueprint", "j_joker"])
        self.assertEqual(float(reward2), 0.0)

    def test_p32_scope_detects_hand_order_while_p14_stays_stable(self) -> None:
        env = SimEnv(seed="TESTHASH")
        env.step({"action_type": "SELECT", "index": 0})

        before = _with_observed(env._state)  # noqa: SLF001 - test-only introspection
        before_p14 = state_hash_p14_real_action_observed_core(before)
        before_p32 = state_hash_p32_real_action_position_observed_core(before)

        hand_cards = (env._state.get("hand") or {}).get("cards") or []  # noqa: SLF001
        permutation = list(reversed(range(len(hand_cards))))
        env.step({"action_type": "REORDER_HAND", "permutation": permutation})

        after = _with_observed(env._state)  # noqa: SLF001
        after_p14 = state_hash_p14_real_action_observed_core(after)
        after_p32 = state_hash_p32_real_action_position_observed_core(after)

        self.assertEqual(before_p14, after_p14)
        self.assertNotEqual(before_p32, after_p32)


if __name__ == "__main__":
    unittest.main()
