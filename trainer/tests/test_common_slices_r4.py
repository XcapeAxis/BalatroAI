from __future__ import annotations

import unittest

from trainer.common.slices import infer_slice_position_sensitive, infer_slice_stateful_joker_present


class R4CommonSlicesTest(unittest.TestCase):
    def test_empty_known_jokers_list_is_not_position_sensitive(self) -> None:
        self.assertFalse(infer_slice_position_sensitive({"jokers": []}))

    def test_empty_known_jokers_list_is_false(self) -> None:
        self.assertFalse(infer_slice_stateful_joker_present({"jokers": []}))

    def test_missing_jokers_key_stays_unknown(self) -> None:
        self.assertEqual(infer_slice_stateful_joker_present({}), "unknown")


if __name__ == "__main__":
    unittest.main()
