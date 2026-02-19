from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import unittest

from sim.core.rank_chips import normalize_rank, rank_chip, sum_rank_chips


class RankChipsTest(unittest.TestCase):
    def test_rank_chip_values(self) -> None:
        self.assertEqual(rank_chip("A"), 11)
        self.assertEqual(rank_chip("K"), 10)
        self.assertEqual(rank_chip("Q"), 10)
        self.assertEqual(rank_chip("J"), 10)
        self.assertEqual(rank_chip("T"), 10)
        self.assertEqual(rank_chip("10"), 10)
        self.assertEqual(rank_chip("9"), 9)
        self.assertEqual(rank_chip("2"), 2)

    def test_normalize_rank(self) -> None:
        self.assertEqual(normalize_rank("t"), "10")
        self.assertEqual(normalize_rank("10"), "10")

    def test_sum_rank_chips(self) -> None:
        cards = [
            {"rank": "A"},
            {"rank": "K"},
            {"rank": "10"},
            {"key": "S_9"},
            {"rank": "2"},
        ]
        self.assertEqual(sum_rank_chips(cards), 42.0)


if __name__ == "__main__":
    unittest.main()
