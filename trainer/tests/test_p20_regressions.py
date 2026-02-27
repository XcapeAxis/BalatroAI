"""Targeted regressions for P20 pipeline fixes."""
from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from trainer.champion_manager import _canary_status_pass
from trainer.expert_policy import ExpertDecision
from trainer.expert_policy_shop import ShopDecision
from trainer.package.verify_model_package import verify_package
from trainer.rollout_distill_p20 import _heuristic_topk


class TestRolloutDistillHeuristicTopK(unittest.TestCase):
    def test_hand_topk_returns_int_action_id(self) -> None:
        state = {
            "hand": {"cards": [{}, {}, {}, {}, {}]},
            "legal_action_ids": [0, 1, 2],
        }
        with patch(
            "trainer.rollout_distill_p20.choose_action",
            return_value=ExpertDecision(phase="SELECTING_HAND", action_type="PLAY", mask_int=1),
        ):
            topk = _heuristic_topk(state, "HAND", k=3)
        self.assertTrue(topk)
        self.assertIsInstance(topk[0], int)

    def test_shop_topk_returns_int_action_id(self) -> None:
        state = {"state": "SHOP", "shop_legal_action_ids": [0, 1, 2]}
        with patch(
            "trainer.rollout_distill_p20.choose_shop_action",
            return_value=ShopDecision(phase="SHOP", action_id=2, action={"action_type": "BUY"}, reason="test"),
        ):
            topk = _heuristic_topk(state, "SHOP", k=3)
        self.assertEqual(topk, [2])


class TestChampionCanaryStatus(unittest.TestCase):
    def test_skip_is_treated_as_pass(self) -> None:
        self.assertTrue(_canary_status_pass("SKIP"))
        self.assertTrue(_canary_status_pass("SKIPPED"))
        self.assertTrue(_canary_status_pass("PASS"))
        self.assertFalse(_canary_status_pass("FAIL"))


class TestRunP20SmokeReliabilityGate(unittest.TestCase):
    def test_no_force_true_in_reliability_status(self) -> None:
        text = Path("scripts/run_p20_smoke.ps1").read_text(encoding="utf-8")
        self.assertNotIn("-or $true", text)


class TestPackageSchemaVerification(unittest.TestCase):
    def test_schema_requires_readme_entry(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            (root / "model").mkdir(parents=True, exist_ok=True)
            (root / "config").mkdir(parents=True, exist_ok=True)
            (root / "metrics").mkdir(parents=True, exist_ok=True)
            (root / "model" / "best.pt").write_bytes(b"stub")
            (root / "metadata.json").write_text(
                (
                    '{"schema":"deploy_package_v1","package_id":"pkg","model_id":"m1",'
                    '"source_strategy":"pv","git_commit":"abc","created_at":"2026-02-26T00:00:00Z",'
                    '"compatibility":{"sim_version":"p20","schema_version":"v1"},"seed_files":["eval_seeds_100.txt"]}'
                ),
                encoding="utf-8",
            )
            (root / "checksums.json").write_text("{}", encoding="utf-8")

            report = verify_package(root)
            self.assertFalse(report["passed"])
            self.assertTrue(any("missing required entry: README.md" in x for x in report["issues"]))


if __name__ == "__main__":
    unittest.main()
