from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trainer.rl.ppo_config import PPOConfig
from trainer.rl.ppo_lite import _build_rollout_seed_schedule, _resolve_hard_case_plan


class PPOHardCaseSamplingTest(unittest.TestCase):
    def test_resolve_hard_case_plan_reads_failure_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "failure_pack_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema": "p40_failure_pack_manifest_v1",
                        "status": "ok",
                        "failures": [
                            {"seed": "AAAAAAA", "failure_types": ["low_score_quantile"]},
                            {"seed": "BBBBBBB", "failure_types": ["champion_regression_segment"]},
                            {"seed": "BBBBBBB", "failure_types": ["low_score_quantile"]},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            cfg = PPOConfig.from_mapping(
                {
                    "hard_case_sampling": {
                        "enabled": True,
                        "failure_pack_manifest": str(manifest_path),
                        "seed_replay_factor": 3,
                        "max_failure_seeds": 2,
                    }
                }
            )
            plan = _resolve_hard_case_plan(cfg=cfg, repo_root=root)
            self.assertEqual(plan["status"], "ok")
            self.assertEqual(plan["selected_failure_count"], 3)
            self.assertEqual(plan["selected_failure_seeds"], ["BBBBBBB", "AAAAAAA"])

    def test_build_rollout_seed_schedule_replays_failure_seeds(self) -> None:
        schedule = _build_rollout_seed_schedule(
            base_seeds=["BASE001"],
            hard_case_plan={
                "status": "ok",
                "include_base_seed": True,
                "seed_replay_factor": 2,
                "selected_failure_seeds": ["AAAAAAA", "BBBBBBB"],
            },
        )
        self.assertEqual(
            schedule,
            ["BASE001", "AAAAAAA", "AAAAAAA", "BBBBBBB", "BBBBBBB"],
        )


if __name__ == "__main__":
    unittest.main()
