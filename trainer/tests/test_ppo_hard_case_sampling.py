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

    def test_bucket_minimum_counts_force_diverse_failure_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "failure_pack_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema": "p40_failure_pack_manifest_v2",
                        "status": "ok",
                        "failures": [
                            {
                                "seed": "AAAAAAA",
                                "episode_id": "risk-1",
                                "failure_types": ["champion_regression_segment"],
                                "failure_bucket": "risk_undercommit",
                                "slice_tags": ["slice_action_type:play"],
                                "risk_tags": ["resource_relaxed"],
                                "replay_weight": 4.0,
                            },
                            {
                                "seed": "AAAAAAA",
                                "episode_id": "risk-2",
                                "failure_types": ["champion_regression_segment"],
                                "failure_bucket": "risk_undercommit",
                                "slice_tags": ["slice_action_type:play"],
                                "risk_tags": ["resource_relaxed"],
                                "replay_weight": 3.5,
                            },
                            {
                                "seed": "BBBBBBB",
                                "episode_id": "resource-1",
                                "failure_types": ["triage_degraded_slice"],
                                "failure_bucket": "resource_pressure_misplay",
                                "slice_tags": ["slice_resource_pressure:high"],
                                "risk_tags": ["resource_tight"],
                                "replay_weight": 1.0,
                            },
                            {
                                "seed": "CCCCCCC",
                                "episode_id": "survival-1",
                                "failure_types": ["low_score_quantile"],
                                "failure_bucket": "low_score_survival",
                                "slice_tags": ["slice_stage:early"],
                                "risk_tags": ["low_score_tail"],
                                "replay_weight": 0.9,
                            },
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
                        "max_failure_cases": 3,
                        "max_failure_seeds": 3,
                        "bucket_sampling_weights": {
                            "risk_undercommit": 1.4,
                            "resource_pressure_misplay": 0.9,
                            "low_score_survival": 0.8,
                        },
                        "bucket_minimum_counts": {
                            "resource_pressure_misplay": 1,
                            "low_score_survival": 1,
                        },
                    }
                }
            )
            plan = _resolve_hard_case_plan(cfg=cfg, repo_root=root)
            self.assertEqual(plan["status"], "ok")
            self.assertEqual(plan["selected_failure_count"], 3)
            self.assertEqual(plan["failure_bucket_counts"]["resource_pressure_misplay"], 1)
            self.assertEqual(plan["failure_bucket_counts"]["low_score_survival"], 1)
            self.assertEqual(plan["bucket_minimum_counts"]["resource_pressure_misplay"], 1)
            self.assertEqual(plan["bucket_minimum_counts"]["low_score_survival"], 1)


if __name__ == "__main__":
    unittest.main()
