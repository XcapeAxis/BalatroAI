from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trainer.closed_loop.failure_buckets import classify_failure_bucket, scarce_failure_buckets
from trainer.rl.ppo_config import PPOConfig
from trainer.rl.ppo_lite import _resolve_hard_case_plan, _select_self_imitation_payload


class R2S2FailureBucketsTest(unittest.TestCase):
    def test_failure_bucket_taxonomy_classifies_resource_and_invalid_cases(self) -> None:
        invalid_row = {
            "invalid_action_rate": 0.2,
            "timeout_rate": 0.0,
            "rounds_survived": 3,
            "total_score": 40.0,
            "slice_labels": {"slice_action_type": "play", "slice_resource_pressure": "high"},
            "bucket_counts": {"risk": {"resource_tight": 5}},
        }
        overcommit_row = {
            "invalid_action_rate": 0.0,
            "timeout_rate": 0.0,
            "rounds_survived": 3,
            "total_score": 35.0,
            "slice_labels": {"slice_action_type": "play", "slice_resource_pressure": "high"},
            "bucket_counts": {"risk": {"resource_tight": 5}},
        }
        resource_row = {
            "invalid_action_rate": 0.0,
            "timeout_rate": 0.0,
            "rounds_survived": 3,
            "total_score": 35.0,
            "slice_labels": {"slice_action_type": "unknown", "slice_resource_pressure": "high"},
            "bucket_counts": {"risk": {"resource_tight": 5}},
        }
        invalid_bucket = classify_failure_bucket(
            row=invalid_row,
            failure_types={"invalid_action"},
            high_risk_round_threshold=2,
            low_score_threshold=50.0,
        )
        overcommit_bucket = classify_failure_bucket(
            row=overcommit_row,
            failure_types={"low_score_quantile"},
            high_risk_round_threshold=2,
            low_score_threshold=50.0,
        )
        resource_bucket = classify_failure_bucket(
            row=resource_row,
            failure_types={"low_score_quantile"},
            high_risk_round_threshold=2,
            low_score_threshold=50.0,
        )
        self.assertEqual(invalid_bucket["failure_bucket"], "invalid_or_wasted_decision")
        self.assertEqual(overcommit_bucket["failure_bucket"], "risk_overcommit")
        self.assertEqual(resource_bucket["failure_bucket"], "resource_pressure_misplay")
        scarce = scarce_failure_buckets({"resource_pressure_misplay": 3, "early_collapse": 1}, threshold=2)
        self.assertIn("early_collapse", scarce)
        self.assertIn("discard_mismanagement", scarce)


class R2S2ReplaySelectionTest(unittest.TestCase):
    def test_bucket_aware_replay_caps_and_weights_apply(self) -> None:
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
                                "episode_id": "ep-1",
                                "failure_types": ["low_score_quantile"],
                                "failure_bucket": "resource_pressure_misplay",
                                "replay_weight": 1.0,
                            },
                            {
                                "seed": "BBBBBBB",
                                "episode_id": "ep-2",
                                "failure_types": ["low_score_quantile"],
                                "failure_bucket": "resource_pressure_misplay",
                                "replay_weight": 0.9,
                            },
                            {
                                "seed": "CCCCCCC",
                                "episode_id": "ep-3",
                                "failure_types": ["champion_regression_segment"],
                                "failure_bucket": "shop_or_economy_misallocation",
                                "replay_weight": 0.5,
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
                            "resource_pressure_misplay": 0.5,
                            "shop_or_economy_misallocation": 2.0,
                        },
                        "bucket_quota_caps": {
                            "resource_pressure_misplay": 1,
                            "shop_or_economy_misallocation": 1,
                        },
                    }
                }
            )
            plan = _resolve_hard_case_plan(cfg=cfg, repo_root=root)
            self.assertEqual(plan["status"], "ok")
            self.assertEqual(plan["selected_failure_count"], 2)
            self.assertEqual(plan["bucket_selected_counts"]["resource_pressure_misplay"], 1)
            self.assertEqual(plan["bucket_selected_counts"]["shop_or_economy_misallocation"], 1)
            self.assertEqual(plan["selected_failures_preview"][0]["failure_bucket"], "shop_or_economy_misallocation")

    def test_self_imitation_is_stage_gated_and_quality_gated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worker_stats_path = root / "worker_stats.json"
            worker_stats_path.write_text(
                json.dumps(
                    [
                        {
                            "episodes": [
                                {
                                    "episode_id": "ep-top",
                                    "seed": "AAAAAAA",
                                    "reward": 12.0,
                                    "final_score": 120.0,
                                    "invalid_action_rate": 0.0,
                                },
                                {
                                    "episode_id": "ep-low",
                                    "seed": "BBBBBBB",
                                    "reward": 2.0,
                                    "final_score": 40.0,
                                    "invalid_action_rate": 0.0,
                                },
                            ]
                        }
                    ]
                ),
                encoding="utf-8",
            )
            steps = [
                {
                    "episode_id": "ep-top",
                    "obs_vector": [0.0, 1.0],
                    "action": 0,
                    "reward": 1.0,
                    "terminated": False,
                    "truncated": False,
                    "action_logprob": 0.0,
                    "value_pred": 0.0,
                    "legal_action_ids": [0, 1],
                },
                {
                    "episode_id": "ep-low",
                    "obs_vector": [0.0, 1.0],
                    "action": 0,
                    "reward": 0.0,
                    "terminated": False,
                    "truncated": False,
                    "action_logprob": 0.0,
                    "value_pred": 0.0,
                    "legal_action_ids": [0, 1],
                },
            ]
            cfg = PPOConfig.from_mapping(
                {
                    "self_imitation": {
                        "enabled": True,
                        "top_k_episodes": 2,
                        "replay_ratio": 0.5,
                        "max_replay_steps": 4,
                        "stage_min": "balance_or_exploit",
                        "quality_threshold": 100.0,
                    }
                }
            )
            blocked = _select_self_imitation_payload(
                steps=steps,
                worker_stats_path=worker_stats_path,
                cfg=cfg,
                stage_payload={"stage": "pressure", "phase_index": 2},
            )
            self.assertEqual(blocked["status"], "disabled")
            allowed = _select_self_imitation_payload(
                steps=steps,
                worker_stats_path=worker_stats_path,
                cfg=cfg,
                stage_payload={"stage": "balance_or_exploit", "phase_index": 3},
            )
            self.assertEqual(allowed["status"], "ok")
            self.assertEqual(allowed["selected_episode_count"], 1)
            self.assertEqual(allowed["selected_episodes"][0]["episode_id"], "ep-top")


if __name__ == "__main__":
    unittest.main()
