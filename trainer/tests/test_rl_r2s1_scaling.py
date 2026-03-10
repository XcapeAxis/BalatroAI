from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trainer.rl.ppo_config import PPOConfig
from trainer.rl.ppo_lite import _build_rollout_seed_schedule, _resolve_hard_case_plan, _select_self_imitation_payload


class R2S1ScalingHelpersTest(unittest.TestCase):
    def test_hard_case_plan_balances_failure_types(self) -> None:
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
                                "episode_id": "ep-a",
                                "failure_types": ["champion_regression_segment"],
                                "failure_bucket": "policy_search_disagreement_failure",
                                "replay_weight": 3.0,
                            },
                            {
                                "seed": "BBBBBBB",
                                "episode_id": "ep-b",
                                "failure_types": ["low_score_quantile"],
                                "failure_bucket": "low_score_survival",
                                "replay_weight": 2.0,
                            },
                            {
                                "seed": "CCCCCCC",
                                "episode_id": "ep-c",
                                "failure_types": ["high_risk_bucket_failure"],
                                "failure_bucket": "high_risk_collapse",
                                "replay_weight": 1.5,
                            },
                            {
                                "seed": "DDDDDDD",
                                "episode_id": "ep-d",
                                "failure_types": ["champion_regression_segment"],
                                "failure_bucket": "policy_search_disagreement_failure",
                                "replay_weight": 0.25,
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
                        "seed_replay_factor": 3,
                        "max_failure_cases": 4,
                        "max_failure_seeds": 3,
                        "balance_across_failure_types": True,
                        "max_failures_per_type": 1,
                        "max_failures_per_seed": 1,
                    }
                }
            )
            plan = _resolve_hard_case_plan(cfg=cfg, repo_root=root)
            self.assertEqual(plan["status"], "ok")
            self.assertEqual(plan["selected_failure_count"], 3)
            self.assertEqual(plan["failure_type_coverage"], 3)
            self.assertEqual(set(plan["selected_failure_seeds"]), {"AAAAAAA", "BBBBBBB", "CCCCCCC"})
            schedule = _build_rollout_seed_schedule(base_seeds=["BASE001"], hard_case_plan=plan)
            self.assertIn("BASE001", schedule)
            self.assertGreaterEqual(schedule.count("AAAAAAA"), 1)
            self.assertGreaterEqual(schedule.count("BBBBBBB"), 1)
            self.assertGreaterEqual(schedule.count("CCCCCCC"), 1)

    def test_self_imitation_selects_top_episodes_under_ratio_cap(self) -> None:
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
                                    "episode_id": "ep-mid",
                                    "seed": "BBBBBBB",
                                    "reward": 8.0,
                                    "final_score": 90.0,
                                    "invalid_action_rate": 0.0,
                                },
                                {
                                    "episode_id": "ep-bad",
                                    "seed": "CCCCCCC",
                                    "reward": 2.0,
                                    "final_score": 10.0,
                                    "invalid_action_rate": 0.2,
                                },
                            ]
                        }
                    ]
                ),
                encoding="utf-8",
            )
            steps: list[dict[str, object]] = []
            for episode_id, reward in (("ep-top", 1.0), ("ep-mid", 0.5), ("ep-bad", -0.1)):
                for step_id in range(4):
                    steps.append(
                        {
                            "episode_id": episode_id,
                            "obs_vector": [0.0, 1.0],
                            "action": 0,
                            "reward": reward,
                            "terminated": step_id == 3,
                            "truncated": False,
                            "action_logprob": 0.0,
                            "value_pred": 0.0,
                            "legal_action_ids": [0, 1],
                        }
                    )
            cfg = PPOConfig.from_mapping(
                {
                    "self_imitation": {
                        "enabled": True,
                        "top_k_episodes": 2,
                        "top_episode_fraction": 0.25,
                        "replay_ratio": 0.25,
                        "max_replay_steps": 4,
                        "max_invalid_action_rate": 0.05,
                    }
                }
            )
            payload = _select_self_imitation_payload(steps=steps, worker_stats_path=worker_stats_path, cfg=cfg)
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["selected_episode_count"], 2)
            self.assertEqual(payload["selected_step_count"], 3)
            self.assertAlmostEqual(payload["replay_ratio"], 0.25)
            selected_episode_ids = {row["episode_id"] for row in payload["selected_episodes"]}
            self.assertEqual(selected_episode_ids, {"ep-top", "ep-mid"})


if __name__ == "__main__":
    unittest.main()
