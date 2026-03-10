from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trainer.closed_loop.failure_buckets import classify_failure_bucket
from trainer.rl.env_adapter import RLEnvAdapter, RLEnvAdapterConfig
from trainer.rl.reward_config import RewardConfig
from trainer.rl.ppo_config import PPOConfig
from trainer.rl.ppo_lite import _resolve_hard_case_plan, _select_self_imitation_payload


class R2S3FailureBucketPriorityTest(unittest.TestCase):
    def test_specific_failure_signals_override_generic_early_collapse(self) -> None:
        row = {
            "invalid_action_rate": 0.0,
            "timeout_rate": 0.0,
            "rounds_survived": 1,
            "total_score": 45.0,
            "slice_labels": {
                "slice_stage": "early",
                "slice_resource_pressure": "medium",
                "slice_action_type": "discard",
                "slice_position_sensitive": "unknown",
                "slice_stateful_joker_present": "unknown",
            },
            "bucket_counts": {
                "risk": {"resource_balanced": 6, "resource_relaxed": 3},
                "slice_action_type": {"discard": 4, "play": 4, "unknown": 1},
            },
        }
        payload = classify_failure_bucket(
            row=row,
            failure_types={"low_score_quantile", "champion_regression_segment"},
            high_risk_round_threshold=2,
            low_score_threshold=50.0,
        )
        self.assertEqual(payload["failure_bucket"], "discard_mismanagement")
        self.assertIn("early_collapse", payload["failure_bucket_candidates"])


class R2S3ReplaySelectionTest(unittest.TestCase):
    def test_slice_and_risk_weighting_affect_replay_priority(self) -> None:
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
                                "failure_types": ["low_score_quantile"],
                                "failure_bucket": "discard_mismanagement",
                                "slice_tags": ["slice_action_type:discard", "slice_stage:early"],
                                "risk_tags": ["resource_balanced"],
                                "replay_weight": 1.0,
                            },
                            {
                                "seed": "BBBBBBB",
                                "episode_id": "ep-b",
                                "failure_types": ["low_score_quantile", "high_risk_bucket_failure"],
                                "failure_bucket": "resource_pressure_misplay",
                                "slice_tags": ["slice_resource_pressure:high", "slice_stage:early"],
                                "risk_tags": ["resource_tight", "early_collapse"],
                                "replay_weight": 0.8,
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
                        "max_failure_cases": 2,
                        "max_failure_seeds": 2,
                        "bucket_sampling_weights": {
                            "discard_mismanagement": 1.0,
                            "resource_pressure_misplay": 1.0,
                        },
                        "slice_sampling_weights": {
                            "slice_resource_pressure:high": 1.8,
                            "slice_action_type:discard": 0.9,
                        },
                        "risk_tag_sampling_weights": {
                            "resource_tight": 1.6,
                        },
                    }
                }
            )
            plan = _resolve_hard_case_plan(cfg=cfg, repo_root=root)
            self.assertEqual(plan["status"], "ok")
            self.assertEqual(plan["selected_failures_preview"][0]["episode_id"], "ep-b")
            self.assertEqual(plan["slice_selected_counts"]["slice_resource_pressure:high"], 1)
            self.assertEqual(plan["risk_selected_counts"]["resource_tight"], 1)


class R2S3SelfImitationFilterTest(unittest.TestCase):
    def test_self_imitation_honors_phase_action_and_slice_allowlists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worker_stats_path = root / "worker_stats.json"
            worker_stats_path.write_text(
                json.dumps(
                    [
                        {
                            "episodes": [
                                {
                                    "episode_id": "ep-good",
                                    "seed": "AAAAAAA",
                                    "reward": 12.0,
                                    "final_score": 120.0,
                                    "invalid_action_rate": 0.0,
                                    "dominant_phase": "SELECTING_HAND",
                                    "dominant_action_type": "DISCARD",
                                    "slice_tags": ["phase:SELECTING_HAND", "action_type:DISCARD"],
                                },
                                {
                                    "episode_id": "ep-off",
                                    "seed": "BBBBBBB",
                                    "reward": 15.0,
                                    "final_score": 130.0,
                                    "invalid_action_rate": 0.0,
                                    "dominant_phase": "SHOP",
                                    "dominant_action_type": "NEXT_ROUND",
                                    "slice_tags": ["phase:SHOP", "action_type:NEXT_ROUND"],
                                },
                            ]
                        }
                    ]
                ),
                encoding="utf-8",
            )
            steps = [
                {
                    "episode_id": "ep-good",
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
                    "episode_id": "ep-off",
                    "obs_vector": [0.0, 1.0],
                    "action": 0,
                    "reward": 1.0,
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
                        "slice_allowlist": ["action_type:DISCARD"],
                        "phase_allowlist": ["SELECTING_HAND"],
                        "action_type_allowlist": ["DISCARD"],
                    }
                }
            )
            payload = _select_self_imitation_payload(
                steps=steps,
                worker_stats_path=worker_stats_path,
                cfg=cfg,
                stage_payload={"stage": "balance_or_exploit", "phase_index": 3},
            )
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["selected_episode_count"], 1)
            self.assertEqual(payload["selected_episodes"][0]["episode_id"], "ep-good")


class R2S3EnvAdapterActionTypeTest(unittest.TestCase):
    def test_step_uses_env_info_action_payload_for_action_type(self) -> None:
        class DummyEnv:
            def step(self, _action: int) -> tuple[dict[str, object], float, bool, dict[str, object]]:
                return (
                    {
                        "action_mask": [1, 1, 0, 0],
                        "action_dim": 4,
                        "phase": "PLAY",
                        "round_num": 2,
                        "ante_num": 1,
                        "score": 30.0,
                    },
                    5.0,
                    False,
                    {
                        "phase_after": "PLAY",
                        "score_after": 30.0,
                        "score_delta": 5.0,
                        "invalid_action": False,
                        "action_payload": {"action_type": "DISCARD"},
                        "episode_length": 1,
                        "episode_return": 5.0,
                    },
                )

        adapter = RLEnvAdapter.__new__(RLEnvAdapter)
        adapter.config = RLEnvAdapterConfig()
        adapter.reward_config = RewardConfig.from_mapping({})
        adapter._env = DummyEnv()
        adapter._last_obs = {
            "action_mask": [1, 1, 0, 0],
            "action_dim": 4,
            "phase": "PLAY",
            "round_num": 1,
            "ante_num": 1,
            "score": 25.0,
        }
        adapter._episode_index = 1
        adapter._episode_shaped_return = 0.0
        adapter._step_index = 0

        next_obs, reward, terminated, truncated, info = adapter.step(1)
        self.assertEqual(next_obs["score"], 30.0)
        self.assertEqual(reward, 5.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["action_type"], "DISCARD")


if __name__ == "__main__":
    unittest.main()
