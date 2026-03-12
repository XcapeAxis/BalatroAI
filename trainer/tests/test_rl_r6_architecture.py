from __future__ import annotations

import unittest

from trainer.rl.ppo_config import PPOConfig
from trainer.rl.ppo_lite import _select_prior_rollout_reuse_payload


class R6ArchitectureConfigTest(unittest.TestCase):
    def test_train_config_parses_actor_lag_and_rollout_reuse_fields(self) -> None:
        cfg = PPOConfig.from_mapping(
            {
                "train": {
                    "actor_refresh_interval_updates": 2,
                    "rollout_reuse_ratio": 0.25,
                    "rollout_reuse_updates": 2,
                    "rollout_reuse_max_steps": 128,
                }
            }
        )
        self.assertEqual(cfg.train.actor_refresh_interval_updates, 2)
        self.assertAlmostEqual(cfg.train.rollout_reuse_ratio, 0.25)
        self.assertEqual(cfg.train.rollout_reuse_updates, 2)
        self.assertEqual(cfg.train.rollout_reuse_max_steps, 128)


class R6RolloutReuseSelectionTest(unittest.TestCase):
    def test_disabled_when_ratio_is_zero(self) -> None:
        cfg = PPOConfig.from_mapping({"train": {"rollout_reuse_ratio": 0.0}})
        payload = _select_prior_rollout_reuse_payload(
            current_step_count=32,
            rollout_history=[{"update_index": 1, "steps": [{"episode_id": "ep-1"}]}],
            cfg=cfg,
        )
        self.assertEqual(payload["status"], "disabled")
        self.assertEqual(payload["selected_step_count"], 0)

    def test_empty_history_returns_stub(self) -> None:
        cfg = PPOConfig.from_mapping(
            {
                "train": {
                    "rollout_reuse_ratio": 0.25,
                    "rollout_reuse_updates": 2,
                    "rollout_reuse_max_steps": 32,
                }
            }
        )
        payload = _select_prior_rollout_reuse_payload(
            current_step_count=64,
            rollout_history=[],
            cfg=cfg,
        )
        self.assertEqual(payload["status"], "stub")
        self.assertEqual(payload["selected_step_count"], 0)

    def test_reuse_selection_is_capped_and_prefers_recent_updates(self) -> None:
        cfg = PPOConfig.from_mapping(
            {
                "train": {
                    "actor_refresh_interval_updates": 2,
                    "rollout_reuse_ratio": 0.25,
                    "rollout_reuse_updates": 2,
                    "rollout_reuse_max_steps": 3,
                }
            }
        )
        rollout_history = [
            {
                "update_index": 1,
                "steps": [
                    {"episode_id": "u1-a", "action": 0},
                    {"episode_id": "u1-b", "action": 1},
                ],
            },
            {
                "update_index": 2,
                "steps": [
                    {"episode_id": "u2-a", "action": 0},
                    {"episode_id": "u2-b", "action": 1},
                ],
            },
            {
                "update_index": 3,
                "steps": [
                    {"episode_id": "u3-a", "action": 0},
                    {"episode_id": "u3-b", "action": 1},
                ],
            },
        ]
        payload = _select_prior_rollout_reuse_payload(
            current_step_count=12,
            rollout_history=rollout_history,
            cfg=cfg,
        )
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["selected_step_count"], 3)
        self.assertEqual(payload["source_update_indices"], [3, 2])
        self.assertAlmostEqual(payload["replay_ratio"], 0.25)
        self.assertEqual(payload["steps"][0]["replay_source"], "prior_rollout_reuse")
        self.assertEqual(payload["steps"][0]["rollout_source_update"], 3)


if __name__ == "__main__":
    unittest.main()
