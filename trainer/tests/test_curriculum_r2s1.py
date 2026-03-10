from __future__ import annotations

import unittest

from trainer.rl.curriculum_rl import load_curriculum_scheduler


class CurriculumR2S1Test(unittest.TestCase):
    def test_curriculum_stage_can_override_replay_and_self_imitation(self) -> None:
        scheduler = load_curriculum_scheduler(
            config={
                "schema": "p44_curriculum_config_v1",
                "schedule": [{"stage": "stage1", "start_iteration": 1, "until_iteration": 0}],
                "stages": {
                    "stage1": {
                        "reward": {"survival_bonus": 0.08},
                        "hard_case_sampling": {"seed_replay_factor": 5, "max_failures_per_type": 2},
                        "self_imitation": {"enabled": True, "replay_ratio": 0.12},
                    }
                },
            }
        )
        merged, payload = scheduler.apply_to_config(
            {
                "env": {"reward": {"survival_bonus": 0.01}},
                "hard_case_sampling": {"enabled": True, "seed_replay_factor": 2},
                "self_imitation": {"enabled": False, "replay_ratio": 0.0},
            },
            training_iteration=1,
        )
        self.assertEqual(payload["stage"], "stage1")
        self.assertEqual(merged["env"]["reward"]["survival_bonus"], 0.08)
        self.assertEqual(merged["hard_case_sampling"]["seed_replay_factor"], 5)
        self.assertEqual(merged["hard_case_sampling"]["max_failures_per_type"], 2)
        self.assertTrue(merged["self_imitation"]["enabled"])
        self.assertEqual(merged["self_imitation"]["replay_ratio"], 0.12)


if __name__ == "__main__":
    unittest.main()
