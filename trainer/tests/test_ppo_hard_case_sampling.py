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

    def test_source_type_quota_caps_force_non_dominant_source_into_selection(self) -> None:
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
                                "episode_id": "arena-1",
                                "failure_types": ["champion_regression_segment"],
                                "failure_bucket": "risk_undercommit",
                                "slice_tags": ["slice_action_type:play"],
                                "risk_tags": ["resource_relaxed"],
                                "replay_weight": 4.0,
                                "source_type": "arena_failure_mining",
                            },
                            {
                                "seed": "BBBBBBB",
                                "episode_id": "arena-2",
                                "failure_types": ["champion_regression_segment"],
                                "failure_bucket": "risk_undercommit",
                                "slice_tags": ["slice_action_type:play"],
                                "risk_tags": ["resource_relaxed"],
                                "replay_weight": 3.5,
                                "source_type": "arena_failure_mining",
                            },
                            {
                                "seed": "CCCCCCC",
                                "episode_id": "candidate-slice-1",
                                "failure_types": ["candidate_slice_failure_seed"],
                                "failure_bucket": "shop_or_economy_misallocation",
                                "slice_tags": ["slice_action_type:shop"],
                                "risk_tags": ["resource_tight"],
                                "replay_weight": 1.1,
                                "source_type": "arena_candidate_slice_seed",
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
                        "max_failure_seeds": 3,
                        "source_type_quota_caps": {
                            "arena_failure_mining": 1,
                            "arena_candidate_slice_seed": 1,
                        },
                        "source_type_sampling_weights": {
                            "arena_candidate_slice_seed": 1.4,
                        },
                    }
                }
            )
            plan = _resolve_hard_case_plan(cfg=cfg, repo_root=root)
            self.assertEqual(plan["status"], "ok")
            self.assertEqual(plan["selected_failure_count"], 2)
            self.assertEqual(plan["source_type_selected_counts"]["arena_failure_mining"], 1)
            self.assertEqual(plan["source_type_selected_counts"]["arena_candidate_slice_seed"], 1)
            self.assertEqual(plan["source_type_quota_caps"]["arena_failure_mining"], 1)

    def test_slice_minimum_counts_force_shop_slice_into_selection(self) -> None:
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
                                "episode_id": "discard-1",
                                "failure_types": ["champion_regression_segment"],
                                "failure_bucket": "risk_undercommit",
                                "slice_tags": ["slice_action_type:discard"],
                                "risk_tags": ["resource_relaxed"],
                                "replay_weight": 4.0,
                                "source_type": "arena_failure_mining",
                            },
                            {
                                "seed": "BBBBBBB",
                                "episode_id": "discard-2",
                                "failure_types": ["champion_regression_segment"],
                                "failure_bucket": "risk_undercommit",
                                "slice_tags": ["slice_action_type:discard"],
                                "risk_tags": ["resource_relaxed"],
                                "replay_weight": 3.5,
                                "source_type": "arena_failure_mining",
                            },
                            {
                                "seed": "CCCCCCC",
                                "episode_id": "shop-1",
                                "failure_types": ["candidate_slice_failure_seed"],
                                "failure_bucket": "shop_or_economy_misallocation",
                                "slice_tags": ["slice_action_type:shop"],
                                "risk_tags": ["resource_tight"],
                                "replay_weight": 0.8,
                                "source_type": "arena_candidate_slice_seed",
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
                        "max_failure_seeds": 3,
                        "slice_minimum_counts": {
                            "slice_action_type:shop": 1,
                        },
                    }
                }
            )
            plan = _resolve_hard_case_plan(cfg=cfg, repo_root=root)
            self.assertEqual(plan["status"], "ok")
            self.assertEqual(plan["selected_failure_count"], 2)
            self.assertEqual(plan["slice_selected_counts"]["slice_action_type:shop"], 1)
            self.assertEqual(plan["slice_minimum_counts"]["slice_action_type:shop"], 1)

    def test_slice_minimum_counts_match_non_primary_tracked_slice(self) -> None:
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
                                "episode_id": "resource-1",
                                "failure_types": ["champion_regression_segment"],
                                "failure_bucket": "resource_pressure_misplay",
                                "slice_tags": ["slice_resource_pressure:high", "slice_action_type:shop"],
                                "risk_tags": ["resource_tight"],
                                "replay_weight": 3.0,
                                "source_type": "arena_slice_gap_seed",
                            },
                            {
                                "seed": "BBBBBBB",
                                "episode_id": "play-1",
                                "failure_types": ["champion_regression_segment"],
                                "failure_bucket": "risk_undercommit",
                                "slice_tags": ["slice_action_type:play"],
                                "risk_tags": ["resource_relaxed"],
                                "replay_weight": 2.5,
                                "source_type": "arena_failure_mining",
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
                        "slice_sampling_weights": {
                            "slice_resource_pressure:high": 1.2,
                            "slice_action_type:shop": 1.4,
                        },
                        "slice_minimum_counts": {
                            "slice_action_type:shop": 1,
                        },
                    }
                }
            )
            plan = _resolve_hard_case_plan(cfg=cfg, repo_root=root)
            self.assertEqual(plan["status"], "ok")
            self.assertEqual(plan["selected_failure_count"], 2)
            self.assertEqual(plan["slice_selected_counts"]["slice_action_type:shop"], 1)
            self.assertEqual(plan["slice_selected_counts"]["slice_resource_pressure:high"], 1)

    def test_source_type_minimum_counts_force_candidate_source_into_selection(self) -> None:
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
                                "episode_id": "arena-1",
                                "failure_types": ["champion_regression_segment"],
                                "failure_bucket": "risk_undercommit",
                                "slice_tags": ["slice_action_type:play"],
                                "risk_tags": ["resource_relaxed"],
                                "replay_weight": 5.0,
                                "source_type": "arena_failure_mining",
                            },
                            {
                                "seed": "BBBBBBB",
                                "episode_id": "arena-2",
                                "failure_types": ["champion_regression_segment"],
                                "failure_bucket": "risk_undercommit",
                                "slice_tags": ["slice_action_type:play"],
                                "risk_tags": ["resource_relaxed"],
                                "replay_weight": 4.5,
                                "source_type": "arena_failure_mining",
                            },
                            {
                                "seed": "CCCCCCC",
                                "episode_id": "candidate-1",
                                "failure_types": ["candidate_slice_failure_seed"],
                                "failure_bucket": "shop_or_economy_misallocation",
                                "slice_tags": ["slice_action_type:shop"],
                                "risk_tags": ["resource_tight"],
                                "replay_weight": 0.7,
                                "source_type": "arena_candidate_slice_seed",
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
                        "max_failure_seeds": 3,
                        "source_type_minimum_counts": {
                            "arena_candidate_slice_seed": 1,
                        },
                    }
                }
            )
            plan = _resolve_hard_case_plan(cfg=cfg, repo_root=root)
            self.assertEqual(plan["status"], "ok")
            self.assertEqual(plan["selected_failure_count"], 2)
            self.assertEqual(plan["source_type_selected_counts"]["arena_candidate_slice_seed"], 1)
            self.assertEqual(plan["source_type_minimum_counts"]["arena_candidate_slice_seed"], 1)

    def test_source_variant_minimum_counts_force_gap_variant_into_selection(self) -> None:
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
                                "episode_id": "discard-1",
                                "failure_types": ["champion_regression_segment"],
                                "failure_bucket": "discard_mismanagement",
                                "slice_tags": ["slice_action_type:discard"],
                                "risk_tags": ["resource_relaxed"],
                                "replay_weight": 5.0,
                                "source_type": "arena_failure_mining",
                                "source_variant": "slice_action_type:discard",
                            },
                            {
                                "seed": "BBBBBBB",
                                "episode_id": "discard-2",
                                "failure_types": ["champion_regression_segment"],
                                "failure_bucket": "discard_mismanagement",
                                "slice_tags": ["slice_action_type:discard"],
                                "risk_tags": ["resource_relaxed"],
                                "replay_weight": 4.5,
                                "source_type": "arena_failure_mining",
                                "source_variant": "slice_action_type:discard",
                            },
                            {
                                "seed": "CCCCCCC",
                                "episode_id": "pos-gap-1",
                                "failure_types": ["candidate_slice_failure_seed"],
                                "failure_bucket": "position_sensitive_misplay",
                                "slice_tags": ["slice_position_sensitive:true"],
                                "risk_tags": ["resource_relaxed"],
                                "replay_weight": 0.8,
                                "source_type": "arena_candidate_slice_seed",
                                "source_variant": "slice_position_sensitive:true",
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
                        "max_failure_seeds": 3,
                        "source_variant_minimum_counts": {
                            "slice_position_sensitive:true": 1,
                        },
                    }
                }
            )
            plan = _resolve_hard_case_plan(cfg=cfg, repo_root=root)
            self.assertEqual(plan["status"], "ok")
            self.assertEqual(plan["selected_failure_count"], 2)
            self.assertEqual(plan["source_variant_selected_counts"]["slice_position_sensitive:true"], 1)
            self.assertEqual(plan["source_variant_minimum_counts"]["slice_position_sensitive:true"], 1)


if __name__ == "__main__":
    unittest.main()
