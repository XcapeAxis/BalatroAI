from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trainer.closed_loop.failure_mining import (
    _bucket_from_slice_tag,
    _build_slice_priority_weights,
    _compound_actionable_slice_tags,
    _infer_source_variant,
    _refine_failure_bucket_for_slice_pressure,
    _resolve_failure_sources,
    _row_selection_matches_policy,
    _source_variant_from_slice_tag,
)


class R2S4SlicePriorityWeightsTest(unittest.TestCase):
    def test_degraded_slice_with_zero_candidate_gets_priority_boost(self) -> None:
        payload = {
            "rows": [
                {
                    "slice_key": "slice_action_type",
                    "slice_label": "play",
                    "candidate_count": 0,
                    "champion_count": 6,
                    "metrics": {"mean_total_score_delta": -341.0},
                    "signals": {"degraded_significant": False},
                }
            ]
        }
        weights = _build_slice_priority_weights(payload)
        self.assertIn("slice_action_type:play", weights)
        self.assertGreater(weights["slice_action_type:play"], 2.5)


class R3SliceBucketMappingTest(unittest.TestCase):
    def test_bucket_from_slice_tag_handles_actionable_and_unknown_tags(self) -> None:
        self.assertEqual(_bucket_from_slice_tag("slice_action_type:discard"), "discard_mismanagement")
        self.assertEqual(_bucket_from_slice_tag("slice_action_type:shop"), "shop_or_economy_misallocation")
        self.assertEqual(_bucket_from_slice_tag("slice_resource_pressure:medium"), "resource_pressure_misplay")
        self.assertEqual(_bucket_from_slice_tag("slice_stage:early"), "early_collapse")
        self.assertEqual(_bucket_from_slice_tag("slice_position_sensitive:unknown"), "")
        self.assertEqual(_source_variant_from_slice_tag("slice_action_type:shop"), "slice_action_type:shop")
        self.assertEqual(_source_variant_from_slice_tag("slice_stateful_joker_present:true"), "slice_stateful_joker_present:true")
        self.assertEqual(_source_variant_from_slice_tag("slice_position_sensitive:false"), "")

    def test_compound_actionable_slice_tags_drop_unknown_and_primary(self) -> None:
        tags = [
            "slice_resource_pressure:high",
            "slice_action_type:shop",
            "slice_position_sensitive:unknown",
            "slice_action_type:shop",
        ]
        self.assertEqual(
            _compound_actionable_slice_tags(tags, primary_tag="slice_resource_pressure:high"),
            ["slice_action_type:shop"],
        )

    def test_source_variant_prefers_action_type_over_generic_pressure(self) -> None:
        self.assertEqual(
            _infer_source_variant(
                slice_tags=[
                    "slice_stage:early",
                    "slice_resource_pressure:medium",
                    "slice_action_type:discard",
                ],
                failure_bucket="risk_undercommit",
            ),
            "slice_action_type:discard",
        )


class R2S4SelectionPolicyTest(unittest.TestCase):
    def test_bucket_and_source_caps_are_enforced(self) -> None:
        payload = {
            "seed": "AAAAAAA",
            "failure_bucket": "discard_mismanagement",
            "source_run_id": "run-a",
            "failure_types": ["low_score_quantile"],
        }
        self.assertFalse(
            _row_selection_matches_policy(
                payload=payload,
                selection_by_type={"low_score_quantile": 1},
                selection_by_seed={"AAAAAAA": 0},
                selection_by_bucket={"discard_mismanagement": 0},
                selection_by_source={"run-a": 0},
                selection_by_source_type={"arena_failure_mining": 0},
                max_failures_per_type=1,
                max_failures_per_seed=0,
                max_failures_per_bucket={},
                max_failures_per_source={},
            )
        )
        self.assertFalse(
            _row_selection_matches_policy(
                payload=payload,
                selection_by_type={},
                selection_by_seed={},
                selection_by_bucket={"discard_mismanagement": 2},
                selection_by_source={"run-a": 0},
                selection_by_source_type={"arena_failure_mining": 0},
                max_failures_per_type=0,
                max_failures_per_seed=0,
                max_failures_per_bucket={"discard_mismanagement": 2},
                max_failures_per_source={},
            )
        )
        self.assertFalse(
            _row_selection_matches_policy(
                payload=payload,
                selection_by_type={},
                selection_by_seed={},
                selection_by_bucket={},
                selection_by_source={"run-a": 3},
                selection_by_source_type={"arena_failure_mining": 0},
                max_failures_per_type=0,
                max_failures_per_seed=0,
                max_failures_per_bucket={},
                max_failures_per_source={"run-a": 3},
            )
        )

    def test_source_type_minimums_can_be_checked_separately(self) -> None:
        payload = {
            "seed": "AAAAAAA",
            "failure_bucket": "shop_or_economy_misallocation",
            "source_run_id": "run-a",
            "source_type": "arena_candidate_slice_seed",
            "failure_types": ["candidate_slice_failure_seed"],
        }
        self.assertTrue(
            _row_selection_matches_policy(
                payload=payload,
                selection_by_type={},
                selection_by_seed={},
                selection_by_bucket={},
                selection_by_source={},
                selection_by_source_type={"arena_candidate_slice_seed": 0},
                max_failures_per_type=0,
                max_failures_per_seed=0,
                max_failures_per_bucket={},
                max_failures_per_source={},
            )
        )


class R2S4FailureSourceResolutionTest(unittest.TestCase):
    def test_multiple_sources_and_promotion_resolution_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arena_a = root / "arena_a"
            arena_b = root / "arena_b"
            arena_a.mkdir()
            arena_b.mkdir()
            promotion = root / "promotion.json"
            slice_breakdown = root / "slice_decision_breakdown.json"
            promotion.write_text(
                (
                    "{"
                    "\"slice_decision_breakdown_json\": "
                    f"\"{str(slice_breakdown).replace('\\', '\\\\')}\""
                    "}"
                ),
                encoding="utf-8",
            )
            slice_breakdown.write_text("{\"rows\":[]}", encoding="utf-8")
            sources = _resolve_failure_sources(
                repo_root=root,
                p39_root=root,
                input_cfg={
                    "arena_run_dirs": [str(arena_a), str(arena_b)],
                    "promotion_decision_json": str(promotion),
                },
                arena_run_dir_override=None,
            )
            self.assertEqual(len(sources), 2)
            self.assertEqual(Path(sources[0]["arena_run_dir"]), arena_a)
            self.assertEqual(Path(sources[1]["arena_run_dir"]), arena_b)
            self.assertEqual(Path(sources[0]["slice_breakdown_path"]), slice_breakdown)


class R2S4SliceAwareBucketRefineTest(unittest.TestCase):
    def test_triage_play_slack_overrides_discard_tie(self) -> None:
        row = {
            "bucket_counts": {
                "slice_action_type": {"play": 4, "discard": 4, "unknown": 1},
                "slice_resource_pressure": {"low": 3, "medium": 6},
            }
        }
        bucket, reason, candidates, signals = _refine_failure_bucket_for_slice_pressure(
            row=row,
            failure_types={"triage_degraded_slice", "low_score_quantile"},
            failure_bucket="discard_mismanagement",
            bucket_reason="discard_dominant_failure",
            failure_bucket_candidates=["discard_mismanagement", "low_score_survival"],
            failure_bucket_signals=["discard_dominant_failure", "fallback_low_score_survival"],
        )
        self.assertEqual(bucket, "risk_undercommit")
        self.assertEqual(reason, "triage_slice_play_under_resource_slack")
        self.assertEqual(candidates[0], "risk_undercommit")
        self.assertEqual(signals[0], "triage_slice_play_under_resource_slack")


if __name__ == "__main__":
    unittest.main()
