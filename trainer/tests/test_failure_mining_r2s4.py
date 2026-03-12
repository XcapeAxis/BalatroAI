from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trainer.closed_loop.failure_mining import (
    _build_overlap_report,
    _bucket_from_slice_tag,
    _build_slice_priority_weights,
    _can_swap_overlap_selected_row,
    _compound_actionable_slice_tags,
    _infer_source_variant,
    _preferred_source_variant,
    _refine_failure_bucket_for_slice_pressure,
    _resolve_failure_sources,
    _row_selection_matches_policy,
    _source_variant_from_slice_tag,
    run_failure_mining,
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
        self.assertEqual(_source_variant_from_slice_tag("slice_stateful_joker_present:yes"), "slice_stateful_joker_present:true")
        self.assertEqual(_source_variant_from_slice_tag("slice_position_sensitive:yes"), "slice_position_sensitive:true")
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

    def test_source_variant_prefers_bucket_family_for_position_sensitive_bucket(self) -> None:
        self.assertEqual(
            _infer_source_variant(
                slice_tags=[
                    "slice_action_type:discard",
                    "slice_position_sensitive:false",
                ],
                failure_bucket="position_sensitive_misplay",
            ),
            "bucket:position_sensitive_misplay",
        )

    def test_source_variant_prefers_bucket_family_for_resource_pressure_bucket(self) -> None:
        self.assertEqual(
            _infer_source_variant(
                slice_tags=[
                    "slice_resource_pressure:high",
                    "slice_action_type:play",
                ],
                failure_bucket="resource_pressure_misplay",
            ),
            "bucket:resource_pressure_misplay",
        )

    def test_source_variant_prefers_bucket_family_for_shop_bucket(self) -> None:
        self.assertEqual(
            _infer_source_variant(
                slice_tags=[
                    "slice_action_type:shop",
                    "slice_stage:early",
                ],
                failure_bucket="shop_or_economy_misallocation",
            ),
            "bucket:shop_or_economy_misallocation",
        )

    def test_preferred_source_variant_uses_bucket_family_before_primary_tag(self) -> None:
        self.assertEqual(
            _preferred_source_variant(
                primary_slice_tag="slice_action_type:shop",
                slice_tags=[
                    "slice_action_type:shop",
                    "slice_resource_pressure:high",
                ],
                failure_bucket="shop_or_economy_misallocation",
            ),
            "bucket:shop_or_economy_misallocation",
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
                selection_by_source_variant={},
                selection_by_overlap_key={},
                max_failures_per_type=1,
                max_failures_per_seed=0,
                max_failures_per_bucket={},
                max_failures_per_source={},
                max_failures_per_source_variant={},
                max_failures_per_overlap_key=0,
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
                selection_by_source_variant={},
                selection_by_overlap_key={},
                max_failures_per_type=0,
                max_failures_per_seed=0,
                max_failures_per_bucket={"discard_mismanagement": 2},
                max_failures_per_source={},
                max_failures_per_source_variant={},
                max_failures_per_overlap_key=0,
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
                selection_by_source_variant={"slice_action_type:discard": 0},
                selection_by_overlap_key={},
                max_failures_per_type=0,
                max_failures_per_seed=0,
                max_failures_per_bucket={},
                max_failures_per_source={"run-a": 3},
                max_failures_per_source_variant={},
                max_failures_per_overlap_key=0,
            )
        )
        self.assertFalse(
            _row_selection_matches_policy(
                payload={**payload, "source_variant": "slice_action_type:discard"},
                selection_by_type={},
                selection_by_seed={},
                selection_by_bucket={},
                selection_by_source={"run-a": 0},
                selection_by_source_type={"arena_failure_mining": 0},
                selection_by_source_variant={"slice_action_type:discard": 2},
                selection_by_overlap_key={},
                max_failures_per_type=0,
                max_failures_per_seed=0,
                max_failures_per_bucket={},
                max_failures_per_source={},
                max_failures_per_source_variant={"slice_action_type:discard": 2},
                max_failures_per_overlap_key=0,
            )
        )

    def test_overlap_key_cap_is_enforced(self) -> None:
        payload = {
            "seed": "AAAAAAA",
            "policy_id": "model_policy",
            "source_run_id": "run-a",
            "episode_index": 1,
            "failure_bucket": "resource_pressure_misplay",
            "source_variant": "slice_resource_pressure:high",
            "failure_types": ["triage_degraded_slice"],
        }
        self.assertFalse(
            _row_selection_matches_policy(
                payload=payload,
                selection_by_type={},
                selection_by_seed={},
                selection_by_bucket={},
                selection_by_source={},
                selection_by_source_type={},
                selection_by_source_variant={},
                selection_by_overlap_key={"run-a|model_policy|AAAAAAA|1": 1},
                max_failures_per_type=0,
                max_failures_per_seed=0,
                max_failures_per_bucket={},
                max_failures_per_source={},
                max_failures_per_source_variant={},
                max_failures_per_overlap_key=1,
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
                selection_by_source_variant={},
                selection_by_overlap_key={},
                max_failures_per_type=0,
                max_failures_per_seed=0,
                max_failures_per_bucket={},
                max_failures_per_source={},
                max_failures_per_source_variant={},
                max_failures_per_overlap_key=0,
            )
        )


class R5FailureOverlapReportTest(unittest.TestCase):
    def test_overlap_report_groups_rows_by_underlying_episode(self) -> None:
        report = _build_overlap_report(
            run_id="r5-overlap-smoke",
            candidate_rows=[
                {
                    "episode_id": "run|gap|model|AAAAAAA|1",
                    "source_run_id": "run",
                    "policy_id": "model",
                    "seed": "AAAAAAA",
                    "episode_index": 1,
                    "source_type": "arena_slice_gap_seed",
                    "source_variant": "slice_resource_pressure:high",
                    "failure_bucket": "resource_pressure_misplay",
                },
                {
                    "episode_id": "run|compound|model|AAAAAAA|1|slice_action_type_shop",
                    "source_run_id": "run",
                    "policy_id": "model",
                    "seed": "AAAAAAA",
                    "episode_index": 1,
                    "source_type": "arena_compound_slice_seed",
                    "source_variant": "slice_action_type:shop",
                    "failure_bucket": "shop_or_economy_misallocation",
                },
            ],
            selected_rows=[
                {
                    "episode_id": "run|gap|model|AAAAAAA|1",
                    "source_run_id": "run",
                    "policy_id": "model",
                    "seed": "AAAAAAA",
                    "episode_index": 1,
                    "source_type": "arena_slice_gap_seed",
                    "source_variant": "slice_resource_pressure:high",
                    "failure_bucket": "resource_pressure_misplay",
                },
                {
                    "episode_id": "run|compound|model|AAAAAAA|1|slice_action_type_shop",
                    "source_run_id": "run",
                    "policy_id": "model",
                    "seed": "AAAAAAA",
                    "episode_index": 1,
                    "source_type": "arena_compound_slice_seed",
                    "source_variant": "slice_action_type:shop",
                    "failure_bucket": "shop_or_economy_misallocation",
                },
            ],
        )
        selected = report["selected_pool"]
        self.assertEqual(selected["overlap_group_count"], 1)
        self.assertEqual(selected["source_variant_overlap_pairs"]["slice_action_type:shop__slice_resource_pressure:high"], 1)
        self.assertEqual(selected["failure_bucket_overlap_pairs"]["resource_pressure_misplay__shop_or_economy_misallocation"], 1)


class R5OverlapSwapRescueTest(unittest.TestCase):
    def test_overlap_swap_can_rescue_underrepresented_source_variant(self) -> None:
        existing = {
            "episode_id": "run|gap|heuristic|AAAAAAA|1",
            "source_run_id": "run",
            "policy_id": "heuristic",
            "seed": "AAAAAAA",
            "episode_index": 1,
            "source_type": "arena_slice_gap_seed",
            "source_variant": "bucket:resource_pressure_misplay",
            "failure_bucket": "resource_pressure_misplay",
            "failure_types": ["slice_coverage_gap_seed", "triage_degraded_slice"],
        }
        candidate = {
            "episode_id": "run|compound_gap|heuristic|AAAAAAA|1|slice_action_type_shop",
            "source_run_id": "run",
            "policy_id": "heuristic",
            "seed": "AAAAAAA",
            "episode_index": 1,
            "source_type": "arena_compound_slice_seed",
            "source_variant": "bucket:shop_or_economy_misallocation",
            "failure_bucket": "shop_or_economy_misallocation",
            "failure_types": ["compound_slice_failure_seed", "triage_degraded_slice"],
        }
        self.assertTrue(
            _can_swap_overlap_selected_row(
                candidate_payload=candidate,
                existing_payload=existing,
                selection_by_type={"slice_coverage_gap_seed": 1, "triage_degraded_slice": 2, "compound_slice_failure_seed": 0},
                selection_by_seed={"AAAAAAA": 1},
                selection_by_bucket={"resource_pressure_misplay": 2},
                selection_by_source={"run": 1},
                selection_by_source_type={"arena_slice_gap_seed": 1},
                selection_by_source_variant={"bucket:resource_pressure_misplay": 2},
                selection_by_overlap_key={"run|heuristic|AAAAAAA|1": 1},
                max_failures_per_type=0,
                max_failures_per_seed=4,
                max_failures_per_bucket={},
                max_failures_per_source={},
                max_failures_per_source_variant={},
                max_failures_per_overlap_key=1,
                min_failures_per_bucket={"resource_pressure_misplay": 1},
                min_failures_per_source_type={},
                min_failures_per_source_variant={
                    "bucket:resource_pressure_misplay": 1,
                    "bucket:shop_or_economy_misallocation": 1,
                },
            )
        )

    def test_overlap_swap_respects_existing_minimums(self) -> None:
        existing = {
            "episode_id": "run|gap|heuristic|AAAAAAA|1",
            "source_run_id": "run",
            "policy_id": "heuristic",
            "seed": "AAAAAAA",
            "episode_index": 1,
            "source_type": "arena_slice_gap_seed",
            "source_variant": "bucket:resource_pressure_misplay",
            "failure_bucket": "resource_pressure_misplay",
            "failure_types": ["slice_coverage_gap_seed"],
        }
        candidate = {
            "episode_id": "run|compound_gap|heuristic|AAAAAAA|1|slice_action_type_shop",
            "source_run_id": "run",
            "policy_id": "heuristic",
            "seed": "AAAAAAA",
            "episode_index": 1,
            "source_type": "arena_compound_slice_seed",
            "source_variant": "bucket:shop_or_economy_misallocation",
            "failure_bucket": "shop_or_economy_misallocation",
            "failure_types": ["compound_slice_failure_seed"],
        }
        self.assertFalse(
            _can_swap_overlap_selected_row(
                candidate_payload=candidate,
                existing_payload=existing,
                selection_by_type={"slice_coverage_gap_seed": 1, "compound_slice_failure_seed": 0},
                selection_by_seed={"AAAAAAA": 1},
                selection_by_bucket={"resource_pressure_misplay": 1},
                selection_by_source={"run": 1},
                selection_by_source_type={"arena_slice_gap_seed": 1},
                selection_by_source_variant={"bucket:resource_pressure_misplay": 1},
                selection_by_overlap_key={"run|heuristic|AAAAAAA|1": 1},
                max_failures_per_type=0,
                max_failures_per_seed=4,
                max_failures_per_bucket={},
                max_failures_per_source={},
                max_failures_per_source_variant={},
                max_failures_per_overlap_key=1,
                min_failures_per_bucket={"resource_pressure_misplay": 1},
                min_failures_per_source_type={},
                min_failures_per_source_variant={
                    "bucket:resource_pressure_misplay": 1,
                    "bucket:shop_or_economy_misallocation": 1,
                },
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


class R5CompoundSourceDecouplingTest(unittest.TestCase):
    def test_compound_source_generation_does_not_require_slice_gap_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arena = root / "arena_run"
            arena.mkdir()
            out_dir = root / "out"

            episode_records = arena / "episode_records.jsonl"
            summary_table = arena / "summary_table.json"
            triage_report = root / "triage_report.json"
            config_path = root / "failure_mining.json"

            champion_row = {
                "policy_id": "heuristic_baseline",
                "seed": "AAAAAAA",
                "episode_index": 0,
                "status": "ok",
                "total_score": 320,
                "rounds_survived": 5,
                "invalid_action_rate": 0.0,
                "timeout_rate": 0.0,
                "phase": "early",
                "action_type": "PLAY",
                "resource_pressure": "high",
                "bucket_counts": {
                    "action_type": {"SHOP": 1, "PLAY": 1},
                    "slice_action_type": {"shop": 1, "play": 1},
                    "position_sensitive": {"true": 1},
                    "slice_position_sensitive": {"true": 1},
                    "resource_pressure": {"high": 1},
                    "slice_resource_pressure": {"high": 1},
                },
            }
            candidate_row = {
                "policy_id": "model_policy",
                "seed": "AAAAAAA",
                "episode_index": 0,
                "status": "ok",
                "total_score": 48,
                "rounds_survived": 1,
                "invalid_action_rate": 0.0,
                "timeout_rate": 0.0,
                "phase": "early",
                "action_type": "PLAY",
                "resource_pressure": "low",
                "bucket_counts": {
                    "action_type": {"PLAY": 1},
                    "resource_pressure": {"low": 1},
                },
            }
            with episode_records.open("w", encoding="utf-8") as fp:
                fp.write(json.dumps(champion_row) + "\n")
                fp.write(json.dumps(candidate_row) + "\n")

            summary_table.write_text(
                json.dumps(
                    [
                        {"policy_id": "model_policy", "mean_total_score": 48.0},
                        {"policy_id": "heuristic_baseline", "mean_total_score": 320.0},
                    ]
                ),
                encoding="utf-8",
            )
            triage_report.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "slice_key": "slice_resource_pressure",
                                "slice_label": "high",
                                "champion_count": 4,
                                "candidate_count": 0,
                                "metrics": {"mean_total_score_delta": -180.0},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            config_path.write_text(
                json.dumps(
                    {
                        "input": {
                            "arena_run_dir": str(arena),
                            "triage_report_json": str(triage_report),
                        },
                        "criteria": {
                            "candidate_policy": "model_policy",
                            "champion_policy": "heuristic_baseline",
                            "max_failures": 8,
                            "max_slice_gap_failures_per_source": 0,
                            "max_candidate_slice_failures_per_source": 0,
                            "max_compound_slice_failures_per_source": 2,
                            "min_failures_per_source_type": {"arena_compound_slice_seed": 1},
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = run_failure_mining(
                config_path=config_path,
                out_dir=out_dir,
                run_id="r5-compound-decouple-test",
            )

            stats = json.loads((out_dir / "failure_pack_stats.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "ok")
            self.assertGreaterEqual(stats["candidate_pool_source_type_counts"].get("arena_compound_slice_seed", 0), 1)
            self.assertGreaterEqual(stats["candidate_pool_bucket_counts"].get("shop_or_economy_misallocation", 0), 1)
            self.assertGreaterEqual(stats["by_source_type"].get("arena_compound_slice_seed", 0), 1)


if __name__ == "__main__":
    unittest.main()
