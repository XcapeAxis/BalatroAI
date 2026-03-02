from __future__ import annotations

from pathlib import Path

import pytest

from trainer.data.trajectory import (
    DecisionStep,
    load_trajectories_from_oracle_traces,
    load_trajectories_from_p13_drift_fixture,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _pick_existing_dir(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def test_load_trajectories_from_oracle_traces_smoke() -> None:
    repo = _repo_root()
    traces_root = _pick_existing_dir(
        [
            repo / "sim/tests/fixtures_runtime/oracle_p3_jokers_v1_regression",
            repo / "sim/tests/fixtures_runtime/oracle_p10_long_v1_regression",
            repo / "sim/tests/fixtures_runtime/oracle_p0_v6_regression",
        ]
    )
    if traces_root is None:
        pytest.skip("no oracle fixture runtime traces available in this checkout")

    trajectories = load_trajectories_from_oracle_traces(traces_root, max_trajectories=2, require_steps=True)
    assert trajectories, "expected at least one parsed trajectory"

    step: DecisionStep = trajectories[0].steps[0]
    assert step.action_type in {
        "PLAY",
        "DISCARD",
        "SHOP",
        "BUY",
        "USE",
        "SELL",
        "REROLL",
        "SKIP",
        "NEXT_ROUND",
        "WAIT",
        "AUTO",
        "SELECT",
        "START",
        "UNKNOWN",
    }
    assert isinstance(step.phase, str) and step.phase
    assert isinstance(step.state_hashes, dict)

    score_steps = [s for t in trajectories for s in t.steps if s.score_delta is not None]
    assert score_steps, "expected at least one step carrying score_delta"


def test_load_trajectories_from_p13_fixture_smoke() -> None:
    repo = _repo_root()
    p13_root = repo / "docs/artifacts/p13"
    if not p13_root.exists():
        pytest.skip("docs/artifacts/p13 not available")

    trajectories = load_trajectories_from_p13_drift_fixture(p13_root, max_trajectories=3, require_steps=False)
    assert isinstance(trajectories, list)
    assert trajectories, "expected at least one p13 fixture trajectory record"
    assert trajectories[0].source == "p13_drift_fixture"

    # P13 fixtures may contain zero actions; ensure schema-level fields still exist.
    t0 = trajectories[0]
    assert "actions_count" in t0.meta
    assert isinstance(t0.meta.get("phase_distribution"), dict)
