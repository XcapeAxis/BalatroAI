from __future__ import annotations

from pathlib import Path

from trainer.replay.ingest_real import ingest_real_replays
from trainer.replay.storage import write_replay_dataset


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_ingest_real_replays_from_p32_smoke() -> None:
    root = _repo_root() / "docs" / "artifacts" / "p32" / "smoke_position_fixture"
    episodes = ingest_real_replays([root], max_episodes=1)
    assert episodes
    first = episodes[0]
    assert first.steps
    assert any(step.action_type != "UNKNOWN" for step in first.steps)


def test_write_replay_dataset_summary(tmp_path: Path) -> None:
    root = _repo_root() / "docs" / "artifacts" / "p32" / "smoke_position_fixture"
    episodes = ingest_real_replays([root], max_episodes=1)
    result = write_replay_dataset(episodes, out_dir=tmp_path / "replay_out")
    assert result["status"] == "ok"
    assert (tmp_path / "replay_out" / "replay_steps.jsonl").exists()
    assert (tmp_path / "replay_out" / "replay_summary.json").exists()
