from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from demo.training_manager import TrainingManager
from demo.training_status import write_status


class _RunningProcess:
    def __init__(self, pid: int = 9876) -> None:
        self.pid = pid
        self.returncode = None

    def poll(self) -> None:
        return None


def _manager_with_temp_status(tmp_path: Path) -> TrainingManager:
    manager = TrainingManager()
    manager.status_path = tmp_path / "latest.json"
    return manager


def test_start_reuses_active_external_training(tmp_path: Path) -> None:
    manager = _manager_with_temp_status(tmp_path)
    write_status(
        {
            "status": "training",
            "status_label": "训练中",
            "message": "已有训练仍在运行。",
            "process_id": 0,
        },
        manager.status_path,
    )

    with (
        patch.object(manager, "_find_external_training_pid", return_value=4321),
        patch("demo.training_manager.subprocess.Popen") as mock_popen,
    ):
        status = manager.start(profile="smoke")

    assert status["status"] == "training"
    assert int(status["process_id"]) == 4321
    mock_popen.assert_not_called()


def test_start_launches_new_training_when_idle(tmp_path: Path) -> None:
    manager = _manager_with_temp_status(tmp_path)
    process = _RunningProcess(pid=2468)

    with patch("demo.training_manager.subprocess.Popen", return_value=process) as mock_popen:
        status = manager.start(profile="smoke")

    assert status["status"] == "queued"
    assert int(status["process_id"]) == 2468
    assert status["profile"] == "smoke"
    mock_popen.assert_called_once()
